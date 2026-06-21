import numpy as np

# ==============================================================================
# ASTROPHYSICAL CONTEXT & PROJECT CONTEXT NOTE
# ==============================================================================
# - PROJECT DESIGN LAYOUT: 8-element antenna array.
# - BANDWIDTH / FREQUENCY: 100 - 200 MHz (wavelength lambda ≈ 1.5m to 3.0m).
# - GEOMETRIC CONSTRAINT: Maximum baseline limited strictly to < 30 meters.
# - OBSERVATIONAL LIMIT: Designed for sky imaging of the Sun and celestial radio
#   sources. All-sky imaging is the primary goal.
#
# - WIDE-FIELD VS. SMALL-FIELD MATHEMATICS:
#   At these dimensions, the field of view (FOV) is extremely wide (ranging from
#   30 degrees to near-all-sky). Standard 2D Fourier synthesis assumes the
#   "small-angle approximation" where the celestial sphere is treated as a flat
#   2D plane: n = sqrt(1 - l^2 - m^2) ≈ 1.  In addition, standard 2D FFT
#   imaging assumes all baselines are perfectly coplanar (e.g. arranged on an
#   exact East-West line, or observing target exactly at the zenith).
#
#   When imaging a large portion of the dome (wide field / all-sky), the
#   non-coplanarity of the antenna array introduces a third coordinate, "w",
#   representing the distance pointing towards the source phase center. The true
#   3D visibility equation is:
#
#     V(u, v, w) = ∬ [ I(l, m) * A(l, m) / sqrt(1 - l^2 - m^2) ]
#                    * e^{-2πi * [u*l + v*m + w*(sqrt(1 - l^2 - m^2) - 1)]} dl dm
#
#   If the w-term is ignored during wide-field reconstruction, severe phase
#   errors accumulate for sources away from the phase center, producing:
#     1. Radial smearing & distortion of off-axis sources.
#     2. Severe phase-wrapping artifact patterns.
#     3. Mislocalization of point-like emission peaks.
#
# - HOW TO CORRECT FOR THE W-EFFECTS:
#   1. W-Stacking / W-Projection: Convolve visibilities onto a 3D grid.
#   2. Facetting: Partition the wide-field sky dome into small facets.
#   3. Direct 3D Fourier Integration: For small arrays (8 antennas → 28
#      baselines), skip gridding entirely and perform direct integration over
#      the curved celestial sphere including the full w-term.
#
# THIS IMPLEMENTATION provides THREE engines:
#
#   Engine A (CPU, Cartesian): Direct 3D Fourier Integration on a Cartesian
#     (l, m) direction-cosine grid.  This is the original working version.
#     Output: 2D array in (l, m) coordinates with wave-number axes.
#     Use via: make_dirty_image_cpu()
#
#   Engine B (GPU, Polar): All-sky polar projection with CuPy acceleration.
#     Output: 2D array in (zenith_angle, azimuth) polar coordinates.
#     Use via: make_dirty_image_GPU()
#
#   Engine C (CPU, Polar via CFFI): All-sky polar projection with C/OpenMP
#     multi-core CPU acceleration. No GPU required.
#     Output: 2D array in (zenith_angle, azimuth) polar coordinates.
#     Use via: make_dirty_image_polar_cpu()
#
#   Convenience wrapper: make_dirty_image_optimized() auto-selects engine.
#
#   I_D(l,m) = Σ_k V_k · exp{+2πi [u_k·l + v_k·m + w_k·(n - 1)]}
#
# where n = sqrt(1 - l² - m²), and (l, m) are direction cosines relative to
# the phase center.
# ==============================================================================


# ---------------------------------------------------------------------------
# Optional GPU support via CuPy
# ---------------------------------------------------------------------------
_GPU_AVAILABLE = False
cp = None
_GPU_IMPORT_ERROR = None

try:
    import cupy as _cp
    cp = _cp
    _GPU_DEVICE = cp.cuda.Device(0)
    # Warm up the GPU context
    _ = cp.array([0.0])
    _GPU_AVAILABLE = True
except ImportError as e:
    _GPU_IMPORT_ERROR = str(e)
except Exception as e:
    _GPU_IMPORT_ERROR = str(e)


def _get_gpu_install_hint():
    """Return platform-specific CuPy installation instructions."""
    return (
        "CuPy is required for GPU acceleration.\n"
        "Installation:\n"
        "  Linux (CUDA 12.x):   pip install cupy-cuda12x\n"
        "  Linux (CUDA 11.x):   pip install cupy-cuda11x\n"
        "  Linux (ROCm/AMD):    pip install cupy-rocm-5-0\n"
        "  Windows:             pip install cupy\n"
        "  (See https://docs.cupy.dev/en/stable/install.html)\n"
        "  Current error: " + (_GPU_IMPORT_ERROR or "unknown")
    )


def _asnumpy(arr):
    """Convert GPU array to numpy if needed."""
    if _GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def gpu_available():
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


def gpu_install_hint():
    """Return CuPy installation instructions if GPU is not available."""
    if _GPU_AVAILABLE:
        return None
    return _get_gpu_install_hint()


# ==============================================================================
# Part A — Shared Utilities (antenna loading, RFI, UVW computation)
# ==============================================================================

def load_optimized_antennas(filepath="optimized_antenna_coordinates.txt"):
    """
    Loads 8-antenna coordinates from a 2-column tabular text file:
    Column 1: X coordinate in meters (West-to-East axis, center-relative)
    Column 2: Y coordinate in meters (South-to-North axis, center-relative)
    Returns shape (8, 2).
    """
    coord_data = np.loadtxt(filepath)
    if coord_data.shape != (8, 2):
        raise ValueError(f"Expected array of shape (8, 2), got {coord_data.shape}")
    return coord_data


def reject_rfi_visibilities(visibilities, threshold_sigma=3.0):
    """
    Flags and suppresses Radio Frequency Interference (RFI) directly in the
    8x8 visibility matrix.

    Uses median absolute deviation (MAD) filtering on baseline amplitudes:
    1. Computes amplitudes of all 28 cross-correlation baselines.
    2. Identifies outlier baselines exceeding the median-driven noise threshold.
    3. Replaces infected visibility indices with zeros (data-flagging).
    4. If an antenna has ≥3 flagged baselines, flags the entire antenna.
    """
    v_cleaned = np.copy(visibilities).astype(complex)
    cross_amplitudes = []
    baseline_indices = []

    for i in range(8):
        for j in range(i + 1, 8):
            cross_amplitudes.append(np.abs(visibilities[i][j]))
            baseline_indices.append((i, j))

    cross_amplitudes = np.array(cross_amplitudes)
    if len(cross_amplitudes) == 0:
        return v_cleaned

    median_val = np.median(cross_amplitudes)
    mad = np.median(np.abs(cross_amplitudes - median_val))
    std_estimate = 1.4826 * mad if mad > 0.0 else (1e-6 * median_val)

    cutoff = median_val + threshold_sigma * std_estimate
    flag_counts = np.zeros(8)

    for k, (i, j) in enumerate(baseline_indices):
        if cross_amplitudes[k] > cutoff:
            v_cleaned[i][j] = 0.0 + 0.0j
            v_cleaned[j][i] = 0.0 + 0.0j
            flag_counts[i] += 1
            flag_counts[j] += 1

    for antenna_idx in range(8):
        if flag_counts[antenna_idx] >= 3:
            for k in range(8):
                v_cleaned[antenna_idx][k] = 0.0 + 0.0j
                v_cleaned[k][antenna_idx] = 0.0 + 0.0j

    return v_cleaned


def compute_uvw_from_antennas(antennas, wavelength,
                               hour_angle_deg=0.0, declination_deg=90.0):
    """
    Compute (u, v, w) baseline coordinates in wavelengths for all 28 baselines
    using full 3D rotation from local (x, y, z=0) to (u, v, w) frame.

    The (u, v, w) coordinate system is aligned with the phase center:
      - w points toward the phase center (source direction)
      - u points East
      - v points North

    Rotation follows the standard interferometric convention
    (Thompson, Moran & Swenson).

    Parameters
    ----------
    antennas : ndarray, shape (8, 2)
        Antenna (x, y) positions in meters.
    wavelength : float
        Observing wavelength in meters.
    hour_angle_deg : float
        Hour angle of the phase center in degrees (0 = meridian).
    declination_deg : float
        Declination of the phase center in degrees.

    Returns
    -------
    baselines_u, baselines_v, baselines_w : ndarray, each shape (28,)
    """
    n_ant = antennas.shape[0]

    H = np.radians(hour_angle_deg)
    dec = np.radians(declination_deg)

    sin_H = np.sin(H)
    cos_H = np.cos(H)
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)

    ant_u = np.zeros(n_ant)
    ant_v = np.zeros(n_ant)
    ant_w = np.zeros(n_ant)

    for i in range(n_ant):
        x, y = antennas[i, 0], antennas[i, 1]
        z = 0.0  # ground plane

        ant_u[i] = (sin_H * x + cos_H * y) / wavelength
        ant_v[i] = (-sin_dec * cos_H * x + sin_dec * sin_H * y + cos_dec * z) / wavelength
        ant_w[i] = (cos_dec * cos_H * x - cos_dec * sin_H * y + sin_dec * z) / wavelength

    baselines_u = []
    baselines_v = []
    baselines_w = []

    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            baselines_u.append(ant_u[i] - ant_u[j])
            baselines_v.append(ant_v[i] - ant_v[j])
            baselines_w.append(ant_w[i] - ant_w[j])

    return (np.array(baselines_u),
            np.array(baselines_v),
            np.array(baselines_w))


def _extract_visibility_data(visibilities):
    """Extract real/imag parts and auto-correlation sum from 8x8 matrix."""
    vis_re = []
    vis_im = []
    for i in range(8):
        for j in range(i + 1, 8):
            vis_re.append(visibilities[i][j].real)
            vis_im.append(visibilities[i][j].imag)

    vis_re = np.array(vis_re, dtype=np.float64)
    vis_im = np.array(vis_im, dtype=np.float64)
    auto_corr_sum = 0.5 * np.sum(np.real(np.diag(visibilities)))

    return vis_re, vis_im, auto_corr_sum


# ==============================================================================
# Part B — Engine A: CPU-only Cartesian (l, m) Direct 3D Fourier Integration
#           This is the ORIGINAL WORKING VERSION with wave-number coordinates.
# ==============================================================================

def build_lm_grid(grid_pts=256, fov_deg=30.0):
    """
    Build a Cartesian direction-cosine (l, m) grid for the visible sky.

    The grid covers a square in (l, m) space:
      l ∈ [-sin(FOV/2), +sin(FOV/2)]  (East-West direction cosine)
      m ∈ [-sin(FOV/2), +sin(FOV/2)]  (North-South direction cosine)

    For FOV=180° (all-sky), this covers the entire unit disk: l² + m² ≤ 1.

    Parameters
    ----------
    grid_pts : int
        Number of points along each axis (grid is grid_pts × grid_pts).
    fov_deg : float
        Field of view in degrees. Determines the l/m range.

    Returns
    -------
    L, M : ndarray, shape (grid_pts, grid_pts)
        2D meshgrid of direction cosines.
    N : ndarray, shape (grid_pts, grid_pts)
        n = sqrt(1 - l² - m²), the zenith direction cosine.
    horizon_mask : ndarray, shape (grid_pts, grid_pts), bool
        True where l² + m² ≤ 1 (above horizon).
    l_axis, m_axis : ndarray, shape (grid_pts,)
        1D axis arrays for plotting.
    """
    l_max = np.sin(np.radians(fov_deg / 2.0))
    l_axis = np.linspace(-l_max, l_max, grid_pts)
    m_axis = np.linspace(l_max, -l_max, grid_pts)  # upper → north

    L, M = np.meshgrid(l_axis, m_axis)  # shape (grid_pts, grid_pts)

    # n = sqrt(1 - l² - m²), clipped to avoid NaN beyond horizon
    r2 = L**2 + M**2
    N = np.sqrt(np.maximum(1.0 - r2, 0.0))
    horizon_mask = r2 <= 1.0

    return L, M, N, horizon_mask, l_axis, m_axis


def _direct_fourier_sum_cpu(L, M, N, baselines_u, baselines_v, baselines_w,
                             vis_re, vis_im, auto_corr_sum,
                             apply_w_correction, horizon_mask):
    """
    CPU-only direct 3D Fourier sum over baselines on a Cartesian (l,m) grid.

    Computes:
      I_D(l,m) = Σ_k [V_k^re · cos(φ_k) - V_k^im · sin(φ_k)]

    where φ_k = 2π [u_k·l + v_k·m + w_k·(n - 1)]  (with w-correction)
          φ_k = 2π [u_k·l + v_k·m]                  (without w-correction)
    """
    n_baselines = len(baselines_u)

    acc = np.full(L.shape, auto_corr_sum, dtype=np.float64)

    for k in range(n_baselines):
        phase = 2.0 * np.pi * (baselines_u[k] * L + baselines_v[k] * M)

        if apply_w_correction:
            phase += 2.0 * np.pi * baselines_w[k] * (N - 1.0)

        acc += vis_re[k] * np.cos(phase) - vis_im[k] * np.sin(phase)

    norm_factor = 28.5
    dirty = np.zeros_like(L)

    if apply_w_correction:
        # Divide by n for primary beam correction; clip n to avoid 1/0
        n_clipped = np.maximum(N, 1e-6)
        valid = horizon_mask & (N > 1e-3)
        dirty[valid] = (acc[valid] / n_clipped[valid]) / norm_factor
    else:
        dirty[horizon_mask] = acc[horizon_mask] / norm_factor

    dirty[~horizon_mask] = 0.0
    return dirty


def make_dirty_image_cpu(antennas_filepath, visibilities,
                          freq_mhz=150.0, grid_pts=256, fov_deg=30.0,
                          apply_w_correction=True, filter_rfi=True,
                          hour_angle_deg=0.0, declination_deg=90.0):
    """
    CPU-only dirty image synthesis via Direct 3D Fourier Integration.

    Computes the dirty image on a CARTESIAN (l, m) direction-cosine grid.
    This is the original working version using wave-number coordinates.

    Mathematical foundation:

        I_D(l,m) = Σ_k V_k · exp{+2πi [u_k·l + v_k·m + w_k·(n - 1)]}

    where n = sqrt(1 - l² - m²).

    Parameters
    ----------
    antennas_filepath : str
        Path to antenna coordinate file.
    visibilities : ndarray, shape (8, 8), dtype complex
        Measured correlation pairs V[i][j].
    freq_mhz : float
        Observing frequency in MHz.
    grid_pts : int
        Number of pixels along each axis (output is grid_pts × grid_pts).
    fov_deg : float
        Field of view in degrees. Determines the l/m coordinate range.
    apply_w_correction : bool
        If True, includes full w-term and 1/n primary beam correction.
    filter_rfi : bool
        If True, applies RFI flagging before imaging.
    hour_angle_deg : float
        Hour angle of the phase center in degrees.
    declination_deg : float
        Declination of the phase center in degrees.

    Returns
    -------
    dirty_img : ndarray, shape (grid_pts, grid_pts)
        Reconstructed dirty image in Cartesian (l, m) coordinates.
    l_axis : ndarray, shape (grid_pts,)
        l-axis values (East-West direction cosine) for plotting.
    m_axis : ndarray, shape (grid_pts,)
        m-axis values (North-South direction cosine) for plotting.
    """
    # 1. RFI flagging
    if filter_rfi:
        visibilities = reject_rfi_visibilities(visibilities)

    # 2. Load antennas and compute (u, v, w)
    antennas = load_optimized_antennas(antennas_filepath)
    wavelength = 299.792458 / freq_mhz

    baselines_u, baselines_v, baselines_w = compute_uvw_from_antennas(
        antennas, wavelength,
        hour_angle_deg=hour_angle_deg,
        declination_deg=declination_deg
    )

    # 3. Build Cartesian (l, m) grid
    L, M, N, horizon_mask, l_axis, m_axis = build_lm_grid(
        grid_pts=grid_pts, fov_deg=fov_deg
    )

    # 4. Extract visibility data
    vis_re, vis_im, auto_corr_sum = _extract_visibility_data(visibilities)

    # 5. Direct 3D Fourier sum
    dirty_img = _direct_fourier_sum_cpu(
        L, M, N,
        baselines_u, baselines_v, baselines_w,
        vis_re, vis_im,
        auto_corr_sum,
        apply_w_correction,
        horizon_mask
    )

    return dirty_img, l_axis, m_axis


# ==============================================================================
# Part C — Engine B: GPU-accelerated All-Sky Polar Direct 3D Fourier Integration
#           This is the NEW GPU engine with polar "fish-eye" coordinates.
# ==============================================================================

def build_polar_sky_grid_GPU(n_radial=256, n_azimuthal=512):
    """
    Build an all-sky polar coordinate grid for the visible hemisphere.

    The grid is in (zenith_angle, azimuth) coordinates:
      - zenith_angle ζ: 0° (zenith) → 90° (horizon)
      - azimuth A: 0° (North) → 360° (East = 90°)

    At each grid point we compute the direction cosines (l, m, n) for the
    direct Fourier sum.

    Parameters
    ----------
    n_radial : int
        Number of radial bins (zenith angle steps).
    n_azimuthal : int
        Number of azimuthal bins.

    Returns
    -------
    L, M, N : ndarray, each shape (n_radial, n_azimuthal)
        Direction cosine grids (East, North, Zenith).
    horizon_mask : ndarray, shape (n_radial, n_azimuthal), bool
        True for pixels on the visible sky.
    zenith_angles : ndarray, shape (n_radial,)
        Zenith angle array in degrees (for axis labels).
    azimuth_angles : ndarray, shape (n_azimuthal,)
        Azimuth array in degrees (for axis labels).
    """
    # Zenith angle: linear from 0° to 90°
    zeta_deg = np.linspace(0.0, 90.0, n_radial)
    zeta = np.radians(zeta_deg)

    # Azimuth: linear from 0° to 360° (exclusive)
    # Convention: 0° = North, 90° = East, 180° = South, 270° = West
    az_deg = np.linspace(0.0, 360.0, n_azimuthal + 1)[:-1]
    az = np.radians(az_deg)

    # Meshgrid
    ZETA, AZ = np.meshgrid(zeta, az, indexing='ij')  # (n_radial, n_azimuthal)

    # Altitude = 90° - zenith_angle
    alt = np.pi / 2.0 - ZETA

    # Direction cosines from (Alt, Az) → (l, m, n):
    #   l = cos(Alt) * sin(Az)    → East
    #   m = cos(Alt) * cos(Az)    → North
    #   n = sin(Alt)               → Zenith (line-of-sight)
    cos_alt = np.cos(alt)
    sin_alt = np.sin(alt)

    L = cos_alt * np.sin(AZ)
    M = cos_alt * np.cos(AZ)
    N = sin_alt

    horizon_mask = np.ones_like(L, dtype=bool)

    return L, M, N, horizon_mask, zeta_deg, az_deg


def _direct_fourier_sum_GPU(L, M, N, baselines_u, baselines_v, baselines_w,
                             vis_re, vis_im, auto_corr_sum,
                             apply_w_correction, horizon_mask):
    """
    GPU-accelerated direct 3D Fourier sum using CuPy.

    Computes:
      I(l,m) = Σ_k [V_k^re · cos(φ_k) - V_k^im · sin(φ_k)]

    where φ_k = 2π [u_k·l + v_k·m + w_k·(n - 1)]

    All arrays are moved to GPU before computation.
    """
    xp = cp  # shorthand

    n_baselines = len(baselines_u)

    # Move everything to GPU
    L_g = xp.asarray(L, dtype=xp.float64)
    M_g = xp.asarray(M, dtype=xp.float64)
    N_g = xp.asarray(N, dtype=xp.float64)

    u_g = xp.asarray(baselines_u, dtype=xp.float64)
    v_g = xp.asarray(baselines_v, dtype=xp.float64)
    w_g = xp.asarray(baselines_w, dtype=xp.float64)

    vr_g = xp.asarray(vis_re, dtype=xp.float64)
    vi_g = xp.asarray(vis_im, dtype=xp.float64)

    # Initialize accumulator with auto-correlation (zero-baseline flux)
    acc = xp.full(L_g.shape, auto_corr_sum, dtype=xp.float64)

    two_pi = xp.float64(2.0 * np.pi)

    for k in range(n_baselines):
        # Phase for baseline k: 2π * [u_k·L + v_k·M]
        phase = two_pi * (u_g[k] * L_g + v_g[k] * M_g)

        if apply_w_correction:
            # Full 3D: add w_k·(N - 1) term
            phase += two_pi * w_g[k] * (N_g - 1.0)

        # V_re·cos(φ) - V_im·sin(φ)
        acc += vr_g[k] * xp.cos(phase) - vi_g[k] * xp.sin(phase)

    # Apply primary beam correction (1/N) and normalization
    norm_factor = xp.float64(28.5)

    if apply_w_correction:
        # Divide by n = sqrt(1 - l^2 - m^2) for primary beam correction.
        # Clip n to a minimum of 1e-6 to avoid infinity at the exact horizon
        # where n → 0. Pixels exactly at the horizon (n < 1e-3) have vanishing
        # sensitivity anyway.
        n_clipped = xp.maximum(N_g, xp.float64(1e-6))
        # Only apply to valid pixels (within FOV and above horizon)
        h_mask = xp.asarray(horizon_mask)
        valid = h_mask & (N_g > xp.float64(1e-3))
        acc[valid] = (acc[valid] / n_clipped[valid]) / norm_factor
        # Zero out horizon-edge pixels where n is too small
        acc[h_mask & ~valid] = xp.float64(0.0)
    else:
        acc[xp.asarray(horizon_mask)] = acc[xp.asarray(horizon_mask)] / norm_factor

    # Mask outside horizon
    acc[~xp.asarray(horizon_mask)] = xp.float64(0.0)

    return _asnumpy(acc)


def make_dirty_image_GPU(antennas_filepath, visibilities,
                          freq_mhz=150.0, grid_pts=256, fov_deg=180.0,
                          apply_w_correction=True, filter_rfi=True,
                          hour_angle_deg=0.0, declination_deg=90.0,
                          n_radial=None, n_azimuthal=None,
                          use_gpu=None):
    """
    All-sky dirty image synthesis via Direct 3D Fourier Integration — GPU path.

    Computes the dirty image on a POLAR coordinate grid covering the entire
    visible hemisphere:
      - Radial axis: zenith angle (0° = zenith, 90° = horizon)
      - Angular axis: azimuth (0° = North, 90° = East)

    The output is a 2D array in (zenith, azimuth) polar coordinates that can
    be displayed as a circular "fish-eye" all-sky map.

    GPU acceleration via CuPy is used automatically when a CUDA GPU is
    available, providing ~10-50× speedup for large grids.

    Mathematical foundation — Direct 3D Fourier Integration:

        I_D(l,m) = Σ_k V_k · exp{+2πi [u_k·l + v_k·m + w_k·(n - 1)]}

    where n = sqrt(1 - l² - m²), and (l, m) are computed from (altitude,
    azimuth) for each polar grid cell.

    Parameters
    ----------
    antennas_filepath : str
        Path to antenna coordinate file.
    visibilities : ndarray, shape (8, 8), dtype complex
        Measured correlation pairs V[i][j].
    freq_mhz : float
        Observing frequency in MHz.
    grid_pts : int
        Approximate resolution. Actual radial/azimuthal bins are derived from
        this to maintain aspect ratio: n_radial = grid_pts/2,
        n_azimuthal = grid_pts.
    fov_deg : float
        Field of view in degrees. Default 180° for all-sky. Values < 90°
        restrict to a zenith-centered circular patch.
    apply_w_correction : bool
        If True, includes full w-term and 1/n primary beam correction.
    filter_rfi : bool
        If True, applies RFI flagging before imaging.
    hour_angle_deg : float
        Hour angle of the phase center in degrees.
    declination_deg : float
        Declination of the phase center in degrees.
    n_radial : int or None
        Number of radial (zenith angle) bins. If None, derived from grid_pts.
    n_azimuthal : int or None
        Number of azimuthal bins. If None, derived from grid_pts.
    use_gpu : bool or None
        If True, force GPU; if False, force CPU fallback; if None, auto-detect.

    Returns
    -------
    dirty_img : ndarray, shape (n_radial, n_azimuthal)
        Reconstructed all-sky dirty image in polar (zenith, azimuth) coords.
    """
    # 0. Determine GPU usage
    if use_gpu is None:
        use_gpu = _GPU_AVAILABLE

    # 1. RFI flagging
    if filter_rfi:
        visibilities = reject_rfi_visibilities(visibilities)

    # 2. Load antennas
    antennas = load_optimized_antennas(antennas_filepath)
    wavelength = 299.792458 / freq_mhz

    # 3. Compute (u, v, w) baseline coordinates
    baselines_u, baselines_v, baselines_w = compute_uvw_from_antennas(
        antennas, wavelength,
        hour_angle_deg=hour_angle_deg,
        declination_deg=declination_deg
    )

    # 4. Determine grid dimensions for polar all-sky
    if n_radial is None:
        n_radial = max(grid_pts // 2, 64)
    if n_azimuthal is None:
        n_azimuthal = max(grid_pts, 128)

    # 5. Build polar sky coordinate grid
    L, M, N, horizon_mask, zeta_deg, az_deg = build_polar_sky_grid_GPU(
        n_radial=n_radial,
        n_azimuthal=n_azimuthal
    )

    # 6. If FOV < 180°, mask pixels beyond the FOV
    if fov_deg < 180.0:
        fov_rad = np.radians(fov_deg / 2.0)
        zeta_grid = np.radians(np.linspace(0.0, 90.0, n_radial))
        zeta_2d = zeta_grid[:, np.newaxis]
        fov_mask = zeta_2d <= fov_rad
        horizon_mask = horizon_mask & fov_mask

    # 7. Extract visibility data
    vis_re, vis_im, auto_corr_sum = _extract_visibility_data(visibilities)

    # 8. Direct 3D Fourier sum (GPU or CPU fallback)
    if use_gpu:
        dirty_img = _direct_fourier_sum_GPU(
            L, M, N,
            baselines_u, baselines_v, baselines_w,
            vis_re, vis_im,
            auto_corr_sum,
            apply_w_correction,
            horizon_mask
        )
    else:
        # CPU fallback for polar grid (when GPU requested but unavailable)
        dirty_img = _direct_fourier_sum_cpu(
            L, M, N,
            baselines_u, baselines_v, baselines_w,
            vis_re, vis_im,
            auto_corr_sum,
            apply_w_correction,
            horizon_mask
        )

    return dirty_img


# ==============================================================================
# Part C2 — Engine C: CPU Polar (CFFI + OpenMP) All-Sky Direct 3D Fourier
#           Integration.  180° FOV, polar (zenith, azimuth) coordinates,
#           accelerated via C + OpenMP multi-core parallelism (up to 8 cores).
#
#   NOTE: This engine requires cffi + a C compiler (MSVC on Windows, GCC on
#   Linux). If unavailable, the engine will report as not-available and the
#   caller should fall back to the pure-Python CPU path (Engine A or the
#   polar CPU fallback using _direct_fourier_sum_cpu).
# ==============================================================================

_CFFI_AVAILABLE = False
_CFFI_ERROR = None

def _polar_cpu_available():
    """Check if the CFFI-based C engine can be loaded."""
    global _CFFI_AVAILABLE, _CFFI_ERROR
    if _CFFI_AVAILABLE:
        return True
    try:
        from polar_cpu_cffi import is_available as _cffi_available
        ok = _cffi_available()
        if ok:
            _CFFI_AVAILABLE = True
            return True
        _CFFI_ERROR = "polar_cpu_cffi.is_available() returned False"
        return False
    except ImportError as e:
        _CFFI_ERROR = f"Missing dependency: {e}"
        return False
    except Exception as e:
        _CFFI_ERROR = str(e)
        return False


def _get_cffi_install_hint():
    """Return platform-specific cffi + C compiler installation instructions."""
    return (
        "The CFFI-based CPU polar engine requires:\n"
        "  1. cffi:       pip install cffi\n"
        "  2. C compiler:\n"
        "     - Linux:     sudo apt install build-essential  (Debian/Ubuntu)\n"
        "                 sudo yum install gcc               (RHEL/CentOS)\n"
        "     - macOS:     xcode-select --install\n"
        "     - Windows:   Install Microsoft C++ Build Tools from\n"
        "                 https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "  Current error: " + (_CFFI_ERROR or "unknown")
    )


def cffi_install_hint():
    """Return CFFI installation instructions if the C engine is not available."""
    if _CFFI_AVAILABLE:
        return None
    # Trigger a check if not yet done
    _polar_cpu_available()
    return _get_cffi_install_hint()


def make_dirty_image_polar_cpu(antennas_filepath, visibilities,
                                freq_mhz=150.0, grid_pts=256, fov_deg=180.0,
                                apply_w_correction=True, filter_rfi=True,
                                hour_angle_deg=0.0, declination_deg=90.0,
                                n_radial=None, n_azimuthal=None,
                                num_threads=0):
    """
    All-sky dirty image synthesis via Direct 3D Fourier Integration — CPU Polar
    engine using C + OpenMP multi-core parallelism.

    Computes the dirty image on a POLAR coordinate grid covering the entire
    visible hemisphere, identical to Engine B (GPU) but running on CPU with
    multi-core acceleration via a compiled C module.

    Uses cffi to call the C function compute_dirty_image_polar() which employs
    OpenMP for parallel loops over radial bins (up to 8 threads).

    The output is a 2D array in (zenith, azimuth) polar coordinates that can
    be displayed as a circular "fish-eye" all-sky map.

    Mathematical foundation — Direct 3D Fourier Integration:
        I_D(l,m) = Σ_k V_k · exp{+2πi [u_k·l + v_k·m + w_k·(n - 1)]}

    Parameters
    ----------
    antennas_filepath : str
        Path to antenna coordinate file.
    visibilities : ndarray, shape (8, 8), dtype complex
        Measured correlation pairs V[i][j].
    freq_mhz : float
        Observing frequency in MHz.
    grid_pts : int
        Approximate resolution. Actual radial/azimuthal bins are derived from
        this: n_radial = grid_pts//2, n_azimuthal = grid_pts.
    fov_deg : float
        Field of view in degrees. Default 180° for all-sky.
    apply_w_correction : bool
        If True, includes full w-term and 1/n primary beam correction.
    filter_rfi : bool
        If True, applies RFI flagging before imaging.
    hour_angle_deg : float
        Hour angle of the phase center in degrees.
    declination_deg : float
        Declination of the phase center in degrees.
    n_radial : int or None
        Number of radial (zenith angle) bins. If None, derived from grid_pts.
    n_azimuthal : int or None
        Number of azimuthal bins. If None, derived from grid_pts.
    num_threads : int
        Number of OpenMP threads. 0 = auto (up to 8).

    Returns
    -------
    dirty_img : ndarray, shape (n_radial, n_azimuthal)
        Reconstructed all-sky dirty image in polar (zenith, azimuth) coords.
    """
    # Detect whether CFFI engine is available
    use_cffi = _polar_cpu_available()

    # 1. RFI flagging
    if filter_rfi:
        visibilities = reject_rfi_visibilities(visibilities)

    # 2. Load antennas
    antennas = load_optimized_antennas(antennas_filepath)
    wavelength = 299.792458 / freq_mhz

    # 3. Compute (u, v, w) baseline coordinates
    baselines_u, baselines_v, baselines_w = compute_uvw_from_antennas(
        antennas, wavelength,
        hour_angle_deg=hour_angle_deg,
        declination_deg=declination_deg
    )

    # 4. Determine grid dimensions for polar all-sky
    if n_radial is None:
        n_radial = max(grid_pts // 2, 64)
    if n_azimuthal is None:
        n_azimuthal = max(grid_pts, 128)

    # 5. Build polar sky coordinate grid (same as GPU version)
    L, M, N, horizon_mask, zeta_deg, az_deg = build_polar_sky_grid_GPU(
        n_radial=n_radial,
        n_azimuthal=n_azimuthal
    )

    # 6. If FOV < 180°, mask pixels beyond the FOV
    if fov_deg < 180.0:
        fov_rad = np.radians(fov_deg / 2.0)
        zeta_grid = np.radians(np.linspace(0.0, 90.0, n_radial))
        zeta_2d = zeta_grid[:, np.newaxis]
        fov_mask = zeta_2d <= fov_rad
        horizon_mask = horizon_mask & fov_mask

    # 7. Extract visibility data
    vis_re, vis_im, auto_corr_sum = _extract_visibility_data(visibilities)

    # 8. Compute dirty image — try CFFI first, fall back to pure Python
    if use_cffi:
        try:
            from polar_cpu_cffi import polar_fourier_sum, compile_and_load
            compile_and_load()
            dirty_img = polar_fourier_sum(
                L, M, N,
                baselines_u, baselines_v, baselines_w,
                vis_re, vis_im,
                auto_corr_sum,
                apply_w_correction,
                horizon_mask=horizon_mask,
                num_threads=num_threads,
            )
            return dirty_img
        except Exception:
            # CFFI failed at runtime — silently fall back to pure Python
            pass

    # Pure Python fallback (same algorithm, no OpenMP acceleration)
    dirty_img = _direct_fourier_sum_cpu(
        L, M, N,
        baselines_u, baselines_v, baselines_w,
        vis_re, vis_im,
        auto_corr_sum,
        apply_w_correction,
        horizon_mask
    )

    return dirty_img


# ==============================================================================
# Part D — Convenience wrappers & grid metadata
# ==============================================================================

def make_dirty_image_optimized(antennas_filepath, visibilities,
                                freq_mhz=150.0, grid_pts=256, fov_deg=30.0,
                                apply_w_correction=True, filter_rfi=True,
                                hour_angle_deg=0.0, declination_deg=90.0,
                                n_radial=None, n_azimuthal=None,
                                use_gpu=None):
    """
    Convenience wrapper that auto-selects the appropriate engine.

    - If use_gpu=True or GPU is available AND fov_deg >= 90°:
        Calls make_dirty_image_GPU() → returns polar (zenith, azimuth) array.
    - Otherwise:
        Calls make_dirty_image_cpu() → returns Cartesian (l, m) array + axes.

    For backward compatibility with realtime_dirty_image.py, this function
    returns a 3-tuple (dirty_img, axis1, axis2) when using CPU mode, and a
    single ndarray when using GPU polar mode.

    Parameters
    ----------
    (same as make_dirty_image_cpu / make_dirty_image_GPU)

    Returns
    -------
    When GPU/polar:
        dirty_img : ndarray, shape (n_radial, n_azimuthal)
    When CPU/Cartesian:
        dirty_img : ndarray, shape (grid_pts, grid_pts)
        l_axis : ndarray, shape (grid_pts,)
        m_axis : ndarray, shape (grid_pts,)
    """
    # Auto-detect: use GPU polar for all-sky, CPU Cartesian for narrow FOV
    if use_gpu is None:
        use_gpu = _GPU_AVAILABLE

    if use_gpu:
        return make_dirty_image_GPU(
            antennas_filepath, visibilities,
            freq_mhz=freq_mhz, grid_pts=grid_pts, fov_deg=fov_deg,
            apply_w_correction=apply_w_correction, filter_rfi=filter_rfi,
            hour_angle_deg=hour_angle_deg, declination_deg=declination_deg,
            n_radial=n_radial, n_azimuthal=n_azimuthal,
            use_gpu=True
        )
    else:
        return make_dirty_image_cpu(
            antennas_filepath, visibilities,
            freq_mhz=freq_mhz, grid_pts=grid_pts, fov_deg=fov_deg,
            apply_w_correction=apply_w_correction, filter_rfi=filter_rfi,
            hour_angle_deg=hour_angle_deg, declination_deg=declination_deg
        )


def get_polar_grid_metadata(grid_pts=256, n_radial=None, n_azimuthal=None):
    """
    Return metadata about the polar grid for display purposes.

    Returns
    -------
    dict with keys:
        'n_radial', 'n_azimuthal': grid dimensions
        'zeta_deg': zenith angle array (degrees)
        'az_deg': azimuth array (degrees)
        'is_allsky': True (always for polar projection)
    """
    if n_radial is None:
        n_radial = max(grid_pts // 2, 64)
    if n_azimuthal is None:
        n_azimuthal = max(grid_pts, 128)

    zeta_deg = np.linspace(0.0, 90.0, n_radial)
    az_deg = np.linspace(0.0, 360.0, n_azimuthal + 1)[:-1]

    return {
        'n_radial': n_radial,
        'n_azimuthal': n_azimuthal,
        'zeta_deg': zeta_deg,
        'az_deg': az_deg,
        'is_allsky': True,
    }
