import numpy as np
from pathlib import Path

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
                          hour_angle_deg=0.0, declination_deg=90.0,
                          subtract_baseline=False):
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
    subtract_baseline : bool
        If True, compute the baseline dirty image (uniform-sky response) and
        apply baseline correction: corrected = dirty / baseline - 1.
        This removes the edge-brightening artifact inherent to wide-field
        polar imaging, showing only features that exceed the uniform-sky level.

    Returns
    -------
    dirty_img : ndarray, shape (grid_pts, grid_pts)
        Reconstructed dirty image in Cartesian (l, m) coordinates.
        If subtract_baseline=True, returns baseline-corrected image.
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

    # 6. Optional baseline correction
    if subtract_baseline:
        baseline = _cached_baseline(
            L, M, N, horizon_mask,
            baselines_u, baselines_v, baselines_w,
            apply_w_correction
        )
        dirty_img = apply_baseline_correction(dirty_img, baseline,
                                               method='relative_excess')

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
                          use_gpu=None, subtract_baseline=False):
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
    subtract_baseline : bool
        If True, apply baseline correction after GPU/CPU imaging.

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

    # 9. Optional baseline correction (CPU-side, applies to both GPU and CPU results)
    if subtract_baseline:
        baseline = _cached_baseline(
            L, M, N, horizon_mask,
            baselines_u, baselines_v, baselines_w,
            apply_w_correction,
            n_radial=n_radial, n_azimuthal=n_azimuthal
        )
        dirty_img = apply_baseline_correction(dirty_img, baseline,
                                               method='relative_excess')

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
                                num_threads=0, subtract_baseline=False):
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
    subtract_baseline : bool
        If True, apply baseline correction after imaging.

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
    dirty_img = None
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
        except Exception:
            # CFFI failed at runtime — silently fall back to pure Python
            pass

    if dirty_img is None:
        # Pure Python fallback
        dirty_img = _direct_fourier_sum_cpu(
            L, M, N,
            baselines_u, baselines_v, baselines_w,
            vis_re, vis_im,
            auto_corr_sum,
            apply_w_correction,
            horizon_mask
        )

    # 9. Optional baseline correction
    if subtract_baseline:
        baseline = _cached_baseline(
            L, M, N, horizon_mask,
            baselines_u, baselines_v, baselines_w,
            apply_w_correction,
            n_radial=n_radial, n_azimuthal=n_azimuthal
        )
        dirty_img = apply_baseline_correction(dirty_img, baseline,
                                               method='relative_excess')

    return dirty_img


# ==============================================================================
# Part D — Convenience wrappers & grid metadata
# ==============================================================================

def make_dirty_image_optimized(antennas_filepath, visibilities,
                                freq_mhz=150.0, grid_pts=256, fov_deg=30.0,
                                apply_w_correction=True, filter_rfi=True,
                                hour_angle_deg=0.0, declination_deg=90.0,
                                n_radial=None, n_azimuthal=None,
                                use_gpu=None, subtract_baseline=False):
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
            use_gpu=True, subtract_baseline=subtract_baseline
        )
    else:
        return make_dirty_image_cpu(
            antennas_filepath, visibilities,
            freq_mhz=freq_mhz, grid_pts=grid_pts, fov_deg=fov_deg,
            apply_w_correction=apply_w_correction, filter_rfi=filter_rfi,
            hour_angle_deg=hour_angle_deg, declination_deg=declination_deg,
            subtract_baseline=subtract_baseline
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


def compute_uv_tracks(antennas, freq_low_mhz=100.0, freq_high_mhz=200.0,
                      n_samples=20, hour_angle_deg=0.0, declination_deg=90.0):
    """
    Compute broadband UV tracks for all 28 baselines across a frequency range.

    For each baseline (i,j), the (u,v) coordinates scale linearly with frequency:
        u(f) = baseline_x * f / c
        v(f) = baseline_y * f / c

    This function samples the frequency range and returns UV coordinate arrays
    suitable for drawing line segments showing the full frequency-dependent
    uv coverage.

    Parameters
    ----------
    antennas : ndarray, shape (8, 2)
        Antenna (x, y) positions in meters.
    freq_low_mhz, freq_high_mhz : float
        Frequency range in MHz (e.g. 100-200 MHz).
    n_samples : int
        Number of frequency samples (endpoints are always included).
    hour_angle_deg : float
        Hour angle in degrees.
    declination_deg : float
        Declination in degrees.

    Returns
    -------
    uv_lines : ndarray, shape (n_baselines, n_samples, 2)
        (u, v) coordinates for each baseline at each frequency sample.
        n_baselines = 28 (all unique pairs of 8 antennas).
    freq_samples_mhz : ndarray, shape (n_samples,)
        Sampled frequencies in MHz.
    uv_center : ndarray, shape (28, 2)
        (u, v) at the center frequency (mid-point), for reference markers.
    """
    n_ant = antennas.shape[0]
    freq_samples = np.linspace(freq_low_mhz, freq_high_mhz, n_samples)

    # Compute (u,v) at each frequency for all 28 baselines
    n_baselines = n_ant * (n_ant - 1) // 2
    uv_lines = np.zeros((n_baselines, n_samples, 2))

    for k, f_mhz in enumerate(freq_samples):
        wavelength = 299.792458 / f_mhz
        bu, bv, _ = compute_uvw_from_antennas(
            antennas, wavelength,
            hour_angle_deg=hour_angle_deg,
            declination_deg=declination_deg
        )
        uv_lines[:, k, 0] = bu
        uv_lines[:, k, 1] = bv

    # Center-frequency uv for markers
    freq_center = (freq_low_mhz + freq_high_mhz) / 2.0
    wavelength_center = 299.792458 / freq_center
    bu_c, bv_c, _ = compute_uvw_from_antennas(
        antennas, wavelength_center,
        hour_angle_deg=hour_angle_deg,
        declination_deg=declination_deg
    )
    uv_center = np.column_stack([bu_c, bv_c])

    return uv_lines, freq_samples, uv_center


def read_frequency_range_from_data(data_dir, center_freq_mhz=150.0):
    """
    Read actual frequency bins from a sample CSV file in data_dir.

    CSV files contain a 'frequency_hz' column with baseband frequencies.
    The actual sky frequency is center_freq_mhz + frequency_hz/1e6.

    Returns
    -------
    (freq_low_sky_mhz, freq_high_sky_mhz) or (None, None) if no data found.
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("correlation_*.csv"))
    if not csv_files:
        return None, None

    try:
        import pandas as pd
        df = pd.read_csv(csv_files[0], comment='#', nrows=5000)
        if 'frequency_hz' in df.columns:
            f_base = df['frequency_hz'].values
            # Real frequency = center + baseband offset
            f_sky = center_freq_mhz + f_base / 1e6
            return float(np.min(f_sky)), float(np.max(f_sky))
    except Exception:
        pass
    return None, None


# ==============================================================================
# Broadband Visibility Loading & Imaging
# ==============================================================================
def _get_channel_indices(pair_name):
    """返回 (row, col) 0-based 索引，从 pair 名称如 'CH1xCH2' 解析"""
    import re
    if '_AUTO' in pair_name:
        m = re.search(r'CH(\d+)_AUTO', pair_name)
        if m:
            ch = int(m.group(1)) - 1
            return ch, ch
    elif 'x' in pair_name:
        m = re.search(r'CH(\d+)xCH(\d+)', pair_name)
        if m:
            a, b = int(m.group(1)) - 1, int(m.group(2)) - 1
            return max(a, b), min(a, b)
    return None, None


def load_broadband_visibilities(file_map, center_freq_mhz=150.0,
                                n_channels=40, max_bins=4096):
    """
    Read visibilities at multiple frequency channels from a set of CSV files.

    Parameters
    ----------
    file_map : dict
        {pair_name: filepath} mapping for one frame.
    center_freq_mhz : float
        Center sky frequency in MHz.
    n_channels : int
        Number of evenly-spaced frequency channels to use.
    max_bins : int
        Maximum number of frequency bins in the CSV files.

    Returns
    -------
    freq_sky_mhz : ndarray, shape (n_channels,)
        Sky frequency for each channel.
    vis_all : ndarray, shape (n_channels, 8, 8), dtype complex128
        Visibility matrix (8x8) for each frequency channel.
    """
    import pandas as pd

    if n_channels >= max_bins:
        n_channels = max_bins

    # Evenly-spaced bin indices
    bin_indices = np.linspace(0, max_bins - 1, n_channels, dtype=int)

    # Pre-allocate: (n_channels, 8 ant, 8 ant)
    vis_all = np.zeros((n_channels, 8, 8), dtype=np.complex128)
    freq_sky_mhz = np.zeros(n_channels, dtype=np.float64)

    # Cache CSV data to avoid repeated I/O
    csv_cache = {}
    for pair_name, filepath in file_map.items():
        try:
            df = pd.read_csv(filepath, comment='#', usecols=['frequency_index',
                                                              'real_part',
                                                              'imag_part'])
            csv_cache[pair_name] = df
        except Exception:
            pass

    # Also read frequency_hz from first file to get actual sky frequencies
    first_file = next(iter(file_map.values()))
    try:
        df_freq = pd.read_csv(first_file, comment='#', usecols=['frequency_hz'])
        f_base_hz = df_freq['frequency_hz'].values
        for ci, bi in enumerate(bin_indices):
            if bi < len(f_base_hz):
                freq_sky_mhz[ci] = center_freq_mhz + f_base_hz[bi] / 1e6
            else:
                freq_sky_mhz[ci] = center_freq_mhz
    except Exception:
        freq_sky_mhz[:] = center_freq_mhz

    # Fill visibility matrices
    for ci, bi in enumerate(bin_indices):
        for pair_name, filepath in file_map.items():
            row_idx, col_idx = _get_channel_indices(pair_name)
            if row_idx is None or col_idx is None:
                continue
            df = csv_cache.get(pair_name)
            if df is None:
                continue
            try:
                actual_idx = df['frequency_index'].values[bi]
                if actual_idx == bi:
                    r = df.iloc[bi]
                    val = complex(r['real_part'], r['imag_part'])
                    vis_all[ci, row_idx, col_idx] = val
                    if row_idx != col_idx:
                        vis_all[ci, col_idx, row_idx] = val.conjugate()
            except (IndexError, KeyError):
                pass

    return freq_sky_mhz, vis_all


def make_dirty_image_broadband_cpu(antennas_filepath, file_map,
                                   center_freq_mhz=150.0,
                                   grid_pts=256, fov_deg=30.0,
                                   apply_w_correction=True,
                                   filter_rfi=True,
                                   hour_angle_deg=0.0, declination_deg=90.0,
                                   n_channels=40, max_bins=4096,
                                   verbose=True,
                                   subtract_baseline=False):
    """
    Broadband dirty image via multi-frequency direct 3D Fourier integration
    on a Cartesian (l, m) grid.

    Instead of using a single frequency bin, this integrates over N evenly-spaced
    frequency channels across the full baseband, properly accounting for
    frequency-dependent (u,v,w) coordinates and visibilities.

    I_broad(l,m) = 1/N Σ_c I_{fc}(l,m)

    Parameters
    ----------
    subtract_baseline : bool
        If True, compute a per-channel baseline dirty image, average, and apply
        correction: corrected = dirty / baseline_avg - 1.

    """
    # 1. Load broadband visibilities
    freq_sky_mhz, vis_all = load_broadband_visibilities(
        file_map, center_freq_mhz=center_freq_mhz,
        n_channels=n_channels, max_bins=max_bins
    )

    if verbose:
        lo, hi = np.min(freq_sky_mhz), np.max(freq_sky_mhz)
        bw_eff = hi - lo
        print(f"  Broadband: {n_channels} channels, "
              f"{lo:.1f}–{hi:.1f} MHz "
              f"(Δf = {bw_eff:.1f} MHz)")

    # 2. Load antennas
    antennas = load_optimized_antennas(antennas_filepath)

    # 3. Build Cartesian (l, m) grid (same for all channels)
    L, M, N, horizon_mask, l_axis, m_axis = build_lm_grid(
        grid_pts=grid_pts, fov_deg=fov_deg
    )

    # 4. Integrate over frequency
    dirty = np.zeros((grid_pts, grid_pts), dtype=np.float64)
    baseline_broad = np.zeros((grid_pts, grid_pts), dtype=np.float64) if subtract_baseline else None

    for ci in range(n_channels):
        vis = vis_all[ci]
        f_mhz = freq_sky_mhz[ci]

        # RFI flagging (per channel)
        if filter_rfi:
            vis = reject_rfi_visibilities(vis)

        # Compute (u,v,w) at this channel's frequency
        wavelength = 299.792458 / f_mhz
        bu, bv, bw = compute_uvw_from_antennas(
            antennas, wavelength,
            hour_angle_deg=hour_angle_deg,
            declination_deg=declination_deg
        )

        # Extract visibility data
        vis_re, vis_im, auto_corr_sum = _extract_visibility_data(vis)

        # Fourier sum for this channel
        channel_dirty = _direct_fourier_sum_cpu(
            L, M, N, bu, bv, bw,
            vis_re, vis_im, auto_corr_sum,
            apply_w_correction, horizon_mask
        )

        dirty += channel_dirty

        # Per-channel baseline for broadband correction
        if subtract_baseline:
            bl = _cached_baseline(
                L, M, N, horizon_mask,
                bu, bv, bw,
                apply_w_correction
            )
            baseline_broad += bl

    dirty /= n_channels

    # Apply broadband baseline correction
    if subtract_baseline:
        baseline_broad /= n_channels
        dirty = apply_baseline_correction(dirty, baseline_broad,
                                           method='relative_excess')

    return dirty, l_axis, m_axis, freq_sky_mhz


def make_dirty_image_broadband_polar_cpu(antennas_filepath, file_map,
                                         center_freq_mhz=150.0,
                                         grid_pts=256, fov_deg=180.0,
                                         apply_w_correction=True,
                                         filter_rfi=True,
                                         hour_angle_deg=0.0,
                                         declination_deg=90.0,
                                         n_channels=40, max_bins=4096,
                                         n_radial=None, n_azimuthal=None,
                                         num_threads=0, verbose=True,
                                         subtract_baseline=False):
    """
    Broadband all-sky dirty image via multi-frequency direct 3D Fourier
    integration on a polar (zenith, azimuth) grid.

    Uses CFFI C module with OpenMP for parallel frequency integration.

    Parameters
    ----------
    subtract_baseline : bool
        If True, compute per-channel baseline and apply broadband correction.
    """
    # 1. Load broadband visibilities
    freq_sky_mhz, vis_all = load_broadband_visibilities(
        file_map, center_freq_mhz=center_freq_mhz,
        n_channels=n_channels, max_bins=max_bins
    )

    if verbose:
        lo, hi = np.min(freq_sky_mhz), np.max(freq_sky_mhz)
        bw_eff = hi - lo
        print(f"  Broadband: {n_channels} channels, "
              f"{lo:.1f}–{hi:.1f} MHz "
              f"(Δf = {bw_eff:.1f} MHz)")

    # 2. Load antennas
    antennas = load_optimized_antennas(antennas_filepath)

    # 3. Grid dimensions
    if n_radial is None:
        n_radial = max(grid_pts // 2, 64)
    if n_azimuthal is None:
        n_azimuthal = max(grid_pts, 128)

    # 4. Build polar sky grid (same for all channels)
    L, M, N, horizon_mask, _, _ = build_polar_sky_grid_GPU(
        n_radial=n_radial, n_azimuthal=n_azimuthal
    )

    if fov_deg < 180.0:
        fov_rad = np.radians(fov_deg / 2.0)
        zeta_grid = np.radians(np.linspace(0.0, 90.0, n_radial))
        zeta_2d = zeta_grid[:, np.newaxis]
        fov_mask = zeta_2d <= fov_rad
        horizon_mask = horizon_mask & fov_mask

    # 5. Integrate over frequency
    dirty = np.zeros((n_radial, n_azimuthal), dtype=np.float64)
    baseline_broad = np.zeros((n_radial, n_azimuthal), dtype=np.float64) if subtract_baseline else None

    use_cffi = _polar_cpu_available()

    for ci in range(n_channels):
        vis = vis_all[ci]
        f_mhz = freq_sky_mhz[ci]

        if filter_rfi:
            vis = reject_rfi_visibilities(vis)

        wavelength = 299.792458 / f_mhz
        bu, bv, bw = compute_uvw_from_antennas(
            antennas, wavelength,
            hour_angle_deg=hour_angle_deg,
            declination_deg=declination_deg
        )

        vis_re, vis_im, auto_corr_sum = _extract_visibility_data(vis)

        if use_cffi:
            try:
                from polar_cpu_cffi import polar_fourier_sum
                channel_dirty = polar_fourier_sum(
                    n_radial, n_azimuthal,
                    bu, bv, bw,
                    vis_re, vis_im, auto_corr_sum,
                    apply_w_correction, num_threads
                )
                # Apply horizon mask
                channel_dirty[~horizon_mask] = 0.0
            except Exception:
                channel_dirty = _direct_fourier_sum_cpu(
                    L, M, N, bu, bv, bw,
                    vis_re, vis_im, auto_corr_sum,
                    apply_w_correction, horizon_mask
                )
        else:
            channel_dirty = _direct_fourier_sum_cpu(
                L, M, N, bu, bv, bw,
                vis_re, vis_im, auto_corr_sum,
                apply_w_correction, horizon_mask
            )

        dirty += channel_dirty

        # Per-channel baseline for broadband correction
        if subtract_baseline:
            bl = _cached_baseline(
                L, M, N, horizon_mask,
                bu, bv, bw,
                apply_w_correction,
                n_radial=n_radial, n_azimuthal=n_azimuthal
            )
            baseline_broad += bl

    dirty /= n_channels

    # Apply broadband baseline correction
    if subtract_baseline:
        baseline_broad /= n_channels
        dirty = apply_baseline_correction(dirty, baseline_broad,
                                           method='relative_excess')

    return dirty, freq_sky_mhz


# ==============================================================================
# Flat-field / Baseline Correction — 均匀天空响应去除
# ==============================================================================
# 180° 视场下, w-修正中的 1/n 因子在地平线附近 (n→0) 发散, 加上稀疏
# uv采样和 w-term 相位累积, 即使对完全均匀的天空亮度分布, 重建脏图也会
# 在视场边缘出现强烈的信号增强。这里计算"基准形状"(均匀天空通过相同
# 管道产生的脏图), 并用它来归一化真实数据, 只保留按比例超出基准的部分。


def _compute_pixel_solid_angles(L, M, N, n_radial=None, n_azimuthal=None):
    """
    计算每个像素的立体角 dΩ。

    对于笛卡尔 (l,m) 网格: dΩ = |Δl·Δm| / n
    对于极坐标 (天顶角, 方位角) 网格: dΩ = sin(ζ) · Δζ · Δφ

    Parameters
    ----------
    L, M, N : ndarray
        方向余弦网格。
    n_radial, n_azimuthal : int or None
        极坐标网格尺寸 (用于判断网格类型)。

    Returns
    -------
    dOmega : ndarray, same shape as L
        每个像素的立体角 (球面度)。
    """
    if n_radial is not None and n_azimuthal is not None:
        # 极坐标网格: dΩ = sin(ζ) · Δζ · Δφ
        d_zeta = np.radians(90.0) / n_radial   # Δζ in radians
        d_phi = np.radians(360.0) / n_azimuthal  # Δφ in radians

        zeta_grid = np.radians(np.linspace(0.0, 90.0, n_radial))
        sin_zeta_2d = np.sin(zeta_grid)[:, np.newaxis]  # (n_radial, 1)

        dOmega = np.full(L.shape, sin_zeta_2d * d_zeta * d_phi)
    else:
        # 笛卡尔 (l,m) 网格: dΩ = |Δl·Δm| / n
        dl = L[0, 1] - L[0, 0] if L.shape[1] > 1 else 0.0
        dm = M[1, 0] - M[0, 0] if M.shape[0] > 1 else 0.0
        pixel_area = abs(dl * dm)

        n_clipped = np.maximum(N, 1e-6)
        dOmega = np.full(L.shape, pixel_area / n_clipped)

    return dOmega


def _compute_uniform_sky_visibilities(L, M, N, horizon_mask, dOmega,
                                       baselines_u, baselines_v, baselines_w):
    """
    正演模拟: 计算亮度均匀为1的天空在每条基线上产生的复可见度。

    可见度方程:
      V_k = ∫∫ B(l,m) · e^(-2πi(u_k·l + v_k·m + w_k·(n-1))) / n  dl dm
          = ∫∫ B(l,m) · e^(-2πi(...))  dΩ

    对均匀天空 B(l,m)=1, 在像素网格上数值积分:
      V_k ≈ Σ_p e^(-2πi(u_k·l_p + v_k·m_p + w_k·(n_p-1))) · dΩ_p

    Parameters
    ----------
    L, M, N : ndarray
        方向余弦网格。
    horizon_mask : ndarray, bool
        有效像素掩膜。
    dOmega : ndarray, same shape as L
        每像素立体角，来自 _compute_pixel_solid_angles()。
    baselines_u, baselines_v, baselines_w : ndarray, shape (28,)
        基线坐标 (波长单位)。

    Returns
    -------
    vis : ndarray, shape (28,), dtype complex128
        均匀单位亮度天空的模拟复可见度。
    """
    n_baselines = len(baselines_u)

    # 选取地平线以上的有效像素
    valid = horizon_mask & (N > 1e-3)
    l_flat = L[valid].ravel()
    m_flat = M[valid].ravel()
    n_flat = N[valid].ravel()
    dOmega_flat = dOmega[valid].ravel()

    vis = np.zeros(n_baselines, dtype=np.complex128)

    # 逐基线正演积分: V_k = Σ_p e^(-2πi(u·l + v·m + w·(n-1))) · dΩ_p
    for k in range(n_baselines):
        # 相位: -2π(u·l + v·m + w·(n-1)) — 正演用负号
        phase = -2.0 * np.pi * (
            baselines_u[k] * l_flat +
            baselines_v[k] * m_flat +
            baselines_w[k] * (n_flat - 1.0)
        )
        # 被积函数: e^(i·phase) · dΩ
        integrand = (np.cos(phase) + 1j * np.sin(phase)) * dOmega_flat
        vis[k] = np.sum(integrand)

    return vis


def _compute_uniform_sky_auto_corr(L, M, N, horizon_mask, dOmega):
    """
    计算均匀单位亮度天空的自相关 (DC) 项。

    单天线自相关 = ∫∫ 1 · dΩ ≈ Σ_p 1 · dΩ_p  (对均匀天空)
    auto_corr_sum = 0.5 · Σ_{i=0}^{7} real(V_auto_i)  (与 _extract_visibility_data 一致)

    Parameters
    ----------
    L, M, N, horizon_mask : ndarray
        网格数据。
    dOmega : ndarray
        每像素立体角。

    Returns
    -------
    auto_corr_sum : float
    """
    valid = horizon_mask & (N > 1e-3)
    # 均匀天空单天线接收总流量 = ∫∫ 1 · dΩ ≈ Σ dΩ_p
    flux_per_antenna = np.sum(dOmega[valid])

    # 8根天线, 每根自相关相同
    auto_corr_sum = 0.5 * 8.0 * flux_per_antenna
    return float(auto_corr_sum)


def _build_uniform_visibility_matrix(vis_cross, auto_corr_sum):
    """
    从互相关可见度和自相关和构建完整的 8x8 复可见度矩阵。

    Parameters
    ----------
    vis_cross : ndarray, shape (28,), dtype complex128
        28条互相关可见度 (上三角, 行优先)。
    auto_corr_sum : float
        自相关和: auto_corr_sum = 0.5 · Σ real(V[i,i])

    Returns
    -------
    V : ndarray, shape (8, 8), dtype complex128
    """
    # 反推对角线值: auto_corr_sum = 0.5 * 8 * diag_val → diag_val = auto_corr_sum / 4
    diag_val = auto_corr_sum / 4.0

    V = np.zeros((8, 8), dtype=np.complex128)
    np.fill_diagonal(V, diag_val)

    idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            V[i, j] = vis_cross[idx]
            V[j, i] = np.conj(vis_cross[idx])
            idx += 1

    return V


def compute_baseline_dirty_image(L, M, N, horizon_mask,
                                  baselines_u, baselines_v, baselines_w,
                                  apply_w_correction=True,
                                  n_radial=None, n_azimuthal=None):
    """
    计算"基准脏图"——仪器对完全均匀单位亮度天空的响应。

    这揭示了成像系统的内在非均匀性:
      - 1/n 几何因子在大天顶角处的增强
      - 非共面基线带来的 w-term 相位结构
      - 稀疏 uv 采样 (脏束) 的旁瓣图案

    计算流程:
      1. 正演模拟均匀天空产生的可见度
      2. 用完全相同的重建管道生成脏图

    Parameters
    ----------
    L, M, N, horizon_mask : ndarray
        来自 build_lm_grid 或 build_polar_sky_grid_GPU 的网格。
    baselines_u, baselines_v, baselines_w : ndarray, shape (28,)
        基线坐标 (波长单位)。
    apply_w_correction : bool
    n_radial, n_azimuthal : int or None
        极坐标网格尺寸。如果同时提供则使用极坐标立体角计算。

    Returns
    -------
    baseline : ndarray, 与 L 同形状
        基准脏图。
    uniform_vis : ndarray, shape (8, 8), dtype complex128
        模拟的均匀天空可见度矩阵 (可供缓存复用)。
    """
    # 计算像素立体角
    dOmega = _compute_pixel_solid_angles(L, M, N,
                                          n_radial=n_radial,
                                          n_azimuthal=n_azimuthal)

    # 1. 正演均匀天空可见度
    vis_cross = _compute_uniform_sky_visibilities(
        L, M, N, horizon_mask, dOmega,
        baselines_u, baselines_v, baselines_w
    )

    # 2. 计算自相关 (DC 项)
    auto_corr = _compute_uniform_sky_auto_corr(L, M, N, horizon_mask, dOmega)

    # 3. 构建完整 8x8 矩阵
    V_uniform = _build_uniform_visibility_matrix(vis_cross, auto_corr)

    # 4. 提取可见度分量
    vis_re, vis_im, auto_from_extract = _extract_visibility_data(V_uniform)

    # 5. 与真实数据完全相同的重建管道
    baseline = _direct_fourier_sum_cpu(
        L, M, N,
        baselines_u, baselines_v, baselines_w,
        vis_re, vis_im,
        auto_from_extract,
        apply_w_correction,
        horizon_mask
    )

    return baseline, V_uniform


def apply_baseline_correction(dirty, baseline, method='relative_excess'):
    """
    从脏图中移除基准形状的影响。

    Parameters
    ----------
    dirty : ndarray
        真实数据的原始脏图。
    baseline : ndarray
        基准脏图 (均匀天空响应)。
    method : str
        'relative_excess': corrected = dirty/baseline - 1
            0 = 与均匀天空相同; 正值 = 比均匀天空更亮。
        'normalize': corrected = dirty/baseline
            1 = 与均匀天空相同。

    Returns
    -------
    corrected : ndarray
        基准校正后的脏图。
    """
    baseline_max = np.max(np.abs(baseline)) if baseline.size > 0 else 1.0
    eps = 1e-10 * max(baseline_max, 1.0)

    # 避免除零
    safe_baseline = np.where(np.abs(baseline) < eps, eps, baseline)

    if method == 'relative_excess':
        result = dirty / safe_baseline - 1.0
    else:  # 'normalize'
        result = dirty / safe_baseline

    # 基准为零的像素 (地平线外) 置零
    result[np.abs(baseline) < eps] = 0.0

    return result


# 基准脏图的内存缓存
# key: (网格形状, uvw 哈希, 是否 w 修正)
_baseline_cache = {}


def _get_baseline_cache_key(L_shape, bu, bv, bw, apply_w_correction,
                             n_radial=None, n_azimuthal=None):
    """生成基准脏图缓存键。"""
    uvw_hash = hash((
        tuple(np.round(bu, 6)),
        tuple(np.round(bv, 6)),
        tuple(np.round(bw, 6)),
    ))
    return (L_shape, uvw_hash, apply_w_correction, n_radial, n_azimuthal)


def _cached_baseline(L, M, N, horizon_mask,
                      bu, bv, bw,
                      apply_w_correction,
                      n_radial=None, n_azimuthal=None,
                      force_recompute=False):
    """计算或从缓存中获取基准脏图。"""
    key = _get_baseline_cache_key(
        L.shape, bu, bv, bw, apply_w_correction,
        n_radial=n_radial, n_azimuthal=n_azimuthal
    )
    if not force_recompute and key in _baseline_cache:
        return _baseline_cache[key]

    baseline, _ = compute_baseline_dirty_image(
        L, M, N, horizon_mask,
        bu, bv, bw,
        apply_w_correction,
        n_radial=n_radial, n_azimuthal=n_azimuthal
    )
    _baseline_cache[key] = baseline
    return baseline


def get_bandwidth_label(center_freq_mhz, n_channels, freq_sky_mhz=None):
    """Generate a bandwidth label string for display."""
    if n_channels <= 1:
        return f"{center_freq_mhz:.0f} MHz (single channel, no bandwidth)"
    if freq_sky_mhz is not None:
        lo = float(np.min(freq_sky_mhz))
        hi = float(np.max(freq_sky_mhz))
        bw = hi - lo
        return f"{lo:.1f}–{hi:.1f} MHz (Δf={bw:.1f} MHz, {n_channels} ch)"
    return f"{center_freq_mhz:.0f} MHz (broadband, {n_channels} ch)"


def transform_display_data(dirty_img, scale="linear"):
    """
    Transform dirty image data for display with the chosen scale.

    Parameters
    ----------
    dirty_img : ndarray
        Raw dirty image (may contain negative values from Fourier synthesis).
    scale : str
        "linear" or "log".

    Returns
    -------
    display_data : ndarray
        Transformed data suitable for imshow/pcolormesh.
    cbar_label : str
        Label for the colorbar.
    """
    if scale == "log":
        # For log display: use SymLogNorm-compatible data.
        # We keep the raw data but signal that a SymLogNorm should be used.
        # The caller should use matplotlib.colors.SymLogNorm for rendering.
        # For simplicity, we return the raw data and a flag; the caller will
        # handle the normalization.
        return dirty_img, "Intensity (log₁₀ scale)"
    else:
        return dirty_img, "Intensity"


def compute_display_norm(dirty_img, scale="linear", linthresh=None):
    """
    Compute an appropriate matplotlib Normalize object for the display scale.

    Uses percentiles to set vmin/vmax for linear, and SymLogNorm for log scale
    (which handles negative values gracefully with a linear region around zero).

    Parameters
    ----------
    dirty_img : ndarray
        Raw dirty image data.
    scale : str
        "linear" or "log".
    linthresh : float or None
        Linear threshold for SymLogNorm. If None, auto-computed from data.

    Returns
    -------
    norm : matplotlib.colors.Normalize
    vmin, vmax : float
        The computed data range.
    """
    from matplotlib.colors import Normalize, SymLogNorm

    valid = dirty_img[dirty_img != 0]
    if len(valid) > 0:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    if scale == "log":
        if linthresh is None:
            # Auto: use ~1% of the positive dynamic range
            pos = valid[valid > 0]
            if len(pos) > 0:
                p1 = float(np.percentile(pos, 1))
                linthresh = max(p1, 1e-6 * vmax)
            else:
                linthresh = vmax * 1e-4
        return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10), vmin, vmax
    else:
        return Normalize(vmin=vmin, vmax=vmax), vmin, vmax
