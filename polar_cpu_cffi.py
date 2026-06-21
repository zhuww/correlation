"""
polar_cpu_cffi.py
=================
C/OpenMP-accelerated polar all-sky direct Fourier sum engine.

This module provides a Python interface to the C function
compute_dirty_image_polar() in direct_fourier_c.c.

Loading strategy:
  1. Try ctypes to load a pre-compiled DLL (.dll/.so).
  2. If DLL not found, try cffi out-of-line compilation.
  3. If compilation fails (no C compiler), raise an informative error
     with instructions for installing MSVC Build Tools.

Usage:
    from polar_cpu_cffi import polar_fourier_sum, is_available

    if is_available():
        dirty_img = polar_fourier_sum(L, M, N, u, v, w, vis_re, vis_im,
                                       auto_corr, True)
"""

import os
import sys
import ctypes
import numpy as np
from pathlib import Path
import subprocess

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Shared library loading
# ---------------------------------------------------------------------------

_lib = None
_lib_loaded = False
_lib_error = None

# Define the C function signature
# int compute_dirty_image_polar(
#     double *dirty_out, int n_radial, int n_azimuthal,
#     const double *u, const double *v, const double *w,
#     const double *vis_re, const double *vis_im,
#     int n_baselines, double auto_corr_sum,
#     int apply_w_correction, int num_threads)


def _find_dll():
    """Find a pre-compiled shared library."""
    candidates = []

    if sys.platform == 'win32':
        names = ['direct_fourier_c.dll', '_polar_fourier_engine.pyd']
        # Also check Release subdirectory (typical MSVC output)
        candidates.append(_HERE / 'Release' / 'direct_fourier_c.dll')
    else:
        names = ['direct_fourier_c.so', '_polar_fourier_engine.so']

    for name in names:
        candidates.append(_HERE / name)

    # Also check for cffi-generated modules
    for p in _HERE.glob('_polar_fourier_engine*'):
        if p.suffix in ('.pyd', '.so'):
            candidates.append(p)

    for path in candidates:
        if path.exists():
            return str(path)
    return None


def _try_load_ctypes():
    """Try to load a pre-compiled DLL using ctypes."""
    dll_path = _find_dll()
    if dll_path is None:
        return None

    try:
        lib = ctypes.CDLL(dll_path)
    except OSError as e:
        return None

    # Set up function signature
    lib.compute_dirty_image_polar.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # dirty_out
        ctypes.c_int,                      # n_radial
        ctypes.c_int,                      # n_azimuthal
        ctypes.POINTER(ctypes.c_double),  # baselines_u
        ctypes.POINTER(ctypes.c_double),  # baselines_v
        ctypes.POINTER(ctypes.c_double),  # baselines_w
        ctypes.POINTER(ctypes.c_double),  # vis_re
        ctypes.POINTER(ctypes.c_double),  # vis_im
        ctypes.c_int,                      # n_baselines
        ctypes.c_double,                   # auto_corr_sum
        ctypes.c_int,                      # apply_w_correction
        ctypes.c_int,                      # num_threads
    ]
    lib.compute_dirty_image_polar.restype = ctypes.c_int

    return lib


def _try_compile_cffi():
    """Try to compile the C source using cffi's out-of-line API."""
    try:
        from cffi import FFI
    except ImportError:
        return None, "cffi not installed. Run: pip install cffi"

    ffi = FFI()

    ffi.cdef("""
        int compute_dirty_image_polar(
            double *dirty_out,
            int n_radial,
            int n_azimuthal,
            const double *baselines_u,
            const double *baselines_v,
            const double *baselines_w,
            const double *vis_re,
            const double *vis_im,
            int n_baselines,
            double auto_corr_sum,
            int apply_w_correction,
            int num_threads
        );
    """)

    source_path = _HERE / 'direct_fourier_c.c'
    if not source_path.exists():
        return None, f"C source not found: {source_path}"

    if sys.platform == 'win32':
        extra_compile_args = ['/O2', '/openmp']
        extra_link_args = []
    else:
        extra_compile_args = ['-O3', '-fopenmp']
        extra_link_args = ['-fopenmp', '-lm']

    ffi.set_source(
        '_polar_fourier_engine',
        source_path.read_text(encoding='utf-8'),
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=[],
    )

    try:
        ffi.compile(tmpdir=str(_HERE), verbose=False)
    except Exception as e:
        msg = str(e)
        if 'Microsoft Visual C++' in msg or 'MSVC' in msg or 'Build Tools' in msg:
            return None, (
                "C compiler not found. On Windows, install Microsoft C++ Build Tools:\n"
                "  https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
                "Then re-run to auto-compile the C engine."
            )
        return None, f"Compilation failed: {e}"

    # Import the compiled module
    import importlib
    sys.path.insert(0, str(_HERE))
    try:
        mod = importlib.import_module('_polar_fourier_engine')
        return mod.lib, mod.ffi
    except ImportError as e:
        return None, f"Failed to import compiled module: {e}"
    finally:
        if str(_HERE) in sys.path:
            sys.path.remove(str(_HERE))


def compile_and_load(force_rebuild=False):
    """
    Load the C polar Fourier engine.

    Tries in order:
      1. ctypes load of a pre-compiled DLL
      2. cffi out-of-line compilation from source

    Returns
    -------
    lib : ctypes.CDLL or cffi.lib
        The loaded library.
    backend : str
        'ctypes' or 'cffi'
    ffi_or_none : cffi.FFI or None
        The FFI instance (only for cffi backend).
    """
    global _lib, _lib_loaded, _lib_error

    if _lib_loaded:
        if _lib is not None:
            return _lib
        else:
            raise RuntimeError(f"C library not available: {_lib_error}")

    # Strategy 1: ctypes with pre-compiled DLL
    lib = _try_load_ctypes()
    if lib is not None:
        _lib = ('ctypes', lib, None)
        _lib_loaded = True
        return _lib

    # Strategy 2: cffi compilation
    result = _try_compile_cffi()
    # _try_compile_cffi returns:
    #   - (lib, ffi) on success
    #   - (None, error_msg) on failure
    if result is not None:
        cffi_lib, cffi_ffi = result
        if cffi_lib is not None:
            _lib = ('cffi', cffi_lib, cffi_ffi)
            _lib_loaded = True
            return _lib
        else:
            _lib_error = cffi_ffi or "Unknown CFFI error"

    _lib_error = _lib_error or "Cannot load C engine (no compiler, no pre-built DLL)"
    _lib_loaded = True
    raise RuntimeError(_lib_error)


def is_available():
    """Check if the C polar engine can be loaded."""
    try:
        compile_and_load()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# High-level Python wrapper
# ---------------------------------------------------------------------------

def polar_fourier_sum(L, M, N, baselines_u, baselines_v, baselines_w,
                      vis_re, vis_im, auto_corr_sum,
                      apply_w_correction, horizon_mask=None,
                      num_threads=0):
    """
    Compute the polar all-sky dirty image using the C engine.

    This calls the C function compute_dirty_image_polar() which uses OpenMP
    for multi-core parallelism (up to 8 cores).

    Parameters
    ----------
    L, M, N : ndarray, shape (n_radial, n_azimuthal)
        Direction cosine grids.
    baselines_u, baselines_v, baselines_w : ndarray, shape (n_baselines,)
        Baseline coordinates in wavelengths.
    vis_re, vis_im : ndarray, shape (n_baselines,)
        Real and imaginary parts of visibilities.
    auto_corr_sum : float
        Sum of auto-correlations (DC term).
    apply_w_correction : bool
        If True, include full w-term and 1/n primary beam correction.
    horizon_mask : ndarray or None
        Not used by C engine; kept for API compatibility.
    num_threads : int
        Number of OpenMP threads. 0 = auto (up to 8).

    Returns
    -------
    dirty_img : ndarray, shape (n_radial, n_azimuthal), dtype float64
    """
    lib_info = compile_and_load()
    backend, lib, ffi = lib_info

    n_radial, n_azimuthal = L.shape
    n_baselines = len(baselines_u)

    # Ensure float64 contiguous arrays
    u_arr  = np.ascontiguousarray(baselines_u, dtype=np.float64)
    v_arr  = np.ascontiguousarray(baselines_v, dtype=np.float64)
    w_arr  = np.ascontiguousarray(baselines_w, dtype=np.float64)
    vr_arr = np.ascontiguousarray(vis_re, dtype=np.float64)
    vi_arr = np.ascontiguousarray(vis_im, dtype=np.float64)

    # Allocate output
    n_total = n_radial * n_azimuthal
    dirty_out = np.zeros(n_total, dtype=np.float64)

    w_corr_flag = 1 if apply_w_correction else 0

    if backend == 'ctypes':
        ret = lib.compute_dirty_image_polar(
            dirty_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_radial),
            ctypes.c_int(n_azimuthal),
            u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            v_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vr_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vi_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(n_baselines),
            ctypes.c_double(float(auto_corr_sum)),
            ctypes.c_int(w_corr_flag),
            ctypes.c_int(num_threads),
        )
    else:  # cffi
        dirty_c = ffi.new('double[]', n_total)
        u_ptr  = ffi.cast('double *', ffi.from_buffer(u_arr))
        v_ptr  = ffi.cast('double *', ffi.from_buffer(v_arr))
        w_ptr  = ffi.cast('double *', ffi.from_buffer(w_arr))
        vr_ptr = ffi.cast('double *', ffi.from_buffer(vr_arr))
        vi_ptr = ffi.cast('double *', ffi.from_buffer(vi_arr))

        ret = lib.compute_dirty_image_polar(
            dirty_c, n_radial, n_azimuthal,
            u_ptr, v_ptr, w_ptr, vr_ptr, vi_ptr,
            n_baselines, float(auto_corr_sum),
            w_corr_flag, num_threads,
        )

        if ret != 0:
            raise RuntimeError(f"compute_dirty_image_polar returned error {ret}")

        buf = ffi.buffer(dirty_c, n_total * 8)
        dirty_out = np.frombuffer(buf, dtype=np.float64).copy()

    if ret != 0:
        raise RuntimeError(f"compute_dirty_image_polar returned error {ret}")

    return dirty_out.reshape(n_radial, n_azimuthal)


# ---------------------------------------------------------------------------
# Build helper: compile the DLL manually
# ---------------------------------------------------------------------------

def print_build_instructions():
    """Print instructions for manually compiling the C module."""
    src = _HERE / 'direct_fourier_c.c'

    print("""
====================================================================
  C Engine Build Instructions
====================================================================

The C source is at: {src}

Option 1 — MSVC (Windows, recommended):
  1. Install Microsoft C++ Build Tools:
     https://visualstudio.microsoft.com/visual-cpp-build-tools/
  2. Open "x64 Native Tools Command Prompt"
  3. Run:
     cd /d {here}
     cl /O2 /openmp /LD direct_fourier_c.c /Fe:direct_fourier_c.dll

Option 2 — MinGW-w64 (Windows):
  1. Install MSYS2: https://www.msys2.org/
  2. In MSYS2 terminal:
     pacman -S mingw-w64-x86_64-gcc
  3. Run:
     cd {here_sh}
     gcc -O3 -fopenmp -shared -o direct_fourier_c.dll direct_fourier_c.c -lm

Option 3 — GCC (Linux/Mac):
  gcc -O3 -fopenmp -shared -fPIC -o direct_fourier_c.so direct_fourier_c.c -lm

After compilation, the Python module will auto-detect the DLL.
====================================================================
""".format(src=src, here=_HERE, here_sh=str(_HERE).replace('\\', '/')))
