/**
 * direct_fourier_c.c
 * ===================
 * CPU-only C implementation of Direct 3D Fourier Integration
 * for all-sky polar coordinate dirty image synthesis.
 *
 * Engine C: Polar CPU — 180° FOV, zenith-angle × azimuth grid,
 *            OpenMP multi-core parallelism (up to 8 cores).
 *
 * Mathematical foundation:
 *   I_D(l,m) = Σ_k V_k · exp{+2πi [u_k·l + v_k·m + w_k·(n - 1)]}
 *
 * where:
 *   l = cos(alt) * sin(az)    (East direction cosine)
 *   m = cos(alt) * cos(az)    (North direction cosine)
 *   n = sin(alt)              (Zenith direction cosine)
 *   alt = 90° - zenith_angle
 *
 * The grid is in polar coordinates:
 *   - Radial axis:    zenith angle ζ ∈ [0°, 90°]  (n_radial bins)
 *   - Azimuthal axis: azimuth A ∈ [0°, 360°)      (n_azimuthal bins)
 *
 * Output: dirty_img[n_radial][n_azimuthal] as float64 (row-major).
 *
 * Compile with (MSVC on Windows):
 *   cl /O2 /openmp /LD direct_fourier_c.c /Fe:direct_fourier_c.dll
 *
 * Compile with (GCC on Linux/Mac):
 *   gcc -O3 -fopenmp -shared -fPIC -o direct_fourier_c.so direct_fourier_c.c -lm
 *
 * Author: Auto-generated for 8-element radio array project.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/* compute_dirty_image_polar                                          */
/* ------------------------------------------------------------------ */
/**
 * Compute the all-sky dirty image via direct 3D Fourier sum on a
 * polar (zenith_angle, azimuth) grid.
 *
 * Parameters
 * ----------
 * dirty_out : double* [output]
 *     Pre-allocated array of size n_radial * n_azimuthal, row-major.
 * n_radial : int
 *     Number of radial (zenith angle) bins.
 * n_azimuthal : int
 *     Number of azimuthal bins.
 *
 * baselines_u : const double* [input]
 *     u coordinates (in wavelengths), length n_baselines.
 * baselines_v : const double* [input]
 *     v coordinates (in wavelengths), length n_baselines.
 * baselines_w : const double* [input]
 *     w coordinates (in wavelengths), length n_baselines.
 * vis_re : const double* [input]
 *     Real part of visibilities, length n_baselines.
 * vis_im : const double* [input]
 *     Imag part of visibilities, length n_baselines.
 * n_baselines : int
 *     Number of baselines (typically 28 for 8 antennas).
 *
 * auto_corr_sum : double
 *     Sum of auto-correlations (DC term / zero-baseline flux).
 * apply_w_correction : int
 *     If non-zero, include full w-term and 1/n primary beam correction.
 * num_threads : int
 *     Number of OpenMP threads to use. If <= 0, uses all available.
 *
 * Returns
 * -------
 * 0 on success, -1 on error.
 */
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
    int num_threads)
{
    if (!dirty_out || !baselines_u || !baselines_v ||
        !baselines_w || !vis_re || !vis_im) {
        return -1;
    }
    if (n_radial <= 0 || n_azimuthal <= 0 || n_baselines <= 0) {
        return -1;
    }

    /* Set number of threads */
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    } else {
        /* Use at most 8 cores as specified */
        int max_threads = omp_get_max_threads();
        if (max_threads > 8) max_threads = 8;
        omp_set_num_threads(max_threads);
    }
#endif

    const int total_pixels = n_radial * n_azimuthal;
    const double deg_to_rad = M_PI / 180.0;
    const double two_pi = 2.0 * M_PI;
    const double norm_factor = 28.5;

    /* Pre-compute zenith angles and azimuths */
    double *zeta_rad = (double *)malloc(n_radial * sizeof(double));
    double *az_rad   = (double *)malloc(n_azimuthal * sizeof(double));
    if (!zeta_rad || !az_rad) {
        free(zeta_rad);
        free(az_rad);
        return -1;
    }

    /* Zenith angle: linear from 0° to 90° (cell centres) */
    for (int i = 0; i < n_radial; i++) {
        double zeta_deg = (i + 0.5) * 90.0 / n_radial;
        zeta_rad[i] = zeta_deg * deg_to_rad;
    }

    /* Azimuth: linear from 0° to 360° (cell centres) */
    for (int j = 0; j < n_azimuthal; j++) {
        double az_deg = (j + 0.5) * 360.0 / n_azimuthal;
        az_rad[j] = az_deg * deg_to_rad;
    }

    /*
     * Parallel over radial bins.
     * Each thread processes a contiguous chunk of radial rows.
     * This keeps memory access patterns cache-friendly and avoids
     * false sharing (each row is independent).
     */
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_radial; i++) {
        double zeta = zeta_rad[i];
        double alt = (M_PI / 2.0) - zeta;   /* altitude = 90° - zenith */
        double sin_alt = sin(alt);
        double cos_alt = cos(alt);

        /* Direction cosine n = sin(altitude) */
        double n_val = sin_alt;

        /* For w-correction: pre-compute (n - 1) */
        double n_minus_1 = n_val - 1.0;

        /* Pre-compute 1/n for primary beam correction (clipped) */
        double inv_n = 0.0;
        if (apply_w_correction) {
            inv_n = (n_val > 1e-6) ? (1.0 / n_val) : 0.0;
        }

        double *row_out = dirty_out + (size_t)i * n_azimuthal;

        for (int j = 0; j < n_azimuthal; j++) {
            double az = az_rad[j];

            /* Direction cosines l, m */
            double l_val = cos_alt * sin(az);
            double m_val = cos_alt * cos(az);

            /* Accumulator starts with auto-correlation (DC term) */
            double acc = auto_corr_sum;

            /* Sum over all baselines */
            for (int k = 0; k < n_baselines; k++) {
                /* Phase φ_k = 2π [u_k·l + v_k·m + w_k·(n - 1)] */
                double phase = two_pi * (baselines_u[k] * l_val +
                                         baselines_v[k] * m_val);

                if (apply_w_correction) {
                    phase += two_pi * baselines_w[k] * n_minus_1;
                }

                /* V_re·cos(φ) - V_im·sin(φ) */
                acc += vis_re[k] * cos(phase) - vis_im[k] * sin(phase);
            }

            /* Apply normalization and primary beam correction */
            double pixel_val;
            if (apply_w_correction) {
                if (n_val > 1e-3 && inv_n > 0.0) {
                    pixel_val = (acc * inv_n) / norm_factor;
                } else {
                    pixel_val = 0.0;  /* horizon edge: zero sensitivity */
                }
            } else {
                pixel_val = acc / norm_factor;
            }

            row_out[j] = pixel_val;
        }
    }

    free(zeta_rad);
    free(az_rad);
    return 0;
}
