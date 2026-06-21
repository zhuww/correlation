#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试: CPU 大视场模式 (Polar CPU, FOV=180°)
=============================================
使用纯 Python/NumPy 引擎，在极坐标(天顶角, 方位角)网格上做
直接 3D Fourier 积分，生成全天空脏图。

注：由于系统缺少 C 编译器，无法使用 C/OpenMP 引擎。
本测试使用 make_dirty_image.py 中已有的 Python CPU 路径
(即 GPU 模式的 CPU fallback)，算法完全一致，只是无 OpenMP 加速。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re
import pandas as pd
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_dirty_image import (
    load_optimized_antennas,
    get_polar_grid_metadata,
    build_polar_sky_grid_GPU,
    _direct_fourier_sum_cpu,
    compute_uvw_from_antennas,
    _extract_visibility_data,
    reject_rfi_visibilities,
)

def build_visibility_matrix(watch_dir="./correlation_results", freq_bin_idx=0):
    watch_dir = Path(watch_dir)
    csv_files = list(watch_dir.glob("correlation_*.csv"))
    if not csv_files:
        print("[ERROR] 没有找到 CSV 文件!")
        return None

    latest_time = max(f.stat().st_mtime for f in csv_files)
    latest_files = [f for f in csv_files if abs(f.stat().st_mtime - latest_time) < 1.0]

    file_map = {}
    for f in latest_files:
        name = f.stem
        match = re.search(r'(CH\d+(?:xCH\d+|_AUTO))', name)
        if match:
            file_map[match.group(1)] = f

    print(f"  找到 {len(file_map)} 个配对文件")

    vis = np.zeros((8, 8), dtype=np.complex128)

    def get_indices(pair_name):
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

    for pair_name, fp in file_map.items():
        row, col = get_indices(pair_name)
        if row is None:
            continue
        try:
            df = pd.read_csv(fp, comment='#')
            if freq_bin_idx < len(df):
                r = df.iloc[freq_bin_idx]
                val = complex(r['real_part'], r['imag_part'])
                vis[row, col] = val
                if row != col:
                    vis[col, row] = val.conjugate()
        except Exception as e:
            print(f"  [WARN] 读取 {fp.name} 失败: {e}")

    print(f"  可见度矩阵非零元素: {np.count_nonzero(vis)}/64")
    return vis


def main():
    print("=" * 60)
    print("  测试: CPU Polar (纯 Python/NumPy) 大视场脏图")
    print("  FOV = 180°, Grid = 256, Freq = 150 MHz")
    print("  引擎: Python CPU (与 C/OpenMP 引擎算法一致)")
    print("  注意: 无 OpenMP 加速，预计比 C 引擎慢 4-8 倍")
    print("=" * 60)

    output_dir = Path("./dirty_image_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载可见度数据
    print("\n[1] 加载可见度数据...")
    vis = build_visibility_matrix()
    if vis is None:
        print("[FAIL] 无法加载可见度数据")
        return 1

    # 2. 准备 UVW 基线和极坐标网格
    print("\n[2] 准备 UVW 基线和极坐标网格...")
    antennas = load_optimized_antennas("optimized_antenna_coordinates.txt")
    wavelength = 299.792458 / 150.0

    baselines_u, baselines_v, baselines_w = compute_uvw_from_antennas(
        antennas, wavelength,
        hour_angle_deg=0.0,
        declination_deg=90.0
    )
    print(f"  基线数: {len(baselines_u)}")

    n_radial = 128
    n_azimuthal = 256
    print(f"  极坐标网格: n_radial={n_radial} (天顶角 0°→90°)")
    print(f"               n_azimuthal={n_azimuthal} (方位角 0°→360°)")

    L, M, N, horizon_mask, zeta_deg, az_deg = build_polar_sky_grid_GPU(
        n_radial=n_radial, n_azimuthal=n_azimuthal
    )

    vis_re, vis_im, auto_corr_sum = _extract_visibility_data(vis)
    print(f"  自相关和 (DC term): {auto_corr_sum:.6f}")

    # 3. 计算脏图
    print("\n[3] 计算全天空脏图 (Python CPU 直接 3D Fourier 积分)...")
    print("  算法: I(l,m) = Σ_k [V_k^re·cos(φ_k) - V_k^im·sin(φ_k)]")
    print("  φ_k = 2π [u_k·l + v_k·m + w_k·(n-1)]")
    print(f"  总像素: {n_radial}×{n_azimuthal} = {n_radial*n_azimuthal}")
    print(f"  每像素遍历 {len(baselines_u)} 条基线")
    print(f"  总运算量: ~{n_radial*n_azimuthal*len(baselines_u):,} 次 sin/cos")
    print("  开始计算...")

    t0 = time.time()
    dirty_img = _direct_fourier_sum_cpu(
        L, M, N,
        baselines_u, baselines_v, baselines_w,
        vis_re, vis_im,
        auto_corr_sum,
        apply_w_correction=True,
        horizon_mask=horizon_mask,
    )
    elapsed = time.time() - t0

    print(f"  耗时: {elapsed:.2f}s")

    # 估算 C/OpenMP 引擎耗时（假设 4-8x 加速）
    if elapsed > 0:
        est_c = elapsed / 4
        print(f"  估算 C/OpenMP 引擎耗时: ~{est_c:.1f}s (假设 4x 加速)")

    print(f"  脏图形状: {dirty_img.shape}  (n_radial={dirty_img.shape[0]}, n_azimuthal={dirty_img.shape[1]})")
    print(f"  总像素:   {dirty_img.size}")

    non_zero = np.count_nonzero(dirty_img)
    print(f"  非零像素: {non_zero}/{dirty_img.size}")

    valid = dirty_img[dirty_img != 0]
    if len(valid) > 0:
        print(f"  强度范围: [{valid.min():.6f}, {valid.max():.6f}]")
        print(f"  峰值:     {valid.max():.6f}")
        print(f"  均值:     {valid.mean():.6f}")
        print(f"  标准差:   {valid.std():.6f}")

        # 定位峰值
        peak_idx = np.unravel_index(np.argmax(dirty_img), dirty_img.shape)
        meta = get_polar_grid_metadata(grid_pts=256, n_radial=128, n_azimuthal=256)
        peak_zeta = meta['zeta_deg'][peak_idx[0]]
        peak_az = meta['az_deg'][peak_idx[1]]
        print(f"  峰值位置: ζ={peak_zeta:.1f}° (天顶角), Az={peak_az:.1f}° (方位角)")
    else:
        print("  [WARN] 脏图全为零!")
        return 1

    # 3. 保存图像
    print("\n[3] 保存测试图像...")
    nr, na = dirty_img.shape
    meta = get_polar_grid_metadata(grid_pts=256, n_radial=nr, n_azimuthal=na)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # 左: 极坐标脏图 (pcolormesh)
    ax1 = axes[0]
    ax1.set_facecolor('black')

    zeta_edges = np.linspace(0.0, 90.0, nr + 1) / 90.0
    az_edges = np.linspace(0.0, 360.0, na + 1)
    az_rad_edges = np.radians(az_edges)
    ZETA_E, AZ_E = np.meshgrid(zeta_edges, az_rad_edges, indexing='ij')
    X_edges = ZETA_E * np.sin(AZ_E)
    Y_edges = ZETA_E * np.cos(AZ_E)

    im = ax1.pcolormesh(X_edges, Y_edges, dirty_img,
                        cmap='inferno', shading='flat', rasterized=True)

    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    im.set_clim(vmin, vmax)

    # 参考网格
    for zc in [15, 30, 45, 60, 75, 90]:
        r = zc / 90.0
        circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                            linewidth=0.5, alpha=0.6, linestyle='--')
        ax1.add_patch(circle)
    for az in [0, 90, 180, 270]:
        theta = np.radians(90 - az)
        ax1.plot([0, np.cos(theta)], [0, np.sin(theta)], color='#336699',
                 linewidth=0.4, alpha=0.5, linestyle=':')

    horizon = plt.Circle((0, 0), 1.0, fill=False, color='#4488cc', linewidth=1.2, alpha=0.8)
    ax1.add_patch(horizon)
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_aspect('equal')
    ax1.set_title("All-Sky Dirty Image — Polar CPU (Python/NumPy)\n150 MHz, FOV=180°, 128×256 polar", fontsize=13)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im, ax=ax1, label='Intensity', fraction=0.046)

    # 右: 天线布局
    ax2 = axes[1]
    antennas = load_optimized_antennas("optimized_antenna_coordinates.txt")
    xs, ys = antennas[:, 0], antennas[:, 1]
    ax2.scatter(xs, ys, c='red', s=100, edgecolors='black', linewidths=1, zorder=5)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax2.annotate(f"CH{i+1}", (x, y), textcoords="offset points",
                      xytext=(8, 8), fontsize=10)
    for i in range(8):
        for j in range(i + 1, 8):
            ax2.plot([xs[i], xs[j]], [ys[i], ys[j]], 'gray', alpha=0.3, linewidth=0.5)
    margin = 2.0
    ax2.set_xlim(xs.min() - margin, xs.max() + margin)
    ax2.set_ylim(ys.min() - margin, ys.max() + margin)
    ax2.set_title("Array Layout (8 Elements)", fontsize=13)
    ax2.set_xlabel("X (m) East →")
    ax2.set_ylabel("Y (m) North →")
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = output_dir / "test_polar_cpu_180deg.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {png_path}")

    print("\n" + "=" * 60)
    print("  CPU 大视场模式 (Polar CPU Python/NumPy) 测试通过!")
    print(f"  耗时: {elapsed:.2f}s, 峰值: {valid.max():.6f}")
    print(f"  峰值位置: ζ={peak_zeta:.1f}°, Az={peak_az:.1f}°")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
