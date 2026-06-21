#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试: GPU 大视场模式 (Polar GPU CuPy, FOV=180°)
================================================
使用 CuPy GPU 加速，在极坐标(天顶角, 方位角)网格上做
直接 3D Fourier 积分，生成全天空脏图。

环境: CuPy 14.1.1 + NVIDIA RTX 4060 Laptop GPU (8GB)
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
    make_dirty_image_GPU,
    load_optimized_antennas,
    get_polar_grid_metadata,
    gpu_available,
    gpu_install_hint,
    _direct_fourier_sum_GPU,
    _direct_fourier_sum_cpu,
    build_polar_sky_grid_GPU,
    compute_uvw_from_antennas,
    _extract_visibility_data,
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
    print("  测试: GPU Polar (CuPy) 大视场脏图")
    print("  FOV = 180°, Grid = 256, Freq = 150 MHz")
    print("  引擎: CuPy GPU (RTX 4060 Laptop 8GB)")
    print("=" * 60)

    output_dir = Path("./dirty_image_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. 检查 GPU
    print("\n[0] 检查 GPU 环境...")
    if not gpu_available():
        print("[FAIL] GPU 不可用! 需要 CuPy + CUDA GPU")
        hint = gpu_install_hint()
        if hint:
            print(hint)
        return 1

    # 安全导入 cupy（此时 gpu_available() 已确认可用）
    try:
        import cupy as cp
    except ImportError as e:
        print(f"[FAIL] 无法导入 cupy: {e}")
        print("安装: pip install cupy")
        print("详见: https://docs.cupy.dev/en/stable/install.html")
        return 1

    dev = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(0)
    dev_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    total_mem = props['totalGlobalMem'] // (1024 * 1024)
    print(f"  GPU: {dev_name}")
    print(f"  显存: {total_mem} MB")
    print(f"  CuPy 版本: {cp.__version__}")
    print("  GPU 可用 [OK]")

    # 1. 加载可见度数据
    print("\n[1] 加载可见度数据...")
    vis = build_visibility_matrix()
    if vis is None:
        print("[FAIL] 无法加载可见度数据")
        return 1

    # 2. 准备基线 + 极坐标网格
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
    total_pixels = n_radial * n_azimuthal
    total_ops = total_pixels * len(baselines_u)
    print(f"  总像素: {n_radial}×{n_azimuthal} = {total_pixels}")
    print(f"  总运算量: ~{total_ops:,} 次 sin/cos")

    # 3. GPU 基准测试
    print("\n[3] GPU 基准: CuPy 直接 3D Fourier 积分...")
    print("  算法: I(l,m) = Σ_k [V_k^re·cos(φ_k) - V_k^im·sin(φ_k)]")
    print("  数据传输: L,M,N → GPU, u,v,w → GPU, V_re,V_im → GPU")
    print("  开始计算...")

    # Warmup: 先跑一次小的让 CUDA context 预热
    _ = cp.array([0.0])

    t0 = time.perf_counter()
    dirty_gpu = _direct_fourier_sum_GPU(
        L, M, N,
        baselines_u, baselines_v, baselines_w,
        vis_re, vis_im,
        auto_corr_sum,
        apply_w_correction=True,
        horizon_mask=horizon_mask,
    )
    # 确保 GPU 同步
    cp.cuda.Stream.null.synchronize()
    elapsed_gpu = time.perf_counter() - t0

    # 4. CPU 对比测试（同网格，同算法）
    print(f"\n  GPU 耗时: {elapsed_gpu:.4f}s")

    print("\n[4] CPU 对比: NumPy 直接 3D Fourier 积分 (同网格)...")
    t0 = time.perf_counter()
    dirty_cpu = _direct_fourier_sum_cpu(
        L, M, N,
        baselines_u, baselines_v, baselines_w,
        vis_re, vis_im,
        auto_corr_sum,
        apply_w_correction=True,
        horizon_mask=horizon_mask,
    )
    elapsed_cpu = time.perf_counter() - t0
    print(f"  CPU 耗时: {elapsed_cpu:.4f}s")

    speedup = elapsed_cpu / elapsed_gpu if elapsed_gpu > 0 else 0
    print(f"  GPU 加速比: {speedup:.1f}×")

    # 5. 数值精度对比
    print("\n[5] 数值精度对比 (GPU vs CPU)...")
    diff = np.abs(dirty_gpu - dirty_cpu)
    valid_mask = (dirty_gpu != 0) | (dirty_cpu != 0)
    if np.any(valid_mask):
        max_diff = diff[valid_mask].max()
        mean_diff = diff[valid_mask].mean()
        rel_diff = diff[valid_mask] / (np.abs(dirty_cpu[valid_mask]) + 1e-10)
        max_rel = rel_diff.max()
        mean_rel = rel_diff.mean()
        print(f"  最大绝对差: {max_diff:.2e}")
        print(f"  平均绝对差: {mean_diff:.2e}")
        print(f"  最大相对差: {max_rel:.2e}")
        print(f"  平均相对差: {mean_rel:.2e}")

        if max_rel < 1e-6:
            print("  精度: 优秀 [OK] (相对误差 < 1e-6)")
        elif max_rel < 1e-4:
            print("  精度: 良好 [OK] (相对误差 < 1e-4)")
        else:
            print("  精度: 可接受 (GPU/CPU 浮点差异)")

    # 6. GPU 脏图统计
    print("\n[6] GPU 脏图统计...")
    dirty_img = dirty_gpu
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
        meta = get_polar_grid_metadata(grid_pts=256, n_radial=n_radial, n_azimuthal=n_azimuthal)
        peak_zeta = meta['zeta_deg'][peak_idx[0]]
        peak_az = meta['az_deg'][peak_idx[1]]
        print(f"  峰值位置: ζ={peak_zeta:.1f}° (天顶角), Az={peak_az:.1f}° (方位角)")
    else:
        print("  [WARN] 脏图全为零!")
        return 1

    # 7. 保存 fish-eye 极坐标图
    print("\n[7] 保存全天空鱼眼图...")
    nr, na = n_radial, n_azimuthal

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # 左: GPU 全天空鱼眼脏图
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
    if len(valid) > 0:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        im.set_clim(vmin, vmax)

    # 天顶角参考圈
    for zc in [15, 30, 45, 60, 75, 90]:
        r = zc / 90.0
        circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                             linewidth=0.5, alpha=0.6, linestyle='--')
        ax1.add_patch(circle)
        if zc < 90:
            ax1.annotate(f'{zc}°', (0, r), color='#336699', fontsize=6,
                          ha='center', va='bottom')

    # 方位角射线 + 标签
    az_rays = [0, 45, 90, 135, 180, 225, 270, 315]
    az_labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
    for az in az_rays:
        theta = np.radians(90 - az)
        dx = np.cos(theta)
        dy = np.sin(theta)
        ax1.plot([0, dx], [0, dy], color='#336699', linewidth=0.4, alpha=0.4, linestyle=':')
        label = az_labels.get(az, '')
        if label:
            ax1.annotate(label, (dx * 1.1, dy * 1.1), color='#336699',
                          fontsize=8, ha='center', va='center')

    # 地平线圈
    horizon_circle = plt.Circle((0, 0), 1.0, fill=False, color='#4488cc', linewidth=1.2)
    ax1.add_patch(horizon_circle)
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(f"All-Sky Dirty Image — GPU (CuPy)\n150 MHz, FOV=180°, 128×256 polar, {speedup:.1f}× speedup",
                  fontsize=13)
    plt.colorbar(im, ax=ax1, label='Intensity', fraction=0.046)

    # 右: 天线布局
    ax2 = axes[1]
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
    png_path = output_dir / "test_polar_gpu_180deg.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {png_path}")

    # 8. 多分辨率 GPU 扩展测试
    print("\n[8] 多分辨率 GPU 扩展测试...")
    resolutions = [
        (64, 128,   "低"),
        (128, 256,  "标准"),
        (256, 512,  "高"),
        (512, 1024, "超高"),
    ]

    print(f"  {'分辨率':<20} {'像素':<12} {'耗时(s)':<12} {'吞吐量':<14} {'显存(MB)':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*14} {'-'*12}")

    for nr_test, na_test, label in resolutions:
        L_t, M_t, N_t, hm_t, _, _ = build_polar_sky_grid_GPU(
            n_radial=nr_test, n_azimuthal=na_test
        )

        t0 = time.perf_counter()
        _ = _direct_fourier_sum_GPU(
            L_t, M_t, N_t,
            baselines_u, baselines_v, baselines_w,
            vis_re, vis_im,
            auto_corr_sum,
            apply_w_correction=True,
            horizon_mask=hm_t,
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0

        n_pix = nr_test * na_test
        throughput = n_pix / elapsed / 1e6 if elapsed > 0 else 0
        # 粗略估计显存: 3×(nr×na)×8 bytes + overhead
        est_mem = (3 * n_pix * 8) / (1024 * 1024)

        print(f"  {nr_test}×{na_test} {label:<12} {n_pix:<12,} {elapsed:<12.4f} {throughput:<14.2f} MPix/s {est_mem:<12.1f}")

    print("\n" + "=" * 60)
    print("  GPU 大视场模式 (Polar GPU CuPy) 测试通过!")
    print(f"  GPU: {dev_name} | CuPy {cp.__version__}")
    print(f"  GPU 耗时: {elapsed_gpu:.4f}s | CPU 耗时: {elapsed_cpu:.4f}s")
    print(f"  加速比: {speedup:.1f}×")
    print(f"  峰值: {valid.max():.6f} @ ζ={peak_zeta:.1f}°, Az={peak_az:.1f}°")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
