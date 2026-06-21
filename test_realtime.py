#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试：验证两种成像引擎
  1. CPU Cartesian (l, m) 波数坐标 —— 原始工作版本
  2. GPU Polar 全天极坐标 —— GPU 加速版本

运行: python test_realtime.py [--mode cpu|gpu|both]
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
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_dirty_image import (make_dirty_image_cpu,
                               load_optimized_antennas,
                               get_polar_grid_metadata)


# 按需导入辅助
def _lazy_import_gpu():
    from make_dirty_image import make_dirty_image_GPU, gpu_available, gpu_install_hint
    return make_dirty_image_GPU, gpu_available, gpu_install_hint


def _lazy_import_polar_cpu():
    from make_dirty_image import make_dirty_image_polar_cpu, _polar_cpu_available, cffi_install_hint
    return make_dirty_image_polar_cpu, _polar_cpu_available, cffi_install_hint

# ==============================================================================
# 辅助函数
# ==============================================================================

def build_visibility_matrix(watch_dir="./correlation_results", freq_bin_idx=0):
    """读取最新一批 CSV，构建 8x8 复可见度矩阵"""
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


# ==============================================================================
# 测试函数
# ==============================================================================

def test_cpu(vis, output_dir):
    """测试 CPU Cartesian (l, m) 引擎 —— 原始工作版本"""
    print("\n" + "=" * 60)
    print("  测试: CPU Cartesian (l, m) 波数坐标脏图")
    print("=" * 60)

    dirty_img, l_axis, m_axis = make_dirty_image_cpu(
        antennas_filepath="optimized_antenna_coordinates.txt",
        visibilities=vis,
        freq_mhz=150.0,
        grid_pts=256,
        fov_deg=30.0,
        apply_w_correction=True,
        filter_rfi=False
    )

    print(f"  脏图形状: {dirty_img.shape}")
    print(f"  非零像素:  {np.count_nonzero(dirty_img)}/{dirty_img.size}")

    valid = dirty_img[dirty_img != 0]
    if len(valid) > 0:
        print(f"  强度范围:  [{valid.min():.4f}, {valid.max():.4f}]")
        print(f"  峰值:      {valid.max():.4f}")

    # 保存图像
    l_max = l_axis[-1]
    m_min, m_max = m_axis[-1], m_axis[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # 左: 脏图
    ax1 = axes[0]
    ax1.set_facecolor('black')
    im = ax1.imshow(dirty_img, extent=[-l_max, l_max, m_min, m_max],
                    origin='upper', cmap='inferno', aspect='equal',
                    interpolation='bilinear')
    if len(valid) > 0:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
        im.set_clim(vmin, vmax)
    ax1.set_xlabel("l (East-West)")
    ax1.set_ylabel("m (South-North)")
    ax1.set_title("Dirty Image — CPU Cartesian (l, m)\n150 MHz, FOV=30°", fontsize=13)
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
    png_path = output_dir / "test_dirty_image_cpu.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {png_path}")
    print("  CPU Cartesian 测试通过!\n")


def test_gpu(vis, output_dir):
    """测试 GPU Polar 全天极坐标引擎"""
    make_dirty_image_GPU, gpu_available_fn, gpu_hint_fn = _lazy_import_gpu()
    gpu_ok = gpu_available_fn()

    print("\n" + "=" * 60)
    print("  测试: GPU Polar 全天极坐标脏图")
    print("=" * 60)
    print(f"  GPU 可用: {gpu_ok}")

    if not gpu_ok:
        hint = gpu_hint_fn()
        if hint:
            print(hint)
        print("  [SKIP] GPU 不可用")
        return

    dirty_img = make_dirty_image_GPU(
        antennas_filepath="optimized_antenna_coordinates.txt",
        visibilities=vis,
        freq_mhz=150.0,
        grid_pts=256,
        fov_deg=180.0,
        apply_w_correction=True,
        filter_rfi=False,
        use_gpu=gpu_ok
    )

    print(f"  脏图形状: {dirty_img.shape} (radial={dirty_img.shape[0]}, azimuthal={dirty_img.shape[1]})")
    print(f"  非零像素:  {np.count_nonzero(dirty_img)}/{dirty_img.size}")

    valid = dirty_img[dirty_img != 0]
    if len(valid) > 0:
        print(f"  强度范围:  [{valid.min():.4f}, {valid.max():.4f}]")
        print(f"  峰值:      {valid.max():.4f}")

    # 保存图像
    meta = get_polar_grid_metadata(grid_pts=256, n_radial=dirty_img.shape[0],
                                    n_azimuthal=dirty_img.shape[1])
    nr = meta['n_radial']
    na = meta['n_azimuthal']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # 左: 极坐标脏图
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

    # 极坐标参考网格
    for zc in [15, 30, 45, 60, 75, 90]:
        r = zc / 90.0
        circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                             linewidth=0.5, alpha=0.6, linestyle='--')
        ax1.add_patch(circle)
        if zc < 90:
            ax1.annotate(f'{zc}°', (0, r), color='#336699', fontsize=6,
                          ha='center', va='bottom')

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

    horizon = plt.Circle((0, 0), 1.0, fill=False, color='#4488cc', linewidth=1.2)
    ax1.add_patch(horizon)
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    backend = "GPU" if gpu_ok else "CPU-fallback"
    ax1.set_title(f"All-Sky Dirty Image (150 MHz, {backend})\nZenith-centred Polar Projection",
                  fontsize=13)
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
    png_path = output_dir / "test_dirty_image_gpu.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {png_path}")
    print("  GPU Polar 测试通过!\n")


def test_polar_cpu(vis, output_dir):
    """测试 CPU Polar 全天极坐标引擎 (C/OpenMP 或纯 Python fallback)"""
    make_dirty_image_polar_cpu, polar_avail_fn, cffi_hint_fn = _lazy_import_polar_cpu()
    polar_ok = polar_avail_fn()

    print("\n" + "=" * 60)
    print("  测试: CPU Polar 全天极坐标脏图")
    print("=" * 60)
    print(f"  C/OpenMP 引擎可用: {polar_ok}")
    if not polar_ok:
        print("  将使用纯 Python fallback (算法一致)")

    dirty_img = make_dirty_image_polar_cpu(
        antennas_filepath="optimized_antenna_coordinates.txt",
        visibilities=vis,
        freq_mhz=150.0,
        grid_pts=256,
        fov_deg=180.0,
        apply_w_correction=True,
        filter_rfi=False
    )

    print(f"  脏图形状: {dirty_img.shape} (radial={dirty_img.shape[0]}, azimuthal={dirty_img.shape[1]})")
    print(f"  非零像素:  {np.count_nonzero(dirty_img)}/{dirty_img.size}")

    valid = dirty_img[dirty_img != 0]
    if len(valid) > 0:
        print(f"  强度范围:  [{valid.min():.4f}, {valid.max():.4f}]")
        print(f"  峰值:      {valid.max():.4f}")

    # 保存图像
    meta = get_polar_grid_metadata(grid_pts=256, n_radial=dirty_img.shape[0],
                                    n_azimuthal=dirty_img.shape[1])
    nr = meta['n_radial']
    na = meta['n_azimuthal']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # 左: 极坐标脏图
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

    for zc in [15, 30, 45, 60, 75, 90]:
        r = zc / 90.0
        circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                             linewidth=0.5, alpha=0.6, linestyle='--')
        ax1.add_patch(circle)
        if zc < 90:
            ax1.annotate(f'{zc}°', (0, r), color='#336699', fontsize=6,
                          ha='center', va='bottom')

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

    horizon = plt.Circle((0, 0), 1.0, fill=False, color='#4488cc', linewidth=1.2)
    ax1.add_patch(horizon)
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_ylim(-1.15, 1.15)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("All-Sky Dirty Image (150 MHz, C/OpenMP)\nZenith-centred Polar Projection",
                  fontsize=13)
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
    png_path = output_dir / "test_dirty_image_polar_cpu.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {png_path}")
    print("  CPU Polar (C/OpenMP) 测试通过!\n")


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test dirty image engines")
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'polar_cpu', 'all'], default='all',
                        help='Which engine to test (default: all)')
    args = parser.parse_args()

    output_dir = Path("./dirty_image_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检测 GPU 可用性（按需导入）
    gpu_ok = False
    try:
        _, gpu_avail_fn, _ = _lazy_import_gpu()
        gpu_ok = gpu_avail_fn()
    except Exception:
        pass

    print("=" * 60)
    print("  脏图成像引擎测试")
    print("=" * 60)
    print(f"  GPU 可用: {gpu_ok}")
    print(f"  测试模式: {args.mode}")

    # 构建可见度矩阵
    vis = build_visibility_matrix()
    if vis is None:
        sys.exit(1)

    # 运行测试
    if args.mode in ('cpu', 'all'):
        test_cpu(vis, output_dir)

    if args.mode in ('gpu', 'all'):
        test_gpu(vis, output_dir)

    if args.mode in ('polar_cpu', 'all'):
        test_polar_cpu(vis, output_dir)

    print("=" * 60)
    print("  所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
