#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试：仅 CPU 小视场模式 (Cartesian l,m, FOV=30°)
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

from make_dirty_image import make_dirty_image_cpu, load_optimized_antennas

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
    print("  测试: CPU Cartesian (l, m) 小视场脏图")
    print("  FOV = 30°, Grid = 256x256, Freq = 150 MHz")
    print("=" * 60)

    output_dir = Path("./dirty_image_frames")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载可见度数据
    print("\n[1] 加载可见度数据...")
    vis = build_visibility_matrix()
    if vis is None:
        print("[FAIL] 无法加载可见度数据")
        return 1

    # 2. 计算脏图
    print("\n[2] 计算 CPU Cartesian 脏图...")
    t0 = time.time()
    dirty_img, l_axis, m_axis = make_dirty_image_cpu(
        antennas_filepath="optimized_antenna_coordinates.txt",
        visibilities=vis,
        freq_mhz=150.0,
        grid_pts=256,
        fov_deg=30.0,
        apply_w_correction=True,
        filter_rfi=False
    )
    elapsed = time.time() - t0

    print(f"  耗时: {elapsed:.2f}s")
    print(f"  脏图形状: {dirty_img.shape}")
    print(f"  非零像素:  {np.count_nonzero(dirty_img)}/{dirty_img.size}")

    valid = dirty_img[dirty_img != 0]
    if len(valid) > 0:
        print(f"  强度范围:  [{valid.min():.6f}, {valid.max():.6f}]")
        print(f"  峰值:      {valid.max():.6f}")
    else:
        print("  [WARN] 脏图全为零!")
        return 1

    # 3. 保存图像
    print("\n[3] 保存测试图像...")
    l_max = l_axis[-1]
    m_min, m_max = m_axis[-1], m_axis[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

    # 左: 脏图
    ax1 = axes[0]
    ax1.set_facecolor('black')
    im = ax1.imshow(dirty_img, extent=[-l_max, l_max, m_min, m_max],
                    origin='upper', cmap='inferno', aspect='equal',
                    interpolation='bilinear')
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    im.set_clim(vmin, vmax)
    ax1.set_xlabel("l (East-West)")
    ax1.set_ylabel("m (South-North)")
    ax1.set_title("Dirty Image — CPU Cartesian (l, m)\n150 MHz, FOV=30°, Grid=256x256", fontsize=13)
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
    png_path = output_dir / "test_cpu_small_fov.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图像已保存: {png_path}")

    print("\n" + "=" * 60)
    print("  CPU 小视场模式测试通过!")
    print(f"  耗时: {elapsed:.2f}s, 峰值: {valid.max():.6f}")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
