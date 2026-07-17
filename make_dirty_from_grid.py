#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 grid_visibilities.py 生成的格点化可见度 (.npz) 计算脏图 (dirty image)。

脏图 = 2D 逆傅里叶变换 (IFFT2) 后的 (l, m) 空间图像：
    I_D(l,m) = FFT^{-1}[ V_grid(u,v) / H_w(u,v) ]

其中 H_w 为权重 hits 计数（即 uv 覆盖采样密度）。空 bin 填 0。

用法:
    python make_dirty_from_grid.py [--npz grid_vis.npz] [--scale linear|log]
                                    [--fov 180] [--output integrated_images]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import glob
import os


def load_gridded_vis(npz_path):
    """加载 grid_visibilities.py 输出的 npz 文件。

    Returns
    -------
    V_grid : ndarray (bins, bins), complex128
    H_w    : ndarray (bins, bins), float64
    xu     : ndarray (bins,), float64
    yv     : ndarray (bins,), float64
    meta   : dict
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # 优先用复数数组
    if 'V_grid' in data:
        V_grid = np.asarray(data['V_grid'], dtype=np.complex128)
    else:
        V_real = data['V_real']
        V_imag = data['V_imag']
        V_grid = V_real + 1j * V_imag
    
    H_w = data['H_w']
    xu = data['xu']
    yv = data['yv']
    
    meta = {
        'uv_lim': float(data.get('uv_lim', np.max(np.abs(xu)))),
        'bins': int(data.get('bins', len(xu))),
        'weight_scheme': str(data.get('weight_scheme', 'unknown')),
        'path': npz_path,
    }
    
    return V_grid, H_w, xu, yv, meta


def compute_dirty_image(V_grid, H_w, uv_lim, hanning_smooth=True):
    """
    从格点化可见度计算脏图。

    算法：
      1. V_avg = V_grid / max(H_w, eps)  → 每个 UV cell 的平均可见度
      2. 零权重 cell → 0
      3. (可选) Hanning 平滑抑制 Gibbs 振铃
      4. dirty_lm = np.fft.ifft2(V_avg)
      5. fftshift → 中心为 DC (l=0,m=0)
      6. 取实部

    坐标关系：
      UV 采样间隔 du = 2*uv_lim / bins
      图像像素大小 dl = 1/(2*uv_lim)  (方向余弦，单位 1)
      l_max = (bins/2) * dl = bins / (4*uv_lim)
      天空仅存在于 |l|,|m| ≤ 1 的圆内，因此脏图有用区域在中心附近。

    Parameters
    ----------
    V_grid : ndarray (bins, bins), complex128
    H_w : ndarray (bins, bins), float64
    uv_lim : float
        UV 网格范围上限 (max |u|)。
    hanning_smooth : bool
        是否对 UV 平面应用 Hanning 窗，抑制边缘截断导致的 Gibbs 振铃。

    Returns
    -------
    dirty : ndarray (bins, bins), float64
    l_axis : ndarray (bins,)
    m_axis : ndarray (bins,)
    """
    bins = V_grid.shape[0]
    
    # 1. Per-cell average visibility
    H_safe = np.maximum(H_w, 1e-12)
    V_avg = np.where(H_w > 0, V_grid / H_safe, 0.0 + 0.0j)
    
    # 2. Hanning 窗 --- 抑制 UV 边缘截断的 Gibbs 振铃
    if hanning_smooth and bins > 2:
        win_u = np.hanning(bins)
        win_v = np.hanning(bins)
        win_2d = np.outer(win_u, win_v)
        V_avg = V_avg * win_2d
    
    # 3. 2D IFFT
    dirty_raw = np.fft.ifft2(V_avg)
    
    # 4. fftshift
    dirty_shifted = np.fft.fftshift(dirty_raw)
    
    # 5. 取实部（理想脏图为实数）
    dirty = np.real(dirty_shifted)
    
    # 6. 构建 (l, m) 坐标轴
    dl = 1.0 / (2.0 * uv_lim)
    l_axis = np.linspace(-bins/2, bins/2 - 1, bins) * dl
    m_axis = np.linspace(-bins/2, bins/2 - 1, bins) * dl
    
    return dirty, l_axis, m_axis


def crop_to_valid_sky(dirty, l_axis, m_axis, l_max=1.0):
    """裁剪脏图到方向余弦 |l|,|m| ≤ l_max 的区域。"""
    l_valid = np.abs(l_axis) <= l_max
    m_valid = np.abs(m_axis) <= l_max
    
    l_idx = np.where(l_valid)[0]
    m_idx = np.where(m_valid)[0]
    
    if len(l_idx) == 0 or len(m_idx) == 0:
        return dirty, l_axis, m_axis
    
    l0, l1 = l_idx[0], l_idx[-1] + 1
    m0, m1 = m_idx[0], m_idx[-1] + 1
    
    dirty_cropped = dirty[m0:m1, l0:l1]
    l_axis_cropped = l_axis[l0:l1]
    m_axis_cropped = m_axis[m0:m1]
    
    return dirty_cropped, l_axis_cropped, m_axis_cropped


def plot_dirty_image(dirty, l_axis, m_axis, output_path, meta,
                      fov_deg=180.0, scale='log', title_note=''):
    """
    保存脏图为 PNG。

    显示范围：方向余弦 |l|,|m| ≤ 1.0（天空边界），并叠加地平线圆。
    """
    from matplotlib.colors import SymLogNorm, Normalize

    bins = dirty.shape[0]
    l_max = np.max(np.abs(l_axis)) if len(l_axis) > 0 else 1.0
    m_max = np.max(np.abs(m_axis)) if len(m_axis) > 0 else 1.0
    display_lmax = min(1.0, l_max)  # 只显示天空有效区域

    # ── 显示变换 ──
    if scale == 'log':
        valid = dirty[dirty != 0]
        if len(valid) > 0:
            vmax = float(np.max(np.abs(valid)))
            linthresh = max(
                float(np.percentile(np.abs(valid[valid > 0]), 1)) if np.any(valid > 0) else vmax * 1e-4,
                vmax * 1e-6
            )
        else:
            vmax, linthresh = 1.0, 1e-6
        norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10)
        cbar_label = 'Intensity (symlog)'
        display_data = dirty
    else:
        valid = dirty[dirty != 0]
        if len(valid) > 0:
            vmin = float(np.percentile(valid, 2))
            vmax = float(np.percentile(valid, 98))
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cbar_label = 'Intensity (linear)'
        display_data = dirty

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1a2e')
    ax.set_facecolor('black')
    ax.set_aspect('equal')

    im = ax.imshow(display_data,
                   extent=[-l_max, l_max, -m_max, m_max],
                   origin='lower', cmap='inferno', aspect='equal',
                   interpolation='bilinear', norm=norm)

    # ── 地平线圆 (l² + m² ≤ 1) ──
    horizon = plt.Circle((0, 0), 1.0, fill=False, color='#4488cc',
                          linewidth=1.5, alpha=0.8, linestyle='--')
    ax.add_patch(horizon)

    # ── 天顶角参考圆 ──
    for zc in [15, 30, 45, 60, 75]:
        r = np.sin(np.radians(zc))  # l = sin(ζ) for unit circle
        if r <= 1.0:
            circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                                linewidth=0.5, alpha=0.5, linestyle=':')
            ax.add_patch(circle)

    # ── 方位角参考线 ──
    az_labels = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
    for az_deg in [0, 90, 180, 270]:
        theta = np.radians(90 - az_deg)
        dx = np.cos(theta)
        dy = np.sin(theta)
        ax.plot([0, dx], [0, dy], color='#336699',
                linewidth=0.4, alpha=0.4, linestyle=':')
        label = az_labels.get(az_deg, f'{az_deg}°')
        ax.annotate(label, (dx * 1.08, dy * 1.08),
                    color='#6699cc', fontsize=8,
                    ha='center', va='center', alpha=0.8)

    ax.set_xlim(-display_lmax * 1.15, display_lmax * 1.15)
    ax.set_ylim(-display_lmax * 1.15, display_lmax * 1.15)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    ax.set_xlabel("l (East-West)", color='white')
    ax.set_ylabel("m (South-North)", color='white')
    ax.tick_params(colors='white', labelsize=8)

    scale_note = f" [{scale.upper()}]"
    title = (f"Dirty Image from Gridded Visibilities{title_note}\n"
             f"Grid: {bins}×{bins} bins  |  Weight: {meta['weight_scheme']}{scale_note}  |  "
             f"UV range: ±{meta['uv_lim']:.1f}λ")
    ax.set_title(title, color='white', fontsize=13)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="从格点化可见度 (.npz) 计算并显示脏图"
    )
    parser.add_argument('--npz', default=None,
                        help='grid_visibilities 输出的 npz 文件路径 '
                             '(默认: 自动寻找 integrated_images 下最新的 4096bins npz)')
    parser.add_argument('--fov', type=float, default=180.0,
                        help='显示视场角 (度, 默认: 180)')
    parser.add_argument('--scale', choices=['linear', 'log'], default='log',
                        help='显示 scale (默认: log)')
    parser.add_argument('--output', default='integrated_images',
                        help='输出目录 (默认: integrated_images)')
    parser.add_argument('--no-hanning', action='store_true',
                        help='关闭 Hanning 窗平滑（保留更多细节但可能增多振铃）')
    parser.add_argument('--npy', action='store_true',
                        help='额外保存脏图为 .npy 文件')
    
    args = parser.parse_args()
    
    # ── 查找 npz 文件 ──
    npz_path = args.npz
    if npz_path is None:
        # 自动查找最新的 4096bins npz
        candidates = sorted(glob.glob('integrated_images/gridded_vis_*_4096bins.npz'))
        if not candidates:
            # 退而求其次找任意 npz
            candidates = sorted(glob.glob('integrated_images/gridded_vis_*.npz'))
        if not candidates:
            print("[ERROR] 未找到任何 gridded_vis_*.npz 文件！")
            print("请先运行 grid_visibilities.py 生成数据，或用 --npz 指定路径。")
            sys.exit(1)
        npz_path = candidates[-1]
        print(f"自动选择: {npz_path}")
    
    # ── 加载数据 ──
    print(f"\n{'='*60}")
    print(f"  加载: {npz_path}")
    V_grid, H_w, xu, yv, meta = load_gridded_vis(npz_path)
    
    bins = meta['bins']
    uv_lim = meta['uv_lim']
    print(f"  UV 网格:     {bins}×{bins}")
    print(f"  UV 范围:     ±{uv_lim:.1f}λ")
    print(f"  非零 bins:   {np.count_nonzero(H_w):,} / {bins*bins:,} ({100*np.count_nonzero(H_w)/(bins*bins):.1f}%)")
    print(f"  max hits:    {int(H_w.max())}")
    print(f"  加权方案:    {meta['weight_scheme']}")
    
    # ── 计算脏图 ──
    print(f"\n  计算脏图 (IFFT2)...")
    dirty, l_axis, m_axis = compute_dirty_image(
        V_grid, H_w, uv_lim, hanning_smooth=not args.no_hanning
    )
    
    # 裁剪到天空有效区域 (|l|,|m| ≤ 1)
    dirty_cropped, l_axis_cropped, m_axis_cropped = crop_to_valid_sky(
        dirty, l_axis, m_axis, l_max=1.0
    )
    
    # 统计信息
    valid = dirty[dirty != 0]
    if len(valid) > 0:
        print(f"  脏图峰值:    {np.max(valid):.4f}")
        print(f"  脏图最小值:  {np.min(valid):.4f}")
        print(f"  动态范围:    {np.max(valid)/np.median(np.abs(valid)):.1f}x (peak/median|V|)")
    
    print(f"  显示范围:    |l|,|m| ≤ 1.0 (方向余弦)")
    print(f"  裁剪后尺寸:  {dirty_cropped.shape[0]}×{dirty_cropped.shape[1]} pixels")
    
    # ── 保存 ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    basename = Path(npz_path).stem  # e.g. gridded_vis_20260630_conj_4096bins
    is_conj = '_conj' in basename
    
    png_name = basename.replace('gridded_vis', 'dirty_image')
    png_path = output_dir / f"{png_name}.png"
    
    conj_note = ' (with conjugates)' if is_conj else ''
    hanning_note = '' if args.no_hanning else ' [Hanning]'
    
    plot_dirty_image(dirty_cropped, l_axis_cropped, m_axis_cropped, png_path, meta,
                      fov_deg=args.fov, scale=args.scale,
                      title_note=conj_note + hanning_note)
    
    if args.npy:
        npy_path = output_dir / f"{png_name}.npy"
        np.save(npy_path, dirty)
        print(f"  保存: {npy_path}")
    
    print(f"\n{'='*60}")
    print(f"  完成！")


if __name__ == "__main__":
    main()
