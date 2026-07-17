#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uv_coverage_plot.py — GPU-accelerated high-precision UV coverage visualization
=====================================================
无格点（no gridding）：直接计算全部帧×通道×基线的 (u,v) 坐标，
以高分辨率 2D 直方图展示 UV 平面覆盖，用于检查条纹/空隙。

数据说明:
  CSV 的 frequency_hz 列记录基带频率 (Hz)，中心频率 150 MHz。
  天空频率 = 150 MHz + frequency_hz/1e6。
  4096 个 FFT bin 排列: 前 2049 个 0→+50 MHz (150→200 MHz),
  后 2047 个 -50→0 MHz (100→150 MHz), Nyquist 处 wrap。
  每 bin 带宽 ≈ 100 MHz/4096 ≈ 24.4 kHz。

原理:
  baseline_uv = (天线差分旋转后的坐标) / wavelength
  每个时刻(时角) × 每个频率(波长) → 28 条基线的 (u,v) 坐标。
  全量 ~1593帧 × 4096ch × 56(含共轭) ≈ 365M 个采样点。

特点:
  - GPU 批量计算：一次广播算完所有帧所有通道的 UVW
  - 不加载可见度数据（只读频率信息），极致速度
  - 高分辨率直方图（4096×4096）log 色阶显示
  - 同时生成全图、中心和象限放大图
"""

import numpy as np
import time
import os
import sys
import argparse
from collections import OrderedDict

# ── 尝试加载 CuPy ──
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ── 从现有模块导入工具函数 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_dirty_image import load_optimized_antennas
from integrate_dirty_image import (
    discover_frames_by_date, compute_hour_angle,
)


# ═══════════════════════════════════════════════════════════════════
# 轻量级频率读取（只读频率列，不加载可见度数据）
# ═══════════════════════════════════════════════════════════════════
def read_sky_frequencies(file_map, center_freq_mhz=150.0,
                         n_channels=0, n_sub_per_channel=1,
                         max_bins=4100):
    """读取全部频率通道，每个通道对应一个 FFT bin。

    数据实际有 4096 个频率 bin，覆盖 -50 ~ +50 MHz（基带），
    对应天空频率 100 ~ 200 MHz（中心 150 MHz）。
    每个 bin 带宽 ≈ 100 MHz / 4096 ≈ 24.4 kHz。

    UV 径向连续性分析：
      max_baseline ≈ 15λ, bin_bw ≈ 24.4 kHz, f_center ≈ 150 MHz
      → du_per_ch = 15 × 24.4k / 150M ≈ 0.0024 λ
      histogram_bin ≈ 30λ / 4096 ≈ 0.0073 λ
      → du_per_ch < histogram_bin → 4096 通道天然径向连续，无需子采样。

    Parameters
    ----------
    file_map : dict
        帧文件映射（仅用于读取频率列）。
    center_freq_mhz : float
        系统中心频率 (MHz)。
    n_channels : int
        使用的通道数。0 = 全部（默认，推荐）；>0 = 从全频段均匀降采样。
    n_sub_per_channel : int
        每个 bin 带宽内的子采样数（1 = 仅用 bin 中心，默认且推荐）。
    max_bins : int
        CSV 读取行数上限。

    Returns
    -------
    freqs_mhz : ndarray (n_channels * n_sub_per_channel,)
        天空频率 (MHz)。
    bin_width_hz : float
        单个 FFT bin 的带宽 (Hz)。
    """
    import pandas as pd
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=max_bins)
    freq_hz = df['frequency_hz'].values.astype(np.float64)

    total_bins = len(freq_hz)
    # FFT 频率数组在 Nyquist 点 wrap（+50→-50 MHz），用相邻 bin 间距计算带宽
    bin_width_hz = abs(freq_hz[1] - freq_hz[0])           # ~24.4 kHz

    # ── 选取通道 ──
    if n_channels <= 0 or n_channels > total_bins:
        n_channels = total_bins
        bin_indices = np.arange(total_bins)               # 使用全部 bin
    else:
        # 从全频段均匀降采样
        bin_indices = np.linspace(0, total_bins - 1, n_channels, dtype=np.int32)

    f_center_hz = freq_hz[bin_indices]                    # (n_channels,)

    # ── 可选：bin 带宽内子采样 ──
    if n_sub_per_channel <= 1:
        all_freqs_hz = f_center_hz                        # 仅用 bin 中心
    else:
        half_bw = bin_width_hz / 2.0
        sub_offsets = np.linspace(-half_bw, half_bw, n_sub_per_channel)
        all_freqs_hz = f_center_hz[:, None] + sub_offsets[None, :]
        all_freqs_hz = all_freqs_hz.ravel()               # (n_ch * n_sub,)

    all_freqs_mhz = center_freq_mhz + all_freqs_hz / 1e6
    return all_freqs_mhz, bin_width_hz


# ═══════════════════════════════════════════════════════════════════
# GPU 批量 UVW 计算
# ═══════════════════════════════════════════════════════════════════
def compute_all_uvw_gpu_full(antennas, hour_angles, wavelengths,
                             latitude_deg, include_conjugates=True):
    """
    GPU 批量计算全部 UVW 坐标（完整版，含赤纬参数）。

    Parameters
    ----------
    antennas : ndarray (8, 2)
    hour_angles : ndarray (n_frames,)
    wavelengths : ndarray (n_channels,)
    latitude_deg : float
        观测纬度（赤纬 = 天顶）。
    include_conjugates : bool

    Returns
    -------
    u_all : ndarray (total_points,)
    v_all : ndarray (total_points,)
    w_all : ndarray (total_points,) or None
    """
    xp = cp
    n_frames = len(hour_angles)
    n_ch = len(wavelengths)
    n_bl = 28

    # ── 天线坐标 ──
    x = xp.asarray(antennas[:, 0], dtype=xp.float64)  # (8,)
    y = xp.asarray(antennas[:, 1], dtype=xp.float64)  # (8,)

    # ── 时角三角函数 ──
    ha_rad = xp.radians(xp.asarray(hour_angles, dtype=xp.float64))  # (F,)
    sin_H = xp.sin(ha_rad)
    cos_H = xp.cos(ha_rad)

    # ── 赤纬三角函数（标量） ──
    dec_rad = xp.float64(np.radians(latitude_deg))
    sin_D = xp.sin(dec_rad)
    cos_D = xp.cos(dec_rad)

    # ── 基线配对 ──
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    idx_i = xp.array([p[0] for p in pairs], dtype=xp.int32)
    idx_j = xp.array([p[1] for p in pairs], dtype=xp.int32)

    # ── 天线坐标 (u,v,w) in meters ──
    # ant_u[f, a] = sin_H[f] * x[a] + cos_H[f] * y[a]
    ant_u = sin_H[:, None] * x[None, :] + cos_H[:, None] * y[None, :]      # (F, 8)
    # ant_v[f, a] = -sin_D * cos_H[f] * x[a] + sin_D * sin_H[f] * y[a]
    ant_v = (-sin_D * cos_H[:, None] * x[None, :]
             + sin_D * sin_H[:, None] * y[None, :])                          # (F, 8)
    # ant_w[f, a] = cos_D * cos_H[f] * x[a] - cos_D * sin_H[f] * y[a]
    ant_w = (cos_D * cos_H[:, None] * x[None, :]
             - cos_D * sin_H[:, None] * y[None, :])                          # (F, 8)

    # ── 基线差分 (in meters) ──
    # bl_u[f, bl] = ant_u[f, i] - ant_u[f, j]
    bl_u_m = ant_u[:, idx_i] - ant_u[:, idx_j]   # (F, 28)
    bl_v_m = ant_v[:, idx_i] - ant_v[:, idx_j]   # (F, 28)
    bl_w_m = ant_w[:, idx_i] - ant_w[:, idx_j]   # (F, 28)

    # ── 除以波长 (broadcast: meters / wavelength → units of λ) ──
    # bl_u[f, c, bl] = bl_u_m[f, bl] / wl[c]
    # shape: (F, 28) → (F, 1, 28) / (C,) → (F, C, 28)
    wl_g = xp.asarray(wavelengths, dtype=xp.float64)  # (C,)
    bl_u = bl_u_m[:, None, :] / wl_g[None, :, None]    # (F, C, 28)
    bl_v = bl_v_m[:, None, :] / wl_g[None, :, None]    # (F, C, 28)

    # ── 可选：加入共轭 ──
    if include_conjugates:
        u_flat = xp.concatenate([
            bl_u.ravel(), -bl_u.ravel()
        ])
        v_flat = xp.concatenate([
            bl_v.ravel(), -bl_v.ravel()
        ])
    else:
        u_flat = bl_u.ravel()
        v_flat = bl_v.ravel()

    return xp.asnumpy(u_flat), xp.asnumpy(v_flat)


# ═══════════════════════════════════════════════════════════════════
# CPU 批量 UVW 计算（纯 numpy，无 GPU 时的回退方案）
# ═══════════════════════════════════════════════════════════════════
def compute_all_uvw_cpu(antennas, hour_angles, wavelengths,
                        latitude_deg, include_conjugates=True):
    """CPU 向量化批量 UVW 计算。"""
    x = antennas[:, 0]  # (8,)
    y = antennas[:, 1]  # (8,)

    ha_rad = np.radians(hour_angles)  # (F,)
    sin_H = np.sin(ha_rad)
    cos_H = np.cos(ha_rad)

    dec_rad = np.radians(latitude_deg)
    sin_D = np.sin(dec_rad)
    cos_D = np.cos(dec_rad)

    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    idx_i = np.array([p[0] for p in pairs], dtype=np.int32)
    idx_j = np.array([p[1] for p in pairs], dtype=np.int32)

    # 天线坐标 (meters)
    ant_u = sin_H[:, None] * x[None, :] + cos_H[:, None] * y[None, :]      # (F, 8)
    ant_v = (-sin_D * cos_H[:, None] * x[None, :]
             + sin_D * sin_H[:, None] * y[None, :])                          # (F, 8)

    # 基线差分 (meters)
    bl_u_m = ant_u[:, idx_i] - ant_u[:, idx_j]   # (F, 28)
    bl_v_m = ant_v[:, idx_i] - ant_v[:, idx_j]   # (F, 28)

    # 除以波长
    bl_u = bl_u_m[:, None, :] / wavelengths[None, :, None]  # (F, C, 28)
    bl_v = bl_v_m[:, None, :] / wavelengths[None, :, None]  # (F, C, 28)

    if include_conjugates:
        u_flat = np.concatenate([bl_u.ravel(), -bl_u.ravel()])
        v_flat = np.concatenate([bl_v.ravel(), -bl_v.ravel()])
    else:
        u_flat = bl_u.ravel()
        v_flat = bl_v.ravel()

    return u_flat, v_flat


# ═══════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════
def plot_uv_coverage(u_all, v_all, output_dir, date_str, bins=4096,
                     include_conjugates=True, dpi=200):
    """生成 UV 覆盖图（高分辨率 2D 直方图）。"""

    os.makedirs(output_dir, exist_ok=True)
    label = "_conj" if include_conjugates else ""
    n_pts = len(u_all)

    # ── 确定 UV 范围（取 99.9% 分位数，剔除极端离群点） ──
    u_lim = np.percentile(np.abs(u_all), 99.9)
    v_lim = np.percentile(np.abs(v_all), 99.9)
    lim = max(u_lim, v_lim) * 1.05

    print(f"\n  UV 范围: ±{lim:.1f} λ  ({n_pts:,} 采样点)")

    # ── 2D 直方图 ──
    t0 = time.time()
    H, xu, yv = np.histogram2d(u_all, v_all, bins=bins,
                               range=[[-lim, lim], [-lim, lim]])
    t_hist = time.time() - t0
    print(f"  2D 直方图: {bins}×{bins} bins ({t_hist:.1f}s)")

    # ── 创建多面板图 ──
    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.28)

    # 全图
    ax0 = fig.add_subplot(gs[0, :2])
    # Log 色阶
    H_log = np.where(H > 0, H, 1)
    vmin = max(H_log[H > 0].min(), 1)
    vmax = H_log.max()
    im = ax0.pcolormesh(xu, yv, H_log.T, norm=LogNorm(vmin=vmin, vmax=vmax),
                         cmap='inferno', rasterized=True)
    ax0.set_xlabel('u (λ)', fontsize=13)
    ax0.set_ylabel('v (λ)', fontsize=13)
    ax0.set_title(f'UV Coverage — {date_str}  ({n_pts:,} pts, '
                  f'{bins}×{bins} bins, log scale)',
                  fontsize=14, fontweight='bold')
    ax0.axhline(0, color='white', lw=0.5, alpha=0.4)
    ax0.axvline(0, color='white', lw=0.5, alpha=0.4)
    ax0.set_aspect('equal')
    plt.colorbar(im, ax=ax0, label='Hits per bin')

    # 中心放大 (×4)
    ax1 = fig.add_subplot(gs[0, 2])
    zoom = lim / 4
    x_mask = (xu[:-1] >= -zoom) & (xu[:-1] <= zoom)
    y_mask = (yv[:-1] >= -zoom) & (yv[:-1] <= zoom)
    H_zoom = H[np.ix_(x_mask, y_mask)]
    if H_zoom.size > 0:
        H_zoom_log = np.where(H_zoom > 0, H_zoom, 1)
        vmin_z = max(H_zoom_log[H_zoom > 0].min() if H_zoom.max() > 0 else 1, 1)
        vmax_z = H_zoom_log.max() if H_zoom.max() > 0 else 1
        im1 = ax1.pcolormesh(xu[:-1][x_mask], yv[:-1][y_mask], H_zoom_log.T,
                              norm=LogNorm(vmin=vmin_z, vmax=vmax_z),
                              cmap='inferno', rasterized=True)
    ax1.set_xlabel('u (λ)', fontsize=11)
    ax1.set_ylabel('v (λ)', fontsize=11)
    ax1.set_title(f'Center zoom (±{zoom:.0f} λ)', fontsize=12)
    ax1.axhline(0, color='white', lw=0.5, alpha=0.4)
    ax1.axvline(0, color='white', lw=0.5, alpha=0.4)
    ax1.set_aspect('equal')

    # 右上象限放大
    ax2 = fig.add_subplot(gs[1, 0])
    qu = lim / 3
    qx = (xu[:-1] >= 0) & (xu[:-1] <= qu)
    qy = (yv[:-1] >= 0) & (yv[:-1] <= qu)
    H_qu = H[np.ix_(qx, qy)]
    if H_qu.size > 0:
        H_qu_log = np.where(H_qu > 0, H_qu, 1)
        vmin_q = max(H_qu_log[H_qu > 0].min() if H_qu.max() > 0 else 1, 1)
        vmax_q = H_qu_log.max() if H_qu.max() > 0 else 1
        ax2.pcolormesh(xu[:-1][qx], yv[:-1][qy], H_qu_log.T,
                        norm=LogNorm(vmin=vmin_q, vmax=vmax_q),
                        cmap='inferno', rasterized=True)
    ax2.set_xlabel('u (λ)', fontsize=11)
    ax2.set_ylabel('v (λ)', fontsize=11)
    ax2.set_title(f'Quadrant (+u,+v) ≤{qu:.0f} λ', fontsize=12)
    ax2.set_aspect('equal')

    # 投影：沿 u 轴的计数分布
    ax3 = fig.add_subplot(gs[1, 1])
    u_proj = H.sum(axis=1)
    u_centers = 0.5 * (xu[:-1] + xu[1:])
    ax3.fill_between(u_centers, u_proj, alpha=0.7, color='steelblue')
    ax3.set_xlabel('u (λ)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('Projection on u-axis', fontsize=12)
    ax3.set_xlim(-lim, lim)

    # 投影：沿 v 轴的计数分布
    ax4 = fig.add_subplot(gs[1, 2])
    v_proj = H.sum(axis=0)
    v_centers = 0.5 * (yv[:-1] + yv[1:])
    ax4.fill_between(v_centers, v_proj, alpha=0.7, color='darkorange')
    ax4.set_xlabel('v (λ)', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Projection on v-axis', fontsize=12)
    ax4.set_xlim(-lim, lim)

    # ── 保存 ──
    fname = f'{output_dir}/uv_coverage_{date_str}{label}_{bins}bins.png'
    fig.savefig(fname, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  保存: {fname}")

    # ── 统计信息 ──
    print(f"\n  UV 覆盖统计:")
    print(f"    总采样点数:      {n_pts:>12,}")
    print(f"    非零 bins:       {np.count_nonzero(H):>12,}  / {bins*bins:,}")
    print(f"    最大 hits/bin:   {int(H.max()):>12}")
    print(f"    中位 hits/bin:   {np.median(H[H > 0]):>12.1f}")
    print(f"    平均 hits/bin:   {H[H > 0].mean():>12.1f}")
    print(f"    UV coverage area: {lim**2:.0f} lambda^2")

    return H, xu, yv


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════
def main():
    # Fix encoding on Windows
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(
        description='UV Coverage Plot — GPU-accelerated, no gridding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python uv_coverage_plot.py                        # 全量 4096ch, GPU
  python uv_coverage_plot.py --bins 8192 --no-conj  # 更高分辨率
  python uv_coverage_plot.py --gpu --date 20260630  # 指定日期
  python uv_coverage_plot.py --cpu --max-frames 100 # CPU 测试
  python uv_coverage_plot.py --nch 256 --nsub 4     # 降采样 256ch×4sub (节省内存)
        """.strip()
    )
    parser.add_argument('--dir', default='correlation_results',
                        help='CSV 数据目录 (default: correlation_results)')
    parser.add_argument('--date', default=None,
                        help='日期 YYYYMMDD (default: 自动选择第一个日期)')
    parser.add_argument('--freq', type=float, default=150.0,
                        help='中心频率 MHz (default: 150)')
    parser.add_argument('--nch', type=int, default=0,
                        help='频率通道数 (0=全部4096个, >0=从全频段均匀降采样)')
    parser.add_argument('--antennas', default='optimized_antenna_coordinates.txt',
                        help='天线坐标文件')
    parser.add_argument('--lat', type=float, default=None,
                        help='观测纬度 (default: 从 observer_location.txt 读取)')
    parser.add_argument('--lon', type=float, default=None,
                        help='观测经度 (default: 从 observer_location.txt 读取)')
    parser.add_argument('--utc-offset', type=float, default=0,
                        help='UTC 时区偏移 (hours)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='最大帧数 (0=全部)')
    parser.add_argument('--bins', type=int, default=4096,
                        help='直方图分辨率 bins (default: 4096)')
    parser.add_argument('--nsub', type=int, default=1,
                        help='每bin带宽内子采样数 (1=仅bin中心, default: 1, 4096ch已径向连续)')
    parser.add_argument('--no-conj', action='store_true',
                        help='不使用共轭基线')
    parser.add_argument('--gpu', action='store_true', default=None,
                        help='强制使用 GPU (默认: 有 GPU 则自动使用)')
    parser.add_argument('--cpu', action='store_true',
                        help='强制使用 CPU')
    parser.add_argument('--output', default='integrated_images',
                        help='输出目录 (default: integrated_images)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='图片 DPI (default: 200)')
    parser.add_argument('--save-raw', action='store_true',
                        help='保存原始 (u,v) 坐标到 .npz 文件')
    args = parser.parse_args()

    # ── 确定使用 GPU 还是 CPU ──
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = GPU_AVAILABLE
        if not GPU_AVAILABLE:
            print("[WARN] CuPy 不可用，回退到 CPU 模式")
    else:
        use_gpu = GPU_AVAILABLE

    engine = "GPU (CuPy)" if use_gpu else "CPU (NumPy)"
    print(f"引擎: {engine}")
    print(f"={60}")

    # ── 读取观测者位置 ──
    if args.lat is None or args.lon is None:
        loc_file = 'observer_location.txt'
        if os.path.exists(loc_file):
            with open(loc_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    parts = line.split(',')
                    if len(parts) == 2:
                        try:
                            args.lat = float(parts[0].strip())
                            args.lon = float(parts[1].strip())
                        except ValueError:
                            continue
        if args.lat is None:
            args.lat = 40.0
        if args.lon is None:
            args.lon = 116.0
    print(f"  观测位置: lat={args.lat}°, lon={args.lon}°")

    # ── 加载天线 ──
    t0 = time.time()
    antennas = load_optimized_antennas(args.antennas)
    print(f"  天线: {args.antennas} (shape={antennas.shape}) "
          f"({time.time()-t0:.1f}s)")

    # ── 发现帧 ──
    t0 = time.time()
    by_date = discover_frames_by_date(args.dir, args.date)
    if not by_date:
        print(f"[ERROR] 在 '{args.dir}' 中未发现数据")
        return
    date_str = args.date or next(iter(by_date.keys()))
    frames_by_ts = by_date[date_str]
    timestamps_all = list(frames_by_ts.keys())
    n_total = len(timestamps_all)
    print(f"  发现日期: {date_str}, {n_total} 帧 ({time.time()-t0:.1f}s)")

    # ── 限制帧数 ──
    if args.max_frames > 0 and args.max_frames < n_total:
        timestamps_all = timestamps_all[-args.max_frames:]
        print(f"  限制为最后 {args.max_frames} 帧")
    n_frames = len(timestamps_all)

    # ── 读取频率信息（含通道内带宽子采样） ──
    t0 = time.time()
    first_file_map = frames_by_ts[timestamps_all[0]]
    freq_sky_mhz, bin_width_hz = read_sky_frequencies(
        first_file_map, args.freq, args.nch,
        n_sub_per_channel=args.nsub
    )
    n_freqs = len(freq_sky_mhz)
    wavelengths = 299.792458 / freq_sky_mhz  # (n_freqs,)
    print(f"  天空频率: {freq_sky_mhz.min():.1f} ~ {freq_sky_mhz.max():.1f} MHz "
          f"({n_freqs:,}通道, 中心={args.freq} MHz, "
          f"bin_bw={bin_width_hz:.1f} Hz, λ={wavelengths.min():.3f}~{wavelengths.max():.3f} m) "
          f"({time.time()-t0:.1f}s)")
    print(f"    通道排列: 前2049个 150→200 MHz, 后2047个 100→150 MHz (Nyquist wrap)")

    # ── 计算所有时角 ──
    t0 = time.time()
    hour_angles = np.array([
        compute_hour_angle(ts, args.lon, args.utc_offset)
        for ts in timestamps_all
    ], dtype=np.float64)
    print(f"  时角: {hour_angles[0]:.1f}° ~ {hour_angles[-1]:.1f}° "
          f"({time.time()-t0:.1f}s)")

    # ── 批量计算 UVW ──
    include_conj = not args.no_conj
    n_bl = 56 if include_conj else 28
    expected_pts = n_frames * n_freqs * n_bl
    print(f"\n  计算 UVW: {n_frames}帧 × {n_freqs:,}ch × {n_bl}基线 "
          f"= {expected_pts:,} 点")

    t0 = time.time()
    if use_gpu:
        u_all, v_all = compute_all_uvw_gpu_full(
            antennas, hour_angles, wavelengths,
            latitude_deg=args.lat,
            include_conjugates=include_conj
        )
    else:
        u_all, v_all = compute_all_uvw_cpu(
            antennas, hour_angles, wavelengths,
            latitude_deg=args.lat,
            include_conjugates=include_conj
        )
    t_uvw = time.time() - t0
    print(f"  UVW 计算完成: {t_uvw:.2f}s "
          f"({t_uvw/n_frames*1000:.1f}ms/frame, {len(u_all):,} 点)")

    # ── 可选：保存原始数据 ──
    if args.save_raw:
        raw_path = f'{args.output}/uv_coverage_raw_{date_str}.npz'
        os.makedirs(args.output, exist_ok=True)
        np.savez_compressed(raw_path, u=u_all, v=v_all,
                            freq_sky_mhz=freq_sky_mhz,
                            hour_angles=hour_angles)
        print(f"  原始数据保存: {raw_path} "
              f"({os.path.getsize(raw_path)/1024**2:.1f} MB)")

    # ── 绘图 ──
    plot_uv_coverage(u_all, v_all, args.output, date_str,
                     bins=args.bins, include_conjugates=include_conj,
                     dpi=args.dpi)

    print(f"\n{'='*60}")
    print(f"完成！总耗时: {time.time() - t0 + t_uvw:.1f}s")


if __name__ == '__main__':
    main()
