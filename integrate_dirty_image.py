#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrate Dirty Image — 同一天连续观测的时间积分天空成像
============================================================
将同一天内连续多个 frame 的可见度数据按基线累加（时间积分），生成一张
信噪比更高的积分天空图像。

提供两种积分模式：
  1. **snapshot**  (不考虑天体位移)：将所有 frame 的可见度矩阵直接叠加取
     平均，等价于假设天空在观测期间静止不变。适用于积分时间较短或
     视场中心指向天极的场景。

  2. **tracked**   (考虑天体位移)：对每个 frame 计算其对应的时角，用
     时变 (u,v,w) 坐标做 Direct 3D Fourier Integration，再将各 frame
     的脏图叠加。适用于长积分时间（分钟级及以上）或视场中心远离天极
     的场景，能补偿地球自转引起的 uv 覆盖旋转。

输出一张 PNG 图像，同时保存积分后的脏图 numpy 数组 (.npy)。

用法:
    python integrate_dirty_image.py [--dir correlation_results]
                                    [--date 20260604]
                                    [--freq 150] [--fov 180]
                                    [--mode polar_cpu|gpu|cpu]
                                    [--grid 256] [--bin 0]
                                    [--lat 40.0] [--lon 116.0]
                                    [--method snapshot|tracked|both]

依赖: numpy, matplotlib, pandas
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re
import time
import sys
import argparse
from collections import defaultdict
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# 复用 make_dirty_image 的成像引擎
# ---------------------------------------------------------------------------
from make_dirty_image import (
    make_dirty_image_cpu,
    make_dirty_image_polar_cpu,
    make_dirty_image_broadband_cpu,
    make_dirty_image_broadband_polar_cpu,
    load_broadband_visibilities,
    get_bandwidth_label,
    load_optimized_antennas,
    compute_uvw_from_antennas,
    compute_uv_tracks,
    read_frequency_range_from_data,
    get_polar_grid_metadata,
    reject_rfi_visibilities,
    _extract_visibility_data,
    _direct_fourier_sum_cpu,
    build_lm_grid,
    build_polar_sky_grid_GPU,
)


# ==============================================================================
# 可见度矩阵构建工具
# ==============================================================================
def _get_channel_indices(pair_name):
    """返回 (row, col) 0-based 索引"""
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


def _load_complex_value(filepath, freq_bin_idx):
    """从 CSV 读取指定频率 bin 的复数值"""
    try:
        df = pd.read_csv(filepath, comment='#')
        if freq_bin_idx < len(df):
            row = df.iloc[freq_bin_idx]
            return complex(row['real_part'], row['imag_part'])
    except Exception as e:
        print(f"  [WARN] Failed to read {filepath}: {e}")
    return 0 + 0j


def build_visibility_matrix_from_files(file_map, freq_bin_idx=0):
    """从一组 CSV 文件构建 8×8 复可见度矩阵"""
    vis = np.zeros((8, 8), dtype=np.complex128)
    active_channels = set()

    for pair_name, filepath in file_map.items():
        row, col = _get_channel_indices(pair_name)
        if row is None or col is None:
            continue
        val = _load_complex_value(filepath, freq_bin_idx)
        vis[row, col] = val
        if row != col:
            vis[col, row] = val.conjugate()
        if val != 0 + 0j:
            active_channels.add(row)
            if row != col:
                active_channels.add(col)

    return vis, active_channels


# ==============================================================================
# Frame 扫描：按日期发现所有 frame
# ==============================================================================
def discover_frames_by_date(watch_dir, date_str=None):
    """
    扫描 correlation_results 目录，按日期分组 frame。

    返回:
        OrderedDict {date_str: OrderedDict {timestamp_str: {pair_name: filepath}}}
    """
    csv_files = sorted(Path(watch_dir).glob("correlation_*.csv"))
    if not csv_files:
        return {}

    # 先按时间戳分组
    frames = defaultdict(dict)
    for f in csv_files:
        m = re.search(r'correlation_(\d{8}_\d{6})', f.name)
        if not m:
            continue
        timestamp = m.group(1)
        name = f.stem
        pair_match = re.search(r'(CH\d+(?:xCH\d+|_AUTO))', name)
        if pair_match:
            frames[timestamp][pair_match.group(1)] = f

    # 按日期分组
    from collections import OrderedDict
    by_date = defaultdict(OrderedDict)
    for ts in sorted(frames.keys()):
        date_key = ts[:8]  # YYYYMMDD
        by_date[date_key][ts] = frames[ts]

    # 如果指定了日期，只返回该日期
    if date_str is not None:
        if date_str in by_date:
            return OrderedDict([(date_str, by_date[date_str])])
        else:
            return OrderedDict()

    return OrderedDict(sorted(by_date.items()))


# ==============================================================================
# 时角计算
# ==============================================================================
def compute_hour_angle(timestamp_str, longitude_deg=116.0):
    """
    从时间戳和观测地经度计算本地时角（以春分点为参考的近似）。

    简化模型：
      - 格林尼治恒星时 (GST) ≈ UTC 时间 + 春分点偏移
      - 本地恒星时 (LST) = GST + longitude / 15
      - 时角 HA = LST - RA

    这里假设观测天顶 (declination = 90°)，RA 任意 → HA = LST（相位中心为天顶）。

    参数:
        timestamp_str: "YYYYMMDD_HHMMSS" 格式
        longitude_deg: 观测地经度（东正）

    返回:
        hour_angle_deg: 时角（度）
    """
    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

    # 简化的格林尼治恒星时计算
    # 春分点: 3月20日左右
    doy = dt.timetuple().tm_yday
    # 春分日约为第 79 天（3月20日）
    vernal_equinox_doy = 79

    # 格林尼治恒星时 (小时)
    ut_hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    # 每天恒星时比太阳时快约 4 分钟 → 每年 24 小时
    gst_hours = (ut_hours + (doy - vernal_equinox_doy) * 24.0 / 365.25 * 1.0027379) % 24

    # 本地恒星时
    lst_hours = (gst_hours + longitude_deg / 15.0) % 24

    # 转换为时角度数 (0h = 0°, 12h = 180°)
    hour_angle_deg = lst_hours * 15.0

    return hour_angle_deg


# ==============================================================================
# 方法 1: Snapshot 积分（不考虑天体位移）
# ==============================================================================
def integrate_snapshot(frames_by_ts, freq_mhz, grid_pts, fov_deg,
                       antennas_file, filter_rfi, apply_w_correction,
                       imaging_mode, n_channels=40, verbose=True):
    """
    Snapshot 模式：将所有 frame 的可见度矩阵直接平均，然后生成一张宽带脏图。

    物理假设：整个观测期间天空亮度分布不变（天体在视场中不移动）。
    数学：V_avg(f) = (1/N) Σ_k V_k(f)，然后 I_D = Σ_f FT(V_avg(f))

    优点：计算快（只需要一次宽带成像）
    缺点：天体运动会造成源展宽/模糊
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Method: SNAPSHOT (no celestial motion compensation)")
        print(f"  Strategy: Average visibilities → broadband imaging")
        print(f"{'='*60}")

    # 收集所有 frame 的宽带可见度
    timestamps = list(frames_by_ts.keys())
    all_vis_channels = []
    active_count = 0
    freq_sky_mhz = None

    for ts in timestamps:
        file_map = frames_by_ts[ts]
        f_sky, vis_channels = load_broadband_visibilities(
            file_map, center_freq_mhz=freq_mhz, n_channels=n_channels
        )
        if freq_sky_mhz is None:
            freq_sky_mhz = f_sky

        # Check if any channel has data
        has_data = np.any(np.abs(vis_channels) > 0)
        if has_data:
            all_vis_channels.append(vis_channels)
            active_count += 1
        elif verbose:
            print(f"  [SKIP] Frame {ts}: no active channels")

    if not all_vis_channels:
        print("  [ERROR] No valid frames to integrate!")
        return None

    # 平均可见度（per channel）
    vis_avg = np.mean(all_vis_channels, axis=0)  # (n_channels, 8, 8)

    if verbose:
        bw_label = get_bandwidth_label(freq_mhz, n_channels, freq_sky_mhz)
        print(f"  Bandwidth: {bw_label}")
        print(f"  Integrated {len(all_vis_channels)}/{len(timestamps)} frames")
        print(f"  Imaging with averaged visibilities...")

    # 宽带成像
    t0 = time.time()

    antennas = load_optimized_antennas(antennas_file)

    if imaging_mode in ("polar_cpu", "gpu"):
        n_radial = max(grid_pts // 2, 64)
        n_azimuthal = max(grid_pts, 128)
        L, M, N, horizon_mask, _, _ = build_polar_sky_grid_GPU(
            n_radial=n_radial, n_azimuthal=n_azimuthal
        )
        if fov_deg < 180.0:
            fov_rad = np.radians(fov_deg / 2.0)
            zeta_grid = np.radians(np.linspace(0.0, 90.0, n_radial))
            zeta_2d = zeta_grid[:, np.newaxis]
            fov_mask = zeta_2d <= fov_rad
            horizon_mask = horizon_mask & fov_mask

        dirty = np.zeros((n_radial, n_azimuthal), dtype=np.float64)

        for ci in range(n_channels):
            vis = vis_avg[ci]
            f_mhz = freq_sky_mhz[ci]

            if filter_rfi:
                vis = reject_rfi_visibilities(vis)

            wavelength = 299.792458 / f_mhz
            bu, bv, bw = compute_uvw_from_antennas(
                antennas, wavelength
            )
            vis_re, vis_im, auto_corr_sum = _extract_visibility_data(vis)
            ch_dirty = _direct_fourier_sum_cpu(
                L, M, N, bu, bv, bw,
                vis_re, vis_im, auto_corr_sum,
                apply_w_correction, horizon_mask
            )
            dirty += ch_dirty

        dirty /= n_channels
    else:
        L, M, N, horizon_mask, l_axis, m_axis = build_lm_grid(
            grid_pts=grid_pts, fov_deg=fov_deg
        )
        dirty = np.zeros((grid_pts, grid_pts), dtype=np.float64)

        for ci in range(n_channels):
            vis = vis_avg[ci]
            f_mhz = freq_sky_mhz[ci]

            if filter_rfi:
                vis = reject_rfi_visibilities(vis)

            wavelength = 299.792458 / f_mhz
            bu, bv, bw = compute_uvw_from_antennas(
                antennas, wavelength
            )
            vis_re, vis_im, auto_corr_sum = _extract_visibility_data(vis)
            ch_dirty = _direct_fourier_sum_cpu(
                L, M, N, bu, bv, bw,
                vis_re, vis_im, auto_corr_sum,
                apply_w_correction, horizon_mask
            )
            dirty += ch_dirty

        dirty /= n_channels

    elapsed = time.time() - t0
    if verbose:
        print(f"  Imaging completed in {elapsed:.2f}s")

    return dirty


# ==============================================================================
# 方法 2: Tracked 积分（考虑天体位移）
# ==============================================================================
def integrate_tracked(frames_by_ts, freq_mhz, grid_pts, fov_deg,
                      antennas_file, filter_rfi, apply_w_correction,
                      imaging_mode, longitude_deg=116.0, n_channels=40,
                      verbose=True):
    """
    Tracked 模式：每个 frame 用其时角对应的 (u,v,w) 独立成像，然后叠加脏图。
    同时每个 frame 使用多频率通道实现宽带 uv 覆盖积分。

    物理假设：天空亮度分布在天球上固定，地球自转导致 uv 覆盖在 (u,v)
    平面旋转。每个 frame × 每个频率通道对应不同的 uv 采样，叠加后获得
    更好的 uv 覆盖。

    数学：
      I_D(l,m) = 1/(N×C) Σ_k Σ_c FT^{-1}[V_k(f_c, u_{k,c}, v_{k,c}, w_{k,c})]
    其中 (u_{k,c}, v_{k,c}, w_{k,c}) 由第 k 个 frame 的时角和第 c 个频率决定。

    优点：补偿地球自转 + 宽带频率积分，uv 覆盖最完整
    缺点：计算量 N_frames × N_channels × 28 baselines
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Method: TRACKED (with celestial motion compensation)")
        print(f"  Strategy: Per-frame × per-channel imaging → stack")
        print(f"{'='*60}")

    antennas = load_optimized_antennas(antennas_file)
    timestamps = list(frames_by_ts.keys())

    # 确定网格
    if imaging_mode in ("polar_cpu", "gpu"):
        n_radial = max(grid_pts // 2, 64)
        n_azimuthal = max(grid_pts, 128)
        integrated = np.zeros((n_radial, n_azimuthal), dtype=np.float64)
    else:
        integrated = np.zeros((grid_pts, grid_pts), dtype=np.float64)

    valid_frames = 0
    total_frames = len(timestamps)
    freq_sky_mhz = None

    for idx, ts in enumerate(timestamps):
        file_map = frames_by_ts[ts]
        f_sky, vis_channels = load_broadband_visibilities(
            file_map, center_freq_mhz=freq_mhz, n_channels=n_channels
        )
        if freq_sky_mhz is None:
            freq_sky_mhz = f_sky

        has_data = np.any(np.abs(vis_channels) > 0)
        if not has_data:
            if verbose:
                print(f"  [{idx+1}/{total_frames}] {ts}: SKIP (no data)")
            continue

        # 计算此时刻的时角
        hour_angle = compute_hour_angle(ts, longitude_deg)

        t0 = time.time()
        frame_dirty = np.zeros_like(integrated)

        for ci in range(n_channels):
            vis = vis_channels[ci]
            f_mhz = freq_sky_mhz[ci]

            if filter_rfi:
                vis = reject_rfi_visibilities(vis)

            # 计算时变 + 频率相关的 (u, v, w)
            wavelength = 299.792458 / f_mhz
            bu, bv, bw = compute_uvw_from_antennas(
                antennas, wavelength,
                hour_angle_deg=hour_angle,
                declination_deg=90.0
            )

            vis_re, vis_im, auto_corr_sum = _extract_visibility_data(vis)

            if imaging_mode in ("polar_cpu", "gpu"):
                L, M, N, horizon_mask, _, _ = build_polar_sky_grid_GPU(
                    n_radial=n_radial, n_azimuthal=n_azimuthal
                )
                if fov_deg < 180.0:
                    fov_rad = np.radians(fov_deg / 2.0)
                    zeta_grid = np.radians(np.linspace(0.0, 90.0, n_radial))
                    zeta_2d = zeta_grid[:, np.newaxis]
                    fov_mask = zeta_2d <= fov_rad
                    horizon_mask = horizon_mask & fov_mask

                ch_dirty = _direct_fourier_sum_cpu(
                    L, M, N, bu, bv, bw,
                    vis_re, vis_im, auto_corr_sum,
                    apply_w_correction, horizon_mask
                )
            else:
                L, M, N, horizon_mask, _, _ = build_lm_grid(
                    grid_pts=grid_pts, fov_deg=fov_deg
                )
                ch_dirty = _direct_fourier_sum_cpu(
                    L, M, N, bu, bv, bw,
                    vis_re, vis_im, auto_corr_sum,
                    apply_w_correction, horizon_mask
                )

            frame_dirty += ch_dirty

        frame_dirty /= n_channels
        elapsed = time.time() - t0

        integrated += frame_dirty
        valid_frames += 1

        if verbose:
            ha_str = f"HA={hour_angle:.1f}°"
            print(f"  [{idx+1}/{total_frames}] {ts}  {ha_str}  "
                  f"imaging={elapsed:.2f}s")

    if valid_frames == 0:
        print("  [ERROR] No valid frames to integrate!")
        return None

    # 平均
    integrated /= valid_frames

    if verbose:
        bw_label = get_bandwidth_label(freq_mhz, n_channels, freq_sky_mhz)
        print(f"  Bandwidth: {bw_label}")
        valid = integrated[integrated != 0]
        peak = float(np.max(valid)) if len(valid) > 0 else 0.0
        print(f"\n  Integrated {valid_frames}/{total_frames} frames")
        print(f"  Peak intensity: {peak:.4f}")

    return integrated


# ==============================================================================
# 可视化与保存
# ==============================================================================
def save_integrated_image(dirty_img, date_str, method, imaging_mode,
                          freq_mhz, fov_deg, grid_pts, output_dir,
                          antennas=None, antennas_file=None,
                          n_channels=40, freq_sky_mhz=None,
                          verbose=True):
    """
    保存积分脏图为 PNG 和 NPY，附带 UV 覆盖图。

    极坐标模式: 鱼眼全天图
    Cartesian 模式: 标准 l/m 投影图
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载天线坐标（如未直接传入）
    if antennas is None and antennas_file is not None:
        antennas = load_optimized_antennas(antennas_file)

    has_uv = antennas is not None

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 9), facecolor='#1a1a2e',
                     layout='constrained' if has_uv else None)

    if has_uv:
        gs = gridspec.GridSpec(2, 2, figure=fig,
                               height_ratios=[1.6, 1],
                               hspace=0.35, wspace=0.35)
        ax_main = fig.add_subplot(gs[0, :], facecolor='black')
        ax_uv = fig.add_subplot(gs[1, 0], facecolor='#0d1b2a')
        ax_ant = fig.add_subplot(gs[1, 1], facecolor='#16213e')

    # 选择主绘图轴
    if has_uv:
        ax = ax_main
    else:
        ax = fig.add_subplot(111, facecolor='black')

    if imaging_mode in ("polar_cpu", "gpu"):
        # ── 极坐标全天图 ──
        nr, na = dirty_img.shape
        meta = get_polar_grid_metadata(
            grid_pts=grid_pts,
            n_radial=nr,
            n_azimuthal=na
        )

        # 坐标变换
        zeta_edges = np.linspace(0.0, 90.0, nr + 1) / 90.0
        az_edges = np.linspace(0.0, 360.0, na + 1)
        az_rad_edges = np.radians(az_edges)
        ZETA_E, AZ_E = np.meshgrid(zeta_edges, az_rad_edges, indexing='ij')
        X_edges = ZETA_E * np.sin(AZ_E)
        Y_edges = ZETA_E * np.cos(AZ_E)

        if not has_uv:
            ax = fig.add_subplot(111, facecolor='black')
        ax.set_aspect('equal')

        # 参考网格
        for zc in [15, 30, 45, 60, 75, 90]:
            r = zc / 90.0
            circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                                linewidth=0.5, alpha=0.6, linestyle='--')
            ax.add_patch(circle)
            if zc < 90:
                ax.annotate(f'{zc}°', (0, r), color='#6699cc',
                            fontsize=6, ha='center', va='bottom', alpha=0.7)

        az_labels = {0: 'N', 45: 'NE', 90: 'E', 135: 'SE',
                     180: 'S', 225: 'SW', 270: 'W', 315: 'NW'}
        for az_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
            theta = np.radians(90 - az_deg)
            dx = np.cos(theta)
            dy = np.sin(theta)
            ax.plot([0, dx], [0, dy], color='#336699',
                    linewidth=0.4, alpha=0.5, linestyle=':')
            label = az_labels.get(az_deg, f'{az_deg}°')
            ax.annotate(label, (dx * 1.08, dy * 1.08),
                        color='#6699cc', fontsize=7,
                        ha='center', va='center', alpha=0.8)

        horizon = plt.Circle((0, 0), 1.0, fill=False, color='#4488cc',
                             linewidth=1.2, alpha=0.8)
        ax.add_patch(horizon)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])

        # 绘图
        valid = dirty_img[dirty_img != 0]
        vmin = float(np.percentile(valid, 2)) if len(valid) > 0 else 0.0
        vmax = float(np.percentile(valid, 98)) if len(valid) > 0 else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

        im = ax.pcolormesh(X_edges, Y_edges, dirty_img,
                           cmap='inferno', shading='flat',
                           vmin=vmin, vmax=vmax, rasterized=True)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        bw_label = get_bandwidth_label(freq_mhz, n_channels, freq_sky_mhz)
        title = (f"Integrated All-Sky Dirty Image — {date_str}\n"
                 f"Method: {method.upper()}  |  {bw_label}  |  "
                 f"FOV={fov_deg:.0f}°  |  Grid: {nr}×{na} polar")
        ax.set_title(title, color='white', fontsize=13)

    else:
        # ── Cartesian l/m 图 ──
        l_max = np.sin(np.radians(fov_deg / 2))
        if not has_uv:
            ax = fig.add_subplot(111, facecolor='black')
        ax.set_aspect('equal')

        valid = dirty_img[dirty_img != 0]
        vmin = float(np.percentile(valid, 2)) if len(valid) > 0 else 0.0
        vmax = float(np.percentile(valid, 98)) if len(valid) > 0 else 1.0
        if vmax <= vmin:
            vmax = vmin + 1e-6

        im = ax.imshow(dirty_img, extent=[-l_max, l_max, -l_max, l_max],
                       origin='upper', cmap='inferno', aspect='equal',
                       interpolation='bilinear', vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Intensity')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')

        ax.set_xlabel("l (East-West)", color='white')
        ax.set_ylabel("m (South-North)", color='white')
        ax.tick_params(colors='white', labelsize=8)

        bw_label = get_bandwidth_label(freq_mhz, n_channels, freq_sky_mhz)
        title = (f"Integrated Dirty Image — {date_str}\n"
                 f"Method: {method.upper()}  |  {bw_label}  |  "
                 f"FOV={fov_deg:.0f}°  |  Grid: {grid_pts}×{grid_pts}")
        ax.set_title(title, color='white', fontsize=13)

    # ── UV 覆盖 + 天线布局子图 ──
    if has_uv:
        _draw_uv_coverage_subplot(ax_uv, antennas, freq_mhz,
                                  data_dir=output_dir.parent / "correlation_results")
        _draw_antenna_layout_subplot(ax_ant, antennas)
    else:
        fig.tight_layout(pad=2.0)

    # 保存
    png_path = output_dir / f"integrated_{date_str}_{method}.png"
    npy_path = output_dir / f"integrated_{date_str}_{method}.npy"

    fig.savefig(png_path, dpi=150, facecolor=fig.get_facecolor())
    np.save(npy_path, dirty_img)
    plt.close(fig)

    if verbose:
        print(f"\n  Saved PNG: {png_path}")
        print(f"  Saved NPY: {npy_path}")

    return png_path, npy_path


def _draw_uv_coverage_subplot(ax_uv, antennas, freq_mhz, data_dir=None):
    """在指定轴上绘制宽带 UV 覆盖图（频率相关的 uv 轨迹线）"""
    # 读取实际频率范围
    flo, fhi = None, None
    if data_dir is not None:
        flo, fhi = read_frequency_range_from_data(
            str(data_dir), center_freq_mhz=freq_mhz
        )
    if flo is None:
        flo = freq_mhz - 50.0
        fhi = freq_mhz + 50.0

    uv_lines, freq_samples, uv_center = compute_uv_tracks(
        antennas, freq_low_mhz=flo, freq_high_mhz=fhi,
        n_samples=20
    )

    ax_uv.set_xlabel("u (λ)", color='white')
    ax_uv.set_ylabel("v (λ)", color='white')
    ax_uv.tick_params(colors='white', labelsize=7)
    ax_uv.set_aspect('equal')
    ax_uv.grid(True, alpha=0.25, color='gray', linestyle=':')

    n_baselines = uv_lines.shape[0]
    cmap = plt.cm.plasma

    for bl_idx in range(n_baselines):
        u_track = uv_lines[bl_idx, :, 0]
        v_track = uv_lines[bl_idx, :, 1]
        for s in range(len(freq_samples) - 1):
            t = s / (len(freq_samples) - 1)
            ax_uv.plot(
                u_track[s:s+2], v_track[s:s+2],
                color=cmap(0.2 + 0.6 * t), linewidth=1.2,
                alpha=0.7, zorder=3
            )
        ax_uv.scatter(
            u_track[0], v_track[0], c=[cmap(0.2)],
            s=15, marker='o', edgecolors='white',
            linewidths=0.3, zorder=5, alpha=0.8
        )

    for bl_idx in range(n_baselines):
        u_track = -uv_lines[bl_idx, :, 0]
        v_track = -uv_lines[bl_idx, :, 1]
        ax_uv.plot(
            u_track, v_track,
            color='#ff8c42', linewidth=1.0, alpha=0.6,
            linestyle='--', zorder=2
        )

    ax_uv.scatter(
        uv_center[:, 0], uv_center[:, 1],
        c='cyan', s=12, marker='D', edgecolors='white',
        linewidths=0.3, zorder=6, alpha=0.9
    )

    ax_uv.axhline(y=0, color='#336699', linewidth=0.5, alpha=0.5)
    ax_uv.axvline(x=0, color='#336699', linewidth=0.5, alpha=0.5)

    uv_max = max(np.max(np.abs(uv_lines[:, 0, :])),
                 np.max(np.abs(uv_lines[:, -1, :]))) * 1.15
    if uv_max > 0:
        ax_uv.set_xlim(-uv_max, uv_max)
        ax_uv.set_ylim(-uv_max, uv_max)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=cmap(0.5), linewidth=1.5,
               label=f'uv track ({flo:.0f}–{fhi:.0f} MHz)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
               markersize=6, label=f'{freq_mhz:.0f} MHz center'),
        Line2D([0], [0], linestyle='--', color='#ff8c42', linewidth=1,
               label='(−u,−v) conjugate'),
    ]
    ax_uv.legend(handles=legend_elements, loc='upper right',
                 fontsize=5.5, facecolor='#16213e',
                 edgecolor='#336699', labelcolor='white')

    ax_uv.set_title(
        f"UV Coverage (28 baselines, {flo:.0f}–{fhi:.0f} MHz)",
        color='white', fontsize=10
    )


def _draw_antenna_layout_subplot(ax_ant, antennas):
    """在指定轴上绘制天线布局图"""
    xs, ys = antennas[:, 0], antennas[:, 1]

    ax_ant.set_title("Array Layout", color='white', fontsize=10)
    ax_ant.set_xlabel("X (m) East →", color='white')
    ax_ant.set_ylabel("Y (m) North →", color='white')
    ax_ant.tick_params(colors='white', labelsize=7)
    ax_ant.set_aspect('equal')
    ax_ant.grid(True, alpha=0.3, color='gray')

    ax_ant.scatter(xs, ys, c='cyan', s=60, edgecolors='white', linewidths=1, zorder=5)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax_ant.annotate(f"CH{i+1}", (x, y), textcoords="offset points",
                         xytext=(5, 5), color='white', fontsize=7)

    for i in range(8):
        for j in range(i + 1, 8):
            ax_ant.plot([xs[i], xs[j]], [ys[i], ys[j]],
                         'gray', alpha=0.25, linewidth=0.5)

    margin = 2.0
    ax_ant.set_xlim(xs.min() - margin, xs.max() + margin)
    ax_ant.set_ylim(ys.min() - margin, ys.max() + margin)


# ==============================================================================
# 主入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Integrate Dirty Image — 同一天连续观测时间积分成像"
    )
    parser.add_argument('--dir', default='correlation_results',
                        help='Directory containing correlation CSV files')
    parser.add_argument('--date', default=None,
                        help='Date string YYYYMMDD (default: auto-detect first date)')
    parser.add_argument('--freq', type=float, default=150.0,
                        help='Observing frequency in MHz (default: 150)')
    parser.add_argument('--fov', type=float, default=180.0,
                        help='Field of view in degrees (default: 180 for all-sky)')
    parser.add_argument('--grid', type=int, default=256,
                        help='Image grid points (default: 256)')
    parser.add_argument('--antennas', default='optimized_antenna_coordinates.txt',
                        help='Antenna coordinate file')
    parser.add_argument('--nch', type=int, default=40,
                        help='Number of broadband frequency channels (default: 40, '
                             '1 = single-channel legacy mode)')
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'polar_cpu'],
                        default='polar_cpu',
                        help='Imaging mode: cpu (Cartesian), gpu (Polar GPU), '
                             'polar_cpu (Polar CPU)')
    parser.add_argument('--method', choices=['snapshot', 'tracked', 'both'],
                        default='both',
                        help='Integration method (default: both)')
    parser.add_argument('--lat', type=float, default=40.0,
                        help='Observer latitude in degrees (default: 40)')
    parser.add_argument('--lon', type=float, default=116.0,
                        help='Observer longitude in degrees (default: 116)')
    parser.add_argument('--output', default='integrated_images',
                        help='Output directory (default: integrated_images)')
    parser.add_argument('--no-rfi', action='store_true',
                        help='Disable RFI filtering')

    args = parser.parse_args()

    # ── 扫描 frames ──
    print("=" * 65)
    print("  Integrate Dirty Image — Time-Integrated All-Sky Imaging")
    print("  8-Element Low-Frequency Radio Array")
    print("=" * 65)

    by_date = discover_frames_by_date(args.dir, args.date)

    if not by_date:
        print(f"\n[ERROR] No frames found in {args.dir}")
        sys.exit(1)

    # 选择日期
    if args.date is None:
        date_str = list(by_date.keys())[0]
        print(f"\nAuto-detected date: {date_str}")
    else:
        date_str = args.date

    frames_by_ts = by_date[date_str]
    n_frames = len(frames_by_ts)
    print(f"Date: {date_str}")
    print(f"Frames: {n_frames}")
    print(f"Frequency: {args.freq} MHz")
    print(f"FOV: {args.fov}°")
    print(f"Grid: {args.grid}×{args.grid}")
    print(f"Mode: {args.mode}")
    print(f"RFI filter: {'OFF' if args.no_rfi else 'ON'}")

    filter_rfi = not args.no_rfi
    apply_w_correction = True
    total_start = time.time()

    # ── 执行积分 ──
    methods_to_run = []
    if args.method in ("snapshot", "both"):
        methods_to_run.append("snapshot")
    if args.method in ("tracked", "both"):
        methods_to_run.append("tracked")

    results = {}
    for method in methods_to_run:
        if method == "snapshot":
            dirty = integrate_snapshot(
                frames_by_ts, args.freq, args.grid, args.fov,
                args.antennas, filter_rfi, apply_w_correction, args.mode,
                n_channels=args.nch
            )
        else:
            dirty = integrate_tracked(
                frames_by_ts, args.freq, args.grid, args.fov,
                args.antennas, filter_rfi, apply_w_correction, args.mode,
                longitude_deg=args.lon, n_channels=args.nch
            )

        if dirty is not None:
            results[method] = dirty

    if not results:
        print("\n[ERROR] No integration results produced!")
        sys.exit(1)

    # ── 保存 ──
    for method, dirty in results.items():
        save_integrated_image(
            dirty, date_str, method, args.mode,
            args.freq, args.fov, args.grid, args.output,
            antennas_file=args.antennas,
            n_channels=args.nch
        )

    total_elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Output directory: {args.output}/")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
