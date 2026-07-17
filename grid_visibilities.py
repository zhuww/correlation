#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_visibilities.py — 精细化 UV 格点化与可见度加权平均
=============================================================
将全部帧×全部通道×全部基线的可见度数据格点化到精细 UV 网格上，
对落入同一格点的可见度进行加权平均，输出格点化后的复可见度二维数组。

原理:
  1. 计算所有 (u,v) 坐标（与 uv_coverage_plot.py 相同的 UVW 计算）
  2. 逐帧加载可见度复数值 real_part + i*imag_part
  3. 用加权 2D 直方图将可见度映射到 UV 网格
  4. 格点化可见度 = Σ(w_i · V_i) / Σ(w_i),  i∈同一格子

数据说明:
  CSV 的 frequency_hz 列为基带频率 (Hz), 天空频率 = 150 MHz + f_hz/1e6。
  4096 个 FFT bin, 100~200 MHz, 每 bin 带宽 ≈ 24.4 kHz。

加权方案:
  - uniform: 所有权重 = 1 (简单算术平均)
  - density: 按 1/√(局部密度) 降权 (抑制过采样区, 近似 robust)

输出:
  - gridded_vis_*.npz: 格点化复可见度 + 权重 + UV 坐标轴
  - gridded_vis_*_scatter.png: 实部 + 虚部 (scatter 散点) + 覆盖 + 径向剖面
  - gridded_vis_*.fits: 实部 FITS 文件

后续可用 numpy FFT 从格点化可见度直接生成脏图。

用法:
  python grid_visibilities.py                                    # 默认: 全帧, GPU, 4096 bins
  python grid_visibilities.py --bins 8192 --weight density      # 高分辨率 + 密度加权
  python grid_visibilities.py --max-frames 100 --cpu            # CPU 测试
  python grid_visibilities.py --nch 256                         # 降采样节省时间
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time
import os
import sys
import argparse
import io
import re
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

# ── GPU 支持 ──
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

# ── 导入现有模块的工具函数 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_dirty_image import load_optimized_antennas, _get_channel_indices
from integrate_dirty_image import discover_frames_by_date, compute_hour_angle


# ═══════════════════════════════════════════════════════════════════
# 频率读取（复用 uv_coverage_plot 的逻辑）
# ═══════════════════════════════════════════════════════════════════
def read_sky_frequencies(file_map, center_freq_mhz=150.0,
                         n_channels=0, max_bins=4100):
    """读取全部频率通道的天空频率。

    4096 FFT bin: 前 2049 个 150→200 MHz, 后 2047 个 100→150 MHz。
    """
    import pandas as pd
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=max_bins)
    freq_hz = df['frequency_hz'].values.astype(np.float64)

    total_bins = len(freq_hz)
    bin_width_hz = abs(freq_hz[1] - freq_hz[0])

    if n_channels <= 0 or n_channels > total_bins:
        n_channels = total_bins
        bin_indices = np.arange(total_bins)
    else:
        bin_indices = np.linspace(0, total_bins - 1, n_channels, dtype=np.int32)

    f_center_hz = freq_hz[bin_indices]
    freqs_mhz = center_freq_mhz + f_center_hz / 1e6
    return freqs_mhz, bin_width_hz


# ═══════════════════════════════════════════════════════════════════
# UVW 计算 —— 单帧 GPU / CPU
# ═══════════════════════════════════════════════════════════════════
def _compute_uvw_single_frame(antennas, hour_angle_deg, wavelengths,
                               latitude_deg, include_conjugates, use_gpu):
    """计算单帧的全部 (u,v) 坐标。

    返回顺序: [ch0_bl0, ch0_bl1, ..., chN_bl27, -ch0_bl0, ..., -chN_bl27]
    """
    ha_rad = np.radians(hour_angle_deg)
    dec_rad = np.radians(latitude_deg)
    sin_H, cos_H = np.sin(ha_rad), np.cos(ha_rad)
    sin_D, cos_D = np.sin(dec_rad), np.cos(dec_rad)

    x = antennas[:, 0]  # (8,)
    y = antennas[:, 1]  # (8,)

    # 基线配对 (i < j)
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    n_bl = len(pairs)  # 28
    idx_i = np.array([p[0] for p in pairs], dtype=np.int32)
    idx_j = np.array([p[1] for p in pairs], dtype=np.int32)

    n_ch = len(wavelengths)

    if use_gpu and GPU_AVAILABLE:
        xp = cp
        x_g = xp.asarray(x, dtype=xp.float64)
        y_g = xp.asarray(y, dtype=xp.float64)
        sin_H_g = xp.float64(sin_H)
        cos_H_g = xp.float64(cos_H)
        sin_D_g = xp.float64(sin_D)
        cos_D_g = xp.float64(cos_D)
        idx_i_g = xp.asarray(idx_i, dtype=xp.int32)
        idx_j_g = xp.asarray(idx_j, dtype=xp.int32)
        wl_g = xp.asarray(wavelengths, dtype=xp.float64)

        # 天线坐标 (m) — 标量时角 → 每个天线一个值
        ant_u = sin_H_g * x_g + cos_H_g * y_g               # (8,)
        ant_v = -sin_D_g * cos_H_g * x_g + sin_D_g * sin_H_g * y_g
        ant_w = cos_D_g * cos_H_g * x_g - cos_D_g * sin_H_g * y_g

        # 基线差分 (m)
        bl_u_m = ant_u[idx_i_g] - ant_u[idx_j_g]            # (28,)
        bl_v_m = ant_v[idx_i_g] - ant_v[idx_j_g]            # (28,)

        # 除以波长 → (u,v) in λ: (28,) / (C,) → broadcast as (C, 28) or (28, C)
        # bl_u: (28,) → (1, 28); wl_g: (C,) → (C, 1) → (C, 28)
        bl_u = bl_u_m[None, :] / wl_g[:, None]              # (C, 28)
        bl_v = bl_v_m[None, :] / wl_g[:, None]              # (C, 28)

        if include_conjugates:
            u_flat = xp.concatenate([bl_u.ravel(), -bl_u.ravel()])
            v_flat = xp.concatenate([bl_v.ravel(), -bl_v.ravel()])
        else:
            u_flat = bl_u.ravel()
            v_flat = bl_v.ravel()

        return xp.asnumpy(u_flat), xp.asnumpy(v_flat)
    else:
        # CPU
        ant_u = sin_H * x + cos_H * y                       # (8,)
        ant_v = -sin_D * cos_H * x + sin_D * sin_H * y

        bl_u_m = ant_u[idx_i] - ant_u[idx_j]                # (28,)
        bl_v_m = ant_v[idx_i] - ant_v[idx_j]                # (28,)

        bl_u = bl_u_m[None, :] / wavelengths[:, None]       # (C, 28)
        bl_v = bl_v_m[None, :] / wavelengths[:, None]       # (C, 28)

        if include_conjugates:
            u_flat = np.concatenate([bl_u.ravel(), -bl_u.ravel()])
            v_flat = np.concatenate([bl_v.ravel(), -bl_v.ravel()])
        else:
            u_flat = bl_u.ravel()
            v_flat = bl_v.ravel()

        return u_flat, v_flat


# ═══════════════════════════════════════════════════════════════════
# 可见度加载 —— 逐帧从 CSV 提取
# ═══════════════════════════════════════════════════════════════════
def load_frame_visibilities(file_map, n_channels, include_conjugates,
                             rfi_mask=None, per_antenna_bandpass=None,
                             amp_smooth_window=0):
    """加载一帧的可见度数据，按 UVW 计算顺序展平输出。

    Parameters
    ----------
    file_map : dict {pair_name: filepath}
    n_channels : int
    include_conjugates : bool
    rfi_mask : ndarray (n_channels,) bool, optional
        True 表示被标记为干扰的通道，对应的可见度会被置零。
    per_antenna_bandpass : ndarray (n_ant, n_channels) float64, optional
        每天线的 bandpass 模板（自相关中位频谱）。
        对基线 (i,j): V_corrected[c] = V_raw[c] / sqrt(bp_i[c] * bp_j[c]) * median(sqrt)
        使得互相关振幅按每天线增益归一化，消除 bandpass 频率结构。
    amp_smooth_window : int
        振幅平滑窗口（通道数）。> 0 时在 bandpass 校正后对每天线对的 |V| 
        做滑动中位数平滑归一化，进一步去除残余频率结构。默认 0（关闭）。

    Returns
    -------
    vis_flat : ndarray (n_channels * 28 * 2,) if conjugates else (n_channels * 28,)
        dtype complex128, 与 _compute_uvw_single_frame 输出顺序一致
    """
    import pandas as pd

    # 基线配对与 UVW 计算一致
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    n_bl = len(pairs)  # 28

    # ── 预读所有 CSV ──
    csv_data = {}
    for pair_name, fp in file_map.items():
        try:
            df = pd.read_csv(fp, comment='#',
                             usecols=['real_part', 'imag_part', 'frequency_index'])
            csv_data[pair_name] = df
        except Exception:
            pass

    # ── 构建 8×8 矩阵索引 → 文件映射 ──
    matrix_from_file = {}  # (i, j) → (real, imag) array of len n_channels
    for pair_name, df in csv_data.items():
        row_idx, col_idx = _get_channel_indices(pair_name)
        if row_idx is None or col_idx is None:
            continue
        re_vals = df['real_part'].values.astype(np.float64)
        im_vals = df['imag_part'].values.astype(np.float64)
        matrix_from_file[(row_idx, col_idx)] = (re_vals, im_vals)

    # ── 每天线 Bandpass 校正因子（每基线独立） ──
    # bp_corr_per_bl[bi, ch] = 校正因子，使得 V_corrected = V_raw * bp_corr
    bp_corr_per_bl = np.ones((n_bl, n_channels), dtype=np.float64)
    if per_antenna_bandpass is not None:
        n_ant = per_antenna_bandpass.shape[0]
        for bi, (i, j) in enumerate(pairs):
            if i < n_ant and j < n_ant:
                expected = np.sqrt(
                    np.maximum(per_antenna_bandpass[i], 1e-30) *
                    np.maximum(per_antenna_bandpass[j], 1e-30)
                )
                med = np.median(expected)
                if med > 0:
                    bp_corr_per_bl[bi] = med / np.maximum(expected, 1e-30)

    # ── 按 UVW 顺序展平 ──
    total = n_channels * n_bl * (2 if include_conjugates else 1)
    vis_flat = np.zeros(total, dtype=np.complex128)

    for ci in range(n_channels):
        ch_masked = (rfi_mask is not None and rfi_mask[ci])

        for bi, (i, j) in enumerate(pairs):
            bp_factor = bp_corr_per_bl[bi, ci]

            if ch_masked:
                val = 0j
            else:
                key_a = (i, j)
                key_b = (j, i)
                if key_a in matrix_from_file:
                    re_arr, im_arr = matrix_from_file[key_a]
                elif key_b in matrix_from_file:
                    re_arr, im_arr = matrix_from_file[key_b]
                else:
                    re_arr, im_arr = None, None

                if re_arr is not None and ci < len(re_arr):
                    val = complex(re_arr[ci] * bp_factor,
                                  im_arr[ci] * bp_factor)
                else:
                    val = 0j

            idx = ci * n_bl + bi           # 正 UV 半平面
            vis_flat[idx] = val

            if include_conjugates:
                idx_c = n_channels * n_bl + idx  # 共轭半平面
                vis_flat[idx_c] = val.conjugate()

    # ── 振幅平滑归一化（每基线跨频率） ──
    if amp_smooth_window > 0 and per_antenna_bandpass is not None:
        half = amp_smooth_window // 2
        for bi in range(n_bl):
            # 正半平面: 每基线 stride=n_bl, 共 n_channels 个元素
            idxs = slice(bi, n_channels * n_bl, n_bl)
            mag = np.abs(vis_flat[idxs])          # (n_channels,)
            # 向量化滑动中位数: pad 后 sliding_window_view
            mag_pad = np.pad(mag, (half, half), mode='reflect')
            windows = sliding_window_view(mag_pad, amp_smooth_window)  # (n_channels, window)
            mag_smooth = np.median(windows, axis=1)
            mag_smooth = np.maximum(mag_smooth, 1e-30)
            scale = np.median(mag_smooth) / mag_smooth
            vis_flat[idxs] *= scale

            if include_conjugates:
                idxs_c = slice(n_channels * n_bl + bi, total, n_bl)
                vis_flat[idxs_c] *= scale

    return vis_flat


# ═══════════════════════════════════════════════════════════════════
# RFI 检测 —— 利用自相关项识别干扰频率
# ═══════════════════════════════════════════════════════════════════
def detect_rfi_channels(file_map, n_channels, threshold=5.0, ratio_threshold=3.0, grow=3):
    """利用 8 路自相关数据 (CHx_AUTO) 检测受干扰的频率通道。

    算法：
      1. 读取每路自相关 magnitude；
      2. 跨天线取中位数，得到各通道的 auto-median；
      3. 滑动中位数滤波（窗口 51）估计平滑谱 baseline；
      4. 残差 = median − baseline，用 MAD 统计量确定阈值；
      5. 同时满足以下两个条件才标记为 RFI：
         median > baseline + threshold × MAD  （绝对偏离）
         median > baseline × ratio_threshold   （相对倍数）
      6. 对 flagged 通道左右各扩展 grow 个通道，防止旁瓣泄漏。

    Parameters
    ----------
    file_map : dict
        单帧的文件映射，需包含 CH1_AUTO … CH8_AUTO。
    n_channels : int
        通道数（<= 4096）。
    threshold : float
        MAD 倍数阈值，默认 5.0。
    ratio_threshold : float
        相对 baseline 的倍数阈值，默认 3.0。
    grow : int
        标志扩展半径，默认 3。

    Returns
    -------
    rfi_mask : ndarray (n_channels,) bool
        True 表示被标记为干扰的通道。
    """
    import pandas as pd

    auto_mags = []
    for ch in range(1, 9):
        key = f"CH{ch}_AUTO"
        if key not in file_map:
            continue
        try:
            df = pd.read_csv(file_map[key], comment='#',
                             usecols=['magnitude'])
            mag = df['magnitude'].values.astype(np.float64)
            if len(mag) >= n_channels:
                mag = mag[:n_channels]
            else:
                pad = np.zeros(n_channels - len(mag), dtype=np.float64)
                mag = np.concatenate([mag, pad])
            auto_mags.append(mag)
        except Exception:
            pass

    if len(auto_mags) == 0:
        return np.zeros(n_channels, dtype=bool)

    auto_mags = np.array(auto_mags)               # (N_auto, n_channels)
    median_per_ch = np.median(auto_mags, axis=0)  # (n_channels,)

    # 滑动中位数 (窗口 51)
    w = 51
    half = w // 2
    smoothed = np.empty_like(median_per_ch)
    for i in range(len(median_per_ch)):
        lo = max(0, i - half)
        hi = min(len(median_per_ch), i + half + 1)
        smoothed[i] = np.median(median_per_ch[lo:hi])

    residual = median_per_ch - smoothed
    mad = np.median(np.abs(residual))
    if mad <= 0:
        mad = np.median(np.abs(median_per_ch)) * 1e-3
        if mad <= 0:
            mad = 1.0

    flagged = np.zeros(n_channels, dtype=bool)
    if threshold > 0:
        flagged |= median_per_ch > (smoothed + threshold * mad)
    if ratio_threshold > 0:
        flagged &= median_per_ch > (smoothed * ratio_threshold)

    # 扩展标志区域
    if grow > 0 and flagged.any():
        grown = np.zeros(n_channels, dtype=bool)
        for i in range(n_channels):
            if flagged[i]:
                lo = max(0, i - grow)
                hi = min(n_channels, i + grow + 1)
                grown[lo:hi] = True
        flagged = grown

    return flagged


# ═══════════════════════════════════════════════════════════════════
# Bandpass 尖峰清理 —— 检测并替换带通模板中的窄带 RFI 尖峰
# ═══════════════════════════════════════════════════════════════════
def clean_bandpass_spikes(bandpass, threshold=5.0, ratio_threshold=3.0,
                           grow=5, window=101):
    """检测 bandpass 模板中的窄带 RFI 尖峰，替换为平滑基线。

    算法：
      1. 滑动中位数滤波（window=101）估计平滑基线；
      2. 残差 = bandpass − 基线，MAD 统计量确定阈值；
      3. 同时满足绝对偏离和相对倍数两个条件才标记为尖峰；
      4. 左右各扩展 grow 通道（防止旁瓣）；
      5. 用平滑基线替换尖峰通道的 bandpass 值。

    这样只校正真实的宽带 bandpass 形状，避免将 RFI 尖峰引入
    互相关校正中。

    Parameters
    ----------
    bandpass : ndarray (n_channels,)
        原始 bandpass 模板。
    threshold : float
        MAD 倍数阈值（默认 5.0）。
    ratio_threshold : float
        相对基线的倍数阈值（默认 3.0）。
    grow : int
        尖峰扩展通道数（默认 5）。
    window : int
        滑动中位数窗口大小（默认 101）。

    Returns
    -------
    bp_clean : ndarray (n_channels,)
        清理后的 bandpass（尖峰被平滑值替换）。
    n_spikes : int
        被标记的尖峰通道数。
    """
    n = len(bandpass)
    half = window // 2

    # 滑动中位数估计平滑基线
    baseline = np.empty_like(bandpass)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        baseline[i] = np.median(bandpass[lo:hi])

    residual = bandpass - baseline
    mad = np.median(np.abs(residual))
    if mad <= 0:
        mad = np.median(np.abs(bandpass)) * 1e-3
        if mad <= 0:
            mad = 1.0

    # 双重条件标记尖峰
    flagged = np.zeros(n, dtype=bool)
    if threshold > 0:
        flagged |= bandpass > (baseline + threshold * mad)
    if ratio_threshold > 0:
        flagged &= bandpass > (baseline * ratio_threshold)

    # 扩展标志区域（防止旁瓣泄漏）
    if grow > 0 and flagged.any():
        grown = np.zeros(n, dtype=bool)
        for i in range(n):
            if flagged[i]:
                lo = max(0, i - grow)
                hi = min(n, i + grow + 1)
                grown[lo:hi] = True
        flagged = grown

    # 用平滑基线替换尖峰区域
    bp_clean = bandpass.copy()
    bp_clean[flagged] = baseline[flagged]

    return bp_clean, flagged.sum()


# ═══════════════════════════════════════════════════════════════════
# Bandpass 模板估计 + 时间域 RFI 帧级检测（第一遍扫描）
# ═══════════════════════════════════════════════════════════════════
def precompute_rfi_references(frames_by_ts, timestamps, n_channels,
                               verbose=True,
                               td_threshold=5.0, td_window=101,
                               bandpass_enable=True,
                               bp_clean_spikes=True):
    """第一遍：扫描全部帧的自相关数据，同时估计每天线 bandpass 和检测时间域 RFI。

    算法：
      Bandpass（每天线独立）：
        1. 逐帧逐天线读取自相关 magnitude → auto_mags[ant, ch]
        2. 按天线累积，跨帧取中位数 → per_antenna_bandpass[ant, ch]
        3. 对每天线 bandpass 做尖峰清理 → 平滑基线替换
        4. 返回每天线 bandpass，用于互相关校正：
           V_corrected[i,j][c] = V_raw[c] / sqrt(bp_i[c] * bp_j[c])
      Time-domain RFI：
        1. 每帧计算帧功率 P = Σ median_ant auto_mags[c]
        2. 滑动中位窗口 (td_window) 平滑 P → 基线
        3. |P - baseline| > td_threshold × MAD 的帧被标记为 bad

    Parameters
    ----------
    frames_by_ts : dict
    timestamps : list
    n_channels : int
    verbose : bool
    td_threshold : float
        时间域 MAD 阈值倍数（默认 5.0，越小越激进）
    td_window : int
        滑动窗口帧数 (默认 101)
    bandpass_enable : bool

    Returns
    -------
    per_antenna_bandpass : ndarray (n_ant, n_channels) or None
        每天线的 bandpass 模板（自相关中位频谱）。用于基线级互相关校正。
    bad_ts_set : set
        被标记为时间域 RFI 的时间戳集合。
    """
    if verbose:
        print(f"\n  [第一遍] 预扫描 {len(timestamps)} 帧自相关数据...")

    import pandas as pd

    all_auto_by_ant = []  # 每帧: (n_ant, n_channels) per-antenna spectra
    frame_powers = []
    frame_ts_valid = []
    n_antennas = 8

    t0 = time.time()
    for fi, ts in enumerate(timestamps):
        file_map = frames_by_ts[ts]
        auto_mags_per_ant = []
        for ch in range(1, n_antennas + 1):
            key = f"CH{ch}_AUTO"
            if key not in file_map:
                continue
            try:
                df = pd.read_csv(file_map[key], comment='#',
                                 usecols=['magnitude'])
                mag = df['magnitude'].values.astype(np.float64)
                if len(mag) >= n_channels:
                    mag = mag[:n_channels]
                else:
                    pad = np.zeros(n_channels - len(mag), dtype=np.float64)
                    mag = np.concatenate([mag, pad])
                auto_mags_per_ant.append(mag)
            except Exception:
                pass

        if len(auto_mags_per_ant) == 0:
            frame_powers.append(0)
            frame_ts_valid.append(ts)
            if bandpass_enable:
                all_auto_by_ant.append(np.zeros((n_antennas, n_channels), dtype=np.float64))
            continue

        auto_mags = np.array(auto_mags_per_ant)          # (N_ant, n_channels)
        frame_spec = np.median(auto_mags, axis=0)        # (n_channels,) 池化用于 TD RFI

        frame_power = np.sum(frame_spec)
        frame_powers.append(frame_power)
        frame_ts_valid.append(ts)

        if bandpass_enable:
            n_ant = auto_mags.shape[0]
            if n_ant < n_antennas:
                pad = np.zeros((n_antennas - n_ant, n_channels), dtype=np.float64)
                auto_mags = np.concatenate([auto_mags, pad], axis=0)
            all_auto_by_ant.append(auto_mags)  # 保留每天线频谱 (固定8×n_ch)

        # 进度
        if verbose and (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (fi + 1) / elapsed
            eta = (len(timestamps) - fi - 1) / fps
            print(f"    扫描 [{fi+1}/{len(timestamps)}] {elapsed:.0f}s, "
                  f"{fps:.1f} fps, ETA {eta:.0f}s")

    # ── 每天线 Bandpass 模板 ──
    per_antenna_bandpass = None
    if bandpass_enable and len(all_auto_by_ant) > 0:
        # 按天线堆叠: (n_frames, n_ant, n_channels)
        all_auto_stack = np.stack(all_auto_by_ant, axis=0)
        # 跨帧取中位数 → 每天线的 bandpass
        per_antenna_bandpass = np.median(all_auto_stack, axis=0)  # (n_ant, n_channels)
        # 天数可能 < 8，补齐
        n_ant_actual = per_antenna_bandpass.shape[0]
        if n_ant_actual < n_antennas:
            pad_bp = np.tile(np.median(per_antenna_bandpass, axis=0, keepdims=True),
                             (n_antennas - n_ant_actual, 1))
            per_antenna_bandpass = np.concatenate([per_antenna_bandpass, pad_bp], axis=0)

        # 池化 bandpass 用于日志
        pooled_bp = np.median(per_antenna_bandpass, axis=0)
        if verbose:
            bp_med = np.median(pooled_bp)
            print(f"    Bandpass 模板 (池化): median={bp_med:.3e}, "
                  f"范围 {pooled_bp.min():.3e}~{pooled_bp.max():.3e}, "
                  f"相对幅度 {pooled_bp.min()/bp_med:.3f}~{pooled_bp.max()/bp_med:.3f}x")

        # ── 每天线独立尖峰清理 ──
        if bp_clean_spikes:
            total_spikes = 0
            for ant in range(n_antennas):
                per_antenna_bandpass[ant], n_spikes = clean_bandpass_spikes(
                    per_antenna_bandpass[ant], threshold=5.0,
                    ratio_threshold=3.0, grow=5, window=101
                )
                total_spikes += n_spikes
            if verbose:
                pooled_clean = np.median(per_antenna_bandpass, axis=0)
                bp_med_clean = np.median(pooled_clean)
                # 每天线 bandpass 的幅度范围
                ant_mins = [per_antenna_bandpass[a].min() for a in range(n_antennas)]
                ant_maxs = [per_antenna_bandpass[a].max() for a in range(n_antennas)]
                print(f"    Bandpass 尖峰清理: 总计 {total_spikes}/{n_antennas*n_channels} 通道 "
                      f"({100*total_spikes/(n_antennas*n_channels):.1f}%), "
                      f"天线幅度范围 {min(ant_mins)/bp_med_clean:.2f}~{max(ant_maxs)/bp_med_clean:.2f}x")

    # ── 时间域 RFI 检测 ──
    bad_ts_set = set()
    if td_threshold > 0 and len(frame_powers) > 0:
        frame_powers = np.array(frame_powers)
        valid_mask = frame_powers > 0
        if valid_mask.sum() > td_window:
            # 滑动中位
            w = td_window
            half = w // 2
            smoothed = np.empty_like(frame_powers)
            for i in range(len(frame_powers)):
                lo = max(0, i - half)
                hi = min(len(frame_powers), i + half + 1)
                smoothed[i] = np.median(frame_powers[lo:hi])
            residual = frame_powers - smoothed
            mad = np.median(np.abs(residual))
            if mad > 0:
                threshold_line = smoothed + td_threshold * mad
                for i, ts in enumerate(frame_ts_valid):
                    if frame_powers[i] > threshold_line[i]:
                        bad_ts_set.add(ts)

    t_scan = time.time() - t0
    if verbose:
        print(f"    扫描完成: {t_scan:.1f}s")
        if td_threshold > 0:
            n_bad = len(bad_ts_set)
            print(f"    时间域 RFI 标记帧: {n_bad} / {len(timestamps)} "
                  f"({100*n_bad/max(len(timestamps),1):.1f}%)")
        else:
            print(f"    时间域 RFI: 已关闭")

    return per_antenna_bandpass, bad_ts_set


# ═══════════════════════════════════════════════════════════════════
# 加权方案
# ═══════════════════════════════════════════════════════════════════
def compute_weights(vis_flat, freq_sky_mhz, scheme='uniform'):
    """计算每个可见度数据点的权重。

    Parameters
    ----------
    vis_flat : ndarray (N,) complex128
    freq_sky_mhz : ndarray (n_channels,)
    scheme : str
        'uniform'  - 所有权重 = 1
        'density'  - 保留为 1 (密度校正在 histogram 阶段完成)

    Returns
    -------
    weights : ndarray (N,) float64
    """
    n = len(vis_flat)
    if scheme == 'uniform':
        return np.ones(n, dtype=np.float64)
    elif scheme == 'density':
        # 密度加权在 histogram 后处理，这里先给 uniform
        return np.ones(n, dtype=np.float64)
    else:
        raise ValueError(f"未知加权方案: {scheme}")


# ═══════════════════════════════════════════════════════════════════
# 核心：逐帧格点化
# ═══════════════════════════════════════════════════════════════════
def grid_visibilities(frames_by_ts, timestamps, antennas,
                      freq_sky_mhz,  wavelengths,
                      latitude_deg, include_conjugates,
                      bins=4096, weight_scheme='uniform',
                      use_gpu=True, verbose=True,
                      rfi_threshold=3.0, rfi_ratio=2.0, rfi_grow=5,
                      td_threshold=5.0, td_window=101,
                      bandpass_enable=True, bp_clean_spikes=True,
                      amp_smooth_window=0):
    """逐帧将可见度格点化到 UV 网格。

    核心流程:
      for each frame:
        1. GPU 计算 (u,v) 坐标
        2. 加载可见度
        3. histogram2d 累积到 (H_real, H_imag, H_w)

    Returns
    -------
    V_grid : ndarray (bins, bins) complex128
    H_w : ndarray (bins, bins) float64
    xu, yv : ndarray (bins+1,)
    """
    n_frames = len(timestamps)
    n_channels = len(freq_sky_mhz)
    n_bl = 28
    n_pts_per_frame = n_channels * n_bl * (2 if include_conjugates else 1)

    # ── 预计算所有时角 ──
    # (这里假设调用者已经传入了正确的 hour_angles，为简单起见在 main 里预先算好)
    # 但我们这里也接收 raw timestamps...

    # ── 确定 UV 范围 ──
    # 先跑前几帧估算
    sample_frames = min(5, n_frames)
    all_u_samples = []
    all_v_samples = []
    for fi in range(sample_frames):
        ts = timestamps[fi]
        ha = compute_hour_angle(ts)
        u_s, v_s = _compute_uvw_single_frame(
            antennas, ha, wavelengths, latitude_deg,
            include_conjugates, use_gpu
        )
        all_u_samples.append(u_s)
        all_v_samples.append(v_s)

    u_concat = np.concatenate(all_u_samples)
    v_concat = np.concatenate(all_v_samples)
    uv_lim = max(np.percentile(np.abs(u_concat), 99.9),
                 np.percentile(np.abs(v_concat), 99.9)) * 1.05
    uv_range = [[-uv_lim, uv_lim], [-uv_lim, uv_lim]]

    if verbose:
        print(f"\n  UV 格点范围: ±{uv_lim:.1f} λ  ({bins}×{bins} bins, "
              f"bin={2*uv_lim/bins:.4f} λ)")
        print(f"  每帧点数: {n_pts_per_frame:,} "
              f"= {n_channels}ch × {n_bl}bl × {2 if include_conjugates else 1}")
        print(f"  累积格点化 ({n_frames} 帧)...")

    # ── 初始化累加器 ──
    H_real = np.zeros((bins, bins), dtype=np.float64)
    H_imag = np.zeros((bins, bins), dtype=np.float64)
    H_w = np.zeros((bins, bins), dtype=np.float64)

    t_start = time.time()
    frames_processed = 0
    total_rfi = 0

    # ── 第一遍：估计每天线 bandpass + 时间域 RFI 检测 ──
    per_antenna_bp, bad_ts_set = precompute_rfi_references(
        frames_by_ts, timestamps, n_channels,
        verbose=verbose,
        td_threshold=td_threshold,
        td_window=td_window,
        bandpass_enable=bandpass_enable,
        bp_clean_spikes=bp_clean_spikes
    )
    if bandpass_enable and per_antenna_bp is None:
        if verbose:
            print("  [WARN] bandpass 估计失败，回退到原始可见度")

    t0_loop = time.time()
    for fi, ts in enumerate(timestamps):
        t_frame = time.time()

        # 0. 跳过时间域 RFI 坏帧
        if ts in bad_ts_set:
            continue

        # 1. 计算 UVW
        ha = compute_hour_angle(ts)
        u, v = _compute_uvw_single_frame(
            antennas, ha, wavelengths, latitude_deg,
            include_conjugates, use_gpu
        )

        # 2. RFI 检测（利用自相关）
        file_map = frames_by_ts[ts]
        rfi_mask = detect_rfi_channels(file_map, n_channels,
                                         threshold=rfi_threshold,
                                         ratio_threshold=rfi_ratio,
                                         grow=rfi_grow)
        n_rfi = int(np.count_nonzero(rfi_mask))
        total_rfi += n_rfi

        # 3. 加载可见度（剔除 RFI + 每天线 bandpass 校正 + 振幅平滑）
        vis = load_frame_visibilities(file_map, n_channels,
                                      include_conjugates,
                                      rfi_mask=rfi_mask,
                                      per_antenna_bandpass=per_antenna_bp,
                                      amp_smooth_window=amp_smooth_window)

        # 4. 计算权重
        w = compute_weights(vis, freq_sky_mhz, weight_scheme)

        # 4. 2D 直方图累积
        h_re, xu, yv = np.histogram2d(
            u, v, bins=bins, range=uv_range,
            weights=vis.real * w
        )
        h_im, _, _ = np.histogram2d(
            u, v, bins=bins, range=uv_range,
            weights=vis.imag * w
        )
        h_cnt, _, _ = np.histogram2d(
            u, v, bins=bins, range=uv_range,
            weights=w
        )

        H_real += h_re
        H_imag += h_im
        H_w += h_cnt

        frames_processed += 1

        # 进度
        if verbose and (fi + 1) % 100 == 0:
            elapsed = time.time() - t_start
            fps = frames_processed / elapsed
            eta = (n_frames - frames_processed) / fps
            print(f"    [{fi+1}/{n_frames}] {elapsed:.0f}s elapsed, "
                  f"{fps:.1f} fps, ETA {eta:.0f}s")

    t_total = time.time() - t_start
    if verbose:
        print(f"  格点化完成: {t_total:.1f}s "
              f"({frames_processed/t_total:.1f} fps, "
              f"{frames_processed * n_pts_per_frame:,} 点)")

    # ── 计算格点化可见度 ──
    mask = H_w > 0
    V_grid = np.zeros((bins, bins), dtype=np.complex128)
    V_grid[mask] = (H_real[mask] + 1j * H_imag[mask]) / H_w[mask]

    # ── 密度加权后处理 ──
    if weight_scheme == 'density':
        # 对过采样的 bin 降权: w_density = 1 / sqrt(n_hits)
        # 等效于 robust weighting 的近似
        density_weight = np.ones((bins, bins), dtype=np.float64)
        density_weight[mask] = 1.0 / np.sqrt(np.maximum(H_w[mask], 1))
        V_grid[mask] *= density_weight[mask]

    # ── 统计 ──
    if verbose:
        non_zero = np.count_nonzero(mask)
        n_bad = len(bad_ts_set)
        n_frames_used = frames_processed
        avg_rfi = total_rfi / max(n_frames_used, 1)
        print(f"\n  格点化统计:")
        print(f"    总 bins:         {bins*bins:>12,}")
        print(f"    填充 bins:       {non_zero:>12,}  ({100*non_zero/(bins*bins):.1f}%)")
        print(f"    最大 hits/bin:   {int(H_w.max()):>12}")
        print(f"    中位 hits/bin:   {np.median(H_w[mask]):>12.1f}")
        print(f"    |V| 范围:        {np.abs(V_grid[mask]).min():.3e} ~ {np.abs(V_grid[mask]).max():.3e}")
        print(f"    总累加权:        {H_w.sum():.3e}")
        print(f"    使用帧数:        {n_frames_used:>12,} / {n_frames} "
              f"(时间域 RFI 跳过 {n_bad}, {100*n_bad/max(n_frames,1):.1f}%)")
        print(f"    频率 RFI 标记:   {total_rfi:>12,} "
              f"(帧均 {avg_rfi:.1f} ch, {100*avg_rfi/n_channels:.1f}%)")
        if bandpass_enable and per_antenna_bp is not None:
            pooled = np.median(per_antenna_bp, axis=0)
            bp_med = np.median(pooled)
            info = f"已启用 (每天线, 池化幅度 {pooled.min()/bp_med:.2f}~{pooled.max()/bp_med:.2f}x"
            if amp_smooth_window > 0:
                info += f", 振幅平滑窗口={amp_smooth_window}"
            info += ")"
            print(f"    Bandpass 校正:   {info}")

    return V_grid, H_w, xu, yv, uv_lim


# ═══════════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════════
def plot_gridded_visibilities(V_grid, H_w, xu, yv, uv_lim, output_dir,
                               date_str, bins, weight_scheme, dpi=200):
    """生成格点化可见度的实部和虚部图（scatter 散点绘制，避免 pcolormesh 方块效应）。"""
    os.makedirs(output_dir, exist_ok=True)

    label = f"_conj" if True else ""  # 目前总是含共轭
    x_centers = 0.5 * (xu[:-1] + xu[1:])
    y_centers = 0.5 * (yv[:-1] + yv[1:])

    V_real = np.real(V_grid)
    V_imag = np.imag(V_grid)
    mask = H_w > 0  # 有采样的 bin

    # ── 提取非零 bin 的坐标和值，用于 scatter 绘制 ──
    rows, cols = np.nonzero(mask)
    sx = x_centers[rows]
    sy = y_centers[cols]
    sv_real = V_real[rows, cols]
    sv_imag = V_imag[rows, cols]
    n_scatter = len(sx)
    # 根据点数自适应 marker 大小
    ms_real = max(0.05, min(2.0, 200.0 / np.sqrt(n_scatter))) if n_scatter > 0 else 1.0
    ms_imag = ms_real

    if n_scatter > 0:
        vlim_real = np.max(np.abs(sv_real))
        vlim_imag = np.max(np.abs(sv_imag))
    else:
        vlim_real = vlim_imag = 1.0

    # ── 2×2 布局: 实部 + 虚部 + 覆盖 + 径向剖面 ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Gridded Visibilities (scatter) — {date_str}  '
                 f'({bins}×{bins}, {weight_scheme} weight)',
                 fontsize=15, fontweight='bold')

    # 实部 (scatter)
    ax = axes[0, 0]
    if n_scatter > 0:
        im = ax.scatter(sx, sy, c=sv_real, cmap='RdBu_r',
                        s=ms_real, marker='.', vmin=-vlim_real, vmax=vlim_real,
                        rasterized=True, edgecolors='none')
    ax.set_xlabel('u (λ)', fontsize=12)
    ax.set_ylabel('v (λ)', fontsize=12)
    ax.set_title(f'Real Part (Re[V])  — {n_scatter:,} pts', fontsize=13)
    ax.axhline(0, color='black', lw=0.5, alpha=0.3)
    ax.axvline(0, color='black', lw=0.5, alpha=0.3)
    ax.set_aspect('equal')
    if n_scatter > 0:
        plt.colorbar(im, ax=ax, label='Re[V]')

    # 虚部 (scatter)
    ax = axes[0, 1]
    if n_scatter > 0:
        im = ax.scatter(sx, sy, c=sv_imag, cmap='RdBu_r',
                        s=ms_imag, marker='.', vmin=-vlim_imag, vmax=vlim_imag,
                        rasterized=True, edgecolors='none')
    ax.set_xlabel('u (λ)', fontsize=12)
    ax.set_ylabel('v (λ)', fontsize=12)
    ax.set_title(f'Imaginary Part (Im[V])  — {n_scatter:,} pts', fontsize=13)
    ax.axhline(0, color='black', lw=0.5, alpha=0.3)
    ax.axvline(0, color='black', lw=0.5, alpha=0.3)
    ax.set_aspect('equal')
    if n_scatter > 0:
        plt.colorbar(im, ax=ax, label='Im[V]')

    # 击中数 (覆盖)
    ax = axes[1, 0]
    H_mask = H_w > 0
    if H_mask.any() and H_w[H_mask].max() > H_w[H_mask].min():
        H_log = np.where(H_mask, H_w, 1.0)
        vmin_h = float(max(H_w[H_mask].min(), 1))
        vmax_h = float(H_w[H_mask].max())
        if vmax_h <= vmin_h:
            vmax_h = vmin_h * 10
        im = ax.pcolormesh(xu, yv, H_log.T, norm=LogNorm(vmin=vmin_h, vmax=vmax_h),
                           cmap='viridis', rasterized=True)
    else:
        im = ax.pcolormesh(xu, yv, H_w.T, cmap='viridis', rasterized=True)
    ax.set_xlabel('u (λ)', fontsize=12)
    ax.set_ylabel('v (λ)', fontsize=12)
    ax.set_title(f'Hits per bin (log scale)', fontsize=13)
    ax.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax.axvline(0, color='white', lw=0.5, alpha=0.3)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Hits')

    # 径向剖面 Re[V] vs |uv|
    ax = axes[1, 1]
    uv_radii = np.sqrt(x_centers[:, None]**2 + y_centers[None, :]**2)
    r_bins = np.linspace(0, np.sqrt(2)*uv_lim, 100)
    r_centers_bin = 0.5 * (r_bins[:-1] + r_bins[1:])
    real_flat = V_real.ravel()
    radii_flat = uv_radii.ravel()
    valid = mask.ravel() & np.isfinite(radii_flat)
    medians = np.zeros(len(r_centers_bin))
    lo = np.zeros(len(r_centers_bin))
    hi = np.zeros(len(r_centers_bin))
    for k in range(len(r_centers_bin)):
        in_bin = valid & (radii_flat >= r_bins[k]) & (radii_flat < r_bins[k+1])
        if in_bin.sum() > 0:
            vals = real_flat[in_bin]
            medians[k] = np.median(vals)
            lo[k] = np.percentile(vals, 16)
            hi[k] = np.percentile(vals, 84)
        else:
            medians[k] = np.nan
            lo[k] = np.nan
            hi[k] = np.nan
    ax.plot(r_centers_bin, medians, 'b-', lw=1.5, label='Median Re[V]')
    ax.fill_between(r_centers_bin, lo, hi, alpha=0.3, color='blue', label='±1σ')
    ax.set_xlabel('|uv| (λ)', fontsize=12)
    ax.set_ylabel('Re[V]', fontsize=12)
    ax.set_title('Radial profile of Re[V]', fontsize=13)
    ax.legend(fontsize=9)
    ax.axhline(0, color='gray', lw=0.5, linestyle='--')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fname = f'{output_dir}/gridded_vis_{date_str}{label}_{bins}bins_scatter.png'
    fig.savefig(fname, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  保存: {fname}")

    # ── 保存 numpy 数据 ──
    npz_path = f'{output_dir}/gridded_vis_{date_str}{label}_{bins}bins.npz'
    np.savez_compressed(
        npz_path,
        V_real=V_grid.real,
        V_imag=V_grid.imag,
        V_grid=V_grid,
        H_w=H_w,
        xu=xu, yv=yv,
        uv_lim=uv_lim,
        bins=bins,
        weight_scheme=weight_scheme
    )
    print(f"  保存: {npz_path} "
          f"({os.path.getsize(npz_path)/1024**2:.1f} MB)")

    # ── 保存 FITS 文件 ──
    fits_path = f'{output_dir}/gridded_vis_{date_str}{label}_{bins}bins.fits'
    try:
        import astropy.io.fits as pyfits
        hdu = pyfits.PrimaryHDU(V_grid.real)
        hdu.writeto(fits_path, overwrite=True)
        print(f"  保存: {fits_path}")
    except ImportError:
        print(f"  [WARN] astropy 未安装，跳过 FITS 输出")


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════
def main():
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                       errors='replace')

    parser = argparse.ArgumentParser(
        description='精细化 UV 格点化 — 加权平均可见度映射到 UV 网格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python grid_visibilities.py                                    # 默认配置
  python grid_visibilities.py --bins 8192 --weight density      # 高分辨率 + 密度加权
  python grid_visibilities.py --max-frames 100 --cpu            # CPU 测试
  python grid_visibilities.py --nch 256                         # 降采样
        """.strip()
    )
    parser.add_argument('--dir', default='correlation_results',
                        help='CSV 数据目录 (default: correlation_results)')
    parser.add_argument('--date', default=None,
                        help='日期 YYYYMMDD (default: 自动)')
    parser.add_argument('--freq', type=float, default=150.0,
                        help='中心频率 MHz (default: 150)')
    parser.add_argument('--nch', type=int, default=0,
                        help='通道数 (0=全部4096)')
    parser.add_argument('--antennas', default='optimized_antenna_coordinates.txt',
                        help='天线坐标文件')
    parser.add_argument('--lat', type=float, default=None,
                        help='观测纬度')
    parser.add_argument('--lon', type=float, default=None,
                        help='观测经度')
    parser.add_argument('--utc-offset', type=float, default=0,
                        help='UTC 时区偏移 (hours)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='最大帧数 (0=全部)')
    parser.add_argument('--bins', type=int, default=4096,
                        help='UV 网格分辨率 (default: 4096)')
    parser.add_argument('--weight', default='uniform',
                        choices=['uniform', 'density'],
                        help='加权方案 (default: uniform)')
    parser.add_argument('--no-conj', action='store_true',
                        help='不使用共轭基线')
    parser.add_argument('--rfi-threshold', type=float, default=3.0,
                        help='频率 RFI MAD 阈值 (0=关闭, 默认 3.0)')
    parser.add_argument('--rfi-ratio', type=float, default=2.0,
                        help='频率 RFI 相对倍数阈值 (0=关闭, 默认 2.0)')
    parser.add_argument('--rfi-grow', type=int, default=5,
                        help='RFI 标志左右扩展通道数 (默认 5)')
    parser.add_argument('--td-threshold', type=float, default=5.0,
                        help='时间域 RFI MAD 阈值 (0=关闭, 默认 5.0)')
    parser.add_argument('--td-window', type=int, default=101,
                        help='时间域滑动窗口帧数 (默认 101)')
    parser.add_argument('--no-bandpass', action='store_true',
                        help='禁用 bandpass 校正')
    parser.add_argument('--no-bp-smooth', action='store_true',
                        help='禁用 bandpass 模板尖峰清理（保留原始带通形状）')
    parser.add_argument('--amp-smooth', type=int, default=0,
                        help='bandpass 校正后振幅平滑窗口（通道数），0=关闭。推荐 101 (约2.5MHz)')
    parser.add_argument('--gpu', action='store_true', default=None,
                        help='强制 GPU')
    parser.add_argument('--cpu', action='store_true',
                        help='强制 CPU')
    parser.add_argument('--output', default='integrated_images',
                        help='输出目录 (default: integrated_images)')
    parser.add_argument('--dpi', type=int, default=200,
                        help='图片 DPI (default: 200)')
    args = parser.parse_args()

    # ── GPU/CPU ──
    if args.cpu:
        use_gpu = False
    elif args.gpu:
        use_gpu = GPU_AVAILABLE
        if not GPU_AVAILABLE:
            print("[WARN] CuPy 不可用，回退 CPU")
    else:
        use_gpu = GPU_AVAILABLE

    engine = "GPU (CuPy)" if use_gpu else "CPU (NumPy)"
    print(f"引擎: {engine}")
    rfi_parts = []
    if args.rfi_threshold > 0 or args.rfi_ratio > 0:
        rfi_parts.append(f"频率: MAD={args.rfi_threshold}, 倍数={args.rfi_ratio}, 扩展={args.rfi_grow}")
    else:
        rfi_parts.append("频率: 关闭")
    if args.td_threshold > 0:
        rfi_parts.append(f"时间: MAD={args.td_threshold}, 窗口={args.td_window}")
    else:
        rfi_parts.append("时间: 关闭")
    bp_status = "已启用" if not args.no_bandpass else "已禁用"
    bp_clean_status = " (尖峰清理)" if (not args.no_bandpass and not args.no_bp_smooth) else ""
    amp_smooth_info = f" (振幅平滑窗口={args.amp_smooth})" if args.amp_smooth > 0 else ""
    print(f"RFI 检测: {' | '.join(rfi_parts)}")
    print(f"Bandpass 校正: {bp_status}{bp_clean_status}{amp_smooth_info}")
    print(f"{'='*60}")

    # ── 纬度/经度 ──
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

    # ── 天线 ──
    t0 = time.time()
    antennas = load_optimized_antennas(args.antennas)
    print(f"  天线: {args.antennas} ({antennas.shape}) "
          f"({time.time()-t0:.1f}s)")

    # ── 发现帧 ──
    t0 = time.time()
    by_date = discover_frames_by_date(args.dir, args.date)
    if not by_date:
        print(f"[ERROR] 未发现数据")
        return
    date_str = args.date or next(iter(by_date.keys()))
    frames_by_ts = by_date[date_str]
    timestamps_all = list(frames_by_ts.keys())
    n_total = len(timestamps_all)
    print(f"  发现日期: {date_str}, {n_total} 帧 ({time.time()-t0:.1f}s)")

    if args.max_frames > 0 and args.max_frames < n_total:
        timestamps_all = timestamps_all[-args.max_frames:]
        print(f"  限制为最后 {args.max_frames} 帧")
    n_frames = len(timestamps_all)

    # ── 频率 ──
    t0 = time.time()
    first_file_map = frames_by_ts[timestamps_all[0]]
    freq_sky_mhz, bin_width_hz = read_sky_frequencies(
        first_file_map, args.freq, args.nch
    )
    n_channels = len(freq_sky_mhz)
    wavelengths = 299.792458 / freq_sky_mhz
    print(f"  天空频率: {freq_sky_mhz.min():.1f} ~ {freq_sky_mhz.max():.1f} MHz "
          f"({n_channels:,}通道, bin_bw={bin_width_hz:.1f} Hz, "
          f"λ={wavelengths.min():.3f}~{wavelengths.max():.3f} m) "
          f"({time.time()-t0:.1f}s)")

    # ── 时角 ──
    t0 = time.time()
    hour_angles = [compute_hour_angle(ts, args.lon, args.utc_offset)
                   for ts in timestamps_all]
    print(f"  时角: {hour_angles[0]:.1f}° ~ {hour_angles[-1]:.1f}° "
          f"(Δ={hour_angles[-1]-hour_angles[0]:.1f}°) ({time.time()-t0:.1f}s)")

    # ── 格点化 ──
    include_conj = not args.no_conj

    V_grid, H_w, xu, yv, uv_lim = grid_visibilities(
        frames_by_ts, timestamps_all, antennas,
        freq_sky_mhz, wavelengths,
        latitude_deg=args.lat,
        include_conjugates=include_conj,
        bins=args.bins,
        weight_scheme=args.weight,
        use_gpu=use_gpu,
        verbose=True,
        rfi_threshold=args.rfi_threshold,
        rfi_ratio=args.rfi_ratio,
        rfi_grow=args.rfi_grow,
        td_threshold=args.td_threshold,
        td_window=args.td_window,
        bandpass_enable=not args.no_bandpass,
        bp_clean_spikes=not args.no_bp_smooth,
        amp_smooth_window=args.amp_smooth
    )

    # ── 可视化 ──
    plot_gridded_visibilities(
        V_grid, H_w, xu, yv, uv_lim,
        args.output, date_str, args.bins, args.weight,
        dpi=args.dpi
    )

    print(f"\n{'='*60}")
    print("完成！")


if __name__ == '__main__':
    main()
