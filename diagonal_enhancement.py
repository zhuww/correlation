"""
瀑布图 2D FFT 对角增强 — 在频域中分离横纵结构与斜向 fringe。

原理：
  - 水平带（时间相干）→ 2D FFT 中沿纵轴（ω_t 轴）的线
  - 垂直线（频率相干）→ 2D FFT 中沿横轴（ω_f 轴）的线
  - 斜向 fringe（不可分解）→ 2D FFT 中沿对角线的峰
  
  通过 PCA 移除前 K 个可分离成分后，计算 2D FFT，
  再掩蔽坐标轴附近的横纵结构，只保留对角区域，
  逆变换得到“仅含对角结构”的图像。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LogNorm, TwoSlopeNorm
from pathlib import Path
import re
import argparse
import time
import os
import sys
from datetime import datetime
from collections import defaultdict, OrderedDict
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ── 中文字体 ──
for fname in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    prop = fm.FontProperties(fname=fname)
    if 'SimHei' in prop.get_name() or 'Microsoft YaHei' in prop.get_name():
        plt.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


# ── 复用函数 ──
def discover_frames(watch_dir, date_str=None):
    csv_files = sorted(Path(watch_dir).glob("correlation_*.csv"))
    if not csv_files: return {}
    frames = defaultdict(dict)
    for f in csv_files:
        m = re.search(r'correlation_(\d{8}_\d{6})', f.name)
        if not m: continue
        ts = m.group(1)
        pair_m = re.search(r'(CH\d+(?:xCH\d+|_AUTO))', f.name)
        if pair_m: frames[ts][pair_m.group(1)] = f
    by_date = defaultdict(OrderedDict)
    for ts in sorted(frames.keys()):
        by_date[ts[:8]][ts] = frames[ts]
    if date_str and date_str in by_date:
        return OrderedDict([(date_str, by_date[date_str])])
    return OrderedDict(sorted(by_date.items()))

def read_auto(file_map, ant_id, n_channels=4096):
    key = f"CH{ant_id}_AUTO"
    if key not in file_map: return None
    try:
        df = pd.read_csv(file_map[key], comment='#', usecols=['magnitude'])
        mag = df['magnitude'].values.astype(np.float64)
        if len(mag) >= n_channels: return mag[:n_channels]
        return np.concatenate([mag, np.zeros(n_channels - len(mag), dtype=np.float64)])
    except Exception: return None

def read_cross(file_map, n_channels=4096):
    key_a, key_b = "CH3xCH8", "CH8xCH3"
    fp = file_map.get(key_a) or file_map.get(key_b)
    if fp is None: return None
    try:
        df = pd.read_csv(fp, comment='#', usecols=['real_part', 'imag_part'])
        re_vals = df['real_part'].values.astype(np.float64)
        im_vals = df['imag_part'].values.astype(np.float64)
        if len(re_vals) >= n_channels:
            return re_vals[:n_channels] + 1j * im_vals[:n_channels]
        return np.zeros(n_channels, dtype=np.complex128)
    except Exception: return None

def detect_freq_rfi(auto3, auto8, threshold=5.0, ratio_threshold=3.0, grow=5, smooth_w=101):
    n_ch = len(auto3)
    median_spec = np.maximum(auto3, auto8)
    half = smooth_w // 2
    mag_pad = np.pad(median_spec, (half, half), mode='reflect')
    baseline = np.median(sliding_window_view(mag_pad, smooth_w), axis=1)
    residual = median_spec - baseline
    mad = np.median(np.abs(residual))
    if mad <= 0: mad = np.median(np.abs(median_spec)) * 1e-3
    if mad <= 0: mad = 1.0
    flagged = np.zeros(n_ch, dtype=bool)
    if threshold > 0: flagged |= median_spec > (baseline + threshold * mad)
    if ratio_threshold > 0: flagged &= median_spec > (baseline * ratio_threshold)
    if grow > 0 and flagged.any():
        grown = np.zeros(n_ch, dtype=bool)
        for i in range(n_ch):
            if flagged[i]:
                lo, hi = max(0, i-grow), min(n_ch, i+grow+1)
                grown[lo:hi] = True
        flagged = grown
    return flagged

def detect_time_rfi(frame_powers, td_threshold=5.0, td_window=101):
    powers = np.array(frame_powers); n = len(powers)
    if n <= td_window: return set()
    half = td_window // 2
    smoothed = np.array([np.median(powers[max(0,i-half):min(n,i+half+1)]) for i in range(n)])
    residual = powers - smoothed
    mad = np.median(np.abs(residual))
    if mad <= 0: mad = np.median(np.abs(powers)) * 1e-3
    if mad <= 0: return set()
    bad = set(np.where(powers > smoothed + td_threshold * mad)[0])
    bad.update(np.where(powers < smoothed - td_threshold * mad * 2)[0])
    return bad

def clean_bandpass_spikes(bp, threshold=5.0, ratio_threshold=3.0, grow=5, window=101):
    n = len(bp); half = window // 2
    mag_pad = np.pad(bp, (half, half), mode='reflect')
    baseline = np.median(sliding_window_view(mag_pad, window), axis=1)
    residual = bp - baseline
    mad = np.median(np.abs(residual))
    if mad <= 0: mad = np.median(np.abs(bp)) * 1e-3
    if mad <= 0: mad = 1.0
    flagged = np.zeros(n, dtype=bool)
    if threshold > 0: flagged |= bp > (baseline + threshold * mad)
    if ratio_threshold > 0: flagged &= bp > (baseline * ratio_threshold)
    if grow > 0 and flagged.any():
        grown = np.zeros(n, dtype=bool)
        for i in range(n):
            if flagged[i]:
                lo, hi = max(0, i-grow), min(n, i+grow+1)
                grown[lo:hi] = True
        flagged = grown
    bp_clean = bp.copy(); bp_clean[flagged] = baseline[flagged]
    return bp_clean

def read_sky_frequencies(file_map, n_channels=4096, center_freq_mhz=150.0):
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=n_channels)
    freq_hz = df['frequency_hz'].values.astype(np.float64)
    return center_freq_mhz + freq_hz / 1e6

def inpaint_masked(data, mask, max_gap=50):
    filled = data.astype(np.float64).copy(); filled[mask] = np.nan
    n_t, n_f = filled.shape
    for fi in range(n_t):
        row = filled[fi]; nan_mask = np.isnan(row)
        if nan_mask.all() or (~nan_mask).sum() < 3: continue
        valid_idx = np.where(~nan_mask)[0]; valid_vals = row[~nan_mask]
        nan_slices = []; in_gap = False; gap_start = 0
        for j in range(n_f):
            if nan_mask[j] and not in_gap: gap_start = j; in_gap = True
            elif not nan_mask[j] and in_gap:
                nan_slices.append((gap_start, j)); in_gap = False
        if in_gap: nan_slices.append((gap_start, n_f))
        for gs, ge in nan_slices:
            if ge - gs <= max_gap: row[gs:ge] = np.interp(np.arange(gs, ge), valid_idx, valid_vals)
            else: row[gs:ge] = 0.0
    for fj in range(n_f):
        col = filled[:, fj]; nan_mask = np.isnan(col)
        if nan_mask.all() or (~nan_mask).sum() < 3:
            col[nan_mask] = 0.0; continue
        valid_idx = np.where(~nan_mask)[0]; valid_vals = col[~nan_mask]
        nan_slices = []; in_gap = False; gap_start = 0
        for i in range(n_t):
            if nan_mask[i] and not in_gap: gap_start = i; in_gap = True
            elif not nan_mask[i] and in_gap:
                nan_slices.append((gap_start, i)); in_gap = False
        if in_gap: nan_slices.append((gap_start, n_t))
        for gs, ge in nan_slices:
            if ge - gs <= max_gap: col[gs:ge] = np.interp(np.arange(gs, ge), valid_idx, valid_vals)
            else: col[gs:ge] = 0.0
        col[np.isnan(col)] = 0.0
    return filled


def main():
    parser = argparse.ArgumentParser(description='2D FFT 对角增强 — 提取斜向 fringe')
    parser.add_argument('--date', default='20260630')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--n-channels', type=int, default=4096)
    parser.add_argument('--center-freq', type=float, default=150.0)
    parser.add_argument('--pca-k', type=int, default=1000,
                        help='PCA 移除前 K 个成分 (default 1000)')
    parser.add_argument('--diag-width', type=int, default=20,
                        help='对角线带宽（像素）(default 20)')
    parser.add_argument('--output', default='integrated_images')
    args = parser.parse_args()

    watch_dir = 'correlation_results'; n_ch = args.n_channels
    K = args.pca_k

    # ── 1. 加载数据 ──
    print(f"扫描 {watch_dir} ...")
    all_data = discover_frames(watch_dir, args.date)
    if args.date not in all_data: print(f"错误: 未找到日期 {args.date}"); sys.exit(1)
    frames_by_ts = all_data[args.date]
    timestamps = list(frames_by_ts.keys())
    if args.max_frames > 0: timestamps = timestamps[:args.max_frames]
    n_frames = len(timestamps)
    print(f"  {n_frames} 帧")

    print(f"\n加载数据...")
    t0 = time.time()
    all_vis, all_auto3, all_auto8, valid_idx = [], [], [], []
    for fi, ts in enumerate(timestamps):
        fm = frames_by_ts[ts]
        vis = read_cross(fm, n_ch); a3 = read_auto(fm, 3, n_ch); a8 = read_auto(fm, 8, n_ch)
        if vis is not None and a3 is not None and a8 is not None:
            all_vis.append(vis); all_auto3.append(a3); all_auto8.append(a8); valid_idx.append(fi)
        if (fi+1) % 500 == 0: print(f"  已读 {fi+1}/{n_frames}")
    n_valid = len(all_vis)
    vis_matrix = np.array(all_vis)        # (n_valid, n_ch) complex
    auto3_s = np.array(all_auto3)
    auto8_s = np.array(all_auto8)
    del all_vis, all_auto3, all_auto8
    print(f"  有效: {n_valid}/{n_frames} ({time.time()-t0:.1f}s)")

    # ── 2. RFI 检测 ──
    freq_rfi = np.zeros((n_valid, n_ch), dtype=bool)
    for fi in range(n_valid):
        freq_rfi[fi] = detect_freq_rfi(auto3_s[fi], auto8_s[fi])
    total_freq_rfi = freq_rfi.sum()
    print(f"  频率 RFI: {total_freq_rfi:,} ({100*total_freq_rfi/(n_valid*n_ch):.2f}%)")

    time_rfi = np.zeros(n_valid, dtype=bool)
    frame_powers = [np.median(np.maximum(auto3_s[fi], auto8_s[fi])) for fi in range(n_valid)]
    bad = detect_time_rfi(frame_powers)
    for bi in bad: time_rfi[bi] = True
    print(f"  时域 RFI: {len(bad)}/{n_valid} 帧")

    # Bandpass
    bp3 = np.median(auto3_s[~time_rfi], axis=0) if (~time_rfi).any() else np.median(auto3_s, axis=0)
    bp8 = np.median(auto8_s[~time_rfi], axis=0) if (~time_rfi).any() else np.median(auto8_s, axis=0)
    bp3 = clean_bandpass_spikes(bp3); bp8 = clean_bandpass_spikes(bp8)
    expected = np.sqrt(np.maximum(bp3, 1e-30) * np.maximum(bp8, 1e-30))
    bp_factor = np.median(expected) / np.maximum(expected, 1e-30)
    vis_matrix = vis_matrix * bp_factor[np.newaxis, :]

    # 掩码
    combined_mask = freq_rfi.copy(); combined_mask[time_rfi] = True
    vis_real = vis_matrix.real
    vis_filled = inpaint_masked(vis_real, combined_mask)

    # 去均值
    vis_centered = vis_filled - vis_filled.mean(axis=0)

    # ── 3. PCA ──
    if K > 0:
        print(f"\nSVD 分解 ({n_valid}×{n_ch})...")
        t_svd = time.time()
        U, S, Vh = np.linalg.svd(vis_centered, full_matrices=False)
        print(f"  SVD ({time.time()-t_svd:.1f}s)")
        residual = np.zeros_like(vis_centered)
        if K < min(n_valid, n_ch):
            for k in range(K, len(S)):
                residual += S[k] * np.outer(U[:, k], Vh[k, :])
        else:
            residual = vis_centered.copy()
        residual = residual + vis_filled.mean(axis=0)
    else:
        residual = vis_filled.copy()

    # ── 4. 重排频率 ──
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, args.center_freq)
    Nc = n_ch
    if Nc % 2 == 0:
        reorder = np.concatenate([np.arange(Nc//2+1, Nc), np.arange(0, Nc//2+1)]).astype(int)
    else:
        reorder = np.arange(Nc)
    freq_disp = freq_raw[reorder]
    residual_r = residual[:, reorder]

    t0_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
    times = np.array([(datetime.strptime(timestamps[idx], "%Y%m%d_%H%M%S") - t0_dt).total_seconds()/60
                      for idx in valid_idx])

    # ── 5. 2D FFT ──
    print(f"\n2D FFT...")
    # 使用窗口减少频谱泄漏
    win_t = np.hanning(n_valid)
    win_f = np.hanning(n_ch)
    win2d = win_t[:, None] * win_f[None, :]
    data_windowed = residual_r * win2d

    # 2D FFT（实数输入，复数输出）
    fft2d = np.fft.fft2(data_windowed)
    # 将零频移到中心
    fft2d_shifted = np.fft.fftshift(fft2d)
    # 对数幅度
    fft_mag = np.abs(fft2d_shifted)
    fft_log = np.log10(np.maximum(fft_mag, 1e-30))

    # 频率轴
    # ω_t: 时间频率（cycles per frame）
    # ω_f: 频率频率（cycles per channel）
    omega_t = np.fft.fftshift(np.fft.fftfreq(n_valid))
    omega_f = np.fft.fftshift(np.fft.fftfreq(n_ch))

    # ── 6. 对角增强滤波 ──
    # 掩蔽坐标轴附近的横纵结构（矩形掩码）
    # 保留对角线附近 ±width 像素区域
    print(f"对角增强滤波 (带宽={args.diag_width})...")
    Nt, Nf = fft2d_shifted.shape
    # 创建对角掩码
    diag_mask = np.zeros((Nt, Nf), dtype=bool)
    # 主对角线附近：|ω_t - ω_f| < thresh（考虑采样率差异）
    # 副对角线附近：|ω_t + ω_f| < thresh
    # 由于 Nt ≠ Nf，需要缩放
    scale_t = Nt / max(Nt, Nf); scale_f = Nf / max(Nt, Nf)
    # 中心在零频
    cy, cx = Nt // 2, Nf // 2
    # 对角线方向：角度 = 45°, 135°
    # 在像素坐标中，创建网格
    yy, xx = np.mgrid[:Nt, :Nf]
    yy_c = yy - cy; xx_c = xx - cx
    # 归一化到 [-1, 1]
    yy_n = yy_c / (Nt/2); xx_n = xx_c / (Nf/2)
    # 对角线角度（45° 和 135°）
    # 以角度判断：arctan2(y, x) = 45° 或 135°
    angle = np.degrees(np.arctan2(yy_c, xx_c))
    # 对角线附近：角度在 (45±w, 135±w) 附近
    # 转换为绝对角度差
    # 45° 对角线：y ≈ x
    # 135° 对角线：y ≈ -x
    diag_angle_width = 15  # 度
    # 掩码：保留对角线附近
    near_45 = np.abs(angle - 45) < diag_angle_width
    near_135 = np.abs(angle - 135) < diag_angle_width
    near_m45 = np.abs(angle + 45) < diag_angle_width
    near_m135 = np.abs(angle + 135) < diag_angle_width
    diag_mask = near_45 | near_135 | near_m45 | near_m135
    # 同时排除非常接近原点的区域（DC+低频横纵结构）
    # 原点周围小圆
    r = np.sqrt((yy_c/(Nt/2))**2 + (xx_c/(Nf/2))**2)
    dc_mask = r < 0.05  # 中心 5% 区域 = DC 和低阶横纵结构
    # 最终对角增强掩码：对角线 + 排除中心 + 排除坐标轴
    # 坐标轴掩码：|y| < 2 或 |x| < 2（横纵结构）
    axial_mask = (np.abs(yy_c) < 3) | (np.abs(xx_c) < 3)
    final_mask = diag_mask & (~dc_mask) & (~axial_mask)
    # 增强 = 保留对角，抑制其他
    fft_diag = fft2d_shifted.copy()
    # 将对角区域以外的能量衰减到 10%
    fft_diag[~final_mask] *= 0.1
    # 或者：完全置零？不，衰减即可，保留微弱信息

    # 逆变换
    diag_enhanced = np.fft.ifft2(np.fft.ifftshift(fft_diag)).real
    # 除以窗口效应（近似）
    diag_enhanced = diag_enhanced / (win2d + 1e-30)
    # 截断极端值（窗口边缘）
    diag_enhanced = np.clip(diag_enhanced, np.percentile(diag_enhanced, 1), np.percentile(diag_enhanced, 99))

    # ── 7. 绘图 ──
    print("\n绘图...")
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], hspace=0.12, wspace=0.08)

    # --- 第1行: 原始瀑布、PCA 清理、对角增强 ---
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_clean = fig.add_subplot(gs[0, 1])
    ax_diag = fig.add_subplot(gs[0, 2])

    vabs = max(abs(residual_r.min()), abs(residual_r.max()))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    ax_orig.pcolormesh(freq_disp, times, residual_r, cmap='RdBu_r', norm=norm,
                        shading='nearest', rasterized=True)
    ax_orig.set_title(f'PCA 清理后 (K={K})\n保留方差 {100*np.sum(S[K:]**2)/np.sum(S**2):.1f}%',
                      fontsize=10, fontweight='bold')
    ax_orig.set_ylabel('时间 [分钟]', fontsize=9)
    ax_orig.set_xlim(freq_disp.min(), freq_disp.max())
    ax_orig.set_ylim(times[0], times[-1])
    ax_orig.ticklabel_format(useOffset=False, style='plain')

    # 对角增强
    dabs = max(abs(diag_enhanced.min()), abs(diag_enhanced.max()))
    norm_d = TwoSlopeNorm(vmin=-dabs, vcenter=0, vmax=dabs)
    ax_diag.pcolormesh(freq_disp, times, diag_enhanced, cmap='RdBu_r', norm=norm_d,
                        shading='nearest', rasterized=True)
    ax_diag.set_title(f'2D FFT 对角增强\n仅保留对角方向，抑制横纵', fontsize=10, fontweight='bold')
    ax_diag.set_xlim(freq_disp.min(), freq_disp.max())
    ax_diag.set_ylim(times[0], times[-1])
    ax_diag.ticklabel_format(useOffset=False, style='plain')

    # 两图差值（对角增强 - 原始）= 仅被增强的部分
    diff = diag_enhanced - residual_r
    dabs2 = max(abs(diff.min()), abs(diff.max()))
    norm_d2 = TwoSlopeNorm(vmin=-dabs2, vcenter=0, vmax=dabs2)
    ax_clean.pcolormesh(freq_disp, times, diff, cmap='RdBu_r', norm=norm_d2,
                         shading='nearest', rasterized=True)
    ax_clean.set_title(f'对角增强 - 原始 = 仅被增强的斜向结构', fontsize=10, fontweight='bold')
    ax_clean.set_xlim(freq_disp.min(), freq_disp.max())
    ax_clean.set_ylim(times[0], times[-1])
    ax_clean.ticklabel_format(useOffset=False, style='plain')

    # --- 第2行: 2D FFT 幅度 ---
    ax_fft = fig.add_subplot(gs[1, 0])
    ax_fft_mask = fig.add_subplot(gs[1, 1])
    ax_fft_diag = fig.add_subplot(gs[1, 2])

    # 2D FFT 对数幅度（全图）
    fft_vmin = np.percentile(fft_log, 5); fft_vmax = np.percentile(fft_log, 95)
    ax_fft.imshow(fft_log, aspect='auto', origin='lower', cmap='viridis',
                  vmin=fft_vmin, vmax=fft_vmax, extent=[omega_f.min(), omega_f.max(), omega_t.min(), omega_t.max()])
    ax_fft.set_title('2D FFT 对数幅度', fontsize=10, fontweight='bold')
    ax_fft.set_xlabel('ω_f [cycles/ch]', fontsize=9)
    ax_fft.set_ylabel('ω_t [cycles/frame]', fontsize=9)
    ax_fft.axhline(0, color='r', alpha=0.3, linewidth=0.5)
    ax_fft.axvline(0, color='r', alpha=0.3, linewidth=0.5)
    ax_fft.grid(True, alpha=0.2, linewidth=0.3)

    # 对角掩码可视化
    ax_fft_mask.imshow(final_mask, aspect='auto', origin='lower', cmap='Greens')
    ax_fft_mask.set_title('对角增强掩码 (绿=保留)', fontsize=10, fontweight='bold')
    ax_fft_mask.set_xlabel('ω_f [cycles/ch]', fontsize=9)
    ax_fft_mask.set_ylabel('ω_t [cycles/frame]', fontsize=9)

    # 2D FFT 对角增强后
    fft_diag_log = np.log10(np.maximum(np.abs(fft_diag), 1e-30))
    ax_fft_diag.imshow(fft_diag_log, aspect='auto', origin='lower', cmap='viridis',
                       vmin=fft_vmin, vmax=fft_vmax, extent=[omega_f.min(), omega_f.max(), omega_t.min(), omega_t.max()])
    ax_fft_diag.set_title('2D FFT 对角增强后', fontsize=10, fontweight='bold')
    ax_fft_diag.set_xlabel('ω_f [cycles/ch]', fontsize=9)
    ax_fft_diag.set_ylabel('ω_t [cycles/frame]', fontsize=9)
    ax_fft_diag.axhline(0, color='r', alpha=0.3, linewidth=0.5)
    ax_fft_diag.axvline(0, color='r', alpha=0.3, linewidth=0.5)
    ax_fft_diag.grid(True, alpha=0.2, linewidth=0.3)

    # --- 第3行: 频谱切片（1D 平均） ---
    # 对角 vs 横纵 的频谱对比
    ax_slice_t = fig.add_subplot(gs[2, 0])
    ax_slice_f = fig.add_subplot(gs[2, 1])
    ax_slice_cum = fig.add_subplot(gs[2, 2])

    # 沿 ω_t 的切片（固定 ω_f ≈ 0 附近 = 横纵结构）
    # 沿 ω_t 的切片（对角区域）
    center_f = Nf // 2; center_t = Nt // 2
    # 横纵切片：ω_f = 0 附近，沿 ω_t
    axial_t_slice = fft_mag[:, center_f:center_f+5].mean(axis=1)
    # 对角切片：ω_f ≈ ω_t 附近，沿 ω_t
    diag_indices = []
    for i in range(Nt):
        j = center_f + int((i - center_t) * Nf / Nt)  # 45° 对角线
        if 0 <= j < Nf: diag_indices.append((i, j))
    if diag_indices:
        diag_vals = [fft_mag[i, j] for i, j in diag_indices]
        diag_t_pos = [omega_t[i] for i, _ in diag_indices]
    else:
        diag_vals = []; diag_t_pos = []

    ax_slice_t.semilogy(omega_t, axial_t_slice, 'b-', alpha=0.6, linewidth=1, label='横纵结构 (ω_f≈0)')
    if diag_vals:
        ax_slice_t.semilogy(diag_t_pos, diag_vals, 'r-', alpha=0.8, linewidth=1.5, label='对角结构 (ω_f≈ω_t)')
    ax_slice_t.set_title('沿 ω_t 的频谱切片', fontsize=10, fontweight='bold')
    ax_slice_t.set_xlabel('ω_t [cycles/frame]', fontsize=9)
    ax_slice_t.set_ylabel('幅度', fontsize=9)
    ax_slice_t.legend(fontsize=8)
    ax_slice_t.grid(True, alpha=0.2)
    ax_slice_t.set_xlim(omega_t.min(), omega_t.max())

    # 沿 ω_f 的切片
    axial_f_slice = fft_mag[center_t:center_t+5, :].mean(axis=0)
    diag_indices_f = []
    for j in range(Nf):
        i = center_t + int((j - center_f) * Nt / Nf)
        if 0 <= i < Nt: diag_indices_f.append((i, j))
    if diag_indices_f:
        diag_vals_f = [fft_mag[i, j] for i, j in diag_indices_f]
        diag_f_pos = [omega_f[j] for _, j in diag_indices_f]
    else:
        diag_vals_f = []; diag_f_pos = []

    ax_slice_f.semilogy(omega_f, axial_f_slice, 'b-', alpha=0.6, linewidth=1, label='横纵结构 (ω_t≈0)')
    if diag_vals_f:
        ax_slice_f.semilogy(diag_f_pos, diag_vals_f, 'r-', alpha=0.8, linewidth=1.5, label='对角结构 (ω_t≈ω_f)')
    ax_slice_f.set_title('沿 ω_f 的频谱切片', fontsize=10, fontweight='bold')
    ax_slice_f.set_xlabel('ω_f [cycles/ch]', fontsize=9)
    ax_slice_f.set_ylabel('幅度', fontsize=9)
    ax_slice_f.legend(fontsize=8)
    ax_slice_f.grid(True, alpha=0.2)
    ax_slice_f.set_xlim(omega_f.min(), omega_f.max())

    # 累积功率分布：对角 vs 横纵
    # 计算对角区域和横纵区域的总功率
    total_power = np.sum(fft_mag**2)
    diag_power = np.sum(fft_mag[final_mask]**2)
    axial_power = np.sum(fft_mag[axial_mask]**2)
    other_power = total_power - diag_power - axial_power
    ax_slice_cum.bar(['横纵', '对角', '其他'], [axial_power, diag_power, other_power],
                      color=['blue', 'red', 'gray'], alpha=0.6)
    ax_slice_cum.set_title('2D FFT 功率分布', fontsize=10, fontweight='bold')
    ax_slice_cum.set_ylabel('功率', fontsize=9)
    # 百分比标注
    for i, (p, label) in enumerate([(axial_power, 'axial'), (diag_power, 'diag'), (other_power, 'other')]):
        pct = 100 * p / total_power
        ax_slice_cum.text(i, p * 1.02, f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle(f'CH3×CH8 瀑布图 2D FFT 对角增强 — {args.date} (K={K})',
                 fontsize=14, fontweight='bold', y=0.98)

    os.makedirs(args.output, exist_ok=True)
    out_path = f'{args.output}/diagonal_enhanced_K{K}_{args.date}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n保存: {out_path}")
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"2D FFT 对角增强汇总:")
    print(f"  矩阵: {n_valid}×{n_ch}")
    print(f"  PCA K: {K}")
    print(f"  对角带宽: {args.diag_width} 像素")
    print(f"  2D FFT 总功率: {total_power:.2e}")
    print(f"  横纵区域功率: {axial_power:.2e} ({100*axial_power/total_power:.1f}%)")
    print(f"  对角区域功率: {diag_power:.2e} ({100*diag_power/total_power:.1f}%)")
    print(f"  其他区域功率: {other_power:.2e} ({100*other_power/total_power:.1f}%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
