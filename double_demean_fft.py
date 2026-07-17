#!/usr/bin/env python
"""
快速脚本：对 PCA 清理后的残余图做行列双去均值 + 2D FFT 聚焦，
在 2D 频域寻找对角结构峰值。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

for fname in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    prop = fm.FontProperties(fname=fname)
    if 'SimHei' in prop.get_name() or 'Microsoft YaHei' in prop.get_name():
        plt.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 加载已保存的 K=1000 PCA 清理图像（复用 diagonal_enhancement 的中间输出）
# 这里直接用 numpy 从文件读取 — 但文件没保存，所以需要重新计算
# 直接运行：python diagonal_enhancement.py 后，读取残余矩阵太复杂
# 所以直接重跑对角增强逻辑的最简版

import re, sys, os, argparse
from datetime import datetime
from collections import defaultdict, OrderedDict
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', default='20260630')
    parser.add_argument('--pca-k', type=int, default=1000)
    parser.add_argument('--output', default='integrated_images')
    args = parser.parse_args()

    watch_dir = 'correlation_results'; n_ch = 4096
    K = args.pca_k

    print("扫描文件...")
    all_data = discover_frames(watch_dir, args.date)
    frames_by_ts = all_data[args.date]
    timestamps = list(frames_by_ts.keys())
    n_frames = len(timestamps)
    print(f"  {n_frames} 帧")

    print("加载数据...")
    all_vis, all_auto3, all_auto8, valid_idx = [], [], [], []
    for fi, ts in enumerate(timestamps):
        fm = frames_by_ts[ts]
        vis = read_cross(fm, n_ch); a3 = read_auto(fm, 3, n_ch); a8 = read_auto(fm, 8, n_ch)
        if vis is not None and a3 is not None and a8 is not None:
            all_vis.append(vis); all_auto3.append(a3); all_auto8.append(a8); valid_idx.append(fi)
    n_valid = len(all_vis)
    vis_matrix = np.array(all_vis); auto3_s = np.array(all_auto3); auto8_s = np.array(all_auto8)
    print(f"  有效: {n_valid}")

    # RFI
    freq_rfi = np.zeros((n_valid, n_ch), dtype=bool)
    for fi in range(n_valid):
        freq_rfi[fi] = detect_freq_rfi(auto3_s[fi], auto8_s[fi])
    time_rfi = np.zeros(n_valid, dtype=bool)
    frame_powers = [np.median(np.maximum(auto3_s[fi], auto8_s[fi])) for fi in range(n_valid)]
    bad = detect_time_rfi(frame_powers)
    for bi in bad: time_rfi[bi] = True
    print(f"  RFI: freq={freq_rfi.sum():,}, time={time_rfi.sum()}")

    # Bandpass
    bp3 = np.median(auto3_s[~time_rfi], axis=0) if (~time_rfi).any() else np.median(auto3_s, axis=0)
    bp8 = np.median(auto8_s[~time_rfi], axis=0) if (~time_rfi).any() else np.median(auto8_s, axis=0)
    bp3 = clean_bandpass_spikes(bp3); bp8 = clean_bandpass_spikes(bp8)
    expected = np.sqrt(np.maximum(bp3, 1e-30) * np.maximum(bp8, 1e-30))
    bp_factor = np.median(expected) / np.maximum(expected, 1e-30)
    vis_matrix = vis_matrix * bp_factor[np.newaxis, :]

    combined_mask = freq_rfi.copy(); combined_mask[time_rfi] = True
    vis_real = vis_matrix.real
    vis_filled = inpaint_masked(vis_real, combined_mask)

    # ── 关键步骤：行列双去均值 ──
    # 1. 列去均值 (每频率通道去时间均值) — 消除水平带
    vis_centered = vis_filled - vis_filled.mean(axis=0, keepdims=True)
    # 2. 行去均值 (每帧去频率均值) — 消除垂直增益变化
    vis_centered = vis_centered - vis_centered.mean(axis=1, keepdims=True)

    # PCA
    print(f"\nSVD (K={K})...")
    U, S, Vh = np.linalg.svd(vis_centered, full_matrices=False)
    residual = np.zeros_like(vis_centered)
    if K < min(n_valid, n_ch):
        for k in range(K, len(S)):
            residual += S[k] * np.outer(U[:, k], Vh[k, :])
    else:
        residual = vis_centered.copy()
    # 恢复均值
    residual = residual + vis_filled.mean(axis=0, keepdims=True) + vis_filled.mean(axis=1, keepdims=True) - vis_filled.mean()

    # 重排频率
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, 150.0)
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

    # 2D FFT (双去均值后)
    print("2D FFT...")
    win_t = np.hanning(n_valid); win_f = np.hanning(n_ch)
    win2d = win_t[:, None] * win_f[None, :]
    data_windowed = residual_r * win2d
    fft2d = np.fft.fft2(data_windowed)
    fft2d_shifted = np.fft.fftshift(fft2d)
    fft_mag = np.abs(fft2d_shifted)
    fft_log = np.log10(np.maximum(fft_mag, 1e-30))
    omega_t = np.fft.fftshift(np.fft.fftfreq(n_valid))
    omega_f = np.fft.fftshift(np.fft.fftfreq(n_ch))

    # 聚焦图：原图 + 2D FFT + 2D FFT 中心放大
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.12, wspace=0.08)

    # 1. 原图（行列双去均值 + PCA）
    ax1 = fig.add_subplot(gs[0, 0])
    vabs = max(abs(residual_r.min()), abs(residual_r.max()))
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    ax1.pcolormesh(freq_disp, times, residual_r, cmap='RdBu_r', norm=norm, shading='nearest', rasterized=True)
    ax1.set_title(f'行列双去均值 + PCA(K={K})', fontsize=10, fontweight='bold')
    ax1.set_ylabel('时间 [分钟]', fontsize=9)
    ax1.set_xlim(freq_disp.min(), freq_disp.max()); ax1.set_ylim(times[0], times[-1])
    ax1.ticklabel_format(useOffset=False, style='plain')

    # 2. 2D FFT 全图
    ax2 = fig.add_subplot(gs[0, 1])
    fft_vmin = np.percentile(fft_log, 2); fft_vmax = np.percentile(fft_log, 98)
    ax2.imshow(fft_log, aspect='auto', origin='lower', cmap='viridis', vmin=fft_vmin, vmax=fft_vmax,
               extent=[omega_f.min(), omega_f.max(), omega_t.min(), omega_t.max()])
    ax2.set_title('2D FFT 对数幅度', fontsize=10, fontweight='bold')
    ax2.set_xlabel('ω_f [cycles/ch]', fontsize=9); ax2.set_ylabel('ω_t [cycles/frame]', fontsize=9)
    ax2.axhline(0, color='r', alpha=0.3, linewidth=0.5); ax2.axvline(0, color='r', alpha=0.3, linewidth=0.5)
    ax2.grid(True, alpha=0.2, linewidth=0.3)

    # 3. 2D FFT 中心放大（±0.05 cycles）
    ax3 = fig.add_subplot(gs[0, 2])
    # 找到 ±0.05 范围内的索引
    f_idx = np.where((omega_f >= -0.05) & (omega_f <= 0.05))[0]
    t_idx = np.where((omega_t >= -0.05) & (omega_t <= 0.05))[0]
    if len(f_idx) > 0 and len(t_idx) > 0:
        zoom_fft = fft_log[t_idx[0]:t_idx[-1]+1, f_idx[0]:f_idx[-1]+1]
        ax3.imshow(zoom_fft, aspect='auto', origin='lower', cmap='viridis', vmin=fft_vmin, vmax=fft_vmax,
                   extent=[omega_f[f_idx[0]], omega_f[f_idx[-1]], omega_t[t_idx[0]], omega_t[t_idx[-1]]])
        ax3.set_title('2D FFT 中心放大 (±0.05)', fontsize=10, fontweight='bold')
        ax3.set_xlabel('ω_f [cycles/ch]', fontsize=9); ax3.set_ylabel('ω_t [cycles/frame]', fontsize=9)
        ax3.axhline(0, color='r', alpha=0.3, linewidth=0.5); ax3.axvline(0, color='r', alpha=0.3, linewidth=0.5)
        ax3.grid(True, alpha=0.2, linewidth=0.3)
    else:
        ax3.text(0.5, 0.5, 'No zoom', ha='center', va='center', transform=ax3.transAxes)

    # 4. 2D FFT 功率分布（环形 vs 轴向）
    ax4 = fig.add_subplot(gs[1, 0])
    Nt, Nf = fft_mag.shape
    cy, cx = Nt // 2, Nf // 2
    yy, xx = np.mgrid[:Nt, :Nf]
    yy_c = yy - cy; xx_c = xx - cx
    r = np.sqrt((yy_c / (Nt/2))**2 + (xx_c / (Nf/2))**2)
    angle = np.degrees(np.arctan2(yy_c, xx_c))
    # 轴向掩码：|angle| < 15 或 |angle - 90| < 15 或 |angle - 180| < 15 等
    axial = (np.abs(angle) < 15) | (np.abs(angle - 90) < 15) | (np.abs(angle - 180) < 15) | (np.abs(angle + 90) < 15) | (np.abs(angle - 270) < 15) | (np.abs(angle + 180) < 15)
    # 对角掩码：45±15 或 135±15
    diagonal = (np.abs(angle - 45) < 15) | (np.abs(angle - 135) < 15) | (np.abs(angle + 45) < 15) | (np.abs(angle + 135) < 15) | (np.abs(angle - 225) < 15) | (np.abs(angle - 315) < 15)
    # 排除 DC
    dc = r < 0.02
    axial = axial & (~dc); diagonal = diagonal & (~dc)
    other = (~axial) & (~diagonal) & (~dc)
    total_power = np.sum(fft_mag**2)
    axial_power = np.sum(fft_mag[axial]**2)
    diag_power = np.sum(fft_mag[diagonal]**2)
    other_power = np.sum(fft_mag[other]**2)
    ax4.bar(['横纵', '对角', '其他'], [axial_power, diag_power, other_power],
            color=['blue', 'red', 'gray'], alpha=0.6)
    ax4.set_title('2D FFT 功率分布 (排除DC)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('功率', fontsize=9)
    for i, p in enumerate([axial_power, diag_power, other_power]):
        pct = 100 * p / total_power
        ax4.text(i, p * 1.02, f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # 5. 沿 ω_t 的切片（聚焦 ±0.05）
    ax5 = fig.add_subplot(gs[1, 1])
    if len(f_idx) > 0:
        center_slice = fft_mag[:, f_idx[0]:f_idx[-1]+1].mean(axis=1)
        ax5.semilogy(omega_t, center_slice, 'b-', alpha=0.6, linewidth=0.8, label='ω_f≈0 附近')
        # 也画对角切片
        diag_slice_vals = []
        diag_t = []
        for i in range(Nt):
            j = cx + int((i - cy) * Nf / Nt)  # 45° 对角线
            if 0 <= j < Nf and abs(omega_t[i]) <= 0.05:
                diag_slice_vals.append(fft_mag[i, j])
                diag_t.append(omega_t[i])
        if diag_slice_vals:
            ax5.semilogy(diag_t, diag_slice_vals, 'r-', alpha=0.8, linewidth=1, label='对角线 (45°)')
        ax5.set_title('沿 ω_t 的切片 (±0.05)', fontsize=10, fontweight='bold')
        ax5.set_xlabel('ω_t [cycles/frame]', fontsize=9); ax5.set_ylabel('幅度', fontsize=9)
        ax5.legend(fontsize=8); ax5.grid(True, alpha=0.2); ax5.set_xlim(-0.05, 0.05)
    else:
        ax5.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax5.transAxes)

    # 6. 沿 ω_f 的切片（聚焦 ±0.05）
    ax6 = fig.add_subplot(gs[1, 2])
    if len(t_idx) > 0:
        center_slice_t = fft_mag[t_idx[0]:t_idx[-1]+1, :].mean(axis=0)
        ax6.semilogy(omega_f, center_slice_t, 'b-', alpha=0.6, linewidth=0.8, label='ω_t≈0 附近')
        diag_slice_vals_f = []
        diag_f = []
        for j in range(Nf):
            i = cy + int((j - cx) * Nt / Nf)
            if 0 <= i < Nt and abs(omega_f[j]) <= 0.05:
                diag_slice_vals_f.append(fft_mag[i, j])
                diag_f.append(omega_f[j])
        if diag_slice_vals_f:
            ax6.semilogy(diag_f, diag_slice_vals_f, 'r-', alpha=0.8, linewidth=1, label='对角线 (45°)')
        ax6.set_title('沿 ω_f 的切片 (±0.05)', fontsize=10, fontweight='bold')
        ax6.set_xlabel('ω_f [cycles/ch]', fontsize=9); ax6.set_ylabel('幅度', fontsize=9)
        ax6.legend(fontsize=8); ax6.grid(True, alpha=0.2); ax6.set_xlim(-0.05, 0.05)
    else:
        ax6.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax6.transAxes)

    fig.suptitle(f'CH3×CH8 行列双去均值 + 2D FFT 聚焦 — {args.date} (K={K})', fontsize=13, fontweight='bold', y=0.98)
    os.makedirs(args.output, exist_ok=True)
    out_path = f'{args.output}/double_demean_fft_K{K}_{args.date}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n保存: {out_path}")
    plt.close(fig)

    print(f"\n{'='*60}")
    print(f"汇总:")
    print(f"  总功率: {total_power:.2e}")
    print(f"  横纵功率: {axial_power:.2e} ({100*axial_power/total_power:.1f}%)")
    print(f"  对角功率: {diag_power:.2e} ({100*diag_power/total_power:.1f}%)")
    print(f"  其他功率: {other_power:.2e} ({100*other_power/total_power:.1f}%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
