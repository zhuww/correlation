"""
完整清理管线：时间中位数相减 + PCA 分解 → 提取天体 fringe。

管线：
  1. 加载数据，RFI 掩码，bandpass 校正
  2. 对每个频率通道，减去复数时间中位数（去除持久窄带 RFI 竖线）
  3. 掩码区域插值填充
  4. SVD/PCA 分解，移除前 K 个主成分（去除剩余可分离横纵结构）
  5. 残余 = 不可分离结构 = 候选天体 fringe
  6. 可视化：原始/中位数移除/PCA移除/PCA残余 + 2D FFT 对角分析
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm
from matplotlib import patches
import time, os, sys, re, argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
from datetime import datetime
import pandas as pd

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


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════
def discover_frames(watch_dir, date_str=None):
    csv_files = sorted(Path(watch_dir).glob("correlation_*.csv"))
    if not csv_files:
        print("错误: 未找到 CSV 文件"); sys.exit(1)
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


def read_sky_frequencies(file_map, n_channels=4096, center_freq_mhz=150.0):
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=n_channels)
    freq_hz = df['frequency_hz'].values.astype(np.float64)
    return center_freq_mhz + freq_hz / 1e6


# ═══════════════════════════════════════════════════════════════════
# RFI / bandpass
# ═══════════════════════════════════════════════════════════════════
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
                lo, hi = max(0, i - grow), min(n_ch, i + grow + 1)
                grown[lo:hi] = True
        flagged = grown
    return flagged


def detect_time_rfi(frame_powers, td_threshold=5.0, td_window=101):
    powers = np.array(frame_powers); n = len(powers)
    if n <= td_window: return set()
    half = td_window // 2
    smoothed = np.array([np.median(powers[max(0, i - half):min(n, i + half + 1)]) for i in range(n)])
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
                lo, hi = max(0, i - grow), min(n, i + grow + 1)
                grown[lo:hi] = True
        flagged = grown
    bp_clean = bp.copy(); bp_clean[flagged] = baseline[flagged]
    return bp_clean


def inpaint_masked(data, mask, max_gap=50):
    """对实数矩阵做双线性插值填充掩码区域。"""
    filled = data.astype(np.float64).copy()
    filled[mask] = np.nan
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
            if ge - gs <= max_gap:
                row[gs:ge] = np.interp(np.arange(gs, ge), valid_idx, valid_vals)
            else:
                row[gs:ge] = 0.0
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
            if ge - gs <= max_gap:
                col[gs:ge] = np.interp(np.arange(gs, ge), valid_idx, valid_vals)
            else:
                col[gs:ge] = 0.0
        col[np.isnan(col)] = 0.0
    return filled


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='时间中位数相减 + PCA → 提取天体 fringe')
    parser.add_argument('--date', default='20260630')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--n-channels', type=int, default=4096)
    parser.add_argument('--center-freq', type=float, default=150.0)
    parser.add_argument('--n-components', type=int, default=200,
                        help='移除的前 N 个 PCA 成分 (default 200)')
    parser.add_argument('--output', default='integrated_images')
    args = parser.parse_args()

    watch_dir = 'correlation_results'; n_ch = args.n_channels; K = args.n_components

    # ═══════════════════════════════════════════════════════════════
    # 第 1 步: 加载数据
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  CH3×CH8: 中位数相减 + PCA 管线")
    print("=" * 70)

    print(f"\n[1/7] 加载数据...")
    all_data = discover_frames(watch_dir, args.date)
    frames_by_ts = all_data[args.date]
    timestamps = list(frames_by_ts.keys())
    if args.max_frames > 0:
        timestamps = timestamps[:args.max_frames]
    n_frames = len(timestamps)
    print(f"  {n_frames} 帧")

    t0 = time.time()
    all_vis, all_auto3, all_auto8, valid_idx = [], [], [], []
    for fi, ts in enumerate(timestamps):
        fm = frames_by_ts[ts]
        vis = read_cross(fm, n_ch)
        a3 = read_auto(fm, 3, n_ch)
        a8 = read_auto(fm, 8, n_ch)
        if vis is not None and a3 is not None and a8 is not None:
            all_vis.append(vis); all_auto3.append(a3); all_auto8.append(a8)
            valid_idx.append(fi)
    n_valid = len(all_vis)
    vis_matrix = np.array(all_vis)       # complex (n_v, n_ch)
    auto3_s = np.array(all_auto3); auto8_s = np.array(all_auto8)
    del all_vis, all_auto3, all_auto8
    print(f"  有效: {n_valid}/{n_frames}")

    # ═══════════════════════════════════════════════════════════════
    # 第 2 步: RFI 掩码 + bandpass
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[2/7] RFI 掩码 + bandpass 校正...")
    freq_rfi = np.zeros((n_valid, n_ch), dtype=bool)
    for fi in range(n_valid):
        freq_rfi[fi] = detect_freq_rfi(auto3_s[fi], auto8_s[fi])

    time_rfi = np.zeros(n_valid, dtype=bool)
    frame_powers = [np.median(np.maximum(auto3_s[fi], auto8_s[fi])) for fi in range(n_valid)]
    for bi in detect_time_rfi(frame_powers): time_rfi[bi] = True

    # bandpass
    good_mask = ~time_rfi
    bp3 = np.median(auto3_s[good_mask], axis=0) if good_mask.any() else np.median(auto3_s, axis=0)
    bp8 = np.median(auto8_s[good_mask], axis=0) if good_mask.any() else np.median(auto8_s, axis=0)
    bp3_c = clean_bandpass_spikes(bp3); bp8_c = clean_bandpass_spikes(bp8)
    expected = np.sqrt(np.maximum(bp3_c, 1e-30) * np.maximum(bp8_c, 1e-30))
    bp_factor = np.median(expected) / np.maximum(expected, 1e-30)
    vis_corr = vis_matrix * bp_factor[np.newaxis, :]  # bandpass 校正后

    combined_mask = freq_rfi.copy()
    combined_mask[time_rfi] = True
    print(f"  RFI: freq={freq_rfi.sum():,}, time={time_rfi.sum()}, combined={combined_mask.sum():,} "
          f"({100*combined_mask.sum()/(n_valid*n_ch):.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # 第 3 步: 时间中位数相减 (复数)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[3/7] 时间中位数相减...")
    complex_median = np.zeros(n_ch, dtype=np.complex128)
    for j in range(n_ch):
        vals = vis_corr[~time_rfi, j] if (~time_rfi).any() else vis_corr[:, j]
        complex_median[j] = np.median(vals.real) + 1j * np.median(vals.imag)

    vis_mediansub = vis_corr - complex_median[np.newaxis, :]

    power_before = np.mean(np.abs(vis_corr)**2)
    med_power = np.mean(np.abs(complex_median)**2)
    power_after_med = np.mean(np.abs(vis_mediansub)**2)
    print(f"  中位数频谱功率: {med_power:.2e} ({100*med_power/power_before:.1f}% 总功率)")
    print(f"  功率: {power_before:.2e} → {power_after_med:.2e} "
          f"(降低 {100*(1-power_after_med/power_before):.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # 第 4 步: 插值填充掩码区域 (PCA 需要完整矩阵)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[4/7] 掩码插值...")
    # 用实部做 PCA（fringe 在实部可见）
    vis_real = vis_mediansub.real
    vis_real_filled = inpaint_masked(vis_real, combined_mask)
    vis_real_filled = np.nan_to_num(vis_real_filled, nan=0.0)

    # 虚部同理
    vis_imag = vis_mediansub.imag
    vis_imag_filled = inpaint_masked(vis_imag, combined_mask)
    vis_imag_filled = np.nan_to_num(vis_imag_filled, nan=0.0)

    # 谱信息
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, args.center_freq)
    t0_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
    times = np.array([(datetime.strptime(timestamps[idx], "%Y%m%d_%H%M%S") - t0_dt).total_seconds() / 60
                      for idx in valid_idx])

    # ═══════════════════════════════════════════════════════════════
    # 第 5 步: PCA 分解 (对实部)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[5/7] SVD 分解 ({n_valid}×{n_ch})...")
    # 先做行+列双去均值，把横纵公共偏移去掉后再 PCA
    vis_centered = vis_real_filled - vis_real_filled.mean(axis=0, keepdims=True)
    vis_centered = vis_centered - vis_centered.mean(axis=1, keepdims=True)

    t_svd = time.time()
    U, S, Vh = np.linalg.svd(vis_centered, full_matrices=False)
    print(f"  SVD 完成 ({time.time()-t_svd:.1f}s), 条件数 {S[0]/S[-1]:.0f}")

    var_total = np.sum(S**2)
    var_cumsum = np.cumsum(S**2) / var_total

    # 移除前 K 个成分
    if K < min(n_valid, n_ch):
        residual = np.zeros_like(vis_centered)
        for k in range(K, len(S)):
            residual += S[k] * np.outer(U[:, k], Vh[k, :])
        pca_removed = vis_centered - residual
    else:
        residual = vis_centered.copy()
        pca_removed = np.zeros_like(vis_centered)

    removed_var = 100 * (1 - np.sum(S[K:]**2) / var_total)
    kept_var = 100 * np.sum(S[K:]**2) / var_total
    print(f"  移除成分: {K}, 移除方差: {removed_var:.1f}%, 保留: {kept_var:.1f}%")

    # 恢复行+列均值
    residual_restored = (residual +
                         vis_real_filled.mean(axis=0, keepdims=True) +
                         vis_real_filled.mean(axis=1, keepdims=True) -
                         vis_real_filled.mean())

    # ═══════════════════════════════════════════════════════════════
    # 第 6 步: 2D FFT 对角分析
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[6/7] 2D FFT 对角分析...")
    # 对 PCA 残余做 2D FFT
    win_t = np.hanning(n_valid); win_f = np.hanning(n_ch)
    win2d = win_t[:, None] * win_f[None, :]
    residual_windowed = residual * win2d  # 使用去均值版本
    fft2d = np.fft.fft2(residual_windowed)
    fft2d_shifted = np.fft.fftshift(fft2d)
    fft_mag = np.abs(fft2d_shifted)
    fft_log = np.log10(np.maximum(fft_mag, 1e-30))
    omega_t = np.fft.fftshift(np.fft.fftfreq(n_valid))
    omega_f = np.fft.fftshift(np.fft.fftfreq(n_ch))

    Nt, Nf = fft_mag.shape
    cy, cx = Nt // 2, Nf // 2
    yy, xx = np.mgrid[:Nt, :Nf]
    yy_c, xx_c = yy - cy, xx - cx
    r = np.sqrt((yy_c / (Nt / 2))**2 + (xx_c / (Nf / 2))**2)
    angle = np.degrees(np.arctan2(yy_c, xx_c))

    axial_mask = ((np.abs(angle) < 15) | (np.abs(angle - 90) < 15) |
                  (np.abs(angle - 180) < 15) | (np.abs(angle + 90) < 15) |
                  (np.abs(angle - 270) < 15) | (np.abs(angle + 180) < 15))
    diag_mask = ((np.abs(angle - 45) < 15) | (np.abs(angle - 135) < 15) |
                 (np.abs(angle + 45) < 15) | (np.abs(angle + 135) < 15) |
                 (np.abs(angle - 225) < 15) | (np.abs(angle - 315) < 15))
    dc_mask = r < 0.02
    axial_mask = axial_mask & (~dc_mask)
    diag_mask = diag_mask & (~dc_mask)
    other_mask = (~axial_mask) & (~diag_mask) & (~dc_mask)

    total_power = np.sum(fft_mag**2)
    axial_power = np.sum(fft_mag[axial_mask]**2)
    diag_power = np.sum(fft_mag[diag_mask]**2)
    other_power = np.sum(fft_mag[other_mask]**2)

    # 沿对角 45° 切片
    diag_vals = []; diag_omega = []
    for i in range(Nt):
        j = cx + int((i - cy) * Nf / Nt)
        if 0 <= j < Nf and abs(omega_t[i]) <= 0.05:
            diag_vals.append(fft_mag[i, j])
            diag_omega.append(omega_t[i])
    diag_vals = np.array(diag_vals)
    diag_omega = np.array(diag_omega)

    # 在 2D FFT 中找显著的对角峰
    diag_region = fft_mag[diag_mask]
    if len(diag_region) > 0:
        diag_rms = np.sqrt(np.mean(diag_region**2))
        diag_peak_idx = np.argmax(diag_region)
        diag_peak_val = diag_region[diag_peak_idx]
    else:
        diag_rms = 0; diag_peak_val = 0

    print(f"  2D FFT 功率分布:")
    print(f"    横纵: {axial_power:.2e} ({100*axial_power/total_power:.1f}%)")
    print(f"    对角: {diag_power:.2e} ({100*diag_power/total_power:.1f}%)")
    print(f"    其他: {other_power:.2e} ({100*other_power/total_power:.1f}%)")
    if diag_peak_val > 0:
        print(f"    对角峰值: {diag_peak_val:.2e}, RMS: {diag_rms:.2e}")

    # ═══════════════════════════════════════════════════════════════
    # 第 7 步: 频率重排 + 绘图
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[7/7] 绘图...")
    Nc = n_ch
    if Nc % 2 == 0:
        reorder = np.concatenate([np.arange(Nc // 2 + 1, Nc), np.arange(0, Nc // 2 + 1)]).astype(int)
    else:
        reorder = np.arange(Nc)
    freq_disp = freq_raw[reorder]

    def reorder2d(m):
        return m[:, reorder]

    def reorder1d(m):
        return m[reorder]

    # 各阶段数据
    vis_orig_r = reorder2d(vis_real)           # 中位数相减后 (实部)
    vis_centered_r = reorder2d(vis_centered)    # 双去均值后
    pca_removed_r = reorder2d(pca_removed)      # PCA 移除的结构
    residual_r = reorder2d(residual_restored)   # PCA 残余 (恢复均值)
    combined_mask_r = reorder2d(combined_mask)

    # 色标
    vabs_orig = max(abs(vis_orig_r[~combined_mask_r].min()) if (~combined_mask_r).any() else 1,
                    abs(vis_orig_r[~combined_mask_r].max()) if (~combined_mask_r).any() else 1)
    vabs_res = max(abs(residual_r[~combined_mask_r].min()) if (~combined_mask_r).any() else 1,
                   abs(residual_r[~combined_mask_r].max()) if (~combined_mask_r).any() else 1)

    # ═══════════════════════════════════════════════════════════════
    # 大图: 8 个子图
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(26, 20))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.5, 1, 1, 1],
                          width_ratios=[1, 0.03, 1, 0.03],
                          hspace=0.25, wspace=0.04)

    # (0,0) 被移除的中位数频谱
    ax_medspec = fig.add_subplot(gs[0, :2])
    cm_r = reorder1d(complex_median)
    ax_medspec.plot(freq_disp, cm_r.real, 'b-', linewidth=0.5, alpha=0.8, label='中位数 实部')
    ax_medspec.plot(freq_disp, cm_r.imag, 'r-', linewidth=0.5, alpha=0.8, label='中位数 虚部')
    ax_medspec.axhline(0, color='gray', alpha=0.3)
    ax_medspec.set_title(f'第1步: 被减去的复数时间中位数频谱\n'
                         f'(持久的窄带 RFI + 静态 bandpass 残余), 功率占比 {100*med_power/power_before:.1f}%',
                         fontsize=9, fontweight='bold')
    ax_medspec.legend(fontsize=8, loc='upper right')
    ax_medspec.set_xlim(freq_disp.min(), freq_disp.max())
    ax_medspec.grid(True, alpha=0.2)

    # (0,2) 奇异值谱
    ax_spec = fig.add_subplot(gs[0, 2:])
    n_sv = min(300, len(S))
    ax_spec.semilogy(np.arange(1, n_sv + 1), S[:n_sv], 'b.-', markersize=2)
    ax_spec.axvline(K, color='red', linestyle='--', alpha=0.7,
                    label=f'K={K}')
    ax_spec.axvspan(1, K, alpha=0.1, color='red')
    ax_spec.set_xlabel('成分编号', fontsize=9)
    ax_spec.set_ylabel('σ_k', fontsize=9)
    ax_spec.set_title(f'奇异值谱 — 前{K}成分={var_cumsum[K-1]*100:.1f}% 累积方差',
                      fontsize=9, fontweight='bold')
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3)

    # (1,0) 中位数相减后 (原始)
    ax_orig = fig.add_subplot(gs[1, 0])
    norm_o = TwoSlopeNorm(vmin=-vabs_orig, vcenter=0, vmax=vabs_orig)
    im1 = ax_orig.pcolormesh(freq_disp, times, vis_orig_r, cmap='RdBu_r',
                              norm=norm_o, shading='nearest', rasterized=True)
    ax_orig.set_title(f'中位数相减后\n(持久 RFI 已移除)', fontsize=10, fontweight='bold')
    ax_orig.set_ylabel('时间 [分钟]', fontsize=10)
    ax_orig.set_xlim(freq_disp.min(), freq_disp.max());
    ax_orig.set_ylim(times[0], times[-1])
    cax1 = fig.add_subplot(gs[1, 1]); plt.colorbar(im1, cax=cax1, label='Re(V)')

    # (1,2) PCA 移除的横纵结构
    ax_pca = fig.add_subplot(gs[1, 2])
    pca_vabs = max(abs(pca_removed_r.min()), abs(pca_removed_r.max()))
    norm_p = TwoSlopeNorm(vmin=-pca_vabs, vcenter=0, vmax=pca_vabs)
    im2 = ax_pca.pcolormesh(freq_disp, times, pca_removed_r, cmap='RdBu_r',
                             norm=norm_p, shading='nearest', rasterized=True)
    ax_pca.set_title(f'PCA 移除的横纵结构\n{K} 成分, {removed_var:.1f}% 方差',
                     fontsize=10, fontweight='bold')
    ax_pca.set_xlim(freq_disp.min(), freq_disp.max());
    ax_pca.set_ylim(times[0], times[-1])
    cax2 = fig.add_subplot(gs[1, 3]); plt.colorbar(im2, cax=cax2, label='Re(V)')

    # (2,0) PCA 残余 (候选 fringe)
    ax_res = fig.add_subplot(gs[2, 0])
    norm_r = TwoSlopeNorm(vmin=-vabs_res, vcenter=0, vmax=vabs_res)
    im3 = ax_res.pcolormesh(freq_disp, times, residual_r, cmap='RdBu_r',
                             norm=norm_r, shading='nearest', rasterized=True)
    ax_res.set_title(f'PCA 残余 — 候选天体 fringe\n'
                     f'保留方差 {kept_var:.1f}%, RMS={np.sqrt(np.mean(residual_r**2)):.1e}',
                     fontsize=10, fontweight='bold')
    ax_res.set_ylabel('时间 [分钟]', fontsize=10)
    ax_res.set_xlim(freq_disp.min(), freq_disp.max());
    ax_res.set_ylim(times[0], times[-1])
    cax3 = fig.add_subplot(gs[2, 1]); plt.colorbar(im3, cax=cax3, label='残余 Re(V)')

    # (2,2) 2D FFT 对数幅度
    ax_fft = fig.add_subplot(gs[2, 2])
    fft_vmin = np.percentile(fft_log, 2);
    fft_vmax = np.percentile(fft_log, 98)
    ax_fft.imshow(fft_log, aspect='auto', origin='lower', cmap='viridis',
                  vmin=fft_vmin, vmax=fft_vmax,
                  extent=[omega_f.min(), omega_f.max(), omega_t.min(), omega_t.max()])
    ax_fft.set_title('2D FFT 对数幅度\n(对角=斜向 fringe)', fontsize=10, fontweight='bold')
    ax_fft.set_xlabel('ω_f [cycles/ch]', fontsize=9)
    ax_fft.set_ylabel('ω_t [cycles/frame]', fontsize=9)
    ax_fft.axhline(0, color='r', alpha=0.3, lw=0.5);
    ax_fft.axvline(0, color='r', alpha=0.3, lw=0.5)

    # 对角方向标注
    xlim = ax_fft.get_xlim(); ylim = ax_fft.get_ylim()
    ax_fft.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]], 'r--', alpha=0.3, lw=0.8)
    ax_fft.plot([xlim[0], xlim[1]], [ylim[1], ylim[0]], 'r--', alpha=0.3, lw=0.8)

    # (3,0) 2D FFT 中心放大
    ax_zoom = fig.add_subplot(gs[3, 0])
    f_idx = np.where((omega_f >= -0.05) & (omega_f <= 0.05))[0]
    t_idx = np.where((omega_t >= -0.05) & (omega_t <= 0.05))[0]
    if len(f_idx) > 0 and len(t_idx) > 0:
        zoom_data = fft_log[t_idx[0]:t_idx[-1] + 1, f_idx[0]:f_idx[-1] + 1]
        ax_zoom.imshow(zoom_data, aspect='auto', origin='lower', cmap='viridis',
                       vmin=fft_vmin, vmax=fft_vmax,
                       extent=[omega_f[f_idx[0]], omega_f[f_idx[-1]],
                               omega_t[t_idx[0]], omega_t[t_idx[-1]]])
        ax_zoom.set_title('2D FFT 中心放大 (±0.05)', fontsize=10, fontweight='bold')
        ax_zoom.set_xlabel('ω_f [cycles/ch]', fontsize=9)
        ax_zoom.set_ylabel('ω_t [cycles/frame]', fontsize=9)
        ax_zoom.axhline(0, color='r', alpha=0.3, lw=0.5);
        ax_zoom.axvline(0, color='r', alpha=0.3, lw=0.5)
        ax_zoom.grid(True, alpha=0.2)

    # (3,1) 功率分布条形图
    ax_pwr = fig.add_subplot(gs[3, 1])
    colors = ['#4472C4', '#ED7D31', '#A5A5A5']
    bars = ax_pwr.bar(['横纵', '对角', '其他'],
                      [axial_power, diag_power, other_power],
                      color=colors, alpha=0.7, width=0.5)
    for i, (p, label) in enumerate([(axial_power, '横纵'), (diag_power, '对角'), (other_power, '其他')]):
        ax_pwr.text(i, p * 1.03, f'{100*p/total_power:.1f}%',
                    ha='center', fontsize=10, fontweight='bold')
    ax_pwr.set_title('2D FFT 功率分布', fontsize=10, fontweight='bold')
    ax_pwr.set_ylabel('功率', fontsize=9)

    # (3,2) 对角切片 vs 轴向切片
    ax_slice = fig.add_subplot(gs[3, 2])
    if len(f_idx) > 0:
        axial_slice = fft_mag[:, f_idx[0]:f_idx[-1] + 1].mean(axis=1)
        ax_slice.semilogy(omega_t, axial_slice, 'b-', alpha=0.5, lw=0.7, label='ω_f≈0 (轴向)')
    if len(diag_omega) > 0:
        ax_slice.semilogy(diag_omega, diag_vals, 'r-', alpha=0.9, lw=1.2, label='对角 45°')
    # 标注可能峰值
    if len(diag_omega) > 0:
        peak_i = np.argmax(np.abs(diag_vals))
        ax_slice.axvline(diag_omega[peak_i], color='orange', linestyle=':', alpha=0.7,
                         label=f'对角峰 ω_t≈{diag_omega[peak_i]:.4f}')
    ax_slice.set_title('沿 ω_t 切片 (±0.05) — 红色=对角方向', fontsize=10, fontweight='bold')
    ax_slice.set_xlabel('ω_t [cycles/frame]', fontsize=9)
    ax_slice.set_ylabel('幅度', fontsize=9)
    ax_slice.legend(fontsize=7)
    ax_slice.grid(True, alpha=0.2)
    ax_slice.set_xlim(-0.05, 0.05)

    fig.suptitle(f'CH3×CH8 — 中位数相减 + PCA(K={K}) → 天体 fringe 检测 — {args.date}',
                 fontsize=13, fontweight='bold', y=0.995)

    os.makedirs(args.output, exist_ok=True)
    out_main = f'{args.output}/median_pca_fringe_K{K}_{args.date}.png'
    fig.savefig(out_main, dpi=200, bbox_inches='tight')
    print(f"\n  保存: {out_main}")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════
    # 额外图: 聚焦低频区 + 2D FFT 残余
    # ═══════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 3, figsize=(22, 14), constrained_layout=True)

    for col, (region, flo, fhi) in enumerate([
            ('100-130 MHz', 100, 130),
            ('130-160 MHz', 130, 160),
            ('160-200 MHz', 160, 200)]):
        fmask = (freq_disp >= flo) & (freq_disp <= fhi)
        if fmask.sum() < 5: continue

        zoom_data = residual_r[:, fmask]
        zabs = max(abs(zoom_data.min()), abs(zoom_data.max()))
        norm_z = TwoSlopeNorm(vmin=-zabs, vcenter=0, vmax=zabs)
        ax_z = axes2[0, col]
        ax_z.pcolormesh(freq_disp[fmask], times, zoom_data,
                        cmap='RdBu_r', norm=norm_z,
                        shading='nearest', rasterized=True)
        ax_z.set_title(f'PCA残余 {region}\nRMS={np.sqrt(np.mean(zoom_data**2)):.1e}',
                       fontsize=9, fontweight='bold')
        ax_z.set_xlabel('频率 [MHz]', fontsize=8)
        if col == 0: ax_z.set_ylabel('时间 [分钟]', fontsize=9)
        ax_z.set_xlim(flo, fhi); ax_z.set_ylim(times[0], times[-1])

    # 2D FFT 残余在不同频率区间
    for col, (region, flo, fhi) in enumerate([
            ('100-140', 100, 140),
            ('140-170', 140, 170),
            ('170-200', 170, 200)]):
        fmask = (freq_raw >= flo) & (freq_raw <= fhi)  # 用原始顺序
        if fmask.sum() < 10: continue
        sub_data = residual[:, fmask]  # 去均值版
        win_t2 = np.hanning(n_valid); win_f2 = np.hanning(fmask.sum())
        sub_win = sub_data * (win_t2[:, None] * win_f2[None, :])
        sub_fft = np.fft.fftshift(np.fft.fft2(sub_win))
        sub_mag = np.abs(sub_fft)

        Nt2, Nf2 = sub_mag.shape
        cy2, cx2 = Nt2 // 2, Nf2 // 2
        yy2, xx2 = np.mgrid[:Nt2, :Nf2]
        yy2_c, xx2_c = yy2 - cy2, xx2 - cx2
        ang2 = np.degrees(np.arctan2(yy2_c, xx2_c))
        r2 = np.sqrt((yy2_c / (Nt2 / 2))**2 + (xx2_c / (Nf2 / 2))**2)
        diag2 = ((np.abs(ang2 - 45) < 15) | (np.abs(ang2 - 135) < 15) |
                 (np.abs(ang2 + 45) < 15) | (np.abs(ang2 + 135) < 15))
        dc2 = r2 < 0.03
        diag2 = diag2 & (~dc2)

        ax_fft2 = axes2[1, col]
        ax_fft2.imshow(np.log10(np.maximum(sub_mag, 1e-30)),
                       aspect='auto', origin='lower', cmap='viridis')
        ax_fft2.set_title(f'2D FFT {region} MHz\n对角功率: '
                          f'{100*sub_mag[diag2].sum()**2/sub_mag.sum()**2:.2f}%',
                          fontsize=9, fontweight='bold')
        ax_fft2.set_xlabel('ω_f', fontsize=8)
        if col == 0: ax_fft2.set_ylabel('ω_t', fontsize=9)
        ax_fft2.axhline(Nt2/2, color='r', alpha=0.3, lw=0.5)
        ax_fft2.axvline(Nf2/2, color='r', alpha=0.3, lw=0.5)

    fig2.suptitle(f'CH3×CH8 — 分频段 PCA 残余分析 — {args.date}', fontsize=13, fontweight='bold')
    out_zoom = f'{args.output}/median_pca_zoom_K{K}_{args.date}.png'
    fig2.savefig(out_zoom, dpi=200, bbox_inches='tight')
    print(f"  保存: {out_zoom}")
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"管线汇总 ({args.date}):")
    print(f"  矩阵: {n_valid} 帧 × {n_ch} 通道 = {n_valid*n_ch:,} 像素")
    print(f"  RFI 掩码: {combined_mask.sum():,} ({100*combined_mask.sum()/(n_valid*n_ch):.1f}%)")
    print(f"  ① 中位数相减: 移除 {100*med_power/power_before:.1f}% 功率 (持久 RFI)")
    print(f"  ② PCA K={K}: 移除 {removed_var:.1f}% 方差 (可分离横纵结构)")
    print(f"  ③ 残余: {kept_var:.1f}% 方差")
    print(f"  2D FFT 功率分布:")
    print(f"    横纵: {100*axial_power/total_power:.1f}%")
    print(f"    对角: {100*diag_power/total_power:.1f}%")
    print(f"    其他: {100*other_power/total_power:.1f}%")
    if diag_peak_val > 0:
        peak_idx = np.argmax(diag_vals)
        print(f"  对角峰值: ω_t={diag_omega[peak_idx]:.5f} cycles/frame "
              f"(周期 {1/abs(diag_omega[peak_idx]):.0f} 帧 ≈ {1/abs(diag_omega[peak_idx])*1.14/60:.1f} 分钟)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
