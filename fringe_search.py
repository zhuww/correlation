"""
逐通道 fringe 频率搜索：对 PCA 清理后的残余，
逐频率通道做 1D FFT，搜索与频率成正比的 fringe 峰。

理论：fringe 频率 f_fringe = f_RF * dτ/dt
  → 在 1D FFT 中，峰位置 ω_t ∝ f_RF
  → 这是一个非常强的预测，可以区分 fringe 和噪声
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm
import time, os, sys, re, argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
from datetime import datetime
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


# ═══════════════════════════════════════════════════════════════════
# 数据加载（复用）
# ═══════════════════════════════════════════════════════════════════
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


def read_sky_frequencies(file_map, n_channels=4096, center_freq_mhz=150.0):
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=n_channels)
    freq_hz = df['frequency_hz'].values.astype(np.float64)
    return center_freq_mhz + freq_hz / 1e6


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
        description='逐通道 1D FFT fringe 搜索：fringe 频率 ∝ RF 频率')
    parser.add_argument('--date', default='20260630')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--n-channels', type=int, default=4096)
    parser.add_argument('--center-freq', type=float, default=150.0)
    parser.add_argument('--n-components', type=int, default=200)
    parser.add_argument('--output', default='integrated_images')
    args = parser.parse_args()

    watch_dir = 'correlation_results'; n_ch = args.n_channels; K = args.n_components

    print("=" * 70)
    print("  CH3×CH8: 逐通道 fringe 频率搜索")
    print("=" * 70)

    # ── 加载数据 ──
    print(f"\n[1/5] 加载数据...")
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
    vis_matrix = np.array(all_vis)
    auto3_s = np.array(all_auto3); auto8_s = np.array(all_auto8)
    print(f"  有效: {n_valid}/{n_frames}")

    # ── RFI + bandpass ──
    print(f"\n[2/5] RFI 掩码 + bandpass 校正...")
    freq_rfi = np.zeros((n_valid, n_ch), dtype=bool)
    for fi in range(n_valid):
        freq_rfi[fi] = detect_freq_rfi(auto3_s[fi], auto8_s[fi])

    time_rfi = np.zeros(n_valid, dtype=bool)
    frame_powers = [np.median(np.maximum(auto3_s[fi], auto8_s[fi])) for fi in range(n_valid)]
    for bi in detect_time_rfi(frame_powers): time_rfi[bi] = True

    good_mask = ~time_rfi
    bp3 = np.median(auto3_s[good_mask], axis=0) if good_mask.any() else np.median(auto3_s, axis=0)
    bp8 = np.median(auto8_s[good_mask], axis=0) if good_mask.any() else np.median(auto8_s, axis=0)
    bp3_c = clean_bandpass_spikes(bp3); bp8_c = clean_bandpass_spikes(bp8)
    expected = np.sqrt(np.maximum(bp3_c, 1e-30) * np.maximum(bp8_c, 1e-30))
    bp_factor = np.median(expected) / np.maximum(expected, 1e-30)
    vis_corr = vis_matrix * bp_factor[np.newaxis, :]

    combined_mask = freq_rfi.copy()
    combined_mask[time_rfi] = True
    print(f"  RFI: {combined_mask.sum():,} ({100*combined_mask.sum()/(n_valid*n_ch):.1f}%)")

    # ── 时间中位数相减 (复数) ──
    print(f"\n[3/5] 时间中位数相减...")
    complex_median = np.zeros(n_ch, dtype=np.complex128)
    for j in range(n_ch):
        vals = vis_corr[~time_rfi, j] if (~time_rfi).any() else vis_corr[:, j]
        complex_median[j] = np.median(vals.real) + 1j * np.median(vals.imag)
    vis_mediansub = vis_corr - complex_median[np.newaxis, :]
    print(f"  中位数功率: {np.mean(np.abs(complex_median)**2):.2e}")

    # ── 插值 + PCA ──
    print(f"\n[4/5] 插值 + PCA (K={K})...")
    vis_real = np.real(vis_mediansub)
    vis_real_filled = inpaint_masked(vis_real, combined_mask)
    vis_real_filled = np.nan_to_num(vis_real_filled, nan=0.0)

    vis_centered = vis_real_filled - vis_real_filled.mean(axis=0, keepdims=True)
    vis_centered = vis_centered - vis_centered.mean(axis=1, keepdims=True)

    t_svd = time.time()
    U, S, Vh = np.linalg.svd(vis_centered, full_matrices=False)
    print(f"  SVD 完成 ({time.time()-t_svd:.1f}s)")

    residual = np.zeros_like(vis_centered)
    if K < min(n_valid, n_ch):
        for k in range(K, len(S)):
            residual += S[k] * np.outer(U[:, k], Vh[k, :])
    residual_restored = (residual +
                         vis_real_filled.mean(axis=0, keepdims=True) +
                         vis_real_filled.mean(axis=1, keepdims=True) -
                         vis_real_filled.mean())

    rms_res = np.sqrt(np.mean(residual_restored**2))
    print(f"  残余 RMS: {rms_res:.2e}")

    # ═══════════════════════════════════════════════════════════════
    # 核心: 逐通道 1D FFT 搜索 fringe
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[5/5] 逐通道 1D FFT fringe 搜索...")
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, args.center_freq)

    # 1D FFT 参数
    n_fft = n_valid
    omega_t = np.fft.rfftfreq(n_fft, d=1.0)  # cycles per frame
    n_omega = len(omega_t)

    # 对每帧加 Hanning 窗，减少频谱泄漏
    win = np.hanning(n_valid)
    power_spectra = np.zeros((n_ch, n_omega), dtype=np.float64)

    for j in range(n_ch):
        ts = residual_restored[:, j] * win
        fft_vals = np.fft.rfft(ts)
        power_spectra[j] = np.abs(fft_vals)**2

    # 只搜索正频率 (0 到 0.05 cycles/frame)
    search_mask = (omega_t >= 0.001) & (omega_t <= 0.05)
    search_idx = np.where(search_mask)[0]

    # 对每个通道找峰值
    peak_freqs = np.zeros(n_ch)
    peak_powers = np.zeros(n_ch)
    for j in range(n_ch):
        if len(search_idx) > 0:
            pwr = power_spectra[j, search_idx]
            max_i = np.argmax(pwr)
            peak_freqs[j] = omega_t[search_idx[max_i]]
            peak_powers[j] = pwr[max_i]

    # 峰值显著性：与周围平均功率比
    peak_snrs = np.zeros(n_ch)
    for j in range(n_ch):
        p = power_spectra[j, search_idx]
        if peak_powers[j] > 0 and len(p) > 10:
            # 排除峰值周围的 5 个 bin
            peak_local_idx = np.argmin(np.abs(omega_t[search_idx] - peak_freqs[j]))
            local_mask = np.ones(len(p), dtype=bool)
            lo = max(0, peak_local_idx - 5)
            hi = min(len(p), peak_local_idx + 6)
            local_mask[lo:hi] = False
            if local_mask.sum() > 0:
                local_mean = np.mean(p[local_mask])
                peak_snrs[j] = peak_powers[j] / local_mean if local_mean > 0 else 0

    # 筛选显著峰 (SNR > 3)
    sig_mask = peak_snrs > 3.0
    print(f"  显著峰 (SNR>3): {sig_mask.sum()}/{n_ch} 通道")

    # 理论预期：fringe 频率 ∝ RF 频率
    # ω_t = f_RF * dτ/dt * Δt_frame
    # 对于 100m 基线：dτ/dt ≈ 1.7e-11 s^-1, Δt ≈ 1.14 s
    # ω_t ≈ f_MHz * 1.0e6 * 1.7e-11 * 1.14 ≈ f_MHz * 1.94e-5 cycles/frame
    # 即 ω_t ≈ 0.00194 * f_MHz
    # 在 150 MHz: ω_t ≈ 0.0029
    expected_slope = 1.94e-5  # cycles/frame per MHz

    # 拟合峰值频率 vs RF 频率
    if sig_mask.sum() >= 10:
        from numpy.polynomial import polynomial as P
        x = freq_raw[sig_mask] - 150.0  # 以 150 MHz 为中心
        y = peak_freqs[sig_mask]
        c, stats = P.polyfit(x, y, 1, full=True)
        fitted_slope = c[1]
        fitted_intercept = c[0]
        print(f"  拟合: ω_t = {fitted_intercept:.5f} + {fitted_slope:.5e} * (f - 150MHz)")
        print(f"  理论斜率: {expected_slope:.2e}, 拟合斜率: {fitted_slope:.2e}")
    else:
        fitted_slope = None
        fitted_intercept = None

    # ═══════════════════════════════════════════════════════════════
    # 绘图
    # ═══════════════════════════════════════════════════════════════
    print(f"\n绘图...")
    t0_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
    times = np.array([(datetime.strptime(timestamps[idx], "%Y%m%d_%H%M%S") - t0_dt).total_seconds() / 60
                      for idx in valid_idx])

    # ── 图 1: 核心图 ──
    fig1 = plt.figure(figsize=(12, 8))
    gs1 = fig1.add_gridspec(2, 3, hspace=0.35, wspace=0.25)

    # (0,0) PCA 残余瀑布图
    ax_res = fig1.add_subplot(gs1[0, 0])
    vabs = max(abs(residual_restored.min()), abs(residual_restored.max()))
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    dec_t = max(1, n_valid // 800)
    dec_f = max(1, n_ch // 800)
    extent = [freq_raw[0], freq_raw[-1], times[0], times[-1]]
    im1 = ax_res.imshow(residual_restored[::dec_t, ::dec_f], aspect='auto', origin='lower',
                        cmap='RdBu_r', norm=norm, extent=extent, interpolation='nearest')
    ax_res.set_title(f'PCA residual (K={K})', fontsize=10, fontweight='bold')
    ax_res.set_ylabel('Time [min]', fontsize=9)
    ax_res.set_xlabel('Freq [MHz]', fontsize=9)
    plt.colorbar(im1, ax=ax_res, label='Re(V)', shrink=0.6)

    # (0,1) 功率谱密度瀑布图
    ax_psd = fig1.add_subplot(gs1[0, 1])
    psd_log = np.log10(np.maximum(power_spectra, 1e-30))
    psd_vmin = np.percentile(psd_log[:, 1:], 5)
    psd_vmax = np.percentile(psd_log[:, 1:], 95)
    im2 = ax_psd.imshow(psd_log.T, aspect='auto', origin='lower', cmap='viridis',
                        vmin=psd_vmin, vmax=psd_vmax,
                        extent=[freq_raw.min(), freq_raw.max(), omega_t[0], omega_t[-1]])
    ax_psd.set_title('逐通道 1D FFT 功率谱', fontsize=10, fontweight='bold')
    ax_psd.set_xlabel('频率 [MHz]', fontsize=9)
    ax_psd.set_ylabel('ω_t [cycles/frame]', fontsize=9)
    ax_psd.set_ylim(0, 0.05)
    plt.colorbar(im2, ax=ax_psd, label='log10(|FFT|²)', shrink=0.6)
    f_range = np.array([freq_raw.min(), freq_raw.max()])
    ax_psd.plot(f_range, expected_slope * f_range * 1e6, 'r--', alpha=0.7, lw=1.5,
                label=f'理论 fringe')
    ax_psd.legend(fontsize=8)

    # (0,2) 峰值频率 vs RF 频率
    ax_peak = fig1.add_subplot(gs1[0, 2])
    ax_peak.scatter(freq_raw[sig_mask], peak_freqs[sig_mask], c='red', s=3, alpha=0.6, label='显著峰 (SNR>3)')
    ax_peak.scatter(freq_raw[~sig_mask], peak_freqs[~sig_mask], c='gray', s=1, alpha=0.3, label='噪声峰')
    ax_peak.plot(f_range, expected_slope * f_range * 1e6, 'b--', alpha=0.7, lw=1.5, label='理论')
    if fitted_slope is not None:
        ax_peak.plot(f_range, fitted_intercept + fitted_slope * (f_range - 150),
                     'g-', alpha=0.7, lw=1.5, label=f'拟合: 斜率={fitted_slope:.1e}')
    ax_peak.set_title('峰值频率 vs RF 频率', fontsize=10, fontweight='bold')
    ax_peak.set_xlabel('RF 频率 [MHz]', fontsize=9)
    ax_peak.set_ylabel('峰值 ω_t [cycles/frame]', fontsize=9)
    ax_peak.set_ylim(0, 0.05)
    ax_peak.legend(fontsize=7)
    ax_peak.grid(True, alpha=0.2)

    # (1,0) 典型通道功率谱
    ax_spec = fig1.add_subplot(gs1[1, 0])
    example_chs = [800, 1200, 1600, 2000, 2400, 2800]
    colors = plt.cm.tab10(np.linspace(0, 1, len(example_chs)))
    for i, ch in enumerate(example_chs):
        f_mhz = freq_raw[ch]
        label = f'{f_mhz:.0f} MHz, 峰@ω_t={peak_freqs[ch]:.4f}'
        ax_spec.semilogy(omega_t[search_idx], power_spectra[ch, search_idx] + 1e-30,
                         color=colors[i], alpha=0.7, lw=0.8, label=label)
    ax_spec.set_title('典型通道功率谱', fontsize=10, fontweight='bold')
    ax_spec.set_xlabel('ω_t [cycles/frame]', fontsize=9)
    ax_spec.set_ylabel('功率', fontsize=9)
    ax_spec.set_xlim(0, 0.05)
    ax_spec.legend(fontsize=7, loc='upper right')
    ax_spec.grid(True, alpha=0.2)

    # (1,1) SNR 分布
    ax_snr = fig1.add_subplot(gs1[1, 1])
    ax_snr.hist(peak_snrs, bins=100, range=(0, 10), color='steelblue', alpha=0.7, edgecolor='none')
    ax_snr.axvline(3, color='red', linestyle='--', alpha=0.7, label='SNR=3')
    ax_snr.set_title('峰值 SNR 分布', fontsize=10, fontweight='bold')
    ax_snr.set_xlabel('SNR', fontsize=9)
    ax_snr.set_ylabel('通道数', fontsize=9)
    ax_snr.legend(fontsize=8)
    ax_snr.grid(True, alpha=0.2)

    # (1,2) 累积功率谱
    ax_avg = fig1.add_subplot(gs1[1, 2])
    avg_psd = np.mean(power_spectra, axis=0)
    ax_avg.semilogy(omega_t[search_idx], avg_psd[search_idx] + 1e-30, 'b-', alpha=0.8, lw=1)
    ax_avg.set_title('所有通道平均功率谱', fontsize=10, fontweight='bold')
    ax_avg.set_xlabel('ω_t [cycles/frame]', fontsize=9)
    ax_avg.set_ylabel('平均功率', fontsize=9)
    ax_avg.set_xlim(0, 0.05)
    ax_avg.grid(True, alpha=0.2)
    for f_mhz in [100, 150, 200]:
        omega_theory = expected_slope * f_mhz * 1e6
        ax_avg.axvline(omega_theory, color='red', linestyle='--', alpha=0.4, lw=0.8)
        ax_avg.text(omega_theory, np.max(avg_psd[search_idx]) * 0.5, f'{f_mhz}MHz',
                    fontsize=7, color='red', ha='center')

    fig1.suptitle(f'CH3xCH8 fringe search (K={K})', fontsize=12, fontweight='bold')

    os.makedirs(args.output, exist_ok=True)
    out_path = f'{args.output}/fringe_search_K{K}_{args.date}.png'
    fig1.savefig(out_path, dpi=120)
    print(f"\n  保存: {out_path}")
    plt.close(fig1)

    # ── 图 2: 时序 + 直方图 ──
    fig2 = plt.figure(figsize=(10, 4))
    gs2 = fig2.add_gridspec(1, 2, wspace=0.25)

    ax_ts = fig2.add_subplot(gs2[0, 0])
    if sig_mask.sum() > 0:
        best_ch = np.argmax(peak_snrs)
        best_freq = freq_raw[best_ch]
        ts = residual_restored[:, best_ch]
        ax_ts.plot(times, ts, 'b-', alpha=0.6, lw=0.5)
        ax_ts.axhline(0, color='gray', alpha=0.3)
        ax_ts.set_title(f'Best SNR: {best_freq:.1f} MHz (SNR={peak_snrs[best_ch]:.1f})',
                        fontsize=10, fontweight='bold')
        ax_ts.set_xlabel('Time [min]', fontsize=9)
        ax_ts.set_ylabel('Re(V)', fontsize=9)
        ax_ts.grid(True, alpha=0.2)

    ax_hist = fig2.add_subplot(gs2[0, 1])
    ax_hist.hist(peak_freqs[sig_mask], bins=50, range=(0, 0.05), color='steelblue', alpha=0.7, edgecolor='none')
    ax_hist.set_title('Peak freq distribution', fontsize=10, fontweight='bold')
    ax_hist.set_xlabel('Peak ω_t [cycles/frame]', fontsize=9)
    ax_hist.set_ylabel('Count', fontsize=9)
    ax_hist.grid(True, alpha=0.2)
    for f_mhz in [100, 150, 200]:
        omega_theory = expected_slope * f_mhz * 1e6
        ax_hist.axvline(omega_theory, color='red', linestyle='--', alpha=0.4, lw=0.8)

    fig2.suptitle(f'Fringe freq hist (K={K})', fontsize=12, fontweight='bold')
    out_path2 = f'{args.output}/fringe_search_hist_K{K}_{args.date}.png'
    fig2.savefig(out_path2, dpi=120)
    print(f"  保存: {out_path2}")
    plt.close(fig2)

    # 汇总
    print(f"\n{'='*70}")
    print(f"Fringe 搜索汇总 ({args.date}):")
    print(f"  通道数: {n_ch}")
    print(f"  搜索范围: ω_t = 0.001 ~ 0.05 cycles/frame")
    print(f"  理论 fringe 斜率: {expected_slope:.2e} cycles/frame/MHz")
    print(f"  显著峰 (SNR>3): {sig_mask.sum()}/{n_ch}")
    if fitted_slope is not None:
        print(f"  拟合斜率: {fitted_slope:.2e} (理论: {expected_slope:.2e})")
        ratio = fitted_slope / expected_slope if expected_slope != 0 else 0
        print(f"  斜率比 (拟合/理论): {ratio:.1f}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
