"""
对每个频率通道减去时间均值，去除纵向窄带 RFI 结构。

原理：
  - 窄带 RFI 在几乎所有帧的同一频率出现 → 时间均值会捕获它
  - 天体 fringe 随时间变化（相位旋转） → 时间均值趋向零
  - 减去时间均值 = 保留 fringe，去除持久 RFI

处理复数 visibility：实部和虚部分别做时间均值减法（等价于复数直接减法）。
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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
# 数据加载（复用现有逻辑）
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
        if pair_m:
            frames[ts][pair_m.group(1)] = f
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
    """双线性插值填充掩码区域（对实部/虚部矩阵分别处理）。"""
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
        description='频率方向时间均值减法，去除纵向窄带 RFI 结构')
    parser.add_argument('--date', default='20260630')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--n-channels', type=int, default=4096)
    parser.add_argument('--center-freq', type=float, default=150.0)
    parser.add_argument('--use-median', action='store_true',
                        help='用时间中位数代替时间均值（更鲁棒）')
    parser.add_argument('--output', default='integrated_images')
    args = parser.parse_args()

    watch_dir = 'correlation_results'; n_ch = args.n_channels

    # ── 1. 加载数据 ──
    print(f"扫描 {watch_dir} ...")
    all_data = discover_frames(watch_dir, args.date)
    if args.date not in all_data:
        print(f"错误: 未找到日期 {args.date}"); sys.exit(1)
    frames_by_ts = all_data[args.date]
    timestamps = list(frames_by_ts.keys())
    if args.max_frames > 0:
        timestamps = timestamps[:args.max_frames]
    n_frames = len(timestamps)
    print(f"  {n_frames} 帧")

    print(f"\n加载数据...")
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
        if (fi + 1) % 500 == 0:
            print(f"  已读 {fi+1}/{n_frames} ({time.time()-t0:.0f}s)")
    n_valid = len(all_vis)
    vis_matrix = np.array(all_vis)               # complex (n_valid, n_ch)
    auto3_s = np.array(all_auto3); auto8_s = np.array(all_auto8)
    del all_vis, all_auto3, all_auto8
    print(f"  有效: {n_valid}/{n_frames} ({time.time()-t0:.1f}s)")

    # ── 2. 频率 RFI ──
    freq_rfi = np.zeros((n_valid, n_ch), dtype=bool)
    total_freq_rfi = 0
    print(f"\n频率 RFI 检测...")
    t1 = time.time()
    for fi in range(n_valid):
        mask = detect_freq_rfi(auto3_s[fi], auto8_s[fi])
        freq_rfi[fi] = mask; total_freq_rfi += mask.sum()
    print(f"  {total_freq_rfi:,} 通道 ({100*total_freq_rfi/(n_valid*n_ch):.2f}%) ({time.time()-t1:.1f}s)")

    # ── 3. 时域 RFI ──
    time_rfi = np.zeros(n_valid, dtype=bool)
    print(f"\n时域 RFI 检测...")
    frame_powers = [np.median(np.maximum(auto3_s[fi], auto8_s[fi])) for fi in range(n_valid)]
    bad = detect_time_rfi(frame_powers)
    for bi in bad: time_rfi[bi] = True
    print(f"  {len(bad)}/{n_valid} 帧 ({100*len(bad)/n_valid:.1f}%)")

    # ── 4. Bandpass 校正 ──
    print(f"\nBandpass 校正...")
    good_mask = ~time_rfi
    if good_mask.any():
        bp3 = np.median(auto3_s[good_mask], axis=0)
        bp8 = np.median(auto8_s[good_mask], axis=0)
        bp3_c = clean_bandpass_spikes(bp3)
        bp8_c = clean_bandpass_spikes(bp8)
        expected = np.sqrt(np.maximum(bp3_c, 1e-30) * np.maximum(bp8_c, 1e-30))
        bp_factor = np.median(expected) / np.maximum(expected, 1e-30)
        
    # ── 5. 掩码 → 插值 ──
    combined_mask = freq_rfi.copy()
    combined_mask[time_rfi] = True
    print(f"合并掩码: {combined_mask.sum():,}/{n_valid*n_ch} ({100*combined_mask.sum()/(n_valid*n_ch):.2f}%)")

    # 分别处理实部和虚部
    vis_real = np.real(vis_matrix)
    vis_imag = np.imag(vis_matrix)
    print(f"\n掩码插值 (实部)...")
    real_filled = inpaint_masked(vis_real, combined_mask)
    real_filled = np.nan_to_num(real_filled, nan=0.0)
    print(f"掩码插值 (虚部)...")
    imag_filled = inpaint_masked(vis_imag, combined_mask)
    imag_filled = np.nan_to_num(imag_filled, nan=0.0)
    vis_filled = real_filled + 1j * imag_filled
    
    # ── 6. 核心：对每个频率通道减去时间均值 ──
    print(f"\n{'='*60}")
    print("核心操作: 对每个频率点 j，减去所有时间上的均值")
    print("  V_clean[t, j] = V[t, j] - (1/N_t) * Σ_i V[i, j]")
    print(f"{'='*60}")
    
    if args.use_median:
        # 用中位数更鲁棒
        time_mean = np.zeros(n_ch, dtype=np.complex128)
        for j in range(n_ch):
            time_mean[j] = np.median(vis_filled[:, j].real) + 1j * np.median(vis_filled[:, j].imag)
        method_name = "时间中位数"
    else:
        time_mean = np.mean(vis_filled, axis=0)
        method_name = "时间均值"
    
    # 减法
    vis_clean = vis_filled - time_mean[np.newaxis, :]
    
    # 统计
    rms_before = np.sqrt(np.mean(np.abs(vis_filled)**2))
    rms_after = np.sqrt(np.mean(np.abs(vis_clean)**2))
    mean_power_time_mean = np.mean(np.abs(time_mean)**2)
    print(f"\n  原始 RMS: {rms_before:.2e}")
    print(f"  {method_name} 功率: {mean_power_time_mean:.2e} ({100*mean_power_time_mean/rms_before**2:.1f}% 的总功率)")
    print(f"  清理后 RMS: {rms_after:.2e}")
    print(f"  功率降低: {100*(1 - rms_after**2/rms_before**2):.1f}%")

    # ── 7. 频率重排 ──
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, args.center_freq)
    Nc = n_ch
    if Nc % 2 == 0:
        reorder = np.concatenate([np.arange(Nc // 2 + 1, Nc), np.arange(0, Nc // 2 + 1)]).astype(int)
    else:
        reorder = np.arange(Nc)
    freq_disp = freq_raw[reorder]

    def reorder_data(m):
        return m[:, reorder] if m.ndim == 2 else m[reorder]

    vis_orig_r = reorder_data(vis_filled.real)
    vis_clean_r = reorder_data(vis_clean.real)
    vis_clean_im_r = reorder_data(vis_clean.imag)
    time_mean_r = reorder_data(time_mean)

    # ── 8. 时间轴 ──
    t0_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
    times = np.array([(datetime.strptime(timestamps[idx], "%Y%m%d_%H%M%S") - t0_dt).total_seconds() / 60
                      for idx in valid_idx])

    # ── 9. 色标 ──
    vabs = max(abs(vis_orig_r.min()), abs(vis_orig_r.max()))
    rabs = max(abs(vis_clean_r.min()), abs(vis_clean_r.max()))
    iabs = max(abs(vis_clean_im_r.min()), abs(vis_clean_im_r.max()))

    # ═══════════════════════════════════════════════════════════════
    # 绘制大图: 5 个子图
    # ═══════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(4, 4, height_ratios=[0.6, 1, 1, 1],
                          width_ratios=[1, 0.03, 1, 0.03],
                          hspace=0.3, wspace=0.05)

    # ── 第 1 行: 被减去的"持久频谱" ──
    ax_spec = fig.add_subplot(gs[0, :2])
    ax_spec.plot(freq_disp, time_mean_r.real, 'b-', linewidth=0.6, alpha=0.8, label=f'{method_name} 实部')
    ax_spec.plot(freq_disp, time_mean_r.imag, 'r-', linewidth=0.6, alpha=0.8, label=f'{method_name} 虚部')
    ax_spec.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax_spec.set_xlabel('天空频率 [MHz]', fontsize=10)
    ax_spec.set_ylabel(f'{method_name}', fontsize=10)
    ax_spec.set_title(f'被减去的"持久频谱": 每个频率通道的 {method_name}\n'
                      f'(这些是时间上不变的结构 → 窄带 RFI + 静态 bandpass)', 
                      fontsize=10, fontweight='bold')
    ax_spec.set_xlim(freq_disp.min(), freq_disp.max())
    ax_spec.legend(fontsize=9)
    ax_spec.grid(True, alpha=0.2)
    ax_spec.ticklabel_format(useOffset=False, style='plain')

    # ── 第 2 行: 原始 vs 清理后 (实部) ──
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_clean = fig.add_subplot(gs[1, 2])

    norm_orig = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im1 = ax_orig.pcolormesh(freq_disp, times, vis_orig_r, cmap='RdBu_r',
                              norm=norm_orig, shading='nearest', rasterized=True)
    ax_orig.set_title(f'原始 (RFI+bandpass 校正后)\n{n_valid} 帧 × {n_ch} 通道', 
                      fontsize=10, fontweight='bold')
    ax_orig.set_ylabel('时间 [分钟]', fontsize=10)
    ax_orig.set_xlim(freq_disp.min(), freq_disp.max())
    ax_orig.set_ylim(times[0], times[-1])
    ax_orig.ticklabel_format(useOffset=False, style='plain')

    norm_clean = TwoSlopeNorm(vmin=-rabs, vcenter=0, vmax=rabs)
    im2 = ax_clean.pcolormesh(freq_disp, times, vis_clean_r, cmap='RdBu_r',
                               norm=norm_clean, shading='nearest', rasterized=True)
    ax_clean.set_title(f'频率方向 {method_name} 减法后 — 实部\n'
                       f'RMS: {rms_before:.2e} → {rms_after:.2e} ',
                       fontsize=10, fontweight='bold')
    ax_clean.set_xlim(freq_disp.min(), freq_disp.max())
    ax_clean.set_ylim(times[0], times[-1])
    ax_clean.ticklabel_format(useOffset=False, style='plain')

    # colorbars
    cax1 = fig.add_subplot(gs[1, 1])
    plt.colorbar(im1, cax=cax1, label='原始 Re(V)')
    cax2 = fig.add_subplot(gs[1, 3])
    plt.colorbar(im2, cax=cax2, label='清理后 Re(V)')

    # ── 第 3 行: 清理后的虚部 ──
    ax_imag = fig.add_subplot(gs[2, 0])
    norm_imag = TwoSlopeNorm(vmin=-iabs, vcenter=0, vmax=iabs)
    im3 = ax_imag.pcolormesh(freq_disp, times, vis_clean_im_r, cmap='RdBu_r',
                              norm=norm_imag, shading='nearest', rasterized=True)
    ax_imag.set_title(f'频率方向 {method_name} 减法后 — 虚部',
                      fontsize=10, fontweight='bold')
    ax_imag.set_ylabel('时间 [分钟]', fontsize=10)
    ax_imag.set_xlabel('天空频率 [MHz]', fontsize=10)
    ax_imag.set_xlim(freq_disp.min(), freq_disp.max())
    ax_imag.set_ylim(times[0], times[-1])
    ax_imag.ticklabel_format(useOffset=False, style='plain')
    cax3 = fig.add_subplot(gs[2, 1])
    plt.colorbar(im3, cax=cax3, label='清理后 Im(V)')

    # ── 第 4 行: 每帧功率变化，RGB 复合图 ──
    ax_power = fig.add_subplot(gs[3, 0])
    power_before = np.mean(np.abs(vis_filled)**2, axis=1)
    power_after = np.mean(np.abs(vis_clean)**2, axis=1)
    ax_power.plot(times, power_before, 'gray', alpha=0.5, linewidth=0.6, label='原始')
    ax_power.plot(times, power_after, 'blue', alpha=0.7, linewidth=0.8, label='清理后')
    ax_power.set_xlabel('时间 [分钟]', fontsize=9)
    ax_power.set_ylabel('平均功率 |V|²', fontsize=9)
    ax_power.set_title('每帧总功率变化', fontsize=9)
    ax_power.legend(fontsize=7)
    ax_power.grid(True, alpha=0.2)

    # ── 右侧: 差分图 (原始 - 清理后 = 被减去的纵向结构) ──
    ax_diff = fig.add_subplot(gs[2, 2])
    diff = vis_orig_r - vis_clean_r  # 实部差分
    dvabs = max(abs(diff.min()), abs(diff.max()))
    norm_diff = TwoSlopeNorm(vmin=-dvabs, vcenter=0, vmax=dvabs)
    im4 = ax_diff.pcolormesh(freq_disp, times, diff, cmap='RdBu_r',
                              norm=norm_diff, shading='nearest', rasterized=True)
    ax_diff.set_title(f'被减去的纵向结构 (原始 − 清理后)\n'
                      f'= 时间不变部分',
                      fontsize=10, fontweight='bold')
    ax_diff.set_xlabel('天空频率 [MHz]', fontsize=10)
    ax_diff.set_xlim(freq_disp.min(), freq_disp.max())
    ax_diff.set_ylim(times[0], times[-1])
    ax_diff.ticklabel_format(useOffset=False, style='plain')
    cax4 = fig.add_subplot(gs[2, 3])
    plt.colorbar(im4, cax=cax4, label='Δ Re(V)')

    # ── 聚焦区域 (低频 100-130 MHz, 清理后实部) ──
    ax_zoom = fig.add_subplot(gs[3, 2])
    f_idx = (freq_disp >= 100) & (freq_disp <= 130)
    if f_idx.sum() > 5:
        zoom_data = vis_clean_r[:, f_idx]
        z_abs = max(abs(zoom_data.min()), abs(zoom_data.max()))
        norm_z = TwoSlopeNorm(vmin=-z_abs, vcenter=0, vmax=z_abs)
        ax_zoom.pcolormesh(freq_disp[f_idx], times, zoom_data,
                            cmap='RdBu_r', norm=norm_z,
                            shading='nearest', rasterized=True)
        ax_zoom.set_title(f'聚焦 100-130 MHz (清理后实部)',
                          fontsize=10, fontweight='bold')
        ax_zoom.set_xlabel('天空频率 [MHz]', fontsize=10)
        ax_zoom.set_xlim(100, 130)
        ax_zoom.set_ylim(times[0], times[-1])

    fig.suptitle(f'CH3×CH8 — 频率方向 {method_name} 减法去除纵向结构 — {args.date}',
                 fontsize=13, fontweight='bold', y=0.99)

    os.makedirs(args.output, exist_ok=True)
    median_tag = "_median" if args.use_median else ""
    out_path = f'{args.output}/freq_demean_time{median_tag}_{args.date}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n保存: {out_path}")
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════
    # 额外图: 单频率通道的时间序列 (证明 fringe 没被破坏)
    # ═══════════════════════════════════════════════════════════════
    # 选一个"干净"频率的通道（比如避开 RFI 的 155 MHz 附近）
    ch_idx = np.argmin(np.abs(freq_raw - 155.0))  # 原始顺序中的 155 MHz
    fig_ts, axes_ts = plt.subplots(3, 1, figsize=(20, 10), sharex=True,
                                    constrained_layout=True)

    ts_real_before = vis_filled[:, ch_idx].real
    ts_real_after = vis_clean[:, ch_idx].real
    ts_imag_before = vis_filled[:, ch_idx].imag
    ts_imag_after = vis_clean[:, ch_idx].imag

    axes_ts[0].plot(times, ts_real_before, 'gray', alpha=0.5, linewidth=0.5, label='原始')
    axes_ts[0].plot(times, ts_real_after, 'blue', alpha=0.8, linewidth=0.7, label='清理后')
    axes_ts[0].axhline(0, color='k', alpha=0.3)
    axes_ts[0].set_ylabel('Re(V)', fontsize=10)
    axes_ts[0].set_title(f'单通道时间序列 ~{freq_raw[ch_idx]:.1f} MHz (避开RFI区)\n'
                         f'—— 减 {method_name} 是否破坏了 fringe？', fontsize=11, fontweight='bold')
    axes_ts[0].legend(fontsize=8)
    axes_ts[0].grid(True, alpha=0.2)

    axes_ts[1].plot(times, ts_imag_before, 'gray', alpha=0.5, linewidth=0.5, label='原始')
    axes_ts[1].plot(times, ts_imag_after, 'red', alpha=0.8, linewidth=0.7, label='清理后')
    axes_ts[1].axhline(0, color='k', alpha=0.3)
    axes_ts[1].set_ylabel('Im(V)', fontsize=10)
    axes_ts[1].legend(fontsize=8)
    axes_ts[1].grid(True, alpha=0.2)

    # 复数幅角 (相位)
    phase_before = np.angle(vis_filled[:, ch_idx])
    phase_after = np.angle(vis_clean[:, ch_idx])
    axes_ts[2].plot(times, np.unwrap(phase_before), 'gray', alpha=0.5, linewidth=0.5, label='原始')
    axes_ts[2].plot(times, np.unwrap(phase_after), 'green', alpha=0.8, linewidth=0.7, label='清理后')
    axes_ts[2].set_xlabel('时间 [分钟]', fontsize=10)
    axes_ts[2].set_ylabel('Unwrapped Phase [rad]', fontsize=10)
    axes_ts[2].set_title('相位时间序列 (unwrap) — 不变 = fringe 未受损', fontsize=10)
    axes_ts[2].legend(fontsize=8)
    axes_ts[2].grid(True, alpha=0.2)

    out_ts = f'{args.output}/freq_demean_ts{median_tag}_{args.date}.png'
    fig_ts.savefig(out_ts, dpi=200, bbox_inches='tight')
    print(f"保存: {out_ts}")
    plt.close(fig_ts)

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"频率方向 {method_name} 减法汇总 ({args.date}):")
    print(f"  矩阵大小: {n_valid} 帧 × {n_ch} 通道")
    print(f"  频率 RFI: {total_freq_rfi:,} 通道 ({100*total_freq_rfi/(n_valid*n_ch):.2f}%)")
    print(f"  时域 RFI: {time_rfi.sum()} 帧 ({100*time_rfi.sum()/n_valid:.1f}%)")
    print(f"  操作: V_clean[t,j] = V[t,j] - {method_name}_t(V[:,j])")
    print(f"  功率变化: {rms_before**2:.2e} → {rms_after**2:.2e} ({100*(1-rms_after**2/rms_before**2):.1f}% 降低)")
    print(f"  被移除的持久结构功率: {mean_power_time_mean:.2e}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
