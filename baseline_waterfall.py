"""
基线 CH3×CH8 瀑布图 — 互相关实部随时间×频率变化。

处理步骤：
  1. 扫描全部帧，读取 CH3×CH8 互相关 + CH3_AUTO + CH8_AUTO
  2. 频率域 RFI 检测：利用自相关中位数频谱 + 滑动中位检测窄带尖峰 → 掩码
  3. 时间域 RFI 检测：利用每帧总功率 + 滑动中位检测异常帧 → 掩码
  4. 每天线 Bandpass 估计：跨所有好帧取自相关中位数 → 调平互相关
  5. 绘制瀑布图（实部），标记被剔除的通道/帧
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm
import time
import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict
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
# 文件扫描
# ═══════════════════════════════════════════════════════════════════
def discover_frames(watch_dir, date_str=None):
    """扫描 correlation_results 目录，按时序返回帧列表。"""
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
        date_key = ts[:8]
        by_date[date_key][ts] = frames[ts]

    if date_str and date_str in by_date:
        return OrderedDict([(date_str, by_date[date_str])])
    return OrderedDict(sorted(by_date.items()))


# ═══════════════════════════════════════════════════════════════════
# 数据读取
# ═══════════════════════════════════════════════════════════════════
def read_auto(file_map, ant_id, n_channels=4096):
    """读取单个天线的自相关 magnitude。"""
    key = f"CH{ant_id}_AUTO"
    if key not in file_map:
        return None
    try:
        df = pd.read_csv(file_map[key], comment='#', usecols=['magnitude'])
        mag = df['magnitude'].values.astype(np.float64)
        if len(mag) >= n_channels:
            return mag[:n_channels]
        pad = np.zeros(n_channels - len(mag), dtype=np.float64)
        return np.concatenate([mag, pad])
    except Exception:
        return None


def read_cross_real(file_map, n_channels=4096):
    """读取 CH3×CH8 互相关实部。"""
    key_a, key_b = "CH3xCH8", "CH8xCH3"
    fp = file_map.get(key_a) or file_map.get(key_b)
    if fp is None:
        return None
    try:
        df = pd.read_csv(fp, comment='#', usecols=['real_part', 'imag_part'])
        re_vals = df['real_part'].values.astype(np.float64)
        im_vals = df['imag_part'].values.astype(np.float64)
        if len(re_vals) >= n_channels:
            return complex(1.0, 0.0) * re_vals[:n_channels] + 1j * im_vals[:n_channels]
        return np.zeros(n_channels, dtype=np.complex128)
    except Exception:
        return None


def read_sky_frequencies(file_map, n_channels=4096, center_freq_mhz=150.0):
    """读取天空频率 (MHz)。"""
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=n_channels)
    freq_hz = df['frequency_hz'].values.astype(np.float64)
    return center_freq_mhz + freq_hz / 1e6


# ═══════════════════════════════════════════════════════════════════
# RFI 检测
# ═══════════════════════════════════════════════════════════════════
def detect_freq_rfi(auto3, auto8, threshold=5.0, ratio_threshold=3.0, grow=5, smooth_w=101):
    """利用 CH3 和 CH8 自相关检测频率域窄带 RFI。

    跨两天线取中位数频谱 → 滑动中位基线 → MAD 阈值检测尖峰。

    Returns:
        rfi_mask: (n_channels,) bool, True=干扰
        baseline: (n_channels,) float, 平滑基线
    """
    n_ch = len(auto3)
    median_spec = np.maximum(auto3, auto8)  # 保守: 取较大者
    # 也试试取平均:
    # median_spec = 0.5 * (auto3 + auto8)

    half = smooth_w // 2
    mag_pad = np.pad(median_spec, (half, half), mode='reflect')
    windows = sliding_window_view(mag_pad, smooth_w)
    baseline = np.median(windows, axis=1)

    residual = median_spec - baseline
    mad = np.median(np.abs(residual))
    if mad <= 0:
        mad = np.median(np.abs(median_spec)) * 1e-3
    if mad <= 0:
        mad = 1.0

    flagged = np.zeros(n_ch, dtype=bool)
    if threshold > 0:
        flagged |= median_spec > (baseline + threshold * mad)
    if ratio_threshold > 0:
        flagged &= median_spec > (baseline * ratio_threshold)

    if grow > 0 and flagged.any():
        grown = np.zeros(n_ch, dtype=bool)
        for i in range(n_ch):
            if flagged[i]:
                lo = max(0, i - grow)
                hi = min(n_ch, i + grow + 1)
                grown[lo:hi] = True
        flagged = grown

    return flagged, baseline


def detect_time_rfi(frame_powers, td_threshold=5.0, td_window=101):
    """利用每帧总功率的滑动中位 MAD 检测时间域异常帧。

    Returns:
        bad_idx: set of frame indices flagged as bad
    """
    powers = np.array(frame_powers)
    n = len(powers)
    if n <= td_window:
        return set()

    half = td_window // 2
    smoothed = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        smoothed[i] = np.median(powers[lo:hi])

    residual = powers - smoothed
    mad = np.median(np.abs(residual))
    if mad <= 0:
        mad = np.median(np.abs(powers)) * 1e-3
    if mad <= 0:
        return set()

    bad_idx = set()
    for i in range(n):
        if powers[i] > smoothed[i] + td_threshold * mad:
            bad_idx.add(i)
        # 也检测功率暴跌（接收机故障）
        if powers[i] < smoothed[i] - td_threshold * mad * 2:
            bad_idx.add(i)

    return bad_idx


# ═══════════════════════════════════════════════════════════════════
# Bandpass 估计
# ═══════════════════════════════════════════════════════════════════
def estimate_bandpass(all_auto3, all_auto8, exclude_idx=set()):
    """跨帧取中位数估计 CH3 和 CH8 的 bandpass。

    Returns:
        bp3, bp8: (n_channels,) 每天线 bandpass
    """
    good_auto3 = [a for i, a in enumerate(all_auto3) if i not in exclude_idx and a is not None]
    good_auto8 = [a for i, a in enumerate(all_auto8) if i not in exclude_idx and a is not None]

    if not good_auto3 or not good_auto8:
        return None, None

    bp3 = np.median(np.array(good_auto3), axis=0)
    bp8 = np.median(np.array(good_auto8), axis=0)
    return bp3, bp8


def clean_bandpass_spikes(bp, threshold=5.0, ratio_threshold=3.0, grow=5, window=101):
    """清理 bandpass 模板中的窄带尖峰。"""
    n = len(bp)
    half = window // 2
    mag_pad = np.pad(bp, (half, half), mode='reflect')
    windows = sliding_window_view(mag_pad, window)
    baseline = np.median(windows, axis=1)

    residual = bp - baseline
    mad = np.median(np.abs(residual))
    if mad <= 0:
        mad = np.median(np.abs(bp)) * 1e-3
    if mad <= 0:
        mad = 1.0

    flagged = np.zeros(n, dtype=bool)
    if threshold > 0:
        flagged |= bp > (baseline + threshold * mad)
    if ratio_threshold > 0:
        flagged &= bp > (baseline * ratio_threshold)

    if grow > 0 and flagged.any():
        grown = np.zeros(n, dtype=bool)
        for i in range(n):
            if flagged[i]:
                lo = max(0, i - grow)
                hi = min(n, i + grow + 1)
                grown[lo:hi] = True
        flagged = grown

    bp_clean = bp.copy()
    bp_clean[flagged] = baseline[flagged]
    return bp_clean, flagged.sum()


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='CH3×CH8 瀑布图 — RFI 消除 + bandpass 调平')
    parser.add_argument('--date', default='20260630', help='观测日期')
    parser.add_argument('--freq-thresh', type=float, default=5.0,
                        help='频率 RFI MAD 阈值 (default 5.0)')
    parser.add_argument('--freq-ratio', type=float, default=3.0,
                        help='频率 RFI 相对倍数 (default 3.0)')
    parser.add_argument('--freq-grow', type=int, default=5,
                        help='频率 RFI 扩展通道 (default 5)')
    parser.add_argument('--freq-smooth', type=int, default=101,
                        help='频率 RFI 滑动窗口 (default 101)')
    parser.add_argument('--time-thresh', type=float, default=5.0,
                        help='时域 RFI MAD 阈值 (default 5.0)')
    parser.add_argument('--time-window', type=int, default=101,
                        help='时域 RFI 滑动窗口帧数 (default 101)')
    parser.add_argument('--no-bandpass', action='store_true',
                        help='跳过 bandpass 调平')
    parser.add_argument('--no-freq-rfi', action='store_true',
                        help='跳过频率 RFI 消除')
    parser.add_argument('--no-time-rfi', action='store_true',
                        help='跳过时域 RFI 消除')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='最大帧数 (0=全部)')
    parser.add_argument('--n-channels', type=int, default=4096,
                        help='通道数 (default 4096)')
    parser.add_argument('--center-freq', type=float, default=150.0,
                        help='中心频率 MHz (default 150)')
    parser.add_argument('--output', default='integrated_images',
                        help='输出目录')
    args = parser.parse_args()

    watch_dir = 'correlation_results'
    n_ch = args.n_channels

    # ── 扫描文件 ──
    print(f"扫描 {watch_dir} ...")
    all_data = discover_frames(watch_dir, args.date)
    if args.date not in all_data:
        print(f"错误: 未找到日期 {args.date} 的数据"); sys.exit(1)

    frames_by_ts = all_data[args.date]
    timestamps = list(frames_by_ts.keys())
    if args.max_frames > 0:
        timestamps = timestamps[:args.max_frames]
    n_frames = len(timestamps)
    print(f"  日期 {args.date}: {n_frames} 帧")

    # ── 通读数据 ──
    print(f"\n通读 {n_frames} 帧数据...")
    t0 = time.time()
    all_vis = []          # 互相关复数: (n_frames, n_ch)
    all_auto3 = []        # CH3 自相关: (n_frames, n_ch)
    all_auto8 = []        # CH8 自相关: (n_frames, n_ch)
    valid_idx = []        # 成功读取的帧索引

    for fi, ts in enumerate(timestamps):
        file_map = frames_by_ts[ts]
        vis = read_cross_real(file_map, n_ch)
        a3 = read_auto(file_map, 3, n_ch)
        a8 = read_auto(file_map, 8, n_ch)

        if vis is not None and a3 is not None and a8 is not None:
            all_vis.append(vis)
            all_auto3.append(a3)
            all_auto8.append(a8)
            valid_idx.append(fi)

        if (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  已读 {fi+1}/{n_frames} ({elapsed:.0f}s)")

    n_valid = len(all_vis)
    vis_matrix = np.array(all_vis)          # (n_valid, n_ch) complex
    auto3_stack = np.array(all_auto3)       # (n_valid, n_ch) float
    auto8_stack = np.array(all_auto8)       # (n_valid, n_ch) float
    del all_vis, all_auto3, all_auto8
    print(f"  有效帧: {n_valid}/{n_frames} ({time.time()-t0:.1f}s)")

    # ── 频率域 RFI 检测（逐帧） ──
    freq_rfi_masks = np.zeros((n_valid, n_ch), dtype=bool)
    total_freq_rfi = 0
    if not args.no_freq_rfi:
        print(f"\n频率域 RFI 检测 (MAD>{args.freq_thresh}, ratio>{args.freq_ratio}, grow={args.freq_grow})...")
        t1 = time.time()
        for fi in range(n_valid):
            mask, _ = detect_freq_rfi(
                auto3_stack[fi], auto8_stack[fi],
                threshold=args.freq_thresh,
                ratio_threshold=args.freq_ratio,
                grow=args.freq_grow,
                smooth_w=args.freq_smooth,
            )
            freq_rfi_masks[fi] = mask
            total_freq_rfi += mask.sum()
        print(f"  标记 {total_freq_rfi:,} 通道 ({100*total_freq_rfi/(n_valid*n_ch):.2f}%) "
              f"({time.time()-t1:.1f}s)")

    # ── 时间域 RFI 检测 ──
    time_rfi_mask = np.zeros(n_valid, dtype=bool)
    if not args.no_time_rfi:
        print(f"\n时间域 RFI 检测 (MAD>{args.time_thresh}, window={args.time_window})...")
        # 每帧功率 = 自相关最大值的中位数
        frame_powers = [np.median(np.maximum(auto3_stack[fi], auto8_stack[fi]))
                        for fi in range(n_valid)]
        bad_idx = detect_time_rfi(frame_powers,
                                  td_threshold=args.time_thresh,
                                  td_window=args.time_window)
        for bi in bad_idx:
            time_rfi_mask[bi] = True
        print(f"  标记 {len(bad_idx)}/{n_valid} 帧 ({100*len(bad_idx)/n_valid:.1f}%)")

    # ── 合并掩码 ──
    combined_mask = freq_rfi_masks.copy()  # (n_valid, n_ch)
    for fi in range(n_valid):
        if time_rfi_mask[fi]:
            combined_mask[fi, :] = True    # 整帧掩码
    n_masked = combined_mask.sum()
    print(f"\n  合并掩码: {n_masked:,}/{n_valid*n_ch} 数据点被标记 ({100*n_masked/(n_valid*n_ch):.2f}%)")

    # ── Bandpass 估计 ──
    bp3, bp8 = None, None
    if not args.no_bandpass:
        print(f"\n每天线 Bandpass 估计...")
        good_idx = np.where(~time_rfi_mask)[0]  # 排除坏帧

        bp3, bp8 = estimate_bandpass(auto3_stack, auto8_stack)
        if bp3 is not None:
            bp3_clean, n3 = clean_bandpass_spikes(bp3)
            bp8_clean, n8 = clean_bandpass_spikes(bp8)
            print(f"  CH3 bandpass 尖峰: {n3}/{n_ch} ({100*n3/n_ch:.1f}%)")
            print(f"  CH8 bandpass 尖峰: {n8}/{n_ch} ({100*n8/n_ch:.1f}%)")

            # 计算校正因子
            expected = np.sqrt(np.maximum(bp3_clean, 1e-30) * np.maximum(bp8_clean, 1e-30))
            med = np.median(expected)
            bp_factor = med / np.maximum(expected, 1e-30)  # (n_ch,)

            # 应用 bandpass 校正
            print("  应用 bandpass 校正...")
            vis_matrix = vis_matrix * bp_factor[np.newaxis, :]

    # ── 应用 RFI 掩码（置零） ──
    vis_real = vis_matrix.real.copy()
    vis_real[combined_mask] = 0.0

    # ── 重排频率轴（FFT 顺序 → 物理 100→200 MHz） ──
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, args.center_freq)
    N = n_ch
    if N % 2 == 0:
        reorder = np.concatenate([
            np.arange(N // 2 + 1, N),   # 负频率 (100→150 MHz)
            np.arange(0, N // 2 + 1)    # 正频率 (150→200 MHz)
        ]).astype(int)
    else:
        reorder = np.arange(N)
    freq_display = freq_raw[reorder]

    # 重排数据
    vis_real = vis_real[:, reorder]
    freq_rfi_masks = freq_rfi_masks[:, reorder]
    combined_mask = combined_mask[:, reorder]

    # ── 计算统计信息 ──
    # 未掩码数据的统计
    good_data = vis_real[~combined_mask]
    vmin = np.percentile(good_data, 1) if len(good_data) > 0 else -1
    vmax = np.percentile(good_data, 99) if len(good_data) > 0 else 1
    vabs = max(abs(vmin), abs(vmax))
    print(f"\n  未掩码数据范围: [{vmin:.2e}, {vmax:.2e}], 对称限: ±{vabs:.2e}")

    # ── 绘制瀑布图 ──
    print("\n绘制瀑布图...")
    fig, axes = plt.subplots(2, 1, figsize=(18, 12),
                             gridspec_kw={'height_ratios': [10, 1]},
                             constrained_layout=True)

    ax_main = axes[0]
    ax_cbar_ax = axes[1]

    # 时间轴
    time_minutes = np.arange(n_valid) * (60.0 / n_frames * n_frames / n_valid)  # 近似
    # 更好的办法：用实际时间戳
    from datetime import datetime
    t0_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
    times = []
    for idx in valid_idx:
        dt = datetime.strptime(timestamps[idx], "%Y%m%d_%H%M%S")
        times.append((dt - t0_dt).total_seconds() / 60.0)
    times = np.array(times)

    # 主图
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax_main.pcolormesh(freq_display, times, vis_real,
                            cmap='RdBu_r', norm=norm,
                            shading='nearest', rasterized=True)

    # 标记频域 RFI 区域
    if not args.no_freq_rfi:
        # 统计每通道被标记的帧占比
        freq_rfi_frac = freq_rfi_masks.sum(axis=0) / n_valid
        # 在底部画频域 RFI 密度条
        pass

    # 标记时域坏帧
    if not args.no_time_rfi and time_rfi_mask.any():
        bad_times = times[time_rfi_mask]
        ax_main.scatter([freq_display[0]] * len(bad_times), bad_times,
                        marker='<', color='lime', s=3, alpha=0.6, zorder=5)

    ax_main.set_ylabel('时间 [分钟]', fontsize=11)
    ax_main.set_xlabel('天空频率 [MHz]', fontsize=11)
    ax_main.set_xlim(freq_display.min(), freq_display.max())
    ax_main.set_ylim(times[0], times[-1])
    ax_main.ticklabel_format(useOffset=False, style='plain')

    title_parts = [f'CH3×CH8 互相关实部瀑布图 — {args.date}']
    title_parts.append(f'{n_valid} 帧, {n_ch} 通道')
    if not args.no_freq_rfi:
        title_parts.append(f'频率RFI={100*total_freq_rfi/(n_valid*n_ch):.1f}%')
    if not args.no_time_rfi:
        title_parts.append(f'时域RFI={time_rfi_mask.sum()}/{n_valid}')
    if not args.no_bandpass and bp3 is not None:
        title_parts.append('bandpass已校正')
    ax_main.set_title(' | '.join(title_parts), fontsize=12, fontweight='bold')

    # colorbar
    plt.colorbar(im, cax=ax_cbar_ax, orientation='horizontal',
                 label='Re(V) [经过 RFI 消除 + Bandpass 校正]')
    ax_cbar_ax.xaxis.set_label_position('top')

    # ── 子图2: 频域 RFI 标记密度 + 频谱 ──  
    # （如果有的话可以额外画，现在先保存主业）

    os.makedirs(args.output, exist_ok=True)
    out_path = f'{args.output}/baseline_3x8_waterfall_{args.date}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n保存: {out_path}")
    plt.close(fig)

    # ── 额外：频域 RFI 密度图 ──
    fig2, ax2 = plt.subplots(figsize=(16, 3))
    freq_rfi_frac = freq_rfi_masks.sum(axis=0) / n_valid
    ax2.fill_between(freq_display, 0, freq_rfi_frac * 100, color='red', alpha=0.3)
    ax2.plot(freq_display, freq_rfi_frac * 100, color='red', linewidth=0.5)
    ax2.set_xlabel('天空频率 [MHz]', fontsize=11)
    ax2.set_ylabel('RFI 占比 [%]', fontsize=11)
    ax2.set_title(f'频率域 RFI 密度 — 每个通道被标记的帧占比 ({args.date})', fontsize=11)
    ax2.set_xlim(freq_display.min(), freq_display.max())
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(useOffset=False, style='plain')
    out_rfi = f'{args.output}/baseline_3x8_rfi_density_{args.date}.png'
    fig2.savefig(out_rfi, dpi=150, bbox_inches='tight')
    print(f"保存: {out_rfi}")
    plt.close(fig2)

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print(f"汇总:")
    print(f"  总帧数: {n_frames} (有效 {n_valid})")
    print(f"  频率 RFI 通道: {total_freq_rfi:,} ({100*total_freq_rfi/(n_valid*n_ch):.2f}%)")
    if not args.no_time_rfi:
        print(f"  时域 RFI 帧: {time_rfi_mask.sum()} ({100*time_rfi_mask.sum()/n_valid:.1f}%)")
    print(f"  合并掩码: {n_masked:,}/{n_valid*n_ch} ({100*n_masked/(n_valid*n_ch):.2f}%)")
    print(f"  未掩码数据范围: [{good_data.min():.4e}, {good_data.max():.4e}]")
    if not args.no_bandpass and bp3 is not None:
        print(f"  Bandpass: CH3 median={np.median(bp3):.2e}, CH8 median={np.median(bp8):.2e}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
