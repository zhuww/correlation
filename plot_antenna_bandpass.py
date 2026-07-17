#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_antenna_bandpass.py — 为每个天线绘制 Bandpass 并标注窄带 RFI 尖峰
=====================================================================
读取所有帧的自相关数据，计算每天线的 bandpass（跨帧中位数频谱），
使用滑动中位数 + MAD 检测窄带尖峰干扰，并在图中高亮标注。

用法:
  python plot_antenna_bandpass.py                          # 默认 20260630 全帧
  python plot_antenna_bandpass.py --max-frames 200         # 快速测试
  python plot_antenna_bandpass.py --date 20260604 --threshold 3.0  # 更激进检测
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Windows 中文字体修复
import matplotlib.font_manager as fm
for fname in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    prop = fm.FontProperties(fname=fname)
    if 'CJK' in prop.get_name() or 'SimHei' in prop.get_name() or 'Microsoft YaHei' in prop.get_name() or 'WenQuanYi' in prop.get_name():
        plt.rcParams['font.sans-serif'] = [prop.get_name(), 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break
else:
    # fallback: try common windows CJK fonts
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
import time
import os
import sys
import argparse
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

# ── 复用现有模块 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integrate_dirty_image import discover_frames_by_date


def read_sky_frequencies(file_map, center_freq_mhz=150.0, n_channels=4096):
    """读取天空频率数组 (MHz)。"""
    import pandas as pd
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=4100)
    freq_hz = df['frequency_hz'].values.astype(np.float64)
    if n_channels > len(freq_hz):
        n_channels = len(freq_hz)
    f_center_hz = freq_hz[:n_channels]
    freqs_mhz = center_freq_mhz + f_center_hz / 1e6
    return freqs_mhz


def detect_spikes(bandpass, threshold=5.0, ratio_threshold=3.0, grow=5, window=101):
    """检测 bandpass 中的窄带尖峰，返回 (flagged_bool, baseline, mad)。

    Returns
    -------
    flagged : ndarray (n,) bool
    baseline : ndarray (n,)
    mad : float
    """
    n = len(bandpass)
    half = window // 2

    # 向量化滑动中位数
    bp_pad = np.pad(bandpass, (half, half), mode='reflect')
    windows = sliding_window_view(bp_pad, window)
    baseline = np.median(windows, axis=1)

    residual = bandpass - baseline
    mad = np.median(np.abs(residual))
    if mad <= 0:
        mad = np.median(np.abs(bandpass)) * 1e-3
        if mad <= 0:
            mad = 1.0

    flagged = np.zeros(n, dtype=bool)
    if threshold > 0:
        flagged |= bandpass > (baseline + threshold * mad)
    if ratio_threshold > 0:
        flagged &= bandpass > (baseline * ratio_threshold)

    # 扩展标志区域
    if grow > 0 and flagged.any():
        grown = np.zeros(n, dtype=bool)
        for i in range(n):
            if flagged[i]:
                lo = max(0, i - grow)
                hi = min(n, i + grow + 1)
                grown[lo:hi] = True
        flagged = grown

    return flagged, baseline, mad


def compute_per_antenna_bandpass(frames_by_date, date_str, n_channels=4096,
                                 max_frames=None, verbose=True):
    """读取全帧自相关，计算每天线的 bandpass（跨帧中位数）。

    Returns
    -------
    per_antenna_bp : ndarray (8, n_channels)  8 天线 × 4096 通道
    freq_mhz : ndarray (n_channels,)
    """
    import pandas as pd

    if date_str not in frames_by_date:
        print(f"[ERROR] 未找到日期 {date_str}")
        return None, None

    frames = frames_by_date[date_str]
    timestamps = list(frames.keys())
    if max_frames and max_frames < len(timestamps):
        timestamps = timestamps[-max_frames:]
        print(f"  限制为最后 {max_frames} 帧")

    n_antennas = 8
    # 累积: 每个天线 (n_frames, n_channels)
    accum = [[] for _ in range(n_antennas)]

    # 读频率
    first_ts = timestamps[0]
    freq_mhz = read_sky_frequencies(frames[first_ts], n_channels=n_channels)

    print(f"  读取 {len(timestamps)} 帧自相关数据...")
    t0 = time.time()
    bad_frames = 0

    for fi, ts in enumerate(timestamps):
        file_map = frames[ts]
        for ch in range(1, n_antennas + 1):
            key = f"CH{ch}_AUTO"
            if key not in file_map:
                accum[ch - 1].append(np.zeros(n_channels, dtype=np.float64))
                continue
            try:
                df = pd.read_csv(file_map[key], comment='#', usecols=['magnitude'])
                mag = df['magnitude'].values.astype(np.float64)
                if len(mag) >= n_channels:
                    mag = mag[:n_channels]
                else:
                    pad = np.zeros(n_channels - len(mag), dtype=np.float64)
                    mag = np.concatenate([mag, pad])
                accum[ch - 1].append(mag)
            except Exception:
                accum[ch - 1].append(np.zeros(n_channels, dtype=np.float64))
                bad_frames += 1

        if verbose and (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (fi + 1) / elapsed
            eta = (len(timestamps) - fi - 1) / fps
            print(f"    [{fi+1}/{len(timestamps)}] {elapsed:.0f}s, {fps:.1f} fps, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"  读取完成: {elapsed:.1f}s (读取出错 {bad_frames} 次)")

    # 跨帧取中位数
    per_antenna_bp = np.zeros((n_antennas, n_channels), dtype=np.float64)
    for ant in range(n_antennas):
        stacked = np.array(accum[ant])
        per_antenna_bp[ant] = np.median(stacked, axis=0)

    return per_antenna_bp, freq_mhz


def plot_antenna_bandpass(per_antenna_bp, freq_mhz, date_str,
                          threshold=5.0, ratio_threshold=3.0,
                          grow=5, window=101, output_dir='integrated_images'):
    """为每个天线绘制 bandpass，标注尖峰干扰。频率轴重排为 100~200 MHz。"""
    n_ant = per_antenna_bp.shape[0]
    n_ch = per_antenna_bp.shape[1]

    # 重排频率：FFT 后半负频率 → 前半正频率 → 显示 100~200 MHz
    N = n_ch
    if N % 2 == 0:
        reorder_idx = np.concatenate([
            np.arange(N // 2 + 1, N),   # 负频率部分 (100→150 MHz)
            np.arange(0, N // 2 + 1)    # 正频率部分 (150→200 MHz)
        ]).astype(int)
    else:
        reorder_idx = np.arange(N)
    freq_display = freq_mhz[reorder_idx]

    fig, axes = plt.subplots(n_ant, 1, figsize=(16, 3.2 * n_ant),
                             sharex=True, constrained_layout=True)

    global_median = np.median(per_antenna_bp)
    colors = plt.cm.tab10(np.linspace(0, 1, n_ant))
    spike_summary = []

    for ant in range(n_ant):
        ax = axes[ant]
        bp = per_antenna_bp[ant]
        bp_display = bp[reorder_idx]
        bp_db = 10 * np.log10(np.maximum(bp_display, 1e-30))

        # 检测尖峰（在原始顺序上检测，然后重排）
        flagged_orig, baseline_orig, mad = detect_spikes(
            bp, threshold=threshold, ratio_threshold=ratio_threshold,
            grow=grow, window=window
        )
        flagged = flagged_orig[reorder_idx]
        baseline = baseline_orig[reorder_idx]
        baseline_db = 10 * np.log10(np.maximum(baseline, 1e-30))
        n_spikes = flagged.sum()
        spike_pct = 100 * n_spikes / n_ch
        spike_summary.append((ant + 1, n_spikes, spike_pct))

        bp_med = np.median(bp)
        bp_range = (bp.min() / bp_med, bp.max() / bp_med)

        # ── 绘制 ──
        ax.plot(freq_display, bp_db, color='lightgray', linewidth=0.4, alpha=0.7,
                label='原始 bandpass')
        ax.plot(freq_display, baseline_db, color='steelblue', linewidth=1.0,
                alpha=0.9, label=f'平滑基线 (窗口={window})')

        if flagged.any():
            ax.scatter(freq_display[flagged], bp_db[flagged],
                       color='red', s=4, alpha=0.7, zorder=5,
                       label=f'尖峰 ({n_spikes} ch, {spike_pct:.1f}%)')

            flagged_regions = []
            in_region = False
            start = 0
            for i in range(n_ch):
                if flagged[i] and not in_region:
                    start = i
                    in_region = True
                elif not flagged[i] and in_region:
                    flagged_regions.append((start, i - 1))
                    in_region = False
            if in_region:
                flagged_regions.append((start, n_ch - 1))

            for (s, e) in flagged_regions:
                f_start = freq_display[max(0, s - 1)]
                f_end = freq_display[min(n_ch - 1, e + 1)]
                ax.axvspan(f_start, f_end, color='red', alpha=0.08, zorder=1)

        ax.set_ylabel(f'天线 {ant+1}\n[dB]', fontsize=9, color=colors[ant])
        ax.set_ylim(baseline_db.min() - 5, bp_db.max() + 3)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)
        ax.ticklabel_format(useOffset=False, style='plain')

        stats_text = (f'median={bp_med:.2e}  |  范围 {bp_range[0]:.2f}~{bp_range[1]:.2f}x median'
                      f'  |  MAD={mad:.2e}')
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                fontsize=6.5, color='dimgray', va='bottom')

    axes[-1].set_xlabel('天空频率 [MHz]', fontsize=10)
    axes[-1].set_xlim(freq_display.min(), freq_display.max())
    axes[-1].ticklabel_format(useOffset=False, style='plain')

    fig.suptitle(f'每天线 Bandpass 与窄带 RFI 尖峰检测  —  {date_str}\n'
                 f'threshold={threshold}×MAD, ratio>{ratio_threshold}×baseline, grow=±{grow}ch',
                 fontsize=12, fontweight='bold', y=1.01)

    os.makedirs(output_dir, exist_ok=True)
    out_path = f'{output_dir}/antenna_bandpass_{date_str}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"\n  保存: {out_path}")
    plt.close(fig)

    print(f"\n  {'天线':^6} {'尖峰通道':>10} {'占比':>8}")
    print(f"  {'-'*26}")
    total = 0
    for ant, n, pct in spike_summary:
        print(f"  CH{ant:>2}   {n:>8}   {pct:>6.1f}%")
        total += n
    print(f"  {'-'*26}")
    print(f"  总计   {total:>8}   {100*total/(n_ant*n_ch):>6.1f}%")

    return spike_summary


def main():
    parser = argparse.ArgumentParser(description='绘制每天线 Bandpass 并标注窄带尖峰')
    parser.add_argument('--watch-dir', default='correlation_results',
                        help='CSV 数据目录')
    parser.add_argument('--date', default='20260630',
                        help='数据日期 YYYYMMDD')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最大读取帧数 (默认全部)')
    parser.add_argument('--nch', type=int, default=4096,
                        help='通道数')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='MAD 倍数阈值 (默认 5.0, 越小越激进)')
    parser.add_argument('--ratio', type=float, default=3.0,
                        help='相对基线倍数阈值 (默认 3.0)')
    parser.add_argument('--grow', type=int, default=5,
                        help='尖峰扩展通道数 (默认 5)')
    parser.add_argument('--window', type=int, default=101,
                        help='滑动中位数窗口 (默认 101)')
    parser.add_argument('--output', default='integrated_images',
                        help='输出目录')
    args = parser.parse_args()

    print("=" * 60)
    print(f"  每天线 Bandpass 尖峰检测")
    print(f"  日期: {args.date}, 通道: {args.nch}")
    if args.max_frames:
        print(f"  帧数限制: {args.max_frames}")
    print(f"  检测参数: MAD阈值={args.threshold}, 倍数>{args.ratio}×, "
          f"扩展=±{args.grow}ch, 窗口={args.window}")
    print("=" * 60)

    # ── 发现帧 ──
    print(f"\n  扫描 {args.watch_dir} ...")
    t0 = time.time()
    frames_by_date = discover_frames_by_date(args.watch_dir, args.date)
    if not frames_by_date:
        print(f"  [ERROR] 未找到 {args.date} 的数据")
        return 1
    n_frames = len(frames_by_date.get(args.date, {}))
    print(f"  发现 {args.date}: {n_frames} 帧 ({time.time()-t0:.1f}s)")

    # ── 计算每天线 bandpass ──
    per_antenna_bp, freq_mhz = compute_per_antenna_bandpass(
        frames_by_date, args.date,
        n_channels=args.nch, max_frames=args.max_frames
    )
    if per_antenna_bp is None:
        return 1

    # ── 绘图 ──
    plot_antenna_bandpass(
        per_antenna_bp, freq_mhz, args.date,
        threshold=args.threshold, ratio_threshold=args.ratio,
        grow=args.grow, window=args.window,
        output_dir=args.output
    )

    print("\n完成！")
    return 0


if __name__ == '__main__':
    sys.exit(main())
