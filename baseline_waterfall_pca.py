"""
基线 CH3×CH8 瀑布图 — PCA 分解去除可分离横纵结构，提取天体 fringe。

原理：
  - 天文 fringe 在时间×频率平面上是斜的对角结构，不可写成 f(t)×g(f) 的外积
  - RFI 残留、bandpass 残余、增益漂移等是横纵结构，可以用外积 f(t)×g(f) 描述
  - SVD/PCA 分解 M = U Σ V^T，每项 σ_k · u_k(t) · v_k(f)^T 是纯可分离外积
  - 移除前 K 个主成分 = 移除最强的可分离信号
  - 残余 = 不可分离结构 = 天体 fringe + 噪声

处理流程：
  1. 读取数据，RFI 掩码 + bandpass 校正（同 baseline_waterfall.py）
  2. 掩码像素插值（局部中位数填充，避免零值干扰 PCA）
  3. SVD 分解
  4. 可视化：奇异值谱、前 N 个主成分的时间/频率特征向量
  5. 去除前 K 个成分，重建残余
  6. 三栏对比：原始 / 被去除的横纵结构 / PCA 清理后
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
# 复用 baseline_waterfall 的函数
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
        date_key = ts[:8]
        by_date[date_key][ts] = frames[ts]
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
        pad = np.zeros(n_channels - len(mag), dtype=np.float64)
        return np.concatenate([mag, pad])
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
    windows = sliding_window_view(mag_pad, smooth_w)
    baseline = np.median(windows, axis=1)
    residual = median_spec - baseline
    mad = np.median(np.abs(residual))
    if mad <= 0: mad = np.median(np.abs(median_spec)) * 1e-3
    if mad <= 0: mad = 1.0
    flagged = np.zeros(n_ch, dtype=bool)
    if threshold > 0:
        flagged |= median_spec > (baseline + threshold * mad)
    if ratio_threshold > 0:
        flagged &= median_spec > (baseline * ratio_threshold)
    if grow > 0 and flagged.any():
        grown = np.zeros(n_ch, dtype=bool)
        for i in range(n_ch):
            if flagged[i]:
                lo, hi = max(0, i-grow), min(n_ch, i+grow+1)
                grown[lo:hi] = True
        flagged = grown
    return flagged, baseline


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
                lo, hi = max(0,i-grow), min(n,i+grow+1)
                grown[lo:hi] = True
        flagged = grown
    bp_clean = bp.copy(); bp_clean[flagged] = baseline[flagged]
    return bp_clean, flagged.sum()


def read_sky_frequencies(file_map, n_channels=4096, center_freq_mhz=150.0):
    first_path = list(file_map.values())[0]
    df = pd.read_csv(first_path, comment='#', usecols=['frequency_hz'], nrows=n_channels)
    freq_hz = df['frequency_hz'].values.astype(np.float64)
    return center_freq_mhz + freq_hz / 1e6


# ═══════════════════════════════════════════════════════════════════
# 掩码像素插值（为 PCA 准备——PCA 不能处理空洞）
# ═══════════════════════════════════════════════════════════════════
def inpaint_masked(data, mask, max_gap=50):
    """双线性插值填充掩码区域。

    先沿时间轴（axis=0）插值 → 再沿频率轴（axis=1）插值。

    Parameters
    ----------
    data : ndarray (n_t, n_f)
        包含 NaN 或零值的数据。
    mask : ndarray (n_t, n_f) bool
        True = 被掩码的像素。
    max_gap : int
        连续掩码通道超过此值则不插值（保留 NaN，后续用 0 填充）。

    Returns
    -------
    filled : ndarray (n_t, n_f)
        插值后的数据（仍可能有 NaN 在超大空洞处）。
    """
    filled = data.astype(np.float64).copy()
    filled[mask] = np.nan

    n_t, n_f = filled.shape

    # 1. 沿频率轴插值（每行独立）
    for fi in range(n_t):
        row = filled[fi]
        nan_mask = np.isnan(row)
        if nan_mask.all() or (~nan_mask).sum() < 3:
            continue
        # 用有效像素线性插值
        valid_idx = np.where(~nan_mask)[0]
        valid_vals = row[~nan_mask]
        # 只对连续间隙 < max_gap 的区域插值
        nan_slices = []
        in_gap = False; gap_start = 0
        for j in range(n_f):
            if nan_mask[j] and not in_gap:
                gap_start = j; in_gap = True
            elif not nan_mask[j] and in_gap:
                nan_slices.append((gap_start, j))
                in_gap = False
        if in_gap: nan_slices.append((gap_start, n_f))
        for gs, ge in nan_slices:
            if ge - gs <= max_gap:
                row[gs:ge] = np.interp(np.arange(gs, ge), valid_idx, valid_vals)
            else:
                row[gs:ge] = 0.0  # 大空洞填 0

    # 2. 沿时间轴插值（每列独立）
    for fj in range(n_f):
        col = filled[:, fj]
        nan_mask = np.isnan(col)
        if nan_mask.all() or (~nan_mask).sum() < 3:
            col[nan_mask] = 0.0
            continue
        valid_idx = np.where(~nan_mask)[0]
        valid_vals = col[~nan_mask]
        nan_slices = []
        in_gap = False; gap_start = 0
        for i in range(n_t):
            if nan_mask[i] and not in_gap:
                gap_start = i; in_gap = True
            elif not nan_mask[i] and in_gap:
                nan_slices.append((gap_start, i))
                in_gap = False
        if in_gap: nan_slices.append((gap_start, n_t))
        for gs, ge in nan_slices:
            if ge - gs <= max_gap:
                col[gs:ge] = np.interp(np.arange(gs, ge), valid_idx, valid_vals)
            else:
                col[gs:ge] = 0.0
        # 剩余的
        col[np.isnan(col)] = 0.0

    return filled


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='PCA 分解去除横纵可分离结构，提取天体 fringe')
    parser.add_argument('--date', default='20260630')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--n-channels', type=int, default=4096)
    parser.add_argument('--center-freq', type=float, default=150.0)
    parser.add_argument('--n-components', type=int, default=10,
                        help='移除的前 N 个 PCA 成分 (default 10)')
    parser.add_argument('--freq-thresh', type=float, default=5.0)
    parser.add_argument('--freq-ratio', type=float, default=3.0)
    parser.add_argument('--freq-grow', type=int, default=5)
    parser.add_argument('--freq-smooth', type=int, default=101)
    parser.add_argument('--time-thresh', type=float, default=5.0)
    parser.add_argument('--time-window', type=int, default=101)
    parser.add_argument('--no-bandpass', action='store_true')
    parser.add_argument('--output', default='integrated_images')
    parser.add_argument('--pca-components-plot', type=int, default=6,
                        help='绘制前 N 个主成分的奇异向量 (default 6)')
    args = parser.parse_args()

    watch_dir = 'correlation_results'
    n_ch = args.n_channels
    K = args.n_components

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
            all_vis.append(vis); all_auto3.append(a3); all_auto8.append(a8); valid_idx.append(fi)
        if (fi+1) % 500 == 0:
            print(f"  已读 {fi+1}/{n_frames} ({time.time()-t0:.0f}s)")
    n_valid = len(all_vis)
    vis_matrix = np.array(all_vis)        # (n_valid, n_ch) complex
    auto3_s = np.array(all_auto3)         # (n_valid, n_ch)
    auto8_s = np.array(all_auto8)
    del all_vis, all_auto3, all_auto8
    print(f"  有效: {n_valid}/{n_frames} ({time.time()-t0:.1f}s)")

    # ── 2. 频率 RFI ──
    freq_rfi = np.zeros((n_valid, n_ch), dtype=bool)
    total_freq_rfi = 0
    print(f"\n频率 RFI 检测...")
    t1 = time.time()
    for fi in range(n_valid):
        mask, _ = detect_freq_rfi(auto3_s[fi], auto8_s[fi],
                                   threshold=args.freq_thresh,
                                   ratio_threshold=args.freq_ratio,
                                   grow=args.freq_grow,
                                   smooth_w=args.freq_smooth)
        freq_rfi[fi] = mask; total_freq_rfi += mask.sum()
    print(f"  {total_freq_rfi:,} 通道 ({100*total_freq_rfi/(n_valid*n_ch):.2f}%) ({time.time()-t1:.1f}s)")

    # ── 3. 时域 RFI ──
    time_rfi = np.zeros(n_valid, dtype=bool)
    print(f"\n时域 RFI 检测...")
    frame_powers = [np.median(np.maximum(auto3_s[fi], auto8_s[fi])) for fi in range(n_valid)]
    bad = detect_time_rfi(frame_powers, args.time_thresh, args.time_window)
    for bi in bad: time_rfi[bi] = True
    print(f"  {len(bad)}/{n_valid} 帧 ({100*len(bad)/n_valid:.1f}%)")

    # ── 4. Bandpass 校正 ──
    if not args.no_bandpass:
        print(f"\nBandpass 校正...")
        good_mask = ~time_rfi
        if good_mask.any():
            bp3 = np.median(auto3_s[good_mask], axis=0)
            bp8 = np.median(auto8_s[good_mask], axis=0)
            bp3_c, n3 = clean_bandpass_spikes(bp3); bp8_c, n8 = clean_bandpass_spikes(bp8)
            print(f"  CH3 尖峰: {n3}/{n_ch} ({100*n3/n_ch:.1f}%), CH8 尖峰: {n8}/{n_ch} ({100*n8/n_ch:.1f}%)")
            expected = np.sqrt(np.maximum(bp3_c,1e-30) * np.maximum(bp8_c,1e-30))
            bp_factor = np.median(expected) / np.maximum(expected, 1e-30)
            vis_matrix = vis_matrix * bp_factor[np.newaxis, :]

    # ── 5. 合并掩码 ──
    combined_mask = freq_rfi.copy()
    combined_mask[time_rfi] = True
    print(f"\n合并掩码: {combined_mask.sum():,}/{n_valid*n_ch} ({100*combined_mask.sum()/(n_valid*n_ch):.2f}%)")

    # ── 6. 提取实部 + 掩码插值 ──
    vis_real = vis_matrix.real
    print(f"\n掩码像素插值...")
    vis_filled = inpaint_masked(vis_real, combined_mask, max_gap=50)
    # 检查剩余 NaN
    nan_remain = np.isnan(vis_filled).sum()
    if nan_remain > 0:
        vis_filled = np.nan_to_num(vis_filled, nan=0.0)
        print(f"  剩余 NaN: {nan_remain}, 已置零")

    # ── 7. 去均值 ──
    vis_centered = vis_filled - vis_filled.mean(axis=0)  # 每频率通道去均值
    # 不对整图去总均值，保留 channel 间的结构信息

    # ── 8. SVD 分解 ──
    print(f"\nSVD 分解 ({n_valid}×{n_ch})...")
    t_svd = time.time()
    U, S, Vh = np.linalg.svd(vis_centered, full_matrices=False)
    # U: (n_valid, n_valid), S: (n_valid,), Vh: (n_valid, n_ch)

    print(f"  SVD 完成 ({time.time()-t_svd:.1f}s)")
    print(f"  奇异值范围: [{S[-1]:.2e}, {S[0]:.2e}]")
    print(f"  条件数 (S0/S_last): {S[0]/S[-1]:.1f}")

    # 累积解释方差
    var_total = np.sum(S**2)
    var_cumsum = np.cumsum(S**2) / var_total

    # ── 9. 移除前 K 个成分 ──
    print(f"\n移除前 {K} 个 PCA 成分...")
    residual = np.zeros_like(vis_centered)
    if K < min(n_valid, n_ch):
        for k in range(K, len(S)):
            residual += S[k] * np.outer(U[:, k], Vh[k, :])
    elif K == 0:
        residual = vis_centered.copy()

    removed = vis_centered - residual  # 被移除的横纵结构
    # 恢复均值
    residual_restored = residual + vis_filled.mean(axis=0)
    removed_restored = removed  # 去均值后的残余均值接近0

    print(f"  移除部分方差: {100*(1-np.sum(S[K:]**2)/var_total):.1f}%")
    print(f"  保留部分方差: {100*np.sum(S[K:]**2)/var_total:.1f}%")

    # ── 10. 重排频率轴 ──
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, args.center_freq)
    Nc = n_ch
    if Nc % 2 == 0:
        reorder = np.concatenate([np.arange(Nc//2+1, Nc), np.arange(0, Nc//2+1)]).astype(int)
    else:
        reorder = np.arange(Nc)
    freq_disp = freq_raw[reorder]

    def reorder_data(m):
        return m[:, reorder] if m.ndim == 2 else m[reorder]

    vis_orig_r = reorder_data(vis_real)
    vis_filled_r = reorder_data(vis_filled)
    removed_r = reorder_data(removed_restored)
    residual_r = reorder_data(residual_restored)
    combined_mask_r = reorder_data(combined_mask)

    # ── 11. 时间轴 ──
    from datetime import datetime
    t0_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
    times = np.array([(datetime.strptime(timestamps[idx], "%Y%m%d_%H%M%S") - t0_dt).total_seconds()/60
                      for idx in valid_idx])

    # ── 12. 绘图: 大图 ──
    print("\n绘制 PCA 分解图...")

    # 统一色标（基于原始数据）
    gd = vis_orig_r[~combined_mask_r]
    vmin = np.percentile(gd, 1) if len(gd) > 0 else -1
    vmax = np.percentile(gd, 99) if len(gd) > 0 else 1
    vabs = max(abs(vmin), abs(vmax))

    # 残余的色标可以更窄
    gd_res = residual_r[~combined_mask_r]
    rmin = np.percentile(gd_res, 1) if len(gd_res) > 0 else -1
    rmax = np.percentile(gd_res, 99) if len(gd_res) > 0 else 1
    rabs = max(abs(rmin), abs(rmax))

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 4, height_ratios=[10, 4, 1],
                          width_ratios=[1, 0.02, 1, 0.02],
                          hspace=0.12, wspace=0.04)

    # --- 第一行: 原始数据 vs PCA 清理后 ---
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_clean = fig.add_subplot(gs[0, 2])

    norm_orig = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im1 = ax_orig.pcolormesh(freq_disp, times, vis_orig_r, cmap='RdBu_r',
                              norm=norm_orig, shading='nearest', rasterized=True)
    ax_orig.set_title(f'原始 (RFI+bandpass 已校正)\n{n_valid} 帧 × {n_ch} 通道', fontsize=10, fontweight='bold')
    ax_orig.set_ylabel('时间 [分钟]', fontsize=10)
    ax_orig.set_xlim(freq_disp.min(), freq_disp.max())
    ax_orig.set_ylim(times[0], times[-1])
    ax_orig.ticklabel_format(useOffset=False, style='plain')
    ax_orig.grid(False)

    norm_res = TwoSlopeNorm(vmin=-rabs, vcenter=0, vmax=rabs)
    im2 = ax_clean.pcolormesh(freq_disp, times, residual_r, cmap='RdBu_r',
                               norm=norm_res, shading='nearest', rasterized=True)
    ax_clean.set_title(f'PCA 清理后 (移除前 {K} 个成分)\n保留方差 {100*np.sum(S[K:]**2)/var_total:.1f}%', 
                       fontsize=10, fontweight='bold')
    ax_clean.set_xlim(freq_disp.min(), freq_disp.max())
    ax_clean.set_ylim(times[0], times[-1])
    ax_clean.ticklabel_format(useOffset=False, style='plain')
    ax_clean.grid(False)

    # colorbars
    cax1 = fig.add_subplot(gs[0, 1])
    cb1 = plt.colorbar(im1, cax=cax1, label='原始 Re(V)')
    cax2 = fig.add_subplot(gs[0, 3])
    cb2 = plt.colorbar(im2, cax=cax2, label='PCA 残余 Re(V)')

    # --- 第二行: 被移除的横纵结构 + 奇异值谱 ---
    ax_removed = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[1, 2])

    im3 = ax_removed.pcolormesh(freq_disp, times, removed_r, cmap='RdBu_r',
                                 norm=norm_orig, shading='nearest', rasterized=True)
    removed_var = 100 * (1 - np.sum(S[K:]**2) / var_total)
    ax_removed.set_title(f'被 PCA 移除的横纵结构 ({K} 个成分, {removed_var:.1f}% 方差)',
                         fontsize=10, fontweight='bold')
    ax_removed.set_xlabel('天空频率 [MHz]', fontsize=10)
    ax_removed.set_ylabel('时间 [分钟]', fontsize=10)
    ax_removed.set_xlim(freq_disp.min(), freq_disp.max())
    ax_removed.set_ylim(times[0], times[-1])
    ax_removed.ticklabel_format(useOffset=False, style='plain')

    # 奇异值谱
    n_sv = min(200, len(S))
    ax_spec.semilogy(np.arange(1, n_sv+1), S[:n_sv], 'b.-', markersize=3)
    # 标注 K
    ax_spec.axvline(K, color='red', linestyle='--', alpha=0.7,
                    label=f'移除 K={K}')
    ax_spec.axvspan(1, K, alpha=0.1, color='red')
    ax_spec.set_xlabel('成分编号', fontsize=10)
    ax_spec.set_ylabel('奇异值 σ_k', fontsize=10)
    ax_spec.set_title(f'奇异值谱 (条件数 {S[0]/S[-1]:.0f})\n累积方差: 前{K}个={var_cumsum[K-1]*100:.1f}%',
                      fontsize=10)
    ax_spec.legend(fontsize=8)
    ax_spec.grid(True, alpha=0.3)

    # --- 第三行: 累积方差 ---
    ax_var = fig.add_subplot(gs[2, 0])
    ax_var.plot(np.arange(1, len(var_cumsum)+1), var_cumsum * 100, 'k-', linewidth=1)
    ax_var.axvline(K, color='red', linestyle='--', alpha=0.7)
    ax_var.set_xlabel('成分编号', fontsize=9)
    ax_var.set_ylabel('累积方差 [%]', fontsize=9)
    ax_var.set_xlim(0, min(200, len(var_cumsum)))
    ax_var.set_ylim(0, 100)
    ax_var.grid(True, alpha=0.3)
    ax_var.set_title('累积解释方差', fontsize=9)

    # ── 保存大图 ──
    os.makedirs(args.output, exist_ok=True)
    out_main = f'{args.output}/pca_waterfall_K{K}_{args.date}.png'
    fig.savefig(out_main, dpi=200, bbox_inches='tight')
    print(f"\n保存: {out_main}")
    plt.close(fig)

    # ── 13. 额外图: 奇异向量可视化 ──
    n_comp_plot = min(args.pca_components_plot, len(S))
    fig_sv, axes_sv = plt.subplots(n_comp_plot, 3, figsize=(18, 2.2 * n_comp_plot),
                                    constrained_layout=True)

    # 列: [时间特征 u_k(t), 频率特征 v_k(f), 外积 u⊗v]
    for k in range(n_comp_plot):
        uk = U[:, k]      # (n_valid,)
        vk = Vh[k, :]     # (n_ch,)
        vk_r = reorder_data(vk)

        # 时间特征
        ax_t = axes_sv[k, 0]
        ax_t.plot(times, uk, linewidth=0.7)
        ax_t.set_ylabel(f'PC{k+1}\nu(t)', fontsize=8)
        ax_t.grid(True, alpha=0.2)
        ax_t.set_xlim(times[0], times[-1])

        # 频率特征
        ax_f = axes_sv[k, 1]
        ax_f.plot(freq_disp, vk_r, linewidth=0.5)
        ax_f.set_ylabel(f'PC{k+1}\nv(f)', fontsize=8)
        ax_f.grid(True, alpha=0.2)
        ax_f.set_xlim(freq_disp.min(), freq_disp.max())
        ax_f.ticklabel_format(useOffset=False, style='plain')

        # 外积: σ_k · u_k ⊗ v_k 的可视化缩影
        ax_uv = axes_sv[k, 2]
        # 取前100列下采样以加速
        outer = np.outer(uk, vk_r)
        v_abs = max(abs(outer.min()), abs(outer.max())) * 0.3
        norm_uv = TwoSlopeNorm(vmin=-v_abs, vcenter=0, vmax=v_abs)
        ax_uv.pcolormesh(freq_disp[::16], times, outer[:, ::16],
                          cmap='RdBu_r', norm=norm_uv,
                          shading='nearest', rasterized=True)
        ax_uv.set_ylabel(f'PC{k+1}\nσ={S[k]:.1e}', fontsize=8)
        ax_uv.set_xlim(freq_disp.min(), freq_disp.max())
        ax_uv.set_ylim(times[0], times[-1])
        # 画一条45度斜线做参考
        t_mid = (times[0] + times[-1]) / 2
        f_mid = (freq_disp[0] + freq_disp[-1]) / 2
        # 标注"外积 = 横×纵"
        ax_uv.text(0.5, 0.95, 'u(t)·v(f)', transform=ax_uv.transAxes,
                   fontsize=7, ha='center', color='white',
                   bbox=dict(boxstyle='round', facecolor='gray', alpha=0.6))

    axes_sv[0, 0].set_title(f'时间特征向量 u_k(t)', fontsize=10, fontweight='bold')
    axes_sv[0, 1].set_title(f'频率特征向量 v_k(f)', fontsize=10, fontweight='bold')
    axes_sv[0, 2].set_title(f'外积 σ_k·u_k(t)·v_k(f)\n→ 纯横×纵结构', fontsize=10, fontweight='bold')
    axes_sv[-1, 0].set_xlabel('时间 [分钟]', fontsize=9)
    axes_sv[-1, 1].set_xlabel('天空频率 [MHz]', fontsize=9)
    axes_sv[-1, 2].set_xlabel('天空频率 [MHz]', fontsize=9)

    fig_sv.suptitle(f'前 {n_comp_plot} 个 PCA 主成分的奇异向量 — {args.date}', 
                    fontsize=12, fontweight='bold', y=1.01)

    out_sv = f'{args.output}/pca_singular_vectors_{args.date}.png'
    fig_sv.savefig(out_sv, dpi=200, bbox_inches='tight')
    print(f"保存: {out_sv}")
    plt.close(fig_sv)

    # ── 14. 聚焦图: 细看残余中的对角结构 ──
    # 在低频和高频各取一段放大
    for region_name, f_lo, f_hi in [('low_100-130MHz', 100, 130),
                                     ('high_170-200MHz', 170, 200)]:
        f_idx = (freq_disp >= f_lo) & (freq_disp <= f_hi)
        if f_idx.sum() < 5: continue

        fig_zoom, (ax_z1, ax_z2) = plt.subplots(1, 2, figsize=(16, 8),
                                                  constrained_layout=True)

        # 原始
        zoom_orig = vis_orig_r[:, f_idx]
        z_vmin = np.percentile(zoom_orig[~combined_mask_r[:, f_idx]], 2)
        z_vmax = np.percentile(zoom_orig[~combined_mask_r[:, f_idx]], 98)
        z_vabs = max(abs(z_vmin), abs(z_vmax))
        norm_z = TwoSlopeNorm(vmin=-z_vabs, vcenter=0, vmax=z_vabs)

        ax_z1.pcolormesh(freq_disp[f_idx], times, zoom_orig,
                          cmap='RdBu_r', norm=norm_z,
                          shading='nearest', rasterized=True)
        ax_z1.set_title(f'原始 — {region_name}', fontsize=11, fontweight='bold')
        ax_z1.set_ylabel('时间 [分钟]', fontsize=10)
        ax_z1.set_xlim(f_lo, f_hi); ax_z1.set_ylim(times[0], times[-1])

        # PCA 清理后
        zoom_res = residual_r[:, f_idx]
        z_rmin = np.percentile(zoom_res[~combined_mask_r[:, f_idx]], 2)
        z_rmax = np.percentile(zoom_res[~combined_mask_r[:, f_idx]], 98)
        z_rabs = max(abs(z_rmin), abs(z_rmax))
        norm_zr = TwoSlopeNorm(vmin=-z_rabs, vcenter=0, vmax=z_rabs)

        ax_z2.pcolormesh(freq_disp[f_idx], times, zoom_res,
                          cmap='RdBu_r', norm=norm_zr,
                          shading='nearest', rasterized=True)
        ax_z2.set_title(f'PCA 清理后 (移除 {K} 成分) — {region_name}', 
                        fontsize=11, fontweight='bold')
        ax_z2.set_xlim(f_lo, f_hi); ax_z2.set_ylim(times[0], times[-1])

        out_zoom = f'{args.output}/pca_zoom_{region_name}_{args.date}.png'
        fig_zoom.savefig(out_zoom, dpi=200, bbox_inches='tight')
        print(f"保存: {out_zoom}")
        plt.close(fig_zoom)

    # ── 15. 汇总 ──
    print(f"\n{'='*60}")
    print(f"PCA 分解汇总 ({args.date}):")
    print(f"  矩阵大小: {n_valid} 帧 × {n_ch} 通道 = {n_valid*n_ch:,} 像素")
    print(f"  频率 RFI: {total_freq_rfi:,} ({100*total_freq_rfi/(n_valid*n_ch):.2f}%)")
    print(f"  时域 RFI: {time_rfi.sum()} 帧 ({100*time_rfi.sum()/n_valid:.1f}%)")
    print(f"  移除成分数: {K}")
    print(f"  前 {K} 个成分方差占比: {var_cumsum[K-1]*100:.1f}%")
    print(f"  保留方差: {100*np.sum(S[K:]**2)/var_total:.1f}%")
    print(f"  残余数据范围: [{residual_r.min():.2e}, {residual_r.max():.2e}]")
    print(f"  奇异值范围: [{S[-1]:.2e}, {S[0]:.2e}]")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
