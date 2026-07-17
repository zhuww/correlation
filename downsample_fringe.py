"""
短基线 fringe 搜索：中位数+PCA 清理 → 复数降采样 → 实部找粗条纹。

核心思路：
  - 短基线的 fringe 条纹非常粗（跨几十 MHz、几十帧），全分辨率下不可见
  - 先中位数去持久 RFI，PCA 去可分离横纵结构
  - 然后大幅降采样复数 visibility（相干平均），放大粗条纹
  - 最后取实部，肉眼可见对角线方向的 fringe

管线:
  1. 加载 + RFI掩码 + bandpass校正
  2. 复数时间中位数相减 (axis=0)
  3. 插值填充掩码 → 实部+虚部分别做 PCA，移除前 K 成分
  4. 重建复数清理数据
  5. 复数降采样 (block-average，频域×时域)
  6. 取实部 → 瀑布图 + fringe 分析
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
# 数据加载
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


def read_cross(file_map, ant_a, ant_b, n_channels=4096):
    key_a, key_b = f"CH{ant_a}xCH{ant_b}", f"CH{ant_b}xCH{ant_a}"
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
def detect_freq_rfi(auto_a, auto_b, threshold=5.0, ratio_threshold=3.0, grow=5, smooth_w=101):
    n_ch = len(auto_a)
    median_spec = np.maximum(auto_a, auto_b)
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


def downsample_complex_2d(data, ds_t=1, ds_f=1):
    """
    对复数 2D 矩阵做块平均降采样（相干平均）。
    data: (n_t, n_f) complex
    ds_t: 时间方向降采样因子
    ds_f: 频率方向降采样因子
    """
    n_t, n_f = data.shape
    new_t = n_t // ds_t
    new_f = n_f // ds_f
    data_trim = data[:new_t * ds_t, :new_f * ds_f]
    reshaped = data_trim.reshape(new_t, ds_t, new_f, ds_f)
    return reshaped.mean(axis=(1, 3))


def cpca_clean(Z_complex, K):
    """
    复数 CPCA 清理：
    双去均值(复数) → SVD → 移除前 K 成分 → 恢复均值。
    np.linalg.svd 原生支持复数矩阵，保留实部/虚部间的相位关系。
    Z: (n_t, n_f) complex128
    返回: (residual, S, var_kept_pct)
    """
    n_t, n_f = Z_complex.shape
    row_mean = Z_complex.mean(axis=1, keepdims=True)
    col_mean = Z_complex.mean(axis=0, keepdims=True)
    total_mean = Z_complex.mean()
    Z_centered = Z_complex - col_mean - row_mean + total_mean

    U, S, Vh = np.linalg.svd(Z_centered, full_matrices=False)
    var_total = np.sum(S**2)

    if K < min(n_t, n_f):
        residual = np.zeros_like(Z_centered, dtype=np.complex128)
        for k in range(K, len(S)):
            residual += S[k] * np.outer(U[:, k], Vh[k, :])
    else:
        residual = Z_centered.copy()

    var_kept = 100 * np.sum(S[K:]**2) / var_total if var_total > 0 else 0
    residual_restored = residual + col_mean + row_mean - total_mean
    return residual_restored, S, var_kept


# ═══════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='短基线 fringe: 中位数+PCA → 复数降采样 → 实部找条纹')
    parser.add_argument('--date', default='20260630')
    parser.add_argument('--start-time', default=None,
                        help='开始时间 (HHMMSS), 从此时间起处理连续时间段')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--n-channels', type=int, default=4096)
    parser.add_argument('--center-freq', type=float, default=150.0)
    parser.add_argument('--n-components', type=int, default=200,
                        help='移除的 PCA 成分数')
    parser.add_argument('--ds-freq', type=int, default=16,
                        help='频率降采样因子 (默认 16, 4096→256 通道)')
    parser.add_argument('--ds-time', type=int, default=16,
                        help='时间降采样因子 (默认 16)')
    parser.add_argument('--skip-pca', action='store_true',
                        help='跳过PCA，只保留中位数减法')
    parser.add_argument('--rms-norm', action='store_true',
                        help='中位数减法后对每个频率通道用时间RMS重新归一化')
    parser.add_argument('--baseline', default='3x8',
                        help='天线对, 如 1x8 表示 CH1×CH8 (默认 3x8)')
    parser.add_argument('--output', default='integrated_images')
    args = parser.parse_args()

    watch_dir = 'correlation_results'; n_ch = args.n_channels; K = args.n_components
    ds_t = args.ds_time; ds_f = args.ds_freq

    # 解析基线
    bl_parts = args.baseline.replace('x', '×').replace('X', '×').split('×')
    ant_a, ant_b = int(bl_parts[0]), int(bl_parts[1])
    baseline_label = f"CH{ant_a}×CH{ant_b}"

    # ═══════════════════════════════════════════════════════════════
    # 第 1 步: 加载
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"  {baseline_label}: 复数降采样 fringe 搜索")
    if args.skip_pca:
        print("  (模式: 仅中位数减法，跳过PCA)")
    print("=" * 70)
    print(f"\n[1/7] 加载数据...")
    all_data = discover_frames(watch_dir, args.date)
    frames_by_ts = all_data[args.date]
    timestamps = list(frames_by_ts.keys())

    # 按开始时间过滤: 只保留 >= start-time 的时间戳 (连续时间段)
    if args.start_time:
        # timestamps 格式: YYYYMMDD_HHMMSS
        filtered = [ts for ts in timestamps if ts[9:] >= args.start_time]
        if not filtered:
            print(f"  错误: 日期 {args.date} 没有 >= {args.start_time} 的数据")
            sys.exit(1)
        first_ts = filtered[0]
        print(f"  从 {first_ts[9:]} 开始 (跳过前 {len(timestamps)-len(filtered)} 帧)")
        timestamps = filtered
    if args.max_frames > 0:
        timestamps = timestamps[:args.max_frames]
    n_frames = len(timestamps)
    print(f"  {n_frames} 帧")

    t0 = time.time()
    all_vis, all_auto_a, all_auto_b, valid_idx = [], [], [], []
    for fi, ts in enumerate(timestamps):
        fm = frames_by_ts[ts]
        vis = read_cross(fm, ant_a, ant_b, n_ch)
        a_a = read_auto(fm, ant_a, n_ch)
        a_b = read_auto(fm, ant_b, n_ch)
        if vis is not None and a_a is not None and a_b is not None:
            all_vis.append(vis); all_auto_a.append(a_a); all_auto_b.append(a_b)
            valid_idx.append(fi)
    n_valid = len(all_vis)
    vis_matrix = np.array(all_vis)  # complex (n_v, n_ch)
    auto_a_s = np.array(all_auto_a); auto_b_s = np.array(all_auto_b)
    del all_vis, all_auto_a, all_auto_b

    # fftshift: 原始数据 ch 0-2047 = 150→200 MHz, ch 2048-4095 = 100→150 MHz
    # 重排为单调升序 100→150→200 MHz，避免后续 block-average 跨频段混叠
    if n_ch % 2 == 0:
        freq_order = np.concatenate([np.arange(n_ch//2+1, n_ch), np.arange(0, n_ch//2+1)]).astype(int)
    else:
        freq_order = np.arange(n_ch)
    vis_matrix = vis_matrix[:, freq_order]
    auto_a_s = auto_a_s[:, freq_order]
    auto_b_s = auto_b_s[:, freq_order]
    print(f"  有效: {n_valid}/{n_frames}")

    # ═══════════════════════════════════════════════════════════════
    # 第 1.5 步: 检测并移除不连续数据 (重复帧 / 时间缺口)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[1.5/7] 检测数据连续性...")

    # 方法 A: 逐帧比较复数 visibility，检测完全相同的内容
    dup_from_exact = n_valid
    for i in range(n_valid - 1, 0, -1):
        if np.array_equal(vis_matrix[i], vis_matrix[i - 1]):
            dup_from_exact = i
        else:
            break

    # 方法 B: 找第一个显著时间缺口，抛弃缺口之后的所有帧
    frame_times = np.array([datetime.strptime(timestamps[valid_idx[i]], "%Y%m%d_%H%M%S").timestamp()
                            for i in range(n_valid)])
    time_diffs = np.diff(frame_times)  # seconds
    # 以中位数时间间隔为基准，超过 5 倍即认为中断
    median_dt = np.median(time_diffs)
    gap_threshold = max(median_dt * 5, 10.0)  # 至少 10 秒
    gap_idx = np.where(time_diffs > gap_threshold)[0]
    gap_from = gap_idx[0] + 1 if len(gap_idx) > 0 else n_valid  # 第一个缺口后的第一帧

    # 取最靠前的截断位置
    cut_idx = min(dup_from_exact, gap_from)
    if cut_idx < n_valid:
        removed = n_valid - cut_idx
        methods = []
        if cut_idx == dup_from_exact and dup_from_exact < n_valid:
            dur = frame_times[-1] - frame_times[cut_idx]
            methods.append(f"{removed} 帧重复内容, ~{dur/60:.1f} min")
        if cut_idx == gap_from and gap_from < n_valid:
            gap_sec = time_diffs[cut_idx - 1]
            dur = frame_times[-1] - frame_times[cut_idx - 1]
            methods.append(f"首个缺口 {gap_sec:.0f}s, 丢弃后续 {removed} 帧 (~{dur/60:.1f}min)")
        print(f"  !! 截断: {'; '.join(methods)}")
        print(f"  截断位置 idx={cut_idx}/{n_valid} ({timestamps[valid_idx[cut_idx-1]]}  T+ {gap_threshold:.0f}s  T  {timestamps[valid_idx[cut_idx]]})")

        vis_matrix = vis_matrix[:cut_idx]
        auto_a_s = auto_a_s[:cut_idx]
        auto_b_s = auto_b_s[:cut_idx]
        valid_idx = valid_idx[:cut_idx]
        n_valid = cut_idx
    else:
        print(f"  数据连续 (中位间隔 {median_dt:.1f}s, 阈值 {gap_threshold:.0f}s)")

    # ═══════════════════════════════════════════════════════════════
    # 第 2 步: RFI + bandpass
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[2/7] RFI掩码 + bandpass...")
    freq_rfi = np.zeros((n_valid, n_ch), dtype=bool)
    for fi in range(n_valid):
        freq_rfi[fi] = detect_freq_rfi(auto_a_s[fi], auto_b_s[fi])

    time_rfi = np.zeros(n_valid, dtype=bool)
    frame_powers = [np.median(np.maximum(auto_a_s[fi], auto_b_s[fi])) for fi in range(n_valid)]
    for bi in detect_time_rfi(frame_powers): time_rfi[bi] = True

    good_mask = ~time_rfi
    bp_a = np.median(auto_a_s[good_mask], axis=0) if good_mask.any() else np.median(auto_a_s, axis=0)
    bp_b = np.median(auto_b_s[good_mask], axis=0) if good_mask.any() else np.median(auto_b_s, axis=0)
    bp_a_c = clean_bandpass_spikes(bp_a); bp_b_c = clean_bandpass_spikes(bp_b)
    expected = np.sqrt(np.maximum(bp_a_c, 1e-30) * np.maximum(bp_b_c, 1e-30))
    bp_factor = np.median(expected) / np.maximum(expected, 1e-30)
    vis_corr = vis_matrix * bp_factor[np.newaxis, :]

    combined_mask = freq_rfi.copy()
    combined_mask[time_rfi] = True
    print(f"  RFI: {combined_mask.sum():,} ({100*combined_mask.sum()/(n_valid*n_ch):.1f}%)")

    # ═══════════════════════════════════════════════════════════════
    # 第 3 步: 复数时间中位数相减
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[3/7] 复数时间中位数相减...")
    complex_median = np.zeros(n_ch, dtype=np.complex128)
    for j in range(n_ch):
        vals = vis_corr[~time_rfi, j] if (~time_rfi).any() else vis_corr[:, j]
        complex_median[j] = np.median(vals.real) + 1j * np.median(vals.imag)
    vis_mediansub = vis_corr - complex_median[np.newaxis, :]

    power_before = np.mean(np.abs(vis_corr)**2)
    med_power = np.mean(np.abs(complex_median)**2)
    print(f"  移除: {100*med_power/power_before:.1f}% 功率")

    # 插值填充：仅用于填补RFI空洞，实/虚分别插值后回组为复数
    vis_real = vis_mediansub.real
    vis_imag = vis_mediansub.imag
    vis_real_filled = inpaint_masked(vis_real, combined_mask)
    vis_real_filled = np.nan_to_num(vis_real_filled, nan=0.0)
    vis_imag_filled = inpaint_masked(vis_imag, combined_mask)
    vis_imag_filled = np.nan_to_num(vis_imag_filled, nan=0.0)
    vis_complex_filled = vis_real_filled + 1j * vis_imag_filled

    # RMS 归一化(复数): 每频率通道除以|Z|时间RMS，消除高变通道主导
    if args.rms_norm:
        rms_c = np.sqrt(np.mean(np.abs(vis_complex_filled)**2, axis=0))
        rms_c = np.maximum(rms_c, np.median(rms_c[rms_c > 0]) * 0.01)
        vis_complex_filled = vis_complex_filled / rms_c[np.newaxis, :]
        print(f"  RMS归一化(复数): |Z| RMS {np.min(rms_c):.2e}~{np.max(rms_c):.2e}")

    # 频率和时间轴 (频率已重排为 100→150→200 MHz 单调升序)
    freq_raw = read_sky_frequencies(frames_by_ts[timestamps[0]], n_ch, args.center_freq)
    if n_ch % 2 == 0:
        freq_order = np.concatenate([np.arange(n_ch//2+1, n_ch), np.arange(0, n_ch//2+1)]).astype(int)
        freq_raw = freq_raw[freq_order]
    t0_dt = datetime.strptime(timestamps[0], "%Y%m%d_%H%M%S")
    times = np.array([(datetime.strptime(timestamps[idx], "%Y%m%d_%H%M%S") - t0_dt).total_seconds() / 60
                      for idx in valid_idx])

    # ═══════════════════════════════════════════════════════════════
    # 第 4 步: CPCA 清理 (复数 SVD，保留相位)
    # ═══════════════════════════════════════════════════════════════
    if args.skip_pca:
        print(f"\n[4/7] 跳过CPCA，仅中位数减后复数数据")
        vis_clean_complex = vis_complex_filled
        var_kept_cpca = 100.0
    else:
        print(f"\n[4/7] CPCA清理 (K={K})...")
        vis_clean_complex, S_cpca, var_kept_cpca = cpca_clean(vis_complex_filled, K)
        print(f"  CPCA: 移除 {100-var_kept_cpca:.1f}% 方差, 保留 {var_kept_cpca:.1f}%")

    # CPCA 标签（用于文件名和标题）
    rms_tag = 'rms_' if args.rms_norm else ''
    pca_label = '中位数 ' if args.skip_pca else f'CPCA_K={K} '
    if args.rms_norm:
        pca_label = 'RMS归一化 ' + pca_label
    pca_tag = 'noCpca_' if args.skip_pca else f'CPCA_K{K}_'
    pca_tag = rms_tag + pca_tag

    # ═══════════════════════════════════════════════════════════════
    # 第 5 步: 复数降采样 (关键步骤!)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[5/7] 复数降采样 (ds_freq={ds_f}, ds_time={ds_t})...")
    vis_ds = downsample_complex_2d(vis_clean_complex, ds_t=ds_t, ds_f=ds_f)
    n_t_ds, n_f_ds = vis_ds.shape

    # 降采样后的频率和时间轴
    freq_ds = freq_raw[:n_f_ds * ds_f].reshape(n_f_ds, ds_f).mean(axis=1)
    times_ds = times[:n_t_ds * ds_t].reshape(n_t_ds, ds_t).mean(axis=1)

    print(f"  降采样后: {n_t_ds} 帧 × {n_f_ds} 通道 (从 {n_valid}×{n_ch})")
    print(f"  频率分辨率: {freq_ds[1]-freq_ds[0]:.2f} MHz/通道")
    print(f"  时间分辨率: {times_ds[1]-times_ds[0]:.2f} 分钟/帧")

    # ═══════════════════════════════════════════════════════════════
    # 第 6 步: 取实部，分析 fringe
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[6/7] 取实部，fringe 分析...")
    vis_real_ds = vis_ds.real

    # 2D FFT on downsampled data
    win_t = np.hanning(n_t_ds); win_f = np.hanning(n_f_ds)
    win2d = win_t[:, None] * win_f[None, :]
    fft2d = np.fft.fft2(vis_real_ds * win2d)
    fft2d_s = np.fft.fftshift(fft2d)
    fft_mag = np.abs(fft2d_s)
    fft_log = np.log10(np.maximum(fft_mag, 1e-30))

    omega_t_ds = np.fft.fftshift(np.fft.fftfreq(n_t_ds, d=times_ds[1]-times_ds[0]))  # cycles/min
    omega_f_ds = np.fft.fftshift(np.fft.fftfreq(n_f_ds, d=freq_ds[1]-freq_ds[0]))   # cycles/MHz

    # 功率分区
    Nt2, Nf2 = fft_mag.shape
    cy, cx = Nt2 // 2, Nf2 // 2
    yy, xx = np.mgrid[:Nt2, :Nf2]
    yy_c, xx_c = yy - cy, xx - cx
    angle = np.degrees(np.arctan2(yy_c, xx_c))
    r = np.sqrt((yy_c/(Nt2/2))**2 + (xx_c/(Nf2/2))**2)

    dc_mask = r < 0.02
    diag_mask = ((np.abs(angle-45)<15) | (np.abs(angle-135)<15) |
                 (np.abs(angle+45)<15) | (np.abs(angle+135)<15) |
                 (np.abs(angle-225)<15) | (np.abs(angle-315)<15))
    diag_mask = diag_mask & (~dc_mask)
    axial_mask = ((np.abs(angle)<15) | (np.abs(angle-90)<15) | (np.abs(angle-180)<15) |
                  (np.abs(angle+90)<15) | (np.abs(angle-270)<15) | (np.abs(angle+180)<15))
    axial_mask = axial_mask & (~dc_mask)

    total_pwr = np.sum(fft_mag**2)
    diag_pwr = np.sum(fft_mag[diag_mask]**2)
    axial_pwr = np.sum(fft_mag[axial_mask]**2)

    print(f"  2D FFT: 对角={100*diag_pwr/total_pwr:.1f}%, 横纵={100*axial_pwr/total_pwr:.1f}%")

    # 沿主对角切片
    diag_slice = []
    diag_omega_slice = []
    for i in range(Nt2):
        j = cx + int((i - cy) * Nf2 / Nt2)
        if 0 <= j < Nf2 and abs(omega_t_ds[i]) <= 0.2:
            diag_slice.append(fft_mag[i, j])
            diag_omega_slice.append(omega_t_ds[i])
    diag_slice = np.array(diag_slice)
    diag_omega_slice = np.array(diag_omega_slice)

    # ═══════════════════════════════════════════════════════════════
    # 第 7 步: 绘图
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[7/7] 绘图...")
    os.makedirs(args.output, exist_ok=True)

    # 数据已在加载时完成频域重排 (100→150→200 MHz 单调升序)，无需再次重排

    # ── 图 1: 全分辨率 vs 降采样对比 (4 panels) ──
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
    fig1.subplots_adjust(hspace=0.3, wspace=0.25)

    # (0,0) 原始实部 (中位数减后, 全分辨率)
    ax0 = axes1[0, 0]
    vlim = np.percentile(np.abs(vis_real[~combined_mask]), 95) if (~combined_mask).any() else 1
    norm0 = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    ax0.pcolormesh(freq_raw, times, vis_real,
                   cmap='RdBu_r', norm=norm0, shading='nearest', rasterized=True)
    ax0.set_title(f'全分辨率 实部 (中位数减后)\n{n_valid}×{n_ch}, RMS={np.sqrt(np.mean(vis_real**2)):.1e}',
                  fontsize=10, fontweight='bold')
    ax0.set_xlabel('Frequency [MHz]', fontsize=9)
    ax0.set_ylabel('Time [min]', fontsize=9)

    # (0,1) CPCA清理后 实部 (全分辨率)
    ax1 = axes1[0, 1]
    cleaned_real = vis_clean_complex.real
    vlim_r = np.percentile(np.abs(cleaned_real[~combined_mask]), 95) if (~combined_mask).any() else 1
    norm1 = TwoSlopeNorm(vmin=-vlim_r, vcenter=0, vmax=vlim_r)
    ax1.pcolormesh(freq_raw, times, cleaned_real,
                   cmap='RdBu_r', norm=norm1, shading='nearest', rasterized=True)
    ax1.set_title(f'CPCA清理后 实部 (K={K})\n保留 {var_kept_cpca:.1f}% 方差',
                  fontsize=10, fontweight='bold')
    ax1.set_xlabel('Frequency [MHz]', fontsize=9)
    ax1.set_ylabel('Time [min]', fontsize=9)

    # (1,0) 降采样后 实部 ← 关键图
    ax2 = axes1[1, 0]
    vlim_ds = np.percentile(np.abs(vis_real_ds), 98)
    norm_ds = TwoSlopeNorm(vmin=-vlim_ds, vcenter=0, vmax=vlim_ds)
    im_ds = ax2.pcolormesh(freq_ds, times_ds, vis_real_ds,
                           cmap='RdBu_r', norm=norm_ds, shading='nearest')
    ax2.set_title(f'复数降采样后 实部 (ds={ds_t}×{ds_f})\n'
                  f'{n_t_ds}×{n_f_ds}, Δf={freq_ds[1]-freq_ds[0]:.1f}MHz, Δt={times_ds[1]-times_ds[0]:.1f}min',
                  fontsize=10, fontweight='bold')
    ax2.set_xlabel('Frequency [MHz]', fontsize=9)
    ax2.set_ylabel('Time [min]', fontsize=9)
    plt.colorbar(im_ds, ax=ax2, label='Re(V) downsamp', shrink=0.7)

    # (1,1) 降采样后 2D FFT
    ax3 = axes1[1, 1]
    fft_vmin = np.percentile(fft_log, 2); fft_vmax = np.percentile(fft_log, 98)
    ax3.imshow(fft_log, aspect='auto', origin='lower', cmap='viridis',
               vmin=fft_vmin, vmax=fft_vmax,
               extent=[omega_f_ds.min(), omega_f_ds.max(),
                       omega_t_ds.min(), omega_t_ds.max()])
    ax3.set_title(f'2D FFT (降采样后)\n对角功率={100*diag_pwr/total_pwr:.1f}%',
                  fontsize=10, fontweight='bold')
    ax3.set_xlabel('ω_f [cycles/MHz]', fontsize=9)
    ax3.set_ylabel('ω_t [cycles/min]', fontsize=9)
    ax3.axhline(0, color='r', alpha=0.3, lw=0.5); ax3.axvline(0, color='r', alpha=0.3, lw=0.5)
    ax3.set_xlim(-0.15, 0.15); ax3.set_ylim(-0.15, 0.15)

    fig1.suptitle(f'{baseline_label} — {pca_label}降采样 fringe (ds={ds_t}×{ds_f}) — {args.date}',
                  fontsize=13, fontweight='bold')
    out1 = f'{args.output}/downsample_fringe_{pca_tag}ds{ds_t}x{ds_f}_{args.date}_{ant_a}x{ant_b}.png'
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"  保存: {out1}")
    plt.close(fig1)

    # ── 图 2: 降采样图放大 + 对角切片 ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.subplots_adjust(wspace=0.3)

    # (0) 降采样 实部大图
    ax_a = axes2[0]
    im_a = ax_a.pcolormesh(freq_ds, times_ds, vis_real_ds,
                           cmap='RdBu_r', norm=norm_ds, shading='nearest')
    ax_a.set_title(f'Downsampled Re(V)\n{n_t_ds}×{n_f_ds}', fontsize=11, fontweight='bold')
    ax_a.set_xlabel('Freq [MHz]', fontsize=10)
    ax_a.set_ylabel('Time [min]', fontsize=10)
    plt.colorbar(im_a, ax=ax_a, shrink=0.8)

    # (1) 沿对角切片
    ax_b = axes2[1]
    ax_b.semilogy(diag_omega_slice, np.maximum(diag_slice, 1e-30), 'r-', lw=1.5, label='Diagonal 45°')
    # 也画轴向切片
    f0_idx = np.argmin(np.abs(omega_f_ds))
    axial_slice = fft_mag[:, max(0,f0_idx-2):min(Nf2,f0_idx+3)].mean(axis=1)
    ax_b.semilogy(omega_t_ds, np.maximum(axial_slice, 1e-30), 'b-', alpha=0.5, lw=1, label='ω_f≈0 (axial)')
    # 标注对角峰
    if len(diag_slice) > 0:
        peak_i = np.argmax(diag_slice)
        ax_b.axvline(diag_omega_slice[peak_i], color='orange', ls=':', alpha=0.7,
                     label=f'Peak ω_t={diag_omega_slice[peak_i]:.3f} c/min')
    ax_b.set_title('FFT slices', fontsize=11, fontweight='bold')
    ax_b.set_xlabel('ω_t [cycles/min]', fontsize=10)
    ax_b.set_ylabel('Magnitude', fontsize=10)
    ax_b.legend(fontsize=8); ax_b.grid(True, alpha=0.2)
    ax_b.set_xlim(-0.2, 0.2)

    # (2) 降采样 虚部（参考）
    ax_c = axes2[2]
    vis_imag_ds = vis_ds.imag
    vlim_im = np.percentile(np.abs(vis_imag_ds), 98)
    norm_im = TwoSlopeNorm(vmin=-vlim_im, vcenter=0, vmax=vlim_im)
    ax_c.pcolormesh(freq_ds, times_ds, vis_imag_ds,
                    cmap='RdBu_r', norm=norm_im, shading='nearest')
    ax_c.set_title(f'Downsampled Im(V)\n(for comparison)', fontsize=11, fontweight='bold')
    ax_c.set_xlabel('Freq [MHz]', fontsize=10)
    ax_c.set_ylabel('Time [min]', fontsize=10)

    fig2.suptitle(f'{baseline_label} — {pca_label}降采样分析 (ds={ds_t}×{ds_f}) — {args.date}',
                  fontsize=13, fontweight='bold')
    out2 = f'{args.output}/downsample_fringe_zoom_{pca_tag}ds{ds_t}x{ds_f}_{args.date}_{ant_a}x{ant_b}.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"  保存: {out2}")
    plt.close(fig2)

    # ═══════════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"汇总 ({args.date}):")
    print(f"  原始: {n_valid}×{n_ch}")
    print(f"  降采样: {n_t_ds}×{n_f_ds} (ds_t={ds_t}, ds_f={ds_f})")
    print(f"  ① 中位数相减: 移除 {100*med_power/power_before:.1f}% 功率")
    if args.rms_norm:
        print(f"  ② RMS归一化: 每频率通道/时间RMS")
    if not args.skip_pca:
        idx_str = "③" if args.rms_norm else "②"
        print(f"  {idx_str} CPCA K={K}: 保留 {var_kept_cpca:.1f}% 方差")
    print(f"  对角功率: {100*diag_pwr/total_pwr:.1f}%")
    if len(diag_slice) > 0:
        peak_i = np.argmax(diag_slice)
        print(f"  对角峰: ω_t={diag_omega_slice[peak_i]:.4f} cycles/min "
              f"(周期={1/abs(diag_omega_slice[peak_i]):.1f} min)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
