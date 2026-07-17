"""批量测试所有天线对的 fringe 检测 (CPCA 模式)"""
import subprocess, sys, re, itertools, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rms-norm', action='store_true', help='启用 RMS 归一化')
parser.add_argument('--n-components', type=int, default=20, help='CPCA 成分数')
parser.add_argument('--date', default='20260630', help='数据日期')
parser.add_argument('--ds-time', type=int, default=16, help='时间降采样因子')
parser.add_argument('--ds-freq', type=int, default=16, help='频率降采样因子')
parser.add_argument('--skip-pca', action='store_true', help='跳过 CPCA')
args_cli = parser.parse_args()

N_ANT = 8
pairs = list(itertools.combinations(range(1, N_ANT + 1), 2))


def run_one(ant_a, ant_b, rms):
    bl = f"{ant_a}x{ant_b}"
    extra = ["--rms-norm"] if rms else []
    if args_cli.skip_pca:
        extra.append("--skip-pca")
    cmd = [sys.executable, "-u", "-W", "ignore", "downsample_fringe.py",
           "--baseline", bl,
           "--n-components", str(args_cli.n_components),
           "--date", args_cli.date,
           "--ds-time", str(args_cli.ds_time),
           "--ds-freq", str(args_cli.ds_freq)] + extra
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    out = proc.stdout + proc.stderr

    rfi_m = re.search(r'RFI:\s*[\d,]+\s*\(([\d.]+)%\)', out)
    diag_m = re.search(r'对角功率\s*[:：]\s*([\d.]+)%', out)
    hw_m = re.search(r'横纵\s*=\s*([\d.]+)%', out)
    cpca_m = re.search(r'CPCA:.*保留\s*([\d.]+)%', out)
    fringe_m = re.search(r'周期\s*=\s*([\d.]+)\s*min', out)
    med_m = re.search(r'中位数相减.*移除\s*([\d.]+)%', out)

    return {
        'rfi': float(rfi_m.group(1)) if rfi_m else None,
        'diag': float(diag_m.group(1)) if diag_m else None,
        'hw': float(hw_m.group(1)) if hw_m else None,
        'cpca_kept': float(cpca_m.group(1)) if cpca_m else None,
        'fringe': float(fringe_m.group(1)) if fringe_m else None,
        'med_removed': float(med_m.group(1)) if med_m else None,
    }


rms_flag = args_cli.rms_norm
mode = "RMS归一化 + " if rms_flag else ""
K = args_cli.n_components
ds_info = f"ds={args_cli.ds_time}x{args_cli.ds_freq}"
pca_mode = "skip-CPCA" if args_cli.skip_pca else f"CPCA_K{K}"
print(f"[{args_cli.date}] {ds_info} {mode}{pca_mode} 批量测试")
print(f"{'基线':>10}  {'RFI%':>7}  {'对角%':>7}  {'横纵%':>7}  {'CPCA保留%':>10}  {'fringe周期':>12}")
print("-" * 82)

results = []
for a, b in pairs:
    bl = f"{a}x{b}"
    try:
        res = run_one(a, b, rms_flag)
        rfi = f"{res['rfi']:.1f}" if res['rfi'] else "?"
        diag = f"{res['diag']:.1f}" if res['diag'] else "?"
        hw = f"{res['hw']:.1f}" if res['hw'] else "?"
        cpca_k = f"{res['cpca_kept']:.1f}" if res['cpca_kept'] else "?"
        fringe = f"{res['fringe']:.1f} min" if res['fringe'] else "无"

        print(f"CH{a}xCH{b}  {rfi:>7}  {diag:>7}  {hw:>7}  {cpca_k:>10}  {fringe:>12}")
        results.append((a, b, res['diag'] or 0, res['fringe'], res['hw'] or 0,
                        res['cpca_kept'] or 0, res['med_removed'] or 0))

    except Exception as e:
        print(f"CH{a}xCH{b}  {'ERR':>7}  {str(e)[:60]}")

print("\n" + "=" * 82)
print(f"按对角功率排序 ({args_cli.date} {ds_info} {mode}{pca_mode}):")
n_fringe = 0
for a, b, diag, fringe, hw, cpca_k, med in sorted(results, key=lambda x: x[2], reverse=True):
    marker = " *** FRINGE ***" if fringe else ""
    hstr = f"hw={hw:.1f}%  med_rm={med:.1f}%"
    fstr = f"fringe={fringe:.1f} min" if fringe else ""
    print(f"  CH{a}xCH{b}:  diag={diag:.1f}%  {hstr}  {fstr}{marker}")
    if fringe:
        n_fringe += 1

print(f"\n检测到 fringe: {n_fringe}/{len(results)} 对基线")
