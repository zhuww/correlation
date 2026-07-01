#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time Dirty Image Monitor — 8-Element Low-Frequency Radio Array
====================================================================
实时监控 correlation_results 目录，每当有新一批相关数据到达时：
  1. 自动读取最新的 8x8 复可见度矩阵
  2. 调用 make_dirty_image.py 的引擎生成脏图
  3. 在 GUI 中实时显示脏图，并自动保存为 PNG 序列

支持三种成像引擎：
  - CPU Cartesian: 波数坐标 (l, m)，直接 3D Fourier 积分（默认，已验证）
  - GPU Polar:     全天极坐标投影，CuPy 加速（需要 NVIDIA GPU）
  - Polar CPU:     全天极坐标投影，C/OpenMP 多核 CPU 加速（需要 cffi + C 编译器）

用法:
    python realtime_dirty_image.py [--dir correlation_results] [--interval 2.0]
                                   [--freq 150] [--fov 30] [--mode cpu|gpu|polar_cpu]

依赖: numpy, matplotlib, pandas, cupy (GPU模式可选), cffi (Polar CPU模式)
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import re
import time
import sys
import argparse
from collections import OrderedDict

# 导入 make_dirty_image 的功能函数
# NOTE: GPU 和 CFFI 相关函数按需延迟导入，确保纯 CPU 模式无需 CuPy/cffi 即可运行
from make_dirty_image import (make_dirty_image_cpu,
                               load_optimized_antennas,
                               get_polar_grid_metadata,
                               compute_uv_tracks,
                               read_frequency_range_from_data)


# ==============================================================================
# 按需导入辅助：GPU / CFFI 函数只在需要时才加载
# ==============================================================================
def _lazy_import_gpu():
    """按需导入 GPU 相关函数。如果 CuPy 不可用会给出安装提示。"""
    from make_dirty_image import make_dirty_image_GPU, gpu_available, gpu_install_hint
    return make_dirty_image_GPU, gpu_available, gpu_install_hint


def _lazy_import_polar_cpu():
    """按需导入 CFFI polar CPU 引擎相关函数。"""
    from make_dirty_image import make_dirty_image_polar_cpu, _polar_cpu_available, cffi_install_hint
    return make_dirty_image_polar_cpu, _polar_cpu_available, cffi_install_hint


# ==============================================================================
# 数据加载器：从 correlation_results 读取最新一批 CSV，构建 8x8 复可见度矩阵
# ==============================================================================
class VisibilityLoader:
    """
    仿照 phase_monitor.py 的文件读取逻辑：
      - 扫描 correlation_results 下所有 correlation_*.csv
      - 按修改时间分组，取最新一批
      - 从每对文件读取对应频率 bin 的复相关值，组装 8x8 矩阵
    """

    def __init__(self, watch_dir="correlation_results", freq_bin_idx=0):
        self.watch_dir = Path(watch_dir)
        self.freq_bin_idx = freq_bin_idx  # 默认取第 0 个频率 bin（DC），也可指定

    def get_latest_csv_files(self):
        """获取最新一批 CSV 文件（与 phase_monitor 逻辑一致）"""
        csv_files = list(self.watch_dir.glob("correlation_*.csv"))
        if not csv_files:
            return {}
        latest_time = max(f.stat().st_mtime for f in csv_files)
        latest_files = [f for f in csv_files if abs(f.stat().st_mtime - latest_time) < 1.0]
        result = {}
        for f in latest_files:
            pair_name = self._extract_pair_name(f)
            if pair_name:
                result[pair_name] = f
        return result

    def _extract_pair_name(self, filepath):
        """从文件名提取配对名称，如 CH1xCH2, CH1_AUTO"""
        name = filepath.stem
        match = re.search(r'(CH\d+(?:xCH\d+|_AUTO))', name)
        return match.group(1) if match else None

    def _get_channel_indices(self, pair_name):
        """返回 (row, col) 索引（0-based）"""
        if '_AUTO' in pair_name:
            match = re.search(r'CH(\d+)_AUTO', pair_name)
            if match:
                ch = int(match.group(1)) - 1
                return ch, ch
        elif 'x' in pair_name:
            match = re.search(r'CH(\d+)xCH(\d+)', pair_name)
            if match:
                a, b = int(match.group(1)) - 1, int(match.group(2)) - 1
                return max(a, b), min(a, b)  # 保证 row >= col（下三角）
        return None, None

    def _load_complex_value(self, filepath, freq_bin_idx):
        """从 CSV 读取指定频率 bin 的复数值"""
        try:
            df = pd.read_csv(filepath, comment='#')
            if freq_bin_idx < len(df):
                row = df.iloc[freq_bin_idx]
                return complex(row['real_part'], row['imag_part'])
        except Exception as e:
            print(f"  [WARN] Failed to read {filepath}: {e}")
        return 0 + 0j

    def build_visibility_matrix(self):
        """
        构建 8x8 复可见度矩阵。
        返回 (matrix, timestamp_str, active_channels)，若无数据则返回 (None, None, set())。
        """
        latest_files = self.get_latest_csv_files()
        if not latest_files:
            return None, None, set()

        # 提取时间戳
        timestamp = None
        for f in latest_files.values():
            m = re.search(r'correlation_(\d{8}_\d{6})', f.name)
            if m:
                timestamp = m.group(1)
                break

        # 初始化 8x8 复矩阵
        vis = np.zeros((8, 8), dtype=np.complex128)
        active_channels = set()  # 记录哪些天线有数据（0-based）

        for pair_name, filepath in latest_files.items():
            row, col = self._get_channel_indices(pair_name)
            if row is None or col is None:
                continue
            val = self._load_complex_value(filepath, self.freq_bin_idx)
            vis[row, col] = val
            if row != col:
                vis[col, row] = val.conjugate()  # 填充上三角共轭对称
            # 标记该天线有数据（非零复值才算有效）
            if val != 0 + 0j:
                active_channels.add(row)
                if row != col:
                    active_channels.add(col)

        return vis, timestamp, active_channels


# ==============================================================================
# 主 GUI 应用
# ==============================================================================
class RealtimeDirtyImageApp:
    def __init__(self, watch_dir="correlation_results", refresh_interval=2.0,
                 freq_mhz=150.0, fov_deg=30.0, grid_pts=256,
                 antennas_file="optimized_antenna_coordinates.txt",
                 save_images=True, output_dir="dirty_image_frames",
                 imaging_mode="cpu"):
        self.watch_dir = Path(watch_dir)
        self.refresh_interval = refresh_interval
        self.freq_mhz = freq_mhz
        self.fov_deg = fov_deg
        self.grid_pts = grid_pts
        self.antennas_file = antennas_file
        self.save_images = save_images
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.imaging_mode = imaging_mode  # "cpu" or "gpu"

        self.loader = VisibilityLoader(watch_dir)
        self.last_timestamp = None
        self.frame_count = 0
        self.running = True
        self.filter_rfi = True  # RFI 消除开关，默认开启
        self.im_dirty = None    # 脏图图像对象（GUI 用）

        # 预加载天线坐标用于显示
        self.antennas = load_optimized_antennas(self.antennas_file)

        # ========================
        # 创建 Tkinter 窗口
        # ========================
        self.root = tk.Tk()
        self.root.title("Real-Time Dirty Image — 8-Element Radio Array")
        self.root.geometry("1350x850")
        self.root.configure(bg='#1a1a2e')

        # 菜单栏
        self._create_menu()
        # 工具栏
        self._create_toolbar()
        # 主绘图区
        self._create_plot_area()
        # 状态栏
        self._create_statusbar()

        # 初始加载
        self.root.after(500, self.refresh)
        # 自动刷新
        self.root.after(int(self.refresh_interval * 1000), self._auto_refresh)

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Current Image", command=self._save_current_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh Now", command=self.force_refresh)
        view_menu.add_command(label="Reset Color Range", command=self._reset_clim)

        # 频率 bin 选择子菜单
        freq_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Frequency Bin", menu=freq_menu)
        for idx in [0, 10, 20, 50, 100, 200, 500, 1000]:
            freq_menu.add_command(
                label=f"Bin {idx}",
                command=lambda i=idx: self._set_freq_bin(i)
            )

    def _create_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Button(toolbar, text="Refresh", command=self.force_refresh).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Save Frame", command=self._save_current_image).pack(side=tk.LEFT, padx=3)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # 成像模式切换
        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=(5, 2))
        self.mode_var = tk.StringVar(value=self.imaging_mode.upper())
        mode_values = ["CPU", "GPU", "POLAR_CPU"]
        mode_combo = ttk.Combobox(toolbar, textvariable=self.mode_var,
                                   values=mode_values, state="readonly", width=10)
        mode_combo.pack(side=tk.LEFT, padx=2)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._switch_mode())

        # 频率设置
        ttk.Label(toolbar, text="Freq (MHz):").pack(side=tk.LEFT, padx=(10, 2))
        self.freq_var = tk.StringVar(value=str(self.freq_mhz))
        freq_entry = ttk.Entry(toolbar, textvariable=self.freq_var, width=7)
        freq_entry.pack(side=tk.LEFT, padx=2)
        freq_entry.bind("<Return>", lambda e: self._update_freq())

        # FOV 设置
        ttk.Label(toolbar, text="FOV (°):").pack(side=tk.LEFT, padx=(10, 2))
        self.fov_var = tk.StringVar(value=str(self.fov_deg))
        fov_entry = ttk.Entry(toolbar, textvariable=self.fov_var, width=6)
        fov_entry.pack(side=tk.LEFT, padx=2)
        fov_entry.bind("<Return>", lambda e: self._update_fov())

        # 网格点数
        ttk.Label(toolbar, text="Grid:").pack(side=tk.LEFT, padx=(10, 2))
        self.grid_var = tk.StringVar(value=str(self.grid_pts))
        grid_entry = ttk.Entry(toolbar, textvariable=self.grid_var, width=5)
        grid_entry.pack(side=tk.LEFT, padx=2)
        grid_entry.bind("<Return>", lambda e: self._update_grid())

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # 刷新间隔
        ttk.Label(toolbar, text="Interval (s):").pack(side=tk.LEFT, padx=(5, 2))
        self.interval_var = tk.StringVar(value=str(self.refresh_interval))
        interval_spin = ttk.Spinbox(toolbar, from_=0.5, to=10.0, increment=0.5,
                                     textvariable=self.interval_var, width=5)
        interval_spin.pack(side=tk.LEFT, padx=2)
        interval_spin.bind("<Return>", lambda e: self._update_interval())

        # 自动刷新开关
        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Auto", variable=self.auto_var).pack(side=tk.LEFT, padx=8)

        # RFI 消除开关
        self.rfi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="RFI Filter", variable=self.rfi_var,
                        command=lambda: setattr(self, 'filter_rfi', self.rfi_var.get())
                        ).pack(side=tk.LEFT, padx=3)

        # 保存开关
        self.save_var = tk.BooleanVar(value=self.save_images)
        ttk.Checkbutton(toolbar, text="Save PNG", variable=self.save_var,
                        command=lambda: setattr(self, 'save_images', self.save_var.get())
                        ).pack(side=tk.LEFT, padx=3)

    def _create_plot_area(self):
        """根据成像模式创建对应的绘图区"""
        self.fig = Figure(figsize=(14, 8), facecolor='#1a1a2e')

        if self.imaging_mode in ("gpu", "polar_cpu"):
            self._create_polar_plot_area()
        else:
            self._create_cartesian_plot_area()

        # 天线布局图 —— 两种模式共用
        self._create_antenna_subplot()

        # UV 覆盖图
        self._create_uv_coverage_subplot()

        self.fig.tight_layout(pad=2.0)

        # 嵌入 Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_cartesian_plot_area(self):
        """Cartesian (l, m) 波数坐标脏图（CPU 模式，原始工作版本）"""
        self.ax_dirty = self.fig.add_subplot(1, 3, 1, facecolor='black')
        self.ax_dirty.set_title("Dirty Image (l, m)", color='white', fontsize=12)
        self.ax_dirty.set_xlabel("l (East-West)", color='white')
        self.ax_dirty.set_ylabel("m (South-North)", color='white')
        self.ax_dirty.tick_params(colors='white', labelsize=8)
        self.ax_dirty.set_aspect('equal')

        # 初始化空图像
        l_max = np.sin(np.radians(self.fov_deg / 2))
        self.im_dirty = self.ax_dirty.imshow(
            np.zeros((self.grid_pts, self.grid_pts)),
            extent=[-l_max, l_max, -l_max, l_max],
            origin='upper', cmap='inferno', aspect='equal',
            interpolation='bilinear'
        )
        self.cbar = self.fig.colorbar(self.im_dirty, ax=self.ax_dirty,
                                       label='Intensity', fraction=0.046, pad=0.04)
        self.cbar.ax.yaxis.label.set_color('white')
        self.cbar.ax.tick_params(colors='white')

    def _create_polar_plot_area(self):
        """全天极坐标脏图（GPU 模式）"""
        self.ax_dirty = self.fig.add_subplot(1, 3, 1, facecolor='black')
        self.ax_dirty.set_title("All-Sky Dirty Image\nZenith-centred Polar Projection",
                                color='white', fontsize=12)
        self.ax_dirty.set_xlabel("", color='white')
        self.ax_dirty.set_ylabel("", color='white')
        self.ax_dirty.tick_params(colors='white', labelsize=7)
        self.ax_dirty.set_aspect('equal')

        # 绘制极坐标参考网格
        self._draw_polar_reference_grid()

        # 初始化空脏图 (pcolormesh for polar data)
        self._init_polar_dirty_image()

        # 颜色条
        self.cbar = self.fig.colorbar(
            plt.cm.ScalarMappable(cmap='inferno'),
            ax=self.ax_dirty, label='Intensity',
            fraction=0.046, pad=0.04
        )
        self.cbar.ax.yaxis.label.set_color('white')
        self.cbar.ax.tick_params(colors='white')

    def _draw_polar_reference_grid(self):
        """绘制极坐标参考网格（天顶角圆 + 方位角射线）"""
        zeta_circles = [15, 30, 45, 60, 75, 90]
        for zc in zeta_circles:
            r = zc / 90.0
            circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                                linewidth=0.5, alpha=0.6, linestyle='--')
            self.ax_dirty.add_patch(circle)
            if zc < 90:
                self.ax_dirty.annotate(f'{zc}°', (0, r), color='#6699cc',
                                       fontsize=6, ha='center', va='bottom',
                                       alpha=0.7)

        az_rays = [0, 45, 90, 135, 180, 225, 270, 315]
        az_labels = {0: 'N', 45: 'NE', 90: 'E', 135: 'SE',
                     180: 'S', 225: 'SW', 270: 'W', 315: 'NW'}
        for az in az_rays:
            theta = np.radians(90 - az)
            dx = np.cos(theta)
            dy = np.sin(theta)
            self.ax_dirty.plot([0, dx], [0, dy], color='#336699',
                               linewidth=0.4, alpha=0.5, linestyle=':')
            label = az_labels.get(az, f'{az}°')
            self.ax_dirty.annotate(label, (dx * 1.08, dy * 1.08),
                                   color='#6699cc', fontsize=7,
                                   ha='center', va='center', alpha=0.8)

        horizon = plt.Circle((0, 0), 1.0, fill=False, color='#4488cc',
                             linewidth=1.2, alpha=0.8)
        self.ax_dirty.add_patch(horizon)

        self.ax_dirty.set_xlim(-1.15, 1.15)
        self.ax_dirty.set_ylim(-1.15, 1.15)
        self.ax_dirty.set_xticks([])
        self.ax_dirty.set_yticks([])

    def _init_polar_dirty_image(self):
        """初始化或重新创建极坐标脏图渲染"""
        meta = get_polar_grid_metadata(
            grid_pts=self.grid_pts,
            n_radial=getattr(self, '_n_radial', None),
            n_azimuthal=getattr(self, '_n_azimuthal', None)
        )
        nr = meta['n_radial']
        na = meta['n_azimuthal']

        zeta_edges = np.linspace(0.0, 90.0, nr + 1) / 90.0
        az_edges = np.linspace(0.0, 360.0, na + 1)
        az_rad_edges = np.radians(az_edges)
        ZETA_E, AZ_E = np.meshgrid(zeta_edges, az_rad_edges, indexing='ij')
        X_edges = ZETA_E * np.sin(AZ_E)
        Y_edges = ZETA_E * np.cos(AZ_E)

        if self.im_dirty is not None:
            self.im_dirty.remove()

        self.im_dirty = self.ax_dirty.pcolormesh(
            X_edges, Y_edges,
            np.zeros((nr, na)),
            cmap='inferno', shading='flat',
            rasterized=True
        )
        self._nr = nr
        self._na = na

    def _create_antenna_subplot(self):
        """天线布局图（右小图）"""
        self.ax_ant = self.fig.add_subplot(1, 3, 2, facecolor='#16213e')
        self.ax_ant.set_title("Array Layout", color='white', fontsize=12)
        self.ax_ant.set_xlabel("X (m) East →", color='white')
        self.ax_ant.set_ylabel("Y (m) North →", color='white')
        self.ax_ant.tick_params(colors='white', labelsize=8)
        self.ax_ant.set_aspect('equal')
        self.ax_ant.grid(True, alpha=0.3, color='gray')

        xs, ys = self.antennas[:, 0], self.antennas[:, 1]
        self.ax_ant.scatter(xs, ys, c='cyan', s=80, edgecolors='white', linewidths=1, zorder=5)
        for i, (x, y) in enumerate(zip(xs, ys)):
            self.ax_ant.annotate(f"CH{i+1}", (x, y), textcoords="offset points",
                                  xytext=(6, 6), color='white', fontsize=8)

        for i in range(8):
            for j in range(i + 1, 8):
                self.ax_ant.plot([xs[i], xs[j]], [ys[i], ys[j]],
                                  'gray', alpha=0.25, linewidth=0.5)

        margin = 2.0
        self.ax_ant.set_xlim(xs.min() - margin, xs.max() + margin)
        self.ax_ant.set_ylim(ys.min() - margin, ys.max() + margin)

    def _create_uv_coverage_subplot(self):
        """UV 覆盖图：显示当前频率下所有 28 条基线的 (u,v) 采样"""
        self.ax_uv = self.fig.add_subplot(1, 3, 3, facecolor='#0d1b2a')
        self.ax_uv.set_xlabel("u (λ)", color='white')
        self.ax_uv.set_ylabel("v (λ)", color='white')
        self.ax_uv.tick_params(colors='white', labelsize=7)
        self.ax_uv.set_aspect('equal')
        self.ax_uv.grid(True, alpha=0.25, color='gray', linestyle=':')

        self._draw_uv_coverage()

    def _draw_uv_coverage(self):
        """计算并绘制宽带 UV 覆盖（频率相关的 uv 轨迹线）"""
        flo, fhi = read_frequency_range_from_data(
            str(self.watch_dir), center_freq_mhz=self.freq_mhz
        )
        if flo is None:
            flo = self.freq_mhz - 50.0
            fhi = self.freq_mhz + 50.0

        uv_lines, freq_samples, uv_center = compute_uv_tracks(
            self.antennas, freq_low_mhz=flo, freq_high_mhz=fhi,
            n_samples=20
        )

        for artist in getattr(self, '_uv_artists', []):
            try:
                artist.remove()
            except Exception:
                pass
        self._uv_artists = []

        n_baselines = uv_lines.shape[0]
        cmap = plt.cm.plasma

        for bl_idx in range(n_baselines):
            u_track = uv_lines[bl_idx, :, 0]
            v_track = uv_lines[bl_idx, :, 1]
            for s in range(len(freq_samples) - 1):
                t = s / (len(freq_samples) - 1)
                line = self.ax_uv.plot(
                    u_track[s:s+2], v_track[s:s+2],
                    color=cmap(0.2 + 0.6 * t), linewidth=1.2,
                    alpha=0.7, zorder=3
                )
                self._uv_artists.append(line[0])
            s_low = self.ax_uv.scatter(
                u_track[0], v_track[0], c=[cmap(0.2)],
                s=15, marker='o', edgecolors='white',
                linewidths=0.3, zorder=5, alpha=0.8
            )
            self._uv_artists.append(s_low)

        for bl_idx in range(n_baselines):
            u_track = -uv_lines[bl_idx, :, 0]
            v_track = -uv_lines[bl_idx, :, 1]
            line = self.ax_uv.plot(
                u_track, v_track,
                color='#ff8c42', linewidth=1.0, alpha=0.6,
                linestyle='--', zorder=2
            )
            self._uv_artists.append(line[0])

        s_center = self.ax_uv.scatter(
            uv_center[:, 0], uv_center[:, 1],
            c='cyan', s=12, marker='D', edgecolors='white',
            linewidths=0.3, zorder=6, alpha=0.9
        )
        self._uv_artists.append(s_center)

        self.ax_uv.axhline(y=0, color='#336699', linewidth=0.5, alpha=0.5)
        self.ax_uv.axvline(x=0, color='#336699', linewidth=0.5, alpha=0.5)

        uv_all = np.vstack([uv_lines[:, 0, :], uv_lines[:, -1, :]])
        uv_max = max(np.max(np.abs(uv_all[:, 0])),
                     np.max(np.abs(uv_all[:, 1]))) * 1.15
        if uv_max > 0:
            self.ax_uv.set_xlim(-uv_max, uv_max)
            self.ax_uv.set_ylim(-uv_max, uv_max)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=cmap(0.5), linewidth=1.5,
                   label=f'uv track ({flo:.0f}–{fhi:.0f} MHz)'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
                   markersize=6, label=f'{self.freq_mhz:.0f} MHz center'),
            Line2D([0], [0], linestyle='--', color='#ff8c42', linewidth=1,
                   label='(−u,−v) conjugate'),
        ]
        leg = self.ax_uv.legend(handles=legend_elements, loc='upper right',
                                fontsize=5.5, facecolor='#16213e',
                                edgecolor='#336699', labelcolor='white')
        self._uv_artists.append(leg)

        self.ax_uv.set_title(
            f"UV Coverage (28 baselines, {flo:.0f}–{fhi:.0f} MHz)",
            color='white', fontsize=10
        )

    def _create_statusbar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=3)

        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                  relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.peak_var = tk.StringVar(value="")
        peak_label = ttk.Label(status_frame, textvariable=self.peak_var,
                                relief=tk.SUNKEN, anchor=tk.E, width=40)
        peak_label.pack(side=tk.RIGHT, padx=(10, 0))

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    def _switch_mode(self):
        """在 CPU / GPU / POLAR_CPU 模式之间切换"""
        new_mode = self.mode_var.get().lower()
        if new_mode == self.imaging_mode:
            return
        if new_mode == "gpu":
            _, gpu_avail, gpu_hint = _lazy_import_gpu()
            if not gpu_avail():
                hint = gpu_hint() or "GPU not available."
                self.status_var.set(hint.split('\n')[0] if hint else "GPU not available!")
                self.mode_var.set(self.imaging_mode.upper())
                return
        if new_mode == "polar_cpu":
            # polar_cpu 模式内部会自动回退到纯 Python，无需检查 CFFI
            pass

        self.imaging_mode = new_mode

        # Clear existing axes
        self.fig.clf()

        # Recreate plot area
        if self.imaging_mode in ("gpu", "polar_cpu"):
            self._create_polar_plot_area()
        else:
            self._create_cartesian_plot_area()
        self._create_antenna_subplot()
        self._create_uv_coverage_subplot()
        self.fig.tight_layout(pad=2.0)

        self.im_dirty = None
        self.last_timestamp = None
        self.refresh()

    # ------------------------------------------------------------------
    # 核心逻辑
    # ------------------------------------------------------------------
    def _load_and_image(self):
        """加载最新数据并计算脏图"""
        vis, timestamp, active_channels = self.loader.build_visibility_matrix()

        if vis is None:
            self.status_var.set("Waiting for data... (no CSV files found)")
            return None, None, set()

        if timestamp == self.last_timestamp:
            self.status_var.set(f"No new data (latest: {timestamp})")
            return None, timestamp, active_channels

        self.last_timestamp = timestamp

        # ---- RFI 过滤安全检查：仅在所有 8 通道到齐时启用 ----
        effective_rfi = self.filter_rfi
        if effective_rfi and len(active_channels) < 8:
            missing = [f"CH{i+1}" for i in range(8) if i not in active_channels]
            print(f"[RFI-SAFETY] 仅 {len(active_channels)}/8 通道到齐，"
                  f"缺失: {', '.join(missing)}，暂时关闭 RFI 过滤")
            effective_rfi = False

        # 调用对应的成像引擎
        try:
            if self.imaging_mode == "gpu":
                make_dirty_image_GPU, _, _ = _lazy_import_gpu()
                dirty_img = make_dirty_image_GPU(
                    antennas_filepath=self.antennas_file,
                    visibilities=vis,
                    freq_mhz=self.freq_mhz,
                    grid_pts=self.grid_pts,
                    fov_deg=self.fov_deg,
                    apply_w_correction=True,
                    filter_rfi=effective_rfi
                )
                return dirty_img, timestamp, active_channels
            elif self.imaging_mode == "polar_cpu":
                make_dirty_image_polar_cpu, _, _ = _lazy_import_polar_cpu()
                dirty_img = make_dirty_image_polar_cpu(
                    antennas_filepath=self.antennas_file,
                    visibilities=vis,
                    freq_mhz=self.freq_mhz,
                    grid_pts=self.grid_pts,
                    fov_deg=self.fov_deg,
                    apply_w_correction=True,
                    filter_rfi=effective_rfi
                )
                return dirty_img, timestamp, active_channels
            else:
                dirty_img, l_axis, m_axis = make_dirty_image_cpu(
                    antennas_filepath=self.antennas_file,
                    visibilities=vis,
                    freq_mhz=self.freq_mhz,
                    grid_pts=self.grid_pts,
                    fov_deg=self.fov_deg,
                    apply_w_correction=True,
                    filter_rfi=effective_rfi
                )
                return (dirty_img, l_axis, m_axis), timestamp, active_channels
        except Exception as e:
            self.status_var.set(f"Image computation error: {e}")
            print(f"[ERROR] make_dirty_image failed: {e}")
            import traceback
            traceback.print_exc()
            return None, timestamp, active_channels

    def refresh(self):
        """执行一次完整的刷新：读取 → 成像 → 显示 → 保存"""
        result = self._load_and_image()

        if result is None:
            return

        dirty_data, timestamp, active_channels = result

        if dirty_data is None:
            return

        n_channels = len(active_channels)

        if self.imaging_mode in ("gpu", "polar_cpu"):
            self._refresh_polar(dirty_data, timestamp, n_channels)
        else:
            dirty_img, l_axis, m_axis = dirty_data
            self._refresh_cartesian(dirty_img, l_axis, m_axis, timestamp, n_channels)

    def _refresh_cartesian(self, dirty_img, l_axis, m_axis, timestamp, n_channels):
        """Cartesian (l, m) 波数坐标脏图显示更新"""
        l_max = l_axis[-1]
        m_min, m_max = m_axis[-1], m_axis[0]

        self.im_dirty.set_extent([-l_max, l_max, m_min, m_max])
        self.im_dirty.set_data(dirty_img)

        # 自动调整颜色范围
        valid = dirty_img[dirty_img != 0]
        if len(valid) > 0:
            vmin = float(np.percentile(valid, 2))
            vmax = float(np.percentile(valid, 98))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            self.im_dirty.set_clim(vmin, vmax)

        # 标题
        rfi_note = self._rfi_status(n_channels)
        self.ax_dirty.set_title(
            f"Dirty Image — {self.freq_mhz:.0f} MHz  FOV={self.fov_deg:.0f}°  "
            f"Ch: {n_channels}/8{rfi_note} [CPU]\n"
            f"Timestamp: {timestamp}  |  Grid: {self.grid_pts}×{self.grid_pts}",
            color='white', fontsize=11
        )

        self.canvas.draw_idle()

        # 峰值信息（用 l, m 坐标表示）
        if len(valid) > 0:
            peak_val = np.max(valid)
            peak_idx = np.unravel_index(np.argmax(dirty_img), dirty_img.shape)
            peak_l = l_axis[peak_idx[1]]
            peak_m = m_axis[peak_idx[0]]
            self.peak_var.set(
                f"Peak: {peak_val:.2f}  @ (l={peak_l:.3f}, m={peak_m:.3f})"
            )

        self._save_and_status(timestamp, n_channels, "CPU")

    def _refresh_polar(self, dirty_img, timestamp, n_channels):
        """全天极坐标脏图显示更新"""
        nr, na = dirty_img.shape

        # Rebuild pcolormesh if dimensions changed
        if self.im_dirty is None or nr != getattr(self, '_nr', 0) or na != getattr(self, '_na', 0):
            self._nr = nr
            self._na = na
            self._init_polar_dirty_image()

        self.im_dirty.set_array(dirty_img.ravel())

        valid = dirty_img[dirty_img != 0]
        if len(valid) > 0:
            vmin = float(np.percentile(valid, 2))
            vmax = float(np.percentile(valid, 98))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            self.im_dirty.set_clim(vmin, vmax)
            self.cbar.mappable.set_clim(vmin, vmax)

        rfi_note = self._rfi_status(n_channels)
        if self.imaging_mode == "polar_cpu":
            backend_note = " [POLAR_CPU]"
        else:
            try:
                _, gpu_avail, _ = _lazy_import_gpu()
                backend_note = " [GPU]" if gpu_avail() else " [CPU-fallback]"
            except Exception:
                backend_note = " [CPU-fallback]"
        self.ax_dirty.set_title(
            f"All-Sky Dirty Image — {self.freq_mhz:.0f} MHz  "
            f"Ch: {n_channels}/8{rfi_note}{backend_note}\n"
            f"Timestamp: {timestamp}  |  Grid: {nr}×{na} polar",
            color='white', fontsize=11
        )

        self.canvas.draw_idle()

        if len(valid) > 0:
            peak_val = np.max(valid)
            peak_idx = np.unravel_index(np.argmax(dirty_img), dirty_img.shape)
            meta = get_polar_grid_metadata(
                grid_pts=self.grid_pts, n_radial=nr, n_azimuthal=na
            )
            peak_zeta = meta['zeta_deg'][peak_idx[0]]
            peak_az = meta['az_deg'][peak_idx[1]]
            self.peak_var.set(
                f"Peak: {peak_val:.2f}  @ (ζ={peak_zeta:.1f}°, Az={peak_az:.1f}°)"
            )

        if self.imaging_mode == "polar_cpu":
            backend_label = "POLAR_CPU"
        else:
            try:
                _, gpu_avail, _ = _lazy_import_gpu()
                backend_label = "GPU" if gpu_avail() else "CPU"
            except Exception:
                backend_label = "CPU"
        self._save_and_status(timestamp, n_channels, backend_label)

    def _rfi_status(self, n_channels):
        if self.filter_rfi and n_channels < 8:
            return " [RFI OFF: waiting for all 8 channels]"
        elif self.filter_rfi:
            return " [RFI ON]"
        else:
            return " [RFI OFF]"

    def _save_and_status(self, timestamp, n_channels, backend):
        if self.save_images:
            self.frame_count += 1
            png_path = self.output_dir / f"dirty_{timestamp}_{self.frame_count:04d}.png"
            self.fig.savefig(png_path, dpi=100, facecolor=self.fig.get_facecolor())
            self.status_var.set(
                f"Updated: {timestamp}  |  Frame #{self.frame_count}  |  "
                f"Ch: {n_channels}/8  |  {backend}  |  Saved: {png_path.name}"
            )
        else:
            self.status_var.set(
                f"Updated: {timestamp}  |  Frame #{self.frame_count}  |  "
                f"Ch: {n_channels}/8  |  {backend}"
            )

        print(f"[{time.strftime('%H:%M:%S')}] Frame #{self.frame_count}  "
              f"timestamp={timestamp}  channels={n_channels}/8  backend={backend}")

    def force_refresh(self):
        """手动强制刷新（忽略时间戳检查）"""
        self.last_timestamp = None
        self.refresh()

    def _auto_refresh(self):
        """自动刷新循环"""
        if not self.running:
            return
        if self.auto_var.get():
            self.refresh()
        self.root.after(int(self.refresh_interval * 1000), self._auto_refresh)

    # ------------------------------------------------------------------
    # 工具栏回调
    # ------------------------------------------------------------------
    def _update_freq(self):
        try:
            self.freq_mhz = float(self.freq_var.get())
            self.last_timestamp = None
            self.refresh()
        except ValueError:
            self.status_var.set("Invalid frequency value")

    def _update_fov(self):
        try:
            self.fov_deg = float(self.fov_var.get())
            self.im_dirty = None
            self.last_timestamp = None
            self.refresh()
        except ValueError:
            self.status_var.set("Invalid FOV value")

    def _update_grid(self):
        try:
            self.grid_pts = int(self.grid_var.get())
            self.im_dirty = None
            self.last_timestamp = None
            self.refresh()
        except ValueError:
            self.status_var.set("Invalid grid size")

    def _update_interval(self):
        try:
            self.refresh_interval = float(self.interval_var.get())
        except ValueError:
            self.status_var.set("Invalid interval value")

    def _set_freq_bin(self, idx):
        self.loader.freq_bin_idx = idx
        self.last_timestamp = None
        self.status_var.set(f"Switched to frequency bin {idx}")
        self.refresh()

    def _reset_clim(self):
        self.im_dirty.autoscale()
        self.canvas.draw_idle()

    def _save_current_image(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        png_path = self.output_dir / f"dirty_manual_{timestamp}.png"
        self.fig.savefig(png_path, dpi=150, facecolor=self.fig.get_facecolor())
        self.status_var.set(f"Manually saved: {png_path.name}")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def on_closing(self):
        self.running = False
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


# ==============================================================================
# 入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Dirty Image Monitor for 8-Element Radio Array"
    )
    parser.add_argument('--dir', default='correlation_results',
                        help='Directory containing correlation CSV files')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Auto-refresh interval in seconds (default: 2.0)')
    parser.add_argument('--freq', type=float, default=150.0,
                        help='Observing frequency in MHz (default: 150)')
    parser.add_argument('--fov', type=float, default=30.0,
                        help='Field of view in degrees (default: 30)')
    parser.add_argument('--grid', type=int, default=256,
                        help='Image grid points (default: 256)')
    parser.add_argument('--antennas', default='optimized_antenna_coordinates.txt',
                        help='Antenna coordinate file')
    parser.add_argument('--output', default='dirty_image_frames',
                        help='Output directory for saved PNG frames')
    parser.add_argument('--no-save', action='store_true',
                        help='Disable automatic PNG saving')
    parser.add_argument('--bin', type=int, default=0,
                        help='Frequency bin index to use (default: 0)')
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'polar_cpu'], default='cpu',
                        help='Imaging mode: cpu (Cartesian l,m), gpu (Polar CuPy), polar_cpu (Polar C/OpenMP)')

    args = parser.parse_args()

    # Validate mode
    gpu_avail = False
    gpu_hint_fn = None
    polar_avail = False
    cffi_hint_fn = None
    if args.mode == 'gpu':
        try:
            _, gpu_avail_fn, gpu_hint_fn = _lazy_import_gpu()
            gpu_avail = gpu_avail_fn()
        except Exception:
            gpu_avail = False
        if not gpu_avail:
            print("[WARN] GPU not available, falling back to CPU mode.")
            if gpu_hint_fn is not None:
                try:
                    hint = gpu_hint_fn()
                    if hint:
                        print(hint)
                except Exception:
                    pass
            args.mode = 'cpu'
    if args.mode == 'polar_cpu':
        # polar_cpu 内部会自动回退到纯 Python，无需因为 CFFI 不可用而降级
        pass

    # 对于 GPU 模式再检测一次（确保 gpu_avail 正确）
    if not gpu_avail and args.mode != 'gpu':
        try:
            _, gpu_avail_fn, _ = _lazy_import_gpu()
            gpu_avail = gpu_avail_fn()
        except Exception:
            gpu_avail = False

    print("=" * 65)
    print("  Real-Time Dirty Image Monitor")
    print("  8-Element Low-Frequency Radio Array")
    print("=" * 65)
    print(f"  Watch dir:      {args.dir}")
    print(f"  Frequency:      {args.freq} MHz")
    print(f"  FOV:            {args.fov}°")
    print(f"  Grid:           {args.grid}×{args.grid}")
    print(f"  Refresh:        {args.interval}s")
    print(f"  Freq bin idx:   {args.bin}")
    print(f"  Save frames:    {not args.no_save}")
    print(f"  Output dir:     {args.output}")
    print(f"  Imaging mode:   {args.mode.upper()}")
    print(f"  GPU available:  {'YES' if gpu_avail else 'NO'}")
    print("=" * 65)

    Path(args.dir).mkdir(parents=True, exist_ok=True)

    app = RealtimeDirtyImageApp(
        watch_dir=args.dir,
        refresh_interval=args.interval,
        freq_mhz=args.freq,
        fov_deg=args.fov,
        grid_pts=args.grid,
        antennas_file=args.antennas,
        save_images=not args.no_save,
        output_dir=args.output,
        imaging_mode=args.mode
    )
    app.loader.freq_bin_idx = args.bin
    app.run()


if __name__ == "__main__":
    main()
