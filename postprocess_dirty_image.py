#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-Process Dirty Image Player — 8-Element Low-Frequency Radio Array
======================================================================
后处理模式：扫描 correlation_results 目录中的所有历史 frame，
逐个计算脏图，然后在 GUI 中动态循环播放。

与 realtime_dirty_image.py 的区别：
  - realtime: 监控目录等待新数据 → 实时成像
  - postprocess: 扫描全部历史 frame → 预处理所有脏图 → 循环播放

用法:
    python postprocess_dirty_image.py [--dir correlation_results]
                                      [--freq 150] [--fov 180] [--mode cpu|gpu|polar_cpu]
                                      [--bin 0] [--interval 0.5] [--grid 256]

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
from collections import defaultdict
import threading

# 导入 make_dirty_image 的功能函数
from make_dirty_image import (make_dirty_image_cpu,
                              load_optimized_antennas,
                              get_polar_grid_metadata)


# ==============================================================================
# 按需导入辅助
# ==============================================================================
def _lazy_import_gpu():
    from make_dirty_image import make_dirty_image_GPU, gpu_available, gpu_install_hint
    return make_dirty_image_GPU, gpu_available, gpu_install_hint


def _lazy_import_polar_cpu():
    from make_dirty_image import make_dirty_image_polar_cpu, _polar_cpu_available, cffi_install_hint
    return make_dirty_image_polar_cpu, _polar_cpu_available, cffi_install_hint


# ==============================================================================
# Frame 扫描器：按时间戳分组，发现所有历史 frame
# ==============================================================================
class FrameScanner:
    """扫描 correlation_results 目录，按时间戳分组所有历史 frame"""

    def __init__(self, watch_dir="correlation_results"):
        self.watch_dir = Path(watch_dir)

    def discover_frames(self):
        """
        扫描所有 correlation_*.csv 文件，按时间戳分组。
        返回: OrderedDict {timestamp_str: {pair_name: filepath}}
        """
        csv_files = sorted(self.watch_dir.glob("correlation_*.csv"))
        if not csv_files:
            return {}

        # 按时间戳分组
        frames = defaultdict(dict)
        for f in csv_files:
            m = re.search(r'correlation_(\d{8}_\d{6})', f.name)
            if not m:
                continue
            timestamp = m.group(1)

            # 提取配对名称
            name = f.stem
            pair_match = re.search(r'(CH\d+(?:xCH\d+|_AUTO))', name)
            if pair_match:
                frames[timestamp][pair_match.group(1)] = f

        # 按时间戳排序
        from collections import OrderedDict
        return OrderedDict(sorted(frames.items()))


# ==============================================================================
# 可见度矩阵构建器（与 realtime 一致）
# ==============================================================================
def build_visibility_matrix_from_files(file_map, freq_bin_idx=0):
    """从一组文件构建 8x8 复可见度矩阵"""
    vis = np.zeros((8, 8), dtype=np.complex128)
    active_channels = set()

    for pair_name, filepath in file_map.items():
        row, col = _get_channel_indices(pair_name)
        if row is None or col is None:
            continue
        val = _load_complex_value(filepath, freq_bin_idx)
        vis[row, col] = val
        if row != col:
            vis[col, row] = val.conjugate()
        if val != 0 + 0j:
            active_channels.add(row)
            if row != col:
                active_channels.add(col)

    return vis, active_channels


def _get_channel_indices(pair_name):
    """返回 (row, col) 索引（0-based）"""
    if '_AUTO' in pair_name:
        m = re.search(r'CH(\d+)_AUTO', pair_name)
        if m:
            ch = int(m.group(1)) - 1
            return ch, ch
    elif 'x' in pair_name:
        m = re.search(r'CH(\d+)xCH(\d+)', pair_name)
        if m:
            a, b = int(m.group(1)) - 1, int(m.group(2)) - 1
            return max(a, b), min(a, b)
    return None, None


def _load_complex_value(filepath, freq_bin_idx):
    """从 CSV 读取指定频率 bin 的复数值"""
    try:
        df = pd.read_csv(filepath, comment='#')
        if freq_bin_idx < len(df):
            row = df.iloc[freq_bin_idx]
            return complex(row['real_part'], row['imag_part'])
    except Exception as e:
        print(f"  [WARN] Failed to read {filepath}: {e}")
    return 0 + 0j


# ==============================================================================
# 后处理播放器 GUI
# ==============================================================================
class PostProcessPlayer:
    def __init__(self, watch_dir="correlation_results",
                 freq_mhz=150.0, fov_deg=180.0, grid_pts=256,
                 antennas_file="optimized_antenna_coordinates.txt",
                 freq_bin_idx=0, imaging_mode="cpu",
                 play_interval=0.5):
        self.watch_dir = Path(watch_dir)
        self.freq_mhz = freq_mhz
        self.fov_deg = fov_deg
        self.grid_pts = grid_pts
        self.antennas_file = antennas_file
        self.freq_bin_idx = freq_bin_idx
        self.imaging_mode = imaging_mode
        self.play_interval = play_interval  # 播放间隔（秒）

        # 帧数据
        self.frame_timestamps = []       # 有序时间戳列表
        self.frame_images = []           # 脏图列表
        self.frame_peaks = []            # 峰值信息列表
        self.frame_channels = []         # 活跃通道数列表
        self.current_frame_idx = 0
        self.total_frames = 0
        self.running = True
        self.playing = False
        self.filter_rfi = True

        # 预加载天线坐标
        self.antennas = load_optimized_antennas(self.antennas_file)

        # ========================
        # 创建 Tkinter 窗口
        # ========================
        self.root = tk.Tk()
        self.root.title("Post-Process Dirty Image Player — 8-Element Radio Array")
        self.root.geometry("1050x850")
        self.root.configure(bg='#1a1a2e')

        # 菜单栏
        self._create_menu()
        # 工具栏
        self._create_toolbar()
        # 主绘图区
        self._create_plot_area()
        # 进度条
        self._create_progress_bar()
        # 状态栏
        self._create_statusbar()

        # 启动后自动扫描并预处理
        self.root.after(300, self._auto_scan_and_preprocess)

    # ------------------------------------------------------------------
    # 界面构建
    # ------------------------------------------------------------------
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Re-scan Frames", command=self._rescan)
        file_menu.add_command(label="Save Current Image", command=self._save_current_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset Color Range", command=self._reset_clim)

    def _create_toolbar(self):
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # 播放控制
        self.play_btn = ttk.Button(toolbar, text="▶ Play", command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=3)

        ttk.Button(toolbar, text="⏮ First", command=self._goto_first).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="◀ Prev", command=self._prev_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Next ▶", command=self._next_frame).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="⏭ Last", command=self._goto_last).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # 帧号显示
        ttk.Label(toolbar, text="Frame:").pack(side=tk.LEFT, padx=(5, 2))
        self.frame_label = tk.StringVar(value="0 / 0")
        ttk.Label(toolbar, textvariable=self.frame_label, width=14).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # 成像模式切换
        ttk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=(5, 2))
        self.mode_var = tk.StringVar(value=self.imaging_mode.upper())
        mode_values = ["CPU", "GPU", "POLAR_CPU"]
        mode_combo = ttk.Combobox(toolbar, textvariable=self.mode_var,
                                   values=mode_values, state="readonly", width=10)
        mode_combo.pack(side=tk.LEFT, padx=2)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._switch_mode())

        # 播放间隔
        ttk.Label(toolbar, text="Speed (s):").pack(side=tk.LEFT, padx=(10, 2))
        self.interval_var = tk.StringVar(value=str(self.play_interval))
        interval_spin = ttk.Spinbox(toolbar, from_=0.1, to=5.0, increment=0.1,
                                     textvariable=self.interval_var, width=5)
        interval_spin.pack(side=tk.LEFT, padx=2)
        interval_spin.bind("<Return>", lambda e: self._update_interval())

        # 频率 bin
        ttk.Label(toolbar, text="Bin:").pack(side=tk.LEFT, padx=(10, 2))
        self.bin_var = tk.StringVar(value=str(self.freq_bin_idx))
        bin_spin = ttk.Spinbox(toolbar, from_=0, to=4095, increment=1,
                                textvariable=self.bin_var, width=6)
        bin_spin.pack(side=tk.LEFT, padx=2)
        bin_spin.bind("<Return>", lambda e: self._update_bin())

        # 频率
        ttk.Label(toolbar, text="Freq (MHz):").pack(side=tk.LEFT, padx=(10, 2))
        self.freq_var = tk.StringVar(value=str(self.freq_mhz))
        freq_entry = ttk.Entry(toolbar, textvariable=self.freq_var, width=7)
        freq_entry.pack(side=tk.LEFT, padx=2)

        # FOV
        ttk.Label(toolbar, text="FOV (°):").pack(side=tk.LEFT, padx=(10, 2))
        self.fov_var = tk.StringVar(value=str(self.fov_deg))
        fov_entry = ttk.Entry(toolbar, textvariable=self.fov_var, width=6)
        fov_entry.pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        # RFI 开关
        self.rfi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="RFI Filter", variable=self.rfi_var,
                        command=lambda: setattr(self, 'filter_rfi', self.rfi_var.get())
                        ).pack(side=tk.LEFT, padx=3)

        # 循环播放开关
        self.loop_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Loop", variable=self.loop_var).pack(side=tk.LEFT, padx=3)

    def _create_plot_area(self):
        self.fig = Figure(figsize=(11, 8), facecolor='#1a1a2e')

        if self.imaging_mode in ("gpu", "polar_cpu"):
            self._create_polar_plot_area()
        else:
            self._create_cartesian_plot_area()

        self._create_antenna_subplot()
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_cartesian_plot_area(self):
        self.ax_dirty = self.fig.add_subplot(1, 2, 1, facecolor='black')
        self.ax_dirty.set_title("Dirty Image (l, m)", color='white', fontsize=12)
        self.ax_dirty.set_xlabel("l (East-West)", color='white')
        self.ax_dirty.set_ylabel("m (South-North)", color='white')
        self.ax_dirty.tick_params(colors='white', labelsize=8)
        self.ax_dirty.set_aspect('equal')

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
        self.ax_dirty = self.fig.add_subplot(1, 2, 1, facecolor='black')
        self.ax_dirty.set_title("All-Sky Dirty Image\nZenith-centred Polar Projection",
                                color='white', fontsize=12)
        self.ax_dirty.set_xlabel("", color='white')
        self.ax_dirty.set_ylabel("", color='white')
        self.ax_dirty.tick_params(colors='white', labelsize=7)
        self.ax_dirty.set_aspect('equal')

        self._draw_polar_reference_grid()
        self._init_polar_dirty_image()

        self.cbar = self.fig.colorbar(
            plt.cm.ScalarMappable(cmap='inferno'),
            ax=self.ax_dirty, label='Intensity',
            fraction=0.046, pad=0.04
        )
        self.cbar.ax.yaxis.label.set_color('white')
        self.cbar.ax.tick_params(colors='white')

    def _draw_polar_reference_grid(self):
        zeta_circles = [15, 30, 45, 60, 75, 90]
        for zc in zeta_circles:
            r = zc / 90.0
            circle = plt.Circle((0, 0), r, fill=False, color='#336699',
                                linewidth=0.5, alpha=0.6, linestyle='--')
            self.ax_dirty.add_patch(circle)
            if zc < 90:
                self.ax_dirty.annotate(f'{zc}°', (0, r), color='#6699cc',
                                       fontsize=6, ha='center', va='bottom', alpha=0.7)

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
        meta = get_polar_grid_metadata(
            grid_pts=self.grid_pts,
            n_radial=getattr(self, '_nr', None),
            n_azimuthal=getattr(self, '_na', None)
        )
        nr = meta['n_radial']
        na = meta['n_azimuthal']

        zeta_edges = np.linspace(0.0, 90.0, nr + 1) / 90.0
        az_edges = np.linspace(0.0, 360.0, na + 1)
        az_rad_edges = np.radians(az_edges)
        ZETA_E, AZ_E = np.meshgrid(zeta_edges, az_rad_edges, indexing='ij')
        X_edges = ZETA_E * np.sin(AZ_E)
        Y_edges = ZETA_E * np.cos(AZ_E)

        if hasattr(self, 'im_dirty') and self.im_dirty is not None:
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
        self.ax_ant = self.fig.add_subplot(1, 2, 2, facecolor='#16213e')
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

    def _create_progress_bar(self):
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.root, variable=self.progress_var,
            maximum=100, mode='determinate'
        )
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=(0, 2))

    def _create_statusbar(self):
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=3)

        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                  relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.peak_var = tk.StringVar(value="")
        peak_label = ttk.Label(status_frame, textvariable=self.peak_var,
                                relief=tk.SUNKEN, anchor=tk.E, width=50)
        peak_label.pack(side=tk.RIGHT, padx=(10, 0))

    # ------------------------------------------------------------------
    # 扫描与预处理
    # ------------------------------------------------------------------
    def _auto_scan_and_preprocess(self):
        """自动扫描 frames 并开始预处理"""
        self.status_var.set("Scanning frames in {}...".format(self.watch_dir))
        self.root.update_idletasks()

        scanner = FrameScanner(self.watch_dir)
        frames = scanner.discover_frames()

        if not frames:
            self.status_var.set("No frames found in {}".format(self.watch_dir))
            return

        self.frame_timestamps = list(frames.keys())
        self.total_frames = len(self.frame_timestamps)
        self.status_var.set(f"Found {self.total_frames} frames. Pre-processing...")

        # 在后台线程中预处理
        thread = threading.Thread(target=self._preprocess_frames, args=(frames,), daemon=True)
        thread.start()

    def _preprocess_frames(self, frames):
        """后台线程：逐个计算所有 frame 的脏图"""
        self.frame_images = [None] * self.total_frames
        self.frame_peaks = [None] * self.total_frames
        self.frame_channels = [0] * self.total_frames

        for idx, ts in enumerate(self.frame_timestamps):
            if not self.running:
                return

            file_map = frames[ts]
            self._update_status_main(f"Processing frame {idx + 1}/{self.total_frames}: {ts}")

            # 构建可见度矩阵
            vis, active_channels = build_visibility_matrix_from_files(
                file_map, self.freq_bin_idx
            )

            # RFI 安全检查
            effective_rfi = self.filter_rfi
            if effective_rfi and len(active_channels) < 8:
                effective_rfi = False

            # 调用成像引擎
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

                # 提取峰值信息
                valid = dirty_img[dirty_img != 0]
                if len(valid) > 0:
                    peak_val = float(np.max(valid))
                    peak_idx = np.unravel_index(np.argmax(dirty_img), dirty_img.shape)
                    if self.imaging_mode in ("gpu", "polar_cpu"):
                        meta = get_polar_grid_metadata(
                            grid_pts=self.grid_pts,
                            n_radial=dirty_img.shape[0],
                            n_azimuthal=dirty_img.shape[1]
                        )
                        peak_info = (peak_val, meta['zeta_deg'][peak_idx[0]],
                                     meta['az_deg'][peak_idx[1]])
                    else:
                        peak_l = l_axis[peak_idx[1]]
                        peak_m = m_axis[peak_idx[0]]
                        peak_info = (peak_val, peak_l, peak_m)
                else:
                    peak_info = (0, 0, 0)

                self.frame_images[idx] = dirty_img
                self.frame_peaks[idx] = peak_info
                self.frame_channels[idx] = len(active_channels)

            except Exception as e:
                print(f"[ERROR] Frame {ts}: {e}")
                import traceback
                traceback.print_exc()

            # 更新进度
            progress_pct = (idx + 1) / self.total_frames * 100
            self._update_progress_main(progress_pct)

        # 预处理完成，自动开始播放
        self._update_status_main(f"Ready: {self.total_frames} frames processed. Starting playback...")
        self._update_progress_main(100)
        self.root.after(200, self._start_playback)

    def _update_status_main(self, msg):
        """从后台线程安全更新状态"""
        if self.running:
            self.root.after(0, lambda: self.status_var.set(msg))

    def _update_progress_main(self, pct):
        """从后台线程安全更新进度条"""
        if self.running:
            self.root.after(0, lambda: self.progress_var.set(pct))

    # ------------------------------------------------------------------
    # 播放控制
    # ------------------------------------------------------------------
    def _start_playback(self):
        """开始播放"""
        self.playing = True
        self.play_btn.config(text="⏸ Pause")
        self._play_loop()

    def _toggle_play(self):
        """切换播放/暂停"""
        if self.playing:
            self.playing = False
            self.play_btn.config(text="▶ Play")
        else:
            self.playing = True
            self.play_btn.config(text="⏸ Pause")
            self._play_loop()

    def _play_loop(self):
        """播放循环"""
        if not self.running or not self.playing:
            return

        if self.total_frames == 0:
            return

        # 显示当前帧
        self._show_frame(self.current_frame_idx)

        # 前进到下一帧
        self.current_frame_idx += 1
        if self.current_frame_idx >= self.total_frames:
            if self.loop_var.get():
                self.current_frame_idx = 0
            else:
                self.playing = False
                self.play_btn.config(text="▶ Play")
                self.status_var.set("Playback complete (loop disabled)")
                return

        # 安排下一帧
        interval_ms = int(self.play_interval * 1000)
        self.root.after(interval_ms, self._play_loop)

    def _show_frame(self, idx):
        """显示指定帧"""
        if idx < 0 or idx >= self.total_frames:
            return

        dirty_img = self.frame_images[idx]
        if dirty_img is None:
            return

        ts = self.frame_timestamps[idx]
        n_channels = self.frame_channels[idx]
        peak_info = self.frame_peaks[idx]

        self.frame_label.set(f"{idx + 1} / {self.total_frames}")

        if self.imaging_mode in ("gpu", "polar_cpu"):
            self._show_polar_frame(dirty_img, ts, n_channels, peak_info)
        else:
            self._show_cartesian_frame(dirty_img, ts, n_channels, peak_info)

    def _show_cartesian_frame(self, dirty_img, ts, n_channels, peak_info):
        """显示 Cartesian 帧"""
        l_max = np.sin(np.radians(self.fov_deg / 2))
        m_axis = np.linspace(l_max, -l_max, dirty_img.shape[0])

        self.im_dirty.set_extent([-l_max, l_max, -l_max, l_max])
        self.im_dirty.set_data(dirty_img)

        valid = dirty_img[dirty_img != 0]
        if len(valid) > 0:
            vmin = float(np.percentile(valid, 2))
            vmax = float(np.percentile(valid, 98))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            self.im_dirty.set_clim(vmin, vmax)

        rfi_note = self._rfi_status(n_channels)
        self.ax_dirty.set_title(
            f"Dirty Image — {self.freq_mhz:.0f} MHz  FOV={self.fov_deg:.0f}°  "
            f"Ch: {n_channels}/8{rfi_note} [CPU]\n"
            f"Timestamp: {ts}  |  Grid: {self.grid_pts}×{self.grid_pts}",
            color='white', fontsize=11
        )

        self.canvas.draw_idle()

        peak_val, peak_l, peak_m = peak_info
        self.peak_var.set(
            f"Peak: {peak_val:.2f}  @ (l={peak_l:.3f}, m={peak_m:.3f})"
        )
        self.status_var.set(
            f"Playing frame {self.current_frame_idx + 1}/{self.total_frames}  |  "
            f"Timestamp: {ts}  |  Ch: {n_channels}/8  |  [CPU]"
        )

    def _show_polar_frame(self, dirty_img, ts, n_channels, peak_info):
        """显示极坐标帧"""
        nr, na = dirty_img.shape

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
            backend_note = " [GPU]"
        self.ax_dirty.set_title(
            f"All-Sky Dirty Image — {self.freq_mhz:.0f} MHz  "
            f"Ch: {n_channels}/8{rfi_note}{backend_note}\n"
            f"Timestamp: {ts}  |  Grid: {nr}×{na} polar",
            color='white', fontsize=11
        )

        self.canvas.draw_idle()

        peak_val, peak_zeta, peak_az = peak_info
        self.peak_var.set(
            f"Peak: {peak_val:.2f}  @ (ζ={peak_zeta:.1f}°, Az={peak_az:.1f}°)"
        )
        self.status_var.set(
            f"Playing frame {self.current_frame_idx + 1}/{self.total_frames}  |  "
            f"Timestamp: {ts}  |  Ch: {n_channels}/8  |  {backend_note.strip()}"
        )

    def _rfi_status(self, n_channels):
        if self.filter_rfi and n_channels < 8:
            return " [RFI OFF: waiting for all 8 channels]"
        elif self.filter_rfi:
            return " [RFI ON]"
        else:
            return " [RFI OFF]"

    # ------------------------------------------------------------------
    # 导航控制
    # ------------------------------------------------------------------
    def _goto_first(self):
        self.current_frame_idx = 0
        self._show_frame(self.current_frame_idx)

    def _goto_last(self):
        self.current_frame_idx = self.total_frames - 1
        self._show_frame(self.current_frame_idx)

    def _prev_frame(self):
        if self.total_frames == 0:
            return
        self.current_frame_idx = (self.current_frame_idx - 1) % self.total_frames
        self._show_frame(self.current_frame_idx)

    def _next_frame(self):
        if self.total_frames == 0:
            return
        self.current_frame_idx = (self.current_frame_idx + 1) % self.total_frames
        self._show_frame(self.current_frame_idx)

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------
    def _switch_mode(self):
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

        self.imaging_mode = new_mode
        self.status_var.set(f"Switched to {new_mode.upper()} mode. Re-scanning...")

        # Clear and recreate
        self.fig.clf()
        if self.imaging_mode in ("gpu", "polar_cpu"):
            self._create_polar_plot_area()
        else:
            self._create_cartesian_plot_area()
        self._create_antenna_subplot()
        self.fig.tight_layout(pad=2.0)

        # Re-scan
        self._rescan()

    # ------------------------------------------------------------------
    # 工具栏回调
    # ------------------------------------------------------------------
    def _rescan(self):
        """重新扫描并预处理"""
        self.playing = False
        self.play_btn.config(text="▶ Play")
        self.frame_images = []
        self.frame_peaks = []
        self.frame_channels = []
        self.current_frame_idx = 0
        self.frame_label.set("0 / 0")
        self.progress_var.set(0)
        self._auto_scan_and_preprocess()

    def _update_interval(self):
        try:
            self.play_interval = float(self.interval_var.get())
        except ValueError:
            pass

    def _update_bin(self):
        try:
            self.freq_bin_idx = int(self.bin_var.get())
            self._rescan()
        except ValueError:
            pass

    def _reset_clim(self):
        if self.im_dirty is not None:
            self.im_dirty.autoscale()
            self.canvas.draw_idle()

    def _save_current_image(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("dirty_image_frames")
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / f"postprocess_{ts}.png"
        self.fig.savefig(png_path, dpi=150, facecolor=self.fig.get_facecolor())
        self.status_var.set(f"Saved: {png_path.name}")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def on_closing(self):
        self.running = False
        self.playing = False
        self.root.destroy()

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


# ==============================================================================
# 入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Post-Process Dirty Image Player — replay historical frames"
    )
    parser.add_argument('--dir', default='correlation_results',
                        help='Directory containing correlation CSV files')
    parser.add_argument('--freq', type=float, default=150.0,
                        help='Observing frequency in MHz (default: 150)')
    parser.add_argument('--fov', type=float, default=180.0,
                        help='Field of view in degrees (default: 180 for all-sky)')
    parser.add_argument('--grid', type=int, default=256,
                        help='Image grid points (default: 256)')
    parser.add_argument('--antennas', default='optimized_antenna_coordinates.txt',
                        help='Antenna coordinate file')
    parser.add_argument('--bin', type=int, default=0,
                        help='Frequency bin index to use (default: 0)')
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'polar_cpu'], default='cpu',
                        help='Imaging mode: cpu (Cartesian l,m), gpu (Polar CuPy), polar_cpu (Polar C/OpenMP)')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='Playback interval between frames in seconds (default: 0.5)')

    args = parser.parse_args()

    # Validate GPU mode
    gpu_avail = False
    if args.mode == 'gpu':
        try:
            _, gpu_avail_fn, gpu_hint_fn = _lazy_import_gpu()
            gpu_avail = gpu_avail_fn()
        except Exception:
            gpu_avail = False
        if not gpu_avail:
            print("[WARN] GPU not available, falling back to CPU mode.")
            args.mode = 'cpu'

    print("=" * 65)
    print("  Post-Process Dirty Image Player")
    print("  8-Element Low-Frequency Radio Array")
    print("=" * 65)
    print(f"  Data dir:      {args.dir}")
    print(f"  Frequency:     {args.freq} MHz")
    print(f"  FOV:           {args.fov}°")
    print(f"  Grid:          {args.grid}×{args.grid}")
    print(f"  Freq bin idx:  {args.bin}")
    print(f"  Play interval: {args.interval}s")
    print(f"  Imaging mode:  {args.mode.upper()}")
    print(f"  GPU available: {'YES' if gpu_avail else 'NO'}")
    print("=" * 65)

    Path(args.dir).mkdir(parents=True, exist_ok=True)

    player = PostProcessPlayer(
        watch_dir=args.dir,
        freq_mhz=args.freq,
        fov_deg=args.fov,
        grid_pts=args.grid,
        antennas_file=args.antennas,
        freq_bin_idx=args.bin,
        imaging_mode=args.mode,
        play_interval=args.interval
    )
    player.run()


if __name__ == "__main__":
    main()
