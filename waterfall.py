#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CH1xCH2 Cross-correlation Spectrogram (Rolling Waterfall)
滚动瀑布图：最新数据始终在最下方，旧数据向上滚动
显示实部、虚部、相位三张图，纵向排列
使用同一批次数据（相同时间戳）
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re
from collections import deque


class RollingWaterfall:
    def __init__(self):
        self.watch_dir = Path("./correlation_results")
        self.max_rows = 10  # 固定显示10个时间点
        # 使用deque存储历史数据，左侧为旧数据，右侧为新数据
        self.data_history = deque(maxlen=self.max_rows)
        self.colorbar_real = None
        self.colorbar_imag = None
        self.colorbar_phase = None
        self.last_processed_timestamp = None  # 记录最后处理的时间戳
        
        self.root = tk.Tk()
        self.root.title("CH1xCH2 Spectrogram (Rolling Waterfall)")
        self.root.geometry("1200x900")
        
        # 创建图形 - 三个子图纵向排列
        self.fig = Figure(figsize=(12, 9), facecolor='black')
        
        # 实部瀑布图
        self.ax_real = self.fig.add_subplot(311, facecolor='black')
        self.ax_real.set_ylabel('Time', color='white')
        self.ax_real.set_title('Real Part (I) - Waterfall', color='white', fontsize=10)
        self.ax_real.tick_params(colors='white')
        self.ax_real.set_xlim(-50, 50)
        
        # 虚部瀑布图
        self.ax_imag = self.fig.add_subplot(312, facecolor='black')
        self.ax_imag.set_ylabel('Time', color='white')
        self.ax_imag.set_title('Imag Part (Q) - Waterfall', color='white', fontsize=10)
        self.ax_imag.tick_params(colors='white')
        self.ax_imag.set_xlim(-50, 50)
        
        # 相位瀑布图
        self.ax_phase = self.fig.add_subplot(313, facecolor='black')
        self.ax_phase.set_xlabel('Frequency (MHz)', color='white')
        self.ax_phase.set_ylabel('Time', color='white')
        self.ax_phase.set_title('Phase (deg) - Waterfall', color='white', fontsize=10)
        self.ax_phase.tick_params(colors='white')
        self.ax_phase.set_xlim(-50, 50)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 控制栏
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Refresh", command=self.force_refresh).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, text="Auto refresh every 2s").pack(side=tk.LEFT, padx=20)
        
        # 颜色映射选择
        ttk.Label(control_frame, text="Colormap:").pack(side=tk.LEFT, padx=(20, 5))
        self.cmap_var = tk.StringVar(value="jet")
        cmap_combo = ttk.Combobox(control_frame, textvariable=self.cmap_var,
                                   values=["jet", "viridis", "plasma", "inferno", "magma", "coolwarm"],
                                   width=10, state="readonly")
        cmap_combo.pack(side=tk.LEFT, padx=5)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self.change_colormap())
        
        self.status_var = tk.StringVar(value="Waiting for data...")
        self.status_bar = ttk.Label(control_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)
        
        # 启动自动刷新
        self.refresh()
        self.root.after(2000, self.auto_refresh)
        self.root.mainloop()
    
    def find_cross_file_by_timestamp(self, timestamp):
        """根据时间戳找到对应的CH1xCH2互相关文件"""
        pattern = f"correlation_{timestamp}_CH1xCH3.csv"
        filepath = self.watch_dir / pattern
        if filepath.exists():
            return filepath
        return None
    
    def extract_timestamp(self, filepath):
        """从文件名提取时间戳"""
        match = re.search(r'correlation_(\d{8}_\d{6})', filepath.name)
        if match:
            return match.group(1)
        return None
    
    def get_latest_timestamp(self):
        """获取最新的时间戳（从自相关或互相关文件）"""
        files = list(self.watch_dir.glob("correlation_*_CH1xCH2.csv"))
        if not files:
            return None
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return self.extract_timestamp(latest_file)
    
    def load_data(self, filepath):
        """加载CSV数据"""
        try:
            df = pd.read_csv(filepath, comment='#')
            freq_mhz = df['frequency_hz'].values / 1e6
            real_part = df['real_part'].values
            imag_part = df['imag_part'].values
            phase_deg = df['phase_deg'].values
            
            # 按频率排序
            idx = np.argsort(freq_mhz)
            freq_mhz = freq_mhz[idx]
            real_part = real_part[idx]
            imag_part = imag_part[idx]
            phase_deg = phase_deg[idx]
            
            return freq_mhz, real_part, imag_part, phase_deg
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None, None, None
    
    def change_colormap(self):
        """更换颜色映射"""
        if len(self.data_history) == 0:
            return
        
        cmap_name = self.cmap_var.get()
        cmap = plt.colormaps.get(cmap_name)
        if cmap is None:
            cmap = plt.cm.jet
        
        self._redraw_all(cmap)
    
    def _redraw_all(self, cmap):
        """重绘所有瀑布图"""
        n_times = len(self.data_history)
        if n_times == 0:
            return
        
        n_freqs = len(self.data_history[0]['freq'])
        
        # 构建矩阵
        real_matrix = np.zeros((n_times, n_freqs))
        imag_matrix = np.zeros((n_times, n_freqs))
        phase_matrix = np.zeros((n_times, n_freqs))
        
        for i, record in enumerate(self.data_history):
            real_matrix[i, :] = record['real']
            imag_matrix[i, :] = record['imag']
            phase_matrix[i, :] = record['phase']
        
        freq_mhz = self.data_history[0]['freq']
        
        # 频率轴刻度
        n_ticks = min(8, n_freqs)
        tick_positions = np.linspace(0, n_freqs-1, n_ticks).astype(int)
        tick_labels = [f'{freq_mhz[pos]:.0f}' for pos in tick_positions]
        
        # 时间轴刻度（从下往上：第0行是最早的数据）
        time_labels = [r['timestamp'][-8:] for r in self.data_history]
        y_ticks = range(n_times)
        
        # ========== 实部 ==========
        self.ax_real.clear()
        im_real = self.ax_real.imshow(real_matrix, aspect='auto', origin='lower',
                                       cmap=cmap, interpolation='bilinear')
        self.ax_real.set_ylabel('Time', color='white')
        self.ax_real.set_title('Real Part (I) - Waterfall', color='white', fontsize=10)
        self.ax_real.set_xticks(tick_positions)
        self.ax_real.set_xticklabels([])  # 只有最下面的图显示x标签
        self.ax_real.set_yticks(y_ticks)
        self.ax_real.set_yticklabels(time_labels, color='white', fontsize=7)
        self.ax_real.tick_params(colors='white')
        
        if self.colorbar_real:
            self.colorbar_real.remove()
        self.colorbar_real = self.fig.colorbar(im_real, ax=self.ax_real, 
                                                label='Real', orientation='horizontal',
                                                pad=0.15, aspect=40)
        self.colorbar_real.ax.xaxis.label.set_color('white')
        self.colorbar_real.ax.tick_params(colors='white')
        
        # ========== 虚部 ==========
        self.ax_imag.clear()
        im_imag = self.ax_imag.imshow(imag_matrix, aspect='auto', origin='lower',
                                       cmap=cmap, interpolation='bilinear')
        self.ax_imag.set_ylabel('Time', color='white')
        self.ax_imag.set_title('Imag Part (Q) - Waterfall', color='white', fontsize=10)
        self.ax_imag.set_xticks(tick_positions)
        self.ax_imag.set_xticklabels([])
        self.ax_imag.set_yticks(y_ticks)
        self.ax_imag.set_yticklabels(time_labels, color='white', fontsize=7)
        self.ax_imag.tick_params(colors='white')
        
        if self.colorbar_imag:
            self.colorbar_imag.remove()
        self.colorbar_imag = self.fig.colorbar(im_imag, ax=self.ax_imag, 
                                               label='Imag', orientation='horizontal',
                                               pad=0.15, aspect=40)
        self.colorbar_imag.ax.xaxis.label.set_color('white')
        self.colorbar_imag.ax.tick_params(colors='white')
        
        # ========== 相位 ==========
        self.ax_phase.clear()
        im_phase = self.ax_phase.imshow(phase_matrix, aspect='auto', origin='lower',
                                         cmap='hsv', interpolation='bilinear', vmin=-180, vmax=180)
        self.ax_phase.set_xlabel('Frequency (MHz)', color='white')
        self.ax_phase.set_ylabel('Time', color='white')
        self.ax_phase.set_title('Phase (deg) - Waterfall', color='white', fontsize=10)
        self.ax_phase.set_xticks(tick_positions)
        self.ax_phase.set_xticklabels(tick_labels, color='white')
        self.ax_phase.set_yticks(y_ticks)
        self.ax_phase.set_yticklabels(time_labels, color='white', fontsize=7)
        self.ax_phase.tick_params(colors='white')
        
        if self.colorbar_phase:
            self.colorbar_phase.remove()
        self.colorbar_phase = self.fig.colorbar(im_phase, ax=self.ax_phase, 
                                                label='Phase (deg)', orientation='horizontal',
                                                pad=0.15, aspect=40)
        self.colorbar_phase.ax.xaxis.label.set_color('white')
        self.colorbar_phase.ax.tick_params(colors='white')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_waterfall(self):
        """更新瀑布图：使用同一批次数据（相同时间戳）"""
        # 获取最新时间戳
        latest_timestamp = self.get_latest_timestamp()
        
        if latest_timestamp is None:
            self.status_var.set("No CH1xCH2 file found")
            return
        
        # 检查是否有新数据（时间戳是否变化）
        if self.last_processed_timestamp == latest_timestamp:
            self.status_var.set(f"Latest: {latest_timestamp} (no new data)")
            return
        
        # 根据时间戳找到对应的互相关文件
        cross_file = self.find_cross_file_by_timestamp(latest_timestamp)
        if cross_file is None:
            self.status_var.set(f"Cannot find file for timestamp: {latest_timestamp}")
            return
        
        # 加载数据
        freq_mhz, real_part, imag_part, phase_deg = self.load_data(cross_file)
        if freq_mhz is None:
            self.status_var.set(f"Failed to load: {cross_file.name}")
            return
        
        # 添加到deque（自动维护最大长度）
        self.data_history.append({
            'timestamp': latest_timestamp,
            'freq': freq_mhz,
            'real': real_part,
            'imag': imag_part,
            'phase': phase_deg
        })
        
        # 更新最后处理的时间戳
        self.last_processed_timestamp = latest_timestamp
        
        # 获取颜色映射并重绘
        cmap_name = self.cmap_var.get()
        cmap = plt.colormaps.get(cmap_name)
        if cmap is None:
            cmap = plt.cm.jet
        
        self._redraw_all(cmap)
        
        # 更新状态栏
        complex_val = real_part + 1j * imag_part
        peak_idx = np.argmax(np.abs(complex_val))
        peak_freq = freq_mhz[peak_idx]
        self.status_var.set(f"Updated: {latest_timestamp} | Total: {len(self.data_history)}/{self.max_rows} | Peak freq: {peak_freq:.2f} MHz")
    
    def refresh(self):
        """刷新数据"""
        self.update_waterfall()
    
    def force_refresh(self):
        """强制刷新"""
        self.update_waterfall()
        self.status_var.set("Manual refresh completed")
    
    def auto_refresh(self):
        """自动刷新"""
        if self.root.winfo_exists():
            self.update_waterfall()
            self.root.after(2000, self.auto_refresh)


if __name__ == "__main__":
    RollingWaterfall()
