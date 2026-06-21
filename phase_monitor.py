#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from pathlib import Path
import re
import time


class PhaseMonitor:
    def __init__(self):
        self.running = True
        self.watch_dir = Path("./correlation_results")
        self.refresh_interval = 2.0
        self.current_phase_range = 180  # 当前相位范围（度）
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Phase/Power Monitor - 8x8 Correlation View")
        self.root.geometry("1500x950")
        
        # 创建界面
        self.create_menu()
        self.create_toolbar()
        self.create_statusbar()
        self.create_plot_area()
        
        # 启动自动刷新
        self.after_id = None
        self.start_auto_refresh()
        
        # 初始加载数据
        self.refresh_all_plots()
    
    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Directory...", command=self.choose_directory)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Refresh", command=self.force_refresh)
        view_menu.add_separator()
        
        # 相位范围子菜单
        phase_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Phase Range", menu=phase_menu)
        for r in [180, 90, 45, 30, 10]:
            phase_menu.add_command(label=f"±{r}°", command=lambda v=r: self.set_phase_range(v))
        
        view_menu.add_separator()
        view_menu.add_command(label="Reset All Views", command=self.reset_all_views)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        ttk.Button(toolbar, text="Refresh", command=self.force_refresh).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Browse", command=self.choose_directory).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # 相位范围选择
        ttk.Label(toolbar, text="Phase Range:").pack(side=tk.LEFT, padx=(5, 2))
        self.phase_range_var = tk.StringVar(value="±180")
        phase_combo = ttk.Combobox(toolbar, textvariable=self.phase_range_var,
                                    values=["±180", "±90", "±45", "±30", "±10"],
                                    width=6, state="readonly")
        phase_combo.pack(side=tk.LEFT, padx=2)
        phase_combo.bind("<<ComboboxSelected>>", lambda e: self._on_phase_range_changed())
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # 刷新间隔
        ttk.Label(toolbar, text="Interval:").pack(side=tk.LEFT, padx=(5, 2))
        self.interval_var = tk.StringVar(value="2.0")
        interval_spin = ttk.Spinbox(toolbar, from_=0.5, to=10.0, increment=0.5,
                                     textvariable=self.interval_var, width=5)
        interval_spin.pack(side=tk.LEFT, padx=2)
        interval_spin.bind("<Return>", lambda e: self.set_interval())
        ttk.Button(toolbar, text="Set", command=self.set_interval).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # 自动刷新开关
        self.auto_refresh_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Auto Refresh", variable=self.auto_refresh_var,
                        command=self.toggle_auto_refresh).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(toolbar, text="Reset Views", command=self.reset_all_views).pack(side=tk.LEFT, padx=2)
    
    def create_statusbar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar(value="Ready")
        statusbar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def create_plot_area(self):
        """创建绘图区域（带滚动条）"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建画布和滚动条
        self.canvas = tk.Canvas(main_frame)
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 可滚动框架
        self.plot_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.plot_frame, anchor=tk.NW)
        self.plot_frame.bind("<Configure>", self._on_frame_configure)
        
        # 存储绘图对象
        self.figures = [[None for _ in range(8)] for _ in range(8)]
        self.canvases = [[None for _ in range(8)] for _ in range(8)]
        self.axes = [[None for _ in range(8)] for _ in range(8)]
        
        # 创建 8x8 网格
        for i in range(8):
            for j in range(8):
                if i >= j:  # 只显示下三角和对角线
                    self._create_plot_cell(i, j)
        
        # 绑定鼠标滚轮
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
    
    def _create_plot_cell(self, i, j):
        """创建单个绘图单元格"""
        # 单元格框架
        cell_frame = ttk.Frame(self.plot_frame, relief=tk.RAISED, borderwidth=1)
        cell_frame.grid(row=i, column=j, padx=2, pady=2, sticky="nsew")
        self.plot_frame.grid_rowconfigure(i, weight=1)
        self.plot_frame.grid_columnconfigure(j, weight=1)
        
        # 创建图形
        fig = Figure(figsize=(2.8, 2.2), dpi=80)
        ax = fig.add_subplot(111)
        
        # 设置标题
        if i == j:
            ax.set_title(f"CH{i+1} (Power)", fontsize=9, fontweight='bold')
        else:
            ax.set_title(f"{j+1}x{i+1} (Phase)", fontsize=8)
        
        ax.set_xlabel("Freq (MHz)", fontsize=6)
        
        if i == j:
            # 对角线：幅度谱（dB）
            ax.set_ylabel("Power (dB)", fontsize=6)
        else:
            # 下三角：相位谱（度）
            ax.set_ylabel("Phase (deg)", fontsize=6)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.6, linewidth=0.8)
        
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        self.figures[i][j] = fig
        self.axes[i][j] = ax
        
        # 嵌入 Tkinter
        canvas_widget = FigureCanvasTkAgg(fig, master=cell_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases[i][j] = canvas_widget
    
    def _on_frame_configure(self, event):
        """框架大小变化时更新滚动区域"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_mousewheel(self, event):
        """鼠标滚轮滚动"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def choose_directory(self):
        """选择监控目录"""
        directory = filedialog.askdirectory(initialdir=str(self.watch_dir))
        if directory:
            self.watch_dir = Path(directory)
            self.status_var.set(f"Directory: {self.watch_dir}")
            self.refresh_all_plots()
    
    def set_interval(self):
        """设置刷新间隔"""
        try:
            self.refresh_interval = float(self.interval_var.get())
            self.status_var.set(f"Refresh interval: {self.refresh_interval}s")
            if self.auto_refresh_var.get():
                self.stop_auto_refresh()
                self.start_auto_refresh()
        except ValueError:
            self.status_var.set("Invalid interval value")
    
    def set_phase_range(self, ymax):
        """设置相位显示范围"""
        self.current_phase_range = ymax
        self.phase_range_var.set(f"±{ymax}")
        self._apply_phase_range()
    
    def _on_phase_range_changed(self):
        """相位范围改变时的回调"""
        range_str = self.phase_range_var.get()
        ymax = int(range_str.replace("±", ""))
        self.current_phase_range = ymax
        self._apply_phase_range()
    
    def _apply_phase_range(self):
        """应用相位范围到互相关图"""
        for i in range(8):
            for j in range(8):
                if i > j and self.axes[i][j] is not None:  # 只对互相关图（下三角，非对角线）
                    self.axes[i][j].set_ylim(-self.current_phase_range, self.current_phase_range)
                    self.canvases[i][j].draw()
        self.status_var.set(f"Phase range: ±{self.current_phase_range}°")
    
    def toggle_auto_refresh(self):
        """切换自动刷新"""
        if self.auto_refresh_var.get():
            self.start_auto_refresh()
            self.status_var.set("Auto refresh enabled")
        else:
            self.stop_auto_refresh()
            self.status_var.set("Auto refresh disabled")
    
    def start_auto_refresh(self):
        """启动自动刷新"""
        if self.after_id:
            self.root.after_cancel(self.after_id)
        self.after_id = self.root.after(int(self.refresh_interval * 1000), self._auto_refresh)
    
    def stop_auto_refresh(self):
        """停止自动刷新"""
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None
    
    def _auto_refresh(self):
        """自动刷新回调"""
        if self.running and self.auto_refresh_var.get():
            self.refresh_all_plots()
            self.after_id = self.root.after(int(self.refresh_interval * 1000), self._auto_refresh)
    
    def force_refresh(self):
        """强制刷新"""
        self.refresh_all_plots()
        self.status_var.set("Manual refresh completed")
    
    def reset_all_views(self):
        """重置所有视图"""
        # 重置相位图
        self._apply_phase_range()
        # 重置频率范围
        for i in range(8):
            for j in range(8):
                if i >= j and self.axes[i][j] is not None:
                    ax = self.axes[i][j]
                    if hasattr(ax, '_original_xlim'):
                        ax.set_xlim(ax._original_xlim)
                        # 对角线需要重新计算Y范围
                        if i == j:
                            self._update_auto_power_range(i)
                        self.canvases[i][j].draw()
        self.status_var.set("All views reset")
    
    def _update_auto_power_range(self, row):
        """更新自相关图的动态Y范围"""
        ax = self.axes[row][row]
        if ax is None:
            return
        
        # 获取当前曲线数据
        lines = ax.get_lines()
        if len(lines) == 0:
            return
        
        ydata = lines[0].get_ydata()
        if len(ydata) == 0:
            return
        
        max_mag_db = np.max(ydata)
        min_mag_db = np.min(ydata)
        y_min = min_mag_db - 100
        y_max = max_mag_db + 5
        
        ax.set_ylim(y_min, y_max)
        
        # 更新标题中的峰值
        ax.set_title(f"CH{row+1} (Power) Peak:{max_mag_db:.1f}dB", fontsize=8, fontweight='bold')
    
    def get_latest_csv_files(self):
        """获取最新的CSV文件组"""
        csv_files = list(self.watch_dir.glob("correlation_*.csv"))
        if not csv_files:
            return {}
        
        # 按文件修改时间分组，取最新的一组
        latest_time = max(f.stat().st_mtime for f in csv_files)
        latest_files = [f for f in csv_files if abs(f.stat().st_mtime - latest_time) < 1.0]
        
        # 按配对名称组织
        result = {}
        for f in latest_files:
            pair_name = self._extract_pair_name(f)
            if pair_name:
                result[pair_name] = f
        return result
    
    def _extract_pair_name(self, filepath):
        """从文件名提取配对名称"""
        name = filepath.stem
        # 匹配 CH1_AUTO 或 CH1xCH2
        match = re.search(r'(CH\d+(?:xCH\d+|_AUTO))', name)
        return match.group(1) if match else None
    
    def _get_channel_indices(self, pair_name):
        """获取通道索引 (row, col) for lower triangle"""
        if '_AUTO' in pair_name:
            match = re.search(r'CH(\d+)_AUTO', pair_name)
            if match:
                ch = int(match.group(1)) - 1
                return ch, ch
        elif 'x' in pair_name:
            match = re.search(r'CH(\d+)xCH(\d+)', pair_name)
            if match:
                ch_i = int(match.group(1)) - 1
                ch_j = int(match.group(2)) - 1
                return max(ch_i, ch_j), min(ch_i, ch_j)
        return None, None
    
    def _load_csv_data(self, filepath):
        """加载CSV数据，返回频率(MHz)、相位(度)、幅度(dB)"""
        try:
            df = pd.read_csv(filepath, comment='#')
            freq_hz = df['frequency_hz'].values
            freq_mhz = freq_hz / 1e6
            phase_deg = df['phase_deg'].values
            
            # 计算幅度(dB)，避免log10(0)
            magnitude = df['magnitude'].values
            mag_db = 20 * np.log10(magnitude + 1e-12)
            
            # 按频率排序
            idx = np.argsort(freq_mhz)
            freq_mhz = freq_mhz[idx]
            phase_deg = phase_deg[idx]
            mag_db = mag_db[idx]
            
            return freq_mhz, phase_deg, mag_db
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None, None
    
    def _update_plot(self, row, col, freq_mhz, phase_deg, mag_db):
        """更新单个子图"""
        if row is None or col is None:
            return
        if row < 0 or row >= 8 or col < 0 or col >= 8:
            return
        if self.axes[row][col] is None:
            return
        
        ax = self.axes[row][col]
        
        # 保存原始X范围
        if not hasattr(ax, '_original_xlim') and freq_mhz is not None and len(freq_mhz) > 0:
            ax._original_xlim = (freq_mhz.min(), freq_mhz.max())
        
        ax.clear()
        
        if freq_mhz is not None and len(freq_mhz) > 0:
            if row == col:
                # 对角线：绘制幅度谱（dB），动态范围
                max_mag_db = np.max(mag_db)
                min_mag_db = np.min(mag_db)
                # 动态范围：从最大值向下80dB，或到最小值，取较大者
                y_min = min_mag_db - 20
                y_max = max_mag_db + 5
                
                ax.plot(freq_mhz, mag_db, 'b-', linewidth=0.7, alpha=0.8)
                ax.fill_between(freq_mhz, mag_db, y_min, alpha=0.2, color='blue')
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel("Power (dB)", fontsize=6)
                ax.set_title(f"CH{row+1} (Power) Peak:{max_mag_db:.1f}dB", fontsize=8, fontweight='bold')
            else:
                # 下三角：绘制相位谱（度）
                ax.plot(freq_mhz, phase_deg, 'b-', linewidth=0.5, alpha=0.7)
                ax.scatter(freq_mhz, phase_deg, c='blue', s=3, alpha=0.4)
                ax.set_ylim(-self.current_phase_range, self.current_phase_range)
                ax.set_ylabel("Phase (deg)", fontsize=6)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.6, linewidth=0.8)
                ax.set_title(f"{col+1}x{row+1} (Phase)", fontsize=8)
            
            ax.set_xlim(freq_mhz.min(), freq_mhz.max())
        else:
            # 无数据时显示提示
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            if row == col:
                ax.set_ylabel("Power (dB)", fontsize=6)
                ax.set_title(f"CH{row+1} (Power)", fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel("Phase (deg)", fontsize=6)
                ax.set_title(f"{col+1}x{row+1} (Phase)", fontsize=8)
        
        # 公共设置
        ax.set_xlabel("Freq (MHz)", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        self.canvases[row][col].draw()
    
    def refresh_all_plots(self):
        """刷新所有图表"""
        if not self.watch_dir.exists():
            self.status_var.set(f"Directory not found: {self.watch_dir}")
            return
        
        # 获取最新文件
        latest_files = self.get_latest_csv_files()
        if not latest_files:
            self.status_var.set(f"No CSV files found in {self.watch_dir}")
            return
        
        # 更新每个子图
        updated = 0
        for pair_name, filepath in latest_files.items():
            row, col = self._get_channel_indices(pair_name)
            if row is not None and col is not None:
                freq_mhz, phase_deg, mag_db = self._load_csv_data(filepath)
                self._update_plot(row, col, freq_mhz, phase_deg, mag_db)
                updated += 1
        
        # 显示状态
        latest_time = max(f.stat().st_mtime for f in latest_files.values())
        time_str = time.strftime("%H:%M:%S", time.localtime(latest_time))
        self.status_var.set(f"Updated: {updated} plots | Latest: {time_str}")
    
    def show_about(self):
        """显示关于对话框"""
        import tkinter.messagebox
        about_text = """Phase/Power Monitor - 8x8 Correlation View

实时监测8通道相关器的频谱图：

- 对角线 (CHi): 功率谱（dB），动态范围自动调整
- 下三角 (CHi x CHj): 相位谱（度）

功能:
- 自动刷新最新数据
- 可调相位显示范围
- 鼠标滚轮滚动查看
- 支持手动/自动刷新

Author: Fast Correlation GPU Project
Version: 2.0"""
        
        tk.messagebox.showinfo("About", about_text)
    
    def on_closing(self):
        """关闭窗口"""
        self.running = False
        self.stop_auto_refresh()
        self.root.destroy()
    
    def run(self):
        """运行主循环"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()


def main():
    import sys
    from pathlib import Path
    
    watch_dir = "./correlation_results"
    if len(sys.argv) > 1:
        watch_dir = sys.argv[1]
    
    print("=" * 60)
    print("Phase/Power Monitor - 8x8 Correlation View")
    print("=" * 60)
    print(f"Monitoring: {watch_dir}")
    print("Diagonal: Power Spectrum (dB) - Auto-scaling")
    print("Lower Triangle: Phase Spectrum (deg)")
    print("=" * 60)
    
    Path(watch_dir).mkdir(parents=True, exist_ok=True)
    
    app = PhaseMonitor()
    app.watch_dir = Path(watch_dir)
    app.run()


if __name__ == "__main__":
    main()
