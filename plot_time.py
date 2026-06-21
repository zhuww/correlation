#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re


class TimeDomainMonitor:
    def __init__(self, watch_dir=".", samples_per_plot=500):
        self.watch_dir = Path(watch_dir)
        self.samples_per_plot = samples_per_plot
        self.channel_data = {}
        
        # Load all debug_ch*.bin files
        self.load_all_channels()
        
        if not self.channel_data:
            print(f"No debug_ch*.bin files found in {watch_dir}")
            print("Please run the CUDA program with debug saving enabled first.")
            sys.exit(1)
        
        self.num_channels = len(self.channel_data)
        print(f"Loaded {self.num_channels} channels")
        
        # Create 8x8 grid plot
        self.create_grid_plot()
    
    def load_all_channels(self):
        """Load all debug_ch*.bin files"""
        bin_files = sorted(self.watch_dir.glob("debug_ch*.bin"))
        
        for bin_file in bin_files:
            # Extract channel number from filename
            match = re.search(r'debug_ch(\d+)', bin_file.name)
            if match:
                ch = int(match.group(1))
                data = np.fromfile(bin_file, dtype=np.int16)
                i_data = data[::2]   # I component
                q_data = data[1::2]  # Q component
                self.channel_data[ch] = {'i': i_data, 'q': q_data}
                print(f"CH{ch}: {len(i_data)} I/Q pairs")
    
    def get_channel_name(self, ch):
        """Get channel name"""
        return f"CH{ch}"
    
    def plot_waveform(self, ax, ch, samples):
        """Plot I and Q waveform for a single channel (auto-correlation)"""
        i_data = self.channel_data[ch]['i'][:samples]
        q_data = self.channel_data[ch]['q'][:samples]
        time_axis = np.arange(len(i_data))
        
        ax.plot(time_axis, i_data, 'b-', linewidth=0.5, alpha=0.7, label='I')
        ax.plot(time_axis, q_data, 'r-', linewidth=0.5, alpha=0.7, label='Q')
        ax.set_title(f"CH{ch}", fontsize=9)
        ax.set_xlabel('Sample', fontsize=6)
        ax.set_ylabel('Amplitude', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=5, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Set y limits based on data
        max_val = max(np.abs(i_data).max(), np.abs(q_data).max())
        ax.set_ylim(-max_val*1.1, max_val*1.1)
    
    def plot_cross_correlation(self, ax, ch_i, ch_j, samples):
        """Plot I and Q waveforms for two channels (cross-correlation comparison)"""
        i_i = self.channel_data[ch_i]['i'][:samples]
        q_i = self.channel_data[ch_i]['q'][:samples]
        i_j = self.channel_data[ch_j]['i'][:samples]
        q_j = self.channel_data[ch_j]['q'][:samples]
        time_axis = np.arange(len(i_i))
        
        # Plot I components
        ax.plot(time_axis, i_i, 'b-', linewidth=0.5, alpha=0.7, label=f'CH{ch_i} I')
        ax.plot(time_axis, i_j, 'c-', linewidth=0.5, alpha=0.7, label=f'CH{ch_j} I')
        # Plot Q components (dashed)
        ax.plot(time_axis, q_i, 'b--', linewidth=0.5, alpha=0.5, label=f'CH{ch_i} Q')
        ax.plot(time_axis, q_j, 'c--', linewidth=0.5, alpha=0.5, label=f'CH{ch_j} Q')
        
        ax.set_title(f"{ch_i} x {ch_j}", fontsize=9)
        ax.set_xlabel('Sample', fontsize=6)
        ax.set_ylabel('Amplitude', fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(fontsize=5, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Set y limits
        max_val_i = max(np.abs(i_i).max(), np.abs(q_i).max())
        max_val_j = max(np.abs(i_j).max(), np.abs(q_j).max())
        ax.set_ylim(-max(max_val_i, max_val_j)*1.1, max(max_val_i, max_val_j)*1.1)
    
    def create_grid_plot(self):
        """Create 8x8 grid with auto-correlation on diagonal, cross-correlation elsewhere"""
        
        # Determine figure size based on number of channels
        figsize = (min(self.num_channels * 1.5, 16), min(self.num_channels * 1.2, 14))
        fig, axes = plt.subplots(self.num_channels, self.num_channels, 
                                  figsize=figsize, squeeze=False)
        
        # For each cell in the grid
        for i in range(self.num_channels):
            for j in range(self.num_channels):
                ax = axes[i][j]
                ch_i = i + 1
                ch_j = j + 1
                
                if ch_i not in self.channel_data or ch_j not in self.channel_data:
                    ax.set_visible(False)
                    continue
                
                if i == j:
                    # Diagonal: Auto-correlation (I and Q of same channel)
                    self.plot_waveform(ax, ch_i, self.samples_per_plot)
                elif i > j:
                    # Lower triangle: Cross-correlation (comparison)
                    self.plot_cross_correlation(ax, ch_i, ch_j, self.samples_per_plot)
                else:
                    # Upper triangle: Empty (to avoid redundancy)
                    ax.set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f"Time Domain Waveforms (First {self.samples_per_plot} samples)", 
                     fontsize=14, y=1.02)
        plt.show()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot 8x8 time domain waveforms')
    parser.add_argument('directory', nargs='?', default='.',
                        help='Directory containing debug_ch*.bin files')
    parser.add_argument('-n', '--samples', type=int, default=500,
                        help='Number of samples to plot (default: 500)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("8x8 Time Domain Waveform Monitor")
    print("=" * 60)
    print(f"Directory: {args.directory}")
    print(f"Samples per plot: {args.samples}")
    print("=" * 60)
    
    app = TimeDomainMonitor(watch_dir=args.directory, 
                             samples_per_plot=args.samples)


if __name__ == "__main__":
    main()
