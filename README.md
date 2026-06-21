# Radio Interferometer Dirty Image Generator

实时射电干涉仪脏图生成器，支持 **CPU 小视场**、**CPU 全天极坐标** 和 **GPU 全天极坐标** 三种成像模式。

## 功能

- **CPU 模式**：基于 C/OpenMP 加速（CFFI）的傅里叶求和，自动回退到纯 Python 实现
- **GPU 模式**：基于 CuPy 的 GPU 加速全天极坐标脏图生成
- **实时成像**：支持从文件或管道实时读取 visibilities 并生成脏图
- **RFI 标记**：自动射频干扰检测与剔除
- **水fall 图**：频域 waterfall 可视化

## 依赖

```
numpy
matplotlib
cupy         # GPU 模式需要
cffi         # CPU C/OpenMP 加速（可选，不可用时自动回退 Python）
```

## 快速开始

```bash
# CPU 小视场模式
python test_cpu_only.py

# CPU 全天极坐标模式（自动选择 C/OpenMP 或纯 Python）
python test_polar_cpu.py

# GPU 全天极坐标模式（需要 CUDA + CuPy）
python test_polar_gpu.py

# 实时成像
python realtime_dirty_image.py --mode cpu
python realtime_dirty_image.py --mode polar_cpu
python realtime_dirty_image.py --mode gpu
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `make_dirty_image.py` | 核心脏图生成引擎 |
| `realtime_dirty_image.py` | 实时成像 GUI |
| `polar_cpu_cffi.py` | C/OpenMP CFFI 接口 |
| `_polar_fourier_engine.c` | C 语言傅里叶求和内核 |
| `direct_fourier_c.c` | C 语言直接傅里叶变换 |
| `waterfall.py` | 频域 waterfall 图 |
| `phase_monitor.py` | 相位监控 |
| `optimized_antenna_coordinates.txt` | 天线坐标数据 |

## License

MIT
