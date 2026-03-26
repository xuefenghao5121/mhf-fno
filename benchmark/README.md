# Benchmark - 基准测试

本目录包含 MHF-FNO 的基准测试脚本用于复现论文结果。

## 目录结构

```
benchmark/
├── README.md              # 本文件
├── BENCHMARK_GUIDE.md     # 完整基准测试指南
├── data_loader.py         # ✨ 通用数据加载器 (支持 PT/H5, 单文件/双文件)
├── run_benchmarks.py      # 完整基准测试入口
├── generate_data.py       # 生成 Darcy Flow 数据集
├── generate_ns_velocity.py # 生成 Navier-Stokes 速度场数据集
├── test_mhf_coda_pino_quick.py # 快速测试最佳配置 (MHF+CoDA+PINO)
├── data/                  # 数据存储目录
└── internal/              # 历史研究脚本 (开发调试用)
```

## 快速开始

### 快速复现最佳结果

复现我们在真实 NS 速度场上的最佳结果：

```bash
cd benchmark

# 1. 生成 NS 速度场数据集 (200 samples, 64x64)
python generate_ns_velocity.py

# 2. 运行 MHF+CoDA+PINO 最佳配置测试
python test_mhf_coda_pino_quick.py
```

生成 Darcy Flow 数据集：

```bash
python generate_data.py
```

### 完整基准测试 (Zenodo 数据集)

如果你已经从 https://zenodo.org/records/13355846 下载了数据集：

```bash
cd benchmark

# Burgers 1D (Re1000) - Zenodo 双文件 H5 格式
python run_benchmarks.py \
    --dataset burgers \
    --format h5 \
    --train_path ./data/1D_Burgers_Re1000_Train.h5 \
    --test_path ./data/1D_Burgers_Re1000_Test.h5

# Navier-Stokes 2D (Re100) - Zenodo 双文件 H5 格式
python run_benchmarks.py \
    --dataset navier_stokes \
    --format h5 \
    --train_path ./data/2D_NS_Re100_Train.h5 \
    --test_path ./data/2D_NS_Re100_Test.h5
```

更多使用说明详见 `BENCHMARK_GUIDE.md`。

## 支持的数据格式

| 格式 | 说明 | 命令行 |
|------|------|--------|
| PT 单文件 | 本地生成 | `--data_path ./data/darcy_train_16.pt` |
| PT 双文件 | train.pt + test.pt | `--train_path ... --test_path ...` |
| H5 单文件 | PDEBench | `--format h5 --data_path ...` |
| **H5 双文件** | **Zenodo ✨** | `--format h5 --train_path ... --test_path ...` |

## 数据集

详见 `BENCHMARK_GUIDE.md`

| 数据集 | 生成方式 | 说明 |
|--------|---------|------|
| Darcy Flow | `generate_data.py` | 内置生成，无需下载 |
| Burgers 1D | 支持 Zenodo 下载 | 从 https://zenodo.org/records/13355846 获取 |
| Navier-Stokes 2D | 支持 Zenodo 下载 | 从 https://zenodo.org/records/13355846 获取 |