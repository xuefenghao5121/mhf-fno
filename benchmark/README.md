# Benchmark 基准测试

本目录包含 MHF-FNO 的基准测试脚本，用于复现论文结果。

## 📁 目录结构

```
benchmark/
├── README.md              # 本文件
├── data_loader.py         # ✨ 通用数据加载器 (支持 PT/H5, 单文件/双文件)
├── run_benchmarks.py      # 完整基准测试入口
├── generate_data.py       # 统一数据生成器 (Darcy/Burgers/NS)
├── quick_test.py          # 快速测试最佳配置 (MHF+CoDA+PINO)
└── data/                  # 数据存储目录
```

## 🚀 快速开始

### 1. 快速复现最佳结果
```bash
cd benchmark

# 生成 Navier-Stokes 速度场数据集 (200 samples, 64x64)
python generate_data.py --dataset navier_stokes --n_train 200

# 运行 MHF+CoDA+PINO 最佳配置测试
python quick_test.py
```

### 2. 生成各类数据集
```bash
# Darcy Flow 2D
python generate_data.py --dataset darcy --n_train 1000 --resolution 64

# Burgers 1D
python generate_data.py --dataset burgers --n_train 1000 --resolution 1024

# Navier-Stokes 2D (含时间序列)
python generate_data.py --dataset navier_stokes --n_train 200 --resolution 64 --time_steps 20

# 生成所有数据集
python generate_data.py --dataset all
```

### 3. 完整基准测试 (Zenodo 数据集)
如果你已经从 [Zenodo](https://zenodo.org/records/13355846) 下载了数据集：
```bash
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

## 📊 支持的数据格式

| 格式 | 说明 | 命令行参数 |
|------|------|------------|
| PT 单文件 | 本地生成 | `--data_path ./data/darcy_train_16.pt` |
| PT 双文件 | train.pt + test.pt | `--train_path ... --test_path ...` |
| H5 单文件 | PDEBench | `--format h5 --data_path ...` |
| H5 双文件 | Zenodo | `--format h5 --train_path ... --test_path ...` |

## 🔧 关键脚本说明

| 脚本 | 功能 |
|------|------|
| **data_loader.py** | 通用数据加载，支持所有格式和分辨率 |
| **run_benchmarks.py** | 完整的基准测试，支持各种配置 |
| **generate_data.py** | 统一生成所有数据集，无需下载 |
| **quick_test.py** | 快速测试 MHF+CoDA+PINO 最佳组合 |
