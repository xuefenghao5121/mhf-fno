# Benchmark 基准测试

本目录包含 MHF-FNO 的基准测试脚本，用于复现论文结果。

## 📁 目录结构

```
benchmark/
├── README.md              # 本文件
├── data_loader.py         # ✨ NeuralOperator 2.0.0 数据加载器 (v1.6.0+)
├── run_benchmarks.py      # 完整基准测试入口
├── generate_data.py       # 统一数据生成器 (Darcy/Burgers/NS)
├── quick_test.py          # 快速测试最佳配置 (MHF+CoDA+PINO)
└── data/                  # 数据存储目录 (自动下载)
```

## 🚀 快速开始

### 1. 快速测试最佳配置
```bash
cd benchmark

# 运行 MHF+CoDA+PINO 最佳配置测试 (Darcy 数据集)
python quick_test.py
```

### 2. 完整基准测试 (自动下载数据)
```bash
# Navier-Stokes 数据集 (64x64 分辨率)
python run_benchmarks.py \
    --dataset navier_stokes \
    --n_train 1000 \
    --n_test 200 \
    --resolution 64

# Darcy Flow 数据集 (64x64 分辨率)
python run_benchmarks.py \
    --dataset darcy \
    --n_train 1000 \
    --n_test 200 \
    --resolution 64
```

### 3. 生成各类数据集 (本地生成)
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

## 📊 支持的数据集 (NeuralOperator 2.0.0)

| 数据集 | 来源 | 分辨率支持 | 使用示例 |
|--------|------|------------|----------|
| **Navier-Stokes** | Zenodo 12825163 | 128, 1024 | `--dataset navier_stokes --resolution 128` |
| **Darcy Flow** | Zenodo 12784353 | 16, 32, 64, 128 | `--dataset darcy --resolution 64` |

## 🔧 数据加载 (v1.6.0 重构)

### 使用 NeuralOperator 2.0.0 数据加载器
从 v1.6.0 开始，直接使用 NeuralOperator 2.0.0 的数据加载器，自动从 Zenodo 下载官方数据集。

**优势**：
- ✅ 稳定可靠（NeuralOperator 官方维护）
- ✅ 自动下载和解压
- ✅ 支持所有官方格式
- ✅ 统一的接口和输出格式
- ✅ 减少维护成本

### API 对比

#### 旧 API (v1.5.x)
```python
# ❌ 需要手动指定文件路径
data = load_dataset(
    dataset_name='navier_stokes',
    data_format='h5',      # ❌ 不再需要
    train_path='./data/NS_Train.h5',  # ❌ 不再需要
    test_path='./data/NS_Test.h5',    # ❌ 不再需要
    n_train=1000,
    n_test=200,
    resolution=None,    # ❌ 现在必需
)
```

#### 新 API (v1.6.0+)
```python
# ✅ 自动从 Zenodo 下载
data = load_dataset(
    dataset_name='navier_stokes',
    n_train=1000,
    n_test=200,
    resolution=64,   # ✅ 现在必需
    download=True,     # ✅ 自动下载
)
```

## 🎛 命令行参数

### 数据集参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集名称 | `darcy` |
| `--resolution` | 数据分辨率 (必需) | `None` |
| `--n_train` | 训练样本数 | `1000` |
| `--n_test` | 测试样本数 | `200` |

### 训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | `50` |
| `--batch_size` | 批次大小 | `32` |
| `--lr` | 学习率 | `0.001` |
| `--seed` | 随机种子 | `42` |

### 物理方程参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--viscosity` | 粘性系数 | `1e-3` |
| `--n_steps` | 时间步数 | `100` |

## 🔧 关键脚本说明

| 脚本 | 功能 | 状态 |
|------|------|------|
| **data_loader.py** | 使用 NeuralOperator 2.0.0 加载数据 | ✅ v1.6.0 重构 (支持自动下载) |
| **run_benchmarks.py** | 完整的基准测试，支持各种配置 | ✅ 已更新到 v1.6.0 API |
| **generate_data.py** | 统一生成所有数据集，无需下载 | ✅ 本地生成 |
| **quick_test.py** | 快速测试 MHF+CoDA+PINO 最佳组合 | ✅ 测试最佳配置 |

## 📦 版本更新说明

### v1.6.0 (2026-03-27) - 重大重构
- **数据加载重构**: 替换为 NeuralOperator 2.0.0 数据加载器
  - 删除 ~1000 行自定义数据加载代码
  - 新增 ~100 行封装代码
  - 代码量减少 90%
  - 稳定性和兼容性大幅提升
  
- **修复的问题**:
  - ✅ 1D H5 加载维度错误
  - ✅ 2D Burgers 数据集未正确检测
  - ✅ 2D Navier-Stokes PDEBench 格式不支持
  - ✅ 多种格式兼容性问题

- **API 变更**:
  - ❌ 移除 `--data_format` 参数
  - ❌ 移除 `--train_path` 参数
  - ❌ 移除 `--test_path` 参数
  - ✅ `--resolution` 变为必需参数
  - ✅ 自动从 Zenodo 下载数据集

## 🎯 迁移指南 (v1.5.x → v1.6.0+)

如果你使用的是 v1.5.x，需要更新命令行参数：

### v1.5.x 用法
```bash
python run_benchmarks.py \
    --dataset navier_stokes \
    --format h5 \
    --train_path ./data/NS_Train.h5 \
    --test_path ./data/NS_Test.h5 \
    --n_train 1000 \
    --n_test 200
```

### v1.6.0+ 用法
```bash
python run_benchmarks.py \
    --dataset navier_stokes \
    --resolution 64 \      # ✅ 现在必需
    --n_train 1000 \
    --n_test 200
    # ✅ 自动从 Zenodo 下载数据
```

## 📊 预期输出示例

```
============================================================
MHF-FNO 基准测试
============================================================

📊 使用 NeuralOperator 2.0.0 加载数据集: navier_stokes
   配置: n_train=1000, n_test=200, resolution=64
   root_dir: /path/to/data
   download: True
   ✅ NeuralOperator 数据集创建成功
      训练集大小: 1000
      测试集大小: 200
   ✅ 数据加载成功
      train_x: torch.Size([1000, 1, 64, 64])
      train_y: torch.Size([1000, 1, 64, 64])
      test_x: torch.Size([200, 1, 64, 64])
      test_y: torch.Size([200, 1, 64, 64])

数据集信息:
  名称: NeuralOperator Navier-Stokes
  分辨率: 64x64
  训练集: 1000
  测试集: 200

============================================================
测试 FNO (基准)
============================================================
参数量: 800,433
  Epoch 10/50: Train 0.0234, Test 0.0215, Time 1.2s
  Epoch 20/50: Train 0.0123, Test 0.0118, Time 1.1s
  ...
```

## 🔗 参考链接

- [NeuralOperator GitHub](https://github.com/neuraloperator/neuraloperator)
- [Zenodo Dataset Records](https://zenodo.org/records/12825163)
- [MHF-FNO GitHub](https://github.com/xuefenghao5121/mhf-fno)
