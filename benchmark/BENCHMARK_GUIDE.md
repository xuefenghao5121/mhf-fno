# MHF-FNO 基准测试指南

本文档提供完整的数据集下载和测试说明。

---

## 数据集信息

### 内置数据（无需下载）

| 数据集 | 分辨率 | 训练集 | 测试集 | 位置 |
|--------|--------|--------|--------|------|
| **Darcy Flow 16** | 16×16 | 1,000 | 50 | NeuralOperator 内置 |

### PDEBench H5 数据（需下载）

| 数据集 | 分辨率 | 训练集 | 测试集 | 大小 |
|--------|--------|--------|--------|------|
| Darcy Flow 2D | 421×421 | 5,000 | 500 | ~500MB |
| Navier-Stokes 2D | 128×128 | 10,000 | 1,000 | ~2GB |
| Burgers 1D | 1024 | 1,000 | 200 | ~50MB |

---

## 数据下载地址

### PDEBench (H5 格式)

官方主页: https://pdebench.github.io/

直接下载:
```bash
# Darcy Flow 2D
wget -O data/2D_DarcyFlow_Train.h5 https://darus.uni-stuttgart.de/api/access/datafile/152002
wget -O data/2D_DarcyFlow_Test.h5 https://darus.uni-stuttgart.de/api/access/datafile/152003

# Navier-Stokes 2D
wget -O data/2D_NS_Train.h5 https://darus.uni-stuttgart.de/api/access/datafile/151996
wget -O data/2D_NS_Test.h5 https://darus.uni-stuttgart.de/api/access/datafile/151997

# Burgers 1D
wget -O data/1D_Burgers.h5 https://darus.uni-stuttgart.de/api/access/datafile/151990
```

### NeuralOperator (PT 格式)

自动下载，无需手动操作。

---

## 运行测试

### PT 格式（默认）

```bash
cd benchmark

# Darcy Flow (内置数据)
python run_benchmarks.py --dataset darcy

# Burgers (自动下载)
python run_benchmarks.py --dataset burgers
```

### H5 格式（PDEBench）

```bash
cd benchmark

# 先下载数据
wget -O ../data/2D_DarcyFlow_Train.h5 https://darus.uni-stuttgart.de/api/access/datafile/152002

# 运行测试
python run_benchmarks.py --dataset darcy --format h5 \
    --data_path ../data/2D_DarcyFlow_Train.h5
```

### 自定义参数

```bash
python run_benchmarks.py \
    --dataset darcy \
    --format h5 \
    --data_path ../data/2D_DarcyFlow_Train.h5 \
    --n_train 500 \
    --n_test 100 \
    --epochs 30
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | darcy | 数据集 (darcy/burgers/navier_stokes/all) |
| `--format` | pt | 数据格式 (pt/h5) |
| `--data_path` | None | H5 文件路径 (format=h5 时需要) |
| `--n_train` | 1000 | 训练集大小 |
| `--n_test` | 200 | 测试集大小 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--output` | benchmark_results.json | 结果文件 |

---

## 测试结果

### Darcy Flow 16×16

| 配置 | 参数减少 | L2 变化 | 推荐 |
|------|----------|---------|------|
| **推荐配置** | **-18.7%** | **+12.2%** | ✅ |
| 精度优先 | -17.8% | +9.8% | |
| 参数优先 | -30.7% | +12.6% | |

---

## 常见问题

### Q: H5 数据下载太慢？
A: 
- 使用代理
- 或使用镜像站点
- 或只运行 PT 格式（内置数据）

### Q: NeuralOperator 报错？
A: 确保版本正确：
```bash
pip install neuralop==0.3.0 h5py
```

### Q: GPU 内存不足？
A: 减小 batch_size：
```bash
python run_benchmarks.py --dataset darcy --batch_size 16
```

---

*更新时间: 2026-03-24*