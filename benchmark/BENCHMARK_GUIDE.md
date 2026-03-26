# MHF-FNO 基准测试指南

本文档提供完整的数据集下载和测试说明。

---

## 快速开始（推荐）

### 自生成数据

**推荐使用数据生成脚本**，避免下载 PDEBench 数据的网络问题：

```bash
cd benchmark

# 生成 Darcy Flow 数据 (默认 16x16)
python generate_data.py --dataset darcy

# 生成 Burgers 数据
python generate_data.py --dataset burgers

# 生成 Navier-Stokes 数据
python generate_data.py --dataset navier_stokes --resolution 64

# 生成所有数据集
python generate_data.py --dataset all
```

生成完成后，直接运行基准测试：

```bash
python run_benchmarks.py --dataset darcy
```

### 生成参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | darcy | 数据集 (darcy/burgers/navier_stokes/all) |
| `--n_train` | 1000 | 训练样本数 |
| `--n_test` | 200 | 测试样本数 |
| `--resolution` | 16 | 空间分辨率 (Darcy/NS) |
| `--n_points` | 1024 | 空间点数 (Burgers) |
| `--viscosity` | 0.1 | 粘性系数 |
| `--n_steps` | 100 | 时间步数 (Navier-Stokes) |
| `--output_dir` | ./data | 输出目录 |
| `--device` | cpu | 计算设备 (cpu/cuda) |

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

### H5 格式

#### 单文件 H5 (PDEBench 原格式)

```bash
cd benchmark

# 先下载数据
wget -O ../data/2D_DarcyFlow_Train.h5 https://darus.uni-stuttgart.de/api/access/datafile/152002

# 运行测试
python run_benchmarks.py --dataset darcy --format h5 \
    --data_path ../data/2D_DarcyFlow_Train.h5
```

#### 双文件 H5 (Zenodo 格式，训练集测试集分开) ✨ **新增**

你从 https://zenodo.org/records/13355846 下载的数据集是训练集和测试集分开的两个文件，可以这样运行：

```bash
cd benchmark

# 假设你已经下载了:
#   data/1D_Burgers_Re1000_Train.h5
#   data/1D_Burgers_Re1000_Test.h5
#   data/2D_NS_Re100_Train.h5
#   data/2D_NS_Re100_Test.h5

# Burgers 1D
python run_benchmarks.py --dataset burgers --format h5 \
    --train_path ../data/1D_Burgers_Re1000_Train.h5 \
    --test_path ../data/1D_Burgers_Re1000_Test.h5

# Navier-Stokes 2D
python run_benchmarks.py --dataset navier_stokes --format h5 \
    --train_path ../data/2D_NS_Re100_Train.h5 \
    --test_path ../data/2D_NS_Re100_Test.h5

# Darcy Flow 2D
python run_benchmarks.py --dataset darcy --format h5 \
    --train_path ../data/2D_DarcyFlow_Train.h5 \
    --test_path ../data/2D_DarcyFlow_Test.h5
```

**Zenodo 下载地址**: https://zenodo.org/records/13355846

| 数据集 | 文件名 |
|--------|--------|
| Burgers 1D | `1D_Burgers_Re1000_Train.h5` + `1D_Burgers_Re1000_Test.h5` |
| Navier-Stokes 2D | `2D_NS_Re100_Train.h5` + `2D_NS_Re100_Test.h5` |
| Darcy Flow 2D | `2D_DarcyFlow_Train.h5` + `2D_DarcyFlow_Test.h5` |

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

## 数据生成原理

### Darcy Flow

求解椭圆 PDE：
```
-∇·(a(x)∇u(x)) = f(x),  x∈[0,1]²
u|∂Ω = 0
```

生成方法：
1. 使用高斯随机场生成渗透系数 a(x)
2. 使用 Jacobi 迭代求解压力场 u(x)

### Burgers 1D

求解对流扩散方程：
```
∂u/∂t + u∂u/∂x = ν∂²u/∂x²
```

生成方法：
1. 随机初始条件
2. 使用伪光谱法时间推进

### Navier-Stokes 2D

求解涡度形式：
```
∂ω/∂t + u·∇ω = ν∇²ω + f
```

生成方法：
1. 随机涡旋叠加作为初始涡度
2. 使用伪光谱法时间推进

---

## 生成 vs 下载对比

| 特性 | 自生成 | PDEBench 下载 |
|------|--------|---------------|
| 网络依赖 | ❌ 无 | ✅ 需要 |
| 数据大小 | 可控 | 固定 (~2GB) |
| 分辨率 | 可调 | 固定 |
| 生成时间 | ~5-30分钟 | 下载时间 |
| 数据质量 | 合成数据 | 真实模拟 |

---

*更新时间: 2026-03-24*