# Data - 数据集目录

本目录用于存放训练和测试数据集。

---

## 数据集下载地址

### PDEBench (H5 格式) ⭐ 推荐

官方主页: https://pdebench.github.io/

| 数据集 | 下载地址 |
|--------|----------|
| Darcy Flow 2D Train | https://darus.uni-stuttgart.de/api/access/datafile/152002 |
| Darcy Flow 2D Test | https://darus.uni-stuttgart.de/api/access/datafile/152003 |
| Navier-Stokes 2D Train | https://darus.uni-stuttgart.de/api/access/datafile/151996 |
| Navier-Stokes 2D Test | https://darus.uni-stuttgart.de/api/access/datafile/151997 |
| Burgers 1D | https://darus.uni-stuttgart.de/api/access/datafile/151990 |

### NeuralOperator (PT 格式)

自动下载到 `~/.neuralop/data/`，无需手动下载。

---

## 快速下载

```bash
cd data

# Darcy Flow 2D
wget https://darus.uni-stuttgart.de/api/access/datafile/152002 -O 2D_DarcyFlow_Train.h5
wget https://darus.uni-stuttgart.de/api/access/datafile/152003 -O 2D_DarcyFlow_Test.h5

# Navier-Stokes 2D
wget https://darus.uni-stuttgart.de/api/access/datafile/151996 -O 2D_NS_Train.h5
wget https://darus.uni-stuttgart.de/api/access/datafile/151997 -O 2D_NS_Test.h5

# Burgers 1D
wget https://darus.uni-stuttgart.de/api/access/datafile/151990 -O 1D_Burgers.h5
```

---

## 目录结构

```
data/
├── 2D_DarcyFlow_Train.h5    # Darcy 训练数据
├── 2D_DarcyFlow_Test.h5     # Darcy 测试数据
├── 2D_NS_Train.h5           # NS 训练数据
├── 2D_NS_Test.h5            # NS 测试数据
├── 1D_Burgers.h5            # Burgers 数据
└── README.md
```

---

## 使用方式

```bash
# PT 格式 (默认，自动下载)
python benchmark/run_benchmarks.py --dataset darcy

# H5 格式 (需手动下载)
python benchmark/run_benchmarks.py --dataset darcy --format h5 \
    --data_path data/2D_DarcyFlow_Train.h5
```

---

## 注意

- 本目录的数据文件不会提交到 Git (已在 .gitignore 中配置)
- 大文件请单独下载