# Data - 数据集目录

本目录用于存放训练和测试数据集。

## 目录结构

```
data/
├── darcy/           # Darcy Flow 数据
│   ├── train.pt
│   └── test.pt
├── burgers/         # Burgers 方程数据
│   └── burgers_data.pt
└── navier_stokes/   # Navier-Stokes 数据
    └── ns_data.pt
```

## 数据获取

### 方式 1：自动下载

运行 `benchmark/run_benchmarks.py` 会自动下载数据到 `~/.neuralop/data/`

### 方式 2：手动下载

从 Zenodo 下载：
- https://zenodo.org/records/10994462

下载后放到本目录或 `~/.neuralop/data/`

## 注意

- 本目录默认不提交到 Git (已在 .gitignore 中配置)
- 大文件请使用 Git LFS 或单独下载