# Benchmark - 基准测试

本目录包含 MHF-FNO 的基准测试脚本用于复现论文结果。

## 目录结构

```
benchmark/
├── README.md              # 本文件
├── BENCHMARK_GUIDE.md     # 完整基准测试指南
├── generate_data.py       # 生成 Darcy Flow 数据集
├── generate_ns_velocity.py # 生成 Navier-Stokes 速度场数据集
├── test_mhf_coda_pino_quick.py # 快速测试最佳配置 (MHF+CoDA+PINO)
├── data/                  # 数据存储目录
└── internal/              # 内部研究脚本 (开发调试用)
```

## 快速开始

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

## 更多研究

完整的研究脚本（包括多轮优化、诊断测试、对比实验等）请参见 `internal/` 目录。这些脚本主要用于开发和研究，商用用户只需关注本目录下的核心文件即可。

## 数据集

详见 `BENCHMARK_GUIDE.md`

| 数据集 | 生成方式 | 说明 |
|--------|---------|------|
| Darcy Flow | `generate_data.py` | 内置生成，无需下载 |
| Navier-Stokes | `generate_ns_velocity.py` | 内置生成，无需下载 |