# Benchmark - 基准测试

本目录包含 MHF-FNO 的基准测试脚本。

## 文件说明

| 文件 | 说明 |
|------|------|
| `run_benchmarks.py` | 多数据集基准测试脚本 |
| `BENCHMARK_GUIDE.md` | 数据集下载指南 |

## 使用方式

```bash
cd benchmark

# 测试 Darcy Flow (内置数据)
python run_benchmarks.py --dataset darcy

# 测试 Burgers (需下载数据)
python run_benchmarks.py --dataset burgers

# 自定义参数
python run_benchmarks.py \
    --dataset darcy \
    --n_train 500 \
    --n_test 100 \
    --epochs 30
```

## 数据集

详见 `BENCHMARK_GUIDE.md`

| 数据集 | 状态 | 说明 |
|--------|------|------|
| Darcy Flow | ✅ 内置 | 无需下载 |
| Burgers | ⚠️ 需下载 | ~100MB |
| Navier-Stokes | ⚠️ 需下载 | ~1.5GB |