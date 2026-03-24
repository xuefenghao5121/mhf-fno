# MHF-FNO 基准测试指南

本指南提供完整的测试脚本，可在任何服务器上运行 MHF-FNO 基准测试。

## 快速开始

### 1. 安装依赖

```bash
pip install neuralop torch numpy
```

### 2. 下载文件

需要以下文件：
- `run_benchmarks.py` - 主测试脚本
- `mhf_fno.py` - MHF-FNO 核心实现

### 3. 运行测试

```bash
# 测试 Darcy Flow (数据内置，最快)
python run_benchmarks.py --dataset darcy

# 测试 Burgers 方程 (需下载数据)
python run_benchmarks.py --dataset burgers

# 测试 Navier-Stokes (需下载大数据)
python run_benchmarks.py --dataset navier_stokes

# 测试所有数据集
python run_benchmarks.py --dataset all
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | darcy | 数据集 (darcy/burgers/navier_stokes/all) |
| `--n_train` | 1000 | 训练集大小 |
| `--n_test` | 200 | 测试集大小 |
| `--epochs` | 50 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 0.001 | 学习率 |
| `--seed` | 42 | 随机种子 |
| `--output` | benchmark_results.json | 输出文件 |

## 数据集说明

### Darcy Flow
- **描述**: 二阶椭圆 PDE，多孔介质流动
- **分辨率**: 16x16 (内置)
- **训练集**: 1000
- **测试集**: 50
- **状态**: ✅ 内置，无需下载

### Burgers Equation
- **描述**: 一维对流扩散方程
- **分辨率**: 128 (空间) x 100 (时间)
- **训练集**: 1000
- **测试集**: 200
- **状态**: ⚠️ 需下载 (~100MB)

### Navier-Stokes
- **描述**: 二维不可压缩流体
- **分辨率**: 64x64
- **训练集**: 1000
- **测试集**: 200
- **状态**: ⚠️ 需下载 (~1.5GB)

## 输出示例

### 控制台输出

```
============================================================
MHF-FNO 基准测试
============================================================
配置: {'n_train': 1000, 'n_test': 200, 'epochs': 50, ...}

############################################################
# 数据集: darcy
############################################################

📊 加载 Darcy Flow (16x16)...
✅ 加载成功: 训练 1000, 测试 200

数据集信息:
  名称: Darcy Flow
  分辨率: 16x16
  训练集: 1000
  测试集: 200

============================================================
测试 FNO (基准)
============================================================
参数量: 133,873
  Epoch 10/50: Train 0.1234, Test 0.1102, Time 2.3s
  ...

============================================================
结果对比
============================================================
指标                 FNO             MHF-FNO         变化            
------------------------------------------------------------
参数量              133,873         103,153         -22.9%
最佳测试Loss        0.1022          0.1072          +4.9%
推理延迟(ms)        3.59            3.26            -9.2%
```

### JSON 输出 (`benchmark_results.json`)

```json
{
  "config": {
    "n_train": 1000,
    "n_test": 200,
    "epochs": 50
  },
  "results": {
    "darcy": {
      "dataset": {
        "name": "Darcy Flow",
        "resolution": "16x16"
      },
      "models": {
        "FNO": {
          "parameters": 133873,
          "best_test_loss": 0.1022,
          "inference_ms": 3.59
        },
        "MHF-FNO": {
          "parameters": 103153,
          "best_test_loss": 0.1072,
          "inference_ms": 3.26
        }
      }
    }
  }
}
```

## 预期结果

基于 Darcy Flow 16x16 的测试结果：

| 指标 | FNO | MHF-FNO | 变化 |
|------|-----|---------|------|
| 参数量 | 133,873 | 103,153 | **-22.9%** |
| L2 误差 | 0.1022 | 0.1072 | +4.9% |
| 推理延迟 | 3.59ms | 3.26ms | **-9.2%** |

## 多头数量敏感性

```bash
# 测试不同头数 (需修改代码中的 n_heads 参数)
n_heads=2: 参数 113,393, Loss 0.1035 (精度优先)
n_heads=4: 参数 103,153, Loss 0.1072 (平衡)
n_heads=8: 参数 98,033,  Loss 0.1083 (参数效率优先)
```

## 常见问题

### Q: 数据下载失败？
A: NeuralOperator 会自动下载数据到 `~/.neuralop/data/`。如果下载慢，可以：
- 使用代理
- 手动下载数据集
- 只测试 Darcy Flow (内置)

### Q: GPU 内存不足？
A: 减小 batch_size：
```bash
python run_benchmarks.py --dataset darcy --batch_size 16
```

### Q: 训练时间太长？
A: 减少 epochs：
```bash
python run_benchmarks.py --dataset darcy --epochs 20
```

## 文件结构

```
mhf_fno_plugin/
├── run_benchmarks.py     # 主测试脚本
├── mhf_fno.py            # MHF-FNO 核心实现
├── BENCHMARK_GUIDE.md    # 本指南
└── benchmark_results.json # 测试结果
```

## GitHub 仓库

- 主仓库: https://github.com/xuefenghao5121/mhf-fno
- 测试脚本: `mhf_fno_plugin/run_benchmarks.py`

---

*更新时间: 2026-03-24*