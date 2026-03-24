# MHF-FNO 基准测试指南

本文档提供完整的数据集下载和测试说明。

---

## 数据集信息

### 内置数据（无需下载）

| 数据集 | 分辨率 | 训练集 | 测试集 | 位置 |
|--------|--------|--------|--------|------|
| **Darcy Flow 16** | 16×16 | 1,000 | 50 | NeuralOperator 内置 |

### 需下载数据

| 数据集 | 分辨率 | 训练集 | 测试集 | 大小 |
|--------|--------|--------|--------|------|
| Burgers | 128 | 1,000 | 200 | ~100MB |
| Navier-Stokes | 64×64 | 1,000 | 200 | ~1.5GB |

---

## 数据下载方式

### 方式 1：自动下载（推荐）

运行测试脚本时会自动下载数据到 `~/.neuralop/data/`：

```bash
python run_benchmarks.py --dataset darcy    # Darcy 内置，无需下载
python run_benchmarks.py --dataset burgers   # 自动下载 Burgers
```

### 方式 2：手动下载

如果自动下载失败，可手动下载：

**Burgers Equation**:
```
下载地址: https://zenodo.org/records/10994462/files/burgers_data.pt
存放位置: ~/.neuralop/data/burgers_data.pt
```

**Navier-Stokes**:
```
下载地址: https://zenodo.org/records/10994462/files/ns_data.pt
存放位置: ~/.neuralop/data/ns_data.pt
```

**Zenodo 数据集主页**:
- https://zenodo.org/records/10994462

---

## 运行测试

### 快速示例（无需下载）

```bash
# 使用合成数据，测试 FNO vs MHF-FNO
python example.py
```

**预期输出**:
```
使用设备: cpu
生成数据: 训练 500, 测试 100, 分辨率 16x16
✅ 数据生成完成

============================================================
结果对比
============================================================
指标              FNO             MHF-FNO         变化            
------------------------------------------------------------
参数量              133,873         108,772         -18.7%
最佳测试Loss        0.0945          0.1060          +12.2%

✅ 测试完成！
```

### 完整基准测试

```bash
# Darcy Flow (内置数据)
python run_benchmarks.py --dataset darcy

# Burgers (需下载)
python run_benchmarks.py --dataset burgers

# 自定义参数
python run_benchmarks.py \
    --dataset darcy \
    --n_train 500 \
    --n_test 100 \
    --epochs 30
```

---

## NeuralOperator 2.0.0 兼容性

本脚本已适配 NeuralOperator 2.0.0 API：

| 旧参数 | 新参数 (2.0.0) |
|--------|----------------|
| `n_test` | `n_tests` (列表) |
| `test_batch_size` | `test_batch_sizes` (列表) |

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

### Q: 数据下载太慢？
A: 
- 使用代理
- 或手动下载到 `~/.neuralop/data/`
- 或只运行 `example.py`（使用合成数据）

### Q: NeuralOperator 报错？
A: 确保版本为 2.0.0：
```bash
pip install neuralop==0.3.0
```

### Q: GPU 内存不足？
A: 减小 batch_size：
```bash
python run_benchmarks.py --dataset darcy --batch_size 16
```

---

*更新时间: 2026-03-24*