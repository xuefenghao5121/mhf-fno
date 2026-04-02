# MHF-FNO 预训练性能优化指南 - 单核利用率问题

## 问题分析

**原始问题**: 使用128核并行训练时，单核CPU使用率只有不到30%，导致整体训练效率低下。

**根本原因分析**:

1. **数据加载瓶颈**: 原始代码使用 `num_workers=0`，所有数据加载都在主进程中完成，导致计算等待IO
2. **线程配置不合理**: 默认让PyTorch/OpenBLAS/MKL自动管理线程，导致超线程竞争
3. **缺乏CPU绑定**: 进程可以在不同核心之间迁移，导致缓存失效
4. **不必要的内存分配**: 每次前向传播都完整分配零张量，增加内存初始化开销

## 优化方案

### 1. 数据加载优化

- **添加 `num_workers` 参数**: 支持多进程并行数据加载
- **添加 `prefetch_factor` 和 `persistent_workers`**: 减少worker重启开销
- **可选 `pin_memory`**: 固定内存加速数据传输

### 2. OpenMP/MKL线程配置

- **可配置 `omp_num_threads`**: 通过环境变量强制设置每个进程的线程数
- **建议配置**: 每个进程使用 **物理核心数**，避免超线程竞争
- **原理**: FFT操作是CPU密集型，超线程不能提供真正的并行加速，反而会导致缓存竞争

### 3. CPU线程绑定

- **添加 `--bind_cpu` 选项**: 使用 `psutil` 将进程绑定到指定CPU核心范围
- **配合MPI**: 每个MPI进程独占一段连续的CPU核心，避免跨核心迁移
- **收益**: 提升L1/L2缓存命中率，减少缓存失效

### 4. 内存分配优化

- **优化频域卷积内存分配**: 只计算需要的频率部分，减少不必要的零初始化
- **减少内存碎片**: 更紧凑的内存分配策略改善缓存局部性

## 使用方法

### 单核单进程训练 (开发调试)

```bash
# 基础用法
python train_pretrained.py --dataset darcy

# 启用基本优化 (16核机器)
python train_pretrained.py --dataset darcy \
    --omp_num_threads 16 \
    --num_workers 4 \
    --bind_cpu
```

### 多进程MPI并行训练 (128核推荐配置)

假设你有 **128个物理核心**，推荐配置：

#### 方案 A: 8进程 × 16线程 (总共128核)

```bash
# 使用mpirun启动
mpirun -np 8 python run_parallel_train.py \
    --dataset darcy \
    --omp_num_threads 16 \
    --num_workers 2 \
    --bind_cpu \
    --epochs 50 \
    --batch_size 64
```

#### 方案 B: 16进程 × 8线程 (总共128核)

```bash
mpirun -np 16 python run_parallel_train.py \
    --dataset darcy \
    --omp_num_threads 8 \
    --num_workers 2 \
    --bind_cpu \
    --epochs 50 \
    --batch_size 64
```

#### 方案 C: 4进程 × 32线程 (总共128核)

```bash
mpirun -np 4 python run_parallel_train.py \
    --dataset navier_stokes \
    --omp_num_threads 32 \
    --num_workers 4 \
    --bind_cpu \
    --epochs 100 \
    --batch_size 32
```

## 参数调优指南

### 如何选择 `omp_num_threads`?

| 总物理核心 | 推荐进程数 | 推荐 omp_num_threads | 说明 |
|-----------|-----------|---------------------|------|
| 16 | 1 | 16 | 单进程，充分利用所有核心 |
| 32 | 2-4 | 16-8 | 平衡数据加载和计算 |
| 64 | 4-8 | 16-8 | 避免过多进程竞争内存带宽 |
| 128 | 8-16 | 16-8 | **推荐 8×16 配置** |
| 256 | 16-32 | 16-8 | 注意内存带宽瓶颈 |

**推荐原则**:
- 每个NUMA节点内设置一个进程，线程数匹配该NUMA节点的物理核心数
- 不推荐超线程: 一个物理核心只分配一个线程

### 如何选择 `num_workers`?

- `num_workers` 用于数据加载，不包含计算线程
- 推荐: `num_workers = omp_num_threads / 4` 到 `omp_num_threads / 2`
- 最大不超过 `omp_num_threads / 2`，否则数据加载会竞争CPU资源

### CPU绑定说明

- `--bind_cpu` 需要安装 `psutil`: `pip install psutil`
- 配合MPI使用时，每个进程自动分配连续的核心范围:
  - rank 0: cores `[0 ... omp_num_threads-1]`
  - rank 1: cores `[omp_num_threads ... 2*omp_num_threads-1]`
  - ... 以此类推

## 预期性能提升

| 优化项 | 单核利用率提升 | 训练速度提升 | 说明 |
|--------|---------------|-------------|------|
| 基础优化 (num_workers + omp_threads) | +10-20% | +15-25% | 解决数据加载瓶颈 |
| + CPU绑定 | +5-15% | +10-20% | 提升缓存命中率 |
| + 内存分配优化 | +2-5% | +3-8% | 减少不必要的内存操作 |
| **总计** | **+20-40%** → 可达 **70-90%** | **+30-50%** | 综合优化效果 |

## 故障排除

### 问题 1: `psutil` 找不到

```
Error: No module named 'psutil'
```

**解决**:
```bash
pip install psutil
```
或者不使用 `--bind_cpu` 选项（仍然可以获得其他优化收益）

### 问题 2: 内存不足

**解决**:
- 减少 `batch_size`
- 增加进程数，减少每个进程的 `omp_num_threads`

### 问题 3: 仍然利用率低

**检查**:
1. 是否设置了正确的 `omp_num_threads`? 环境变量应该正确传递
2. 是否使用了超线程? 尝试减少线程数到物理核心数
3. 数据加载是否瓶颈? 可以尝试增加 `num_workers`

### 问题 4: MPI绑定失败

**替代方案**:
如果MPI环境不支持CPU绑定，可以手动设置每个进程的 `--start_core`:

```bash
# 手动启动两个进程，分别绑定到不同核心范围
python train_pretrained.py --omp_num_threads 16 --start_core 0 --bind_cpu &
python train_pretrained.py --omp_num_threads 16 --start_core 16 --bind_cpu &
wait
```

## 验证优化效果

建议测试以下几种配置，记录单核利用率和每epoch训练时间:

| 配置 | omp_num_threads | num_workers | bind_cpu | 预期单核利用率 |
|------|-----------------|-------------|----------|---------------|
| 原始 | (auto) | 0 | 否 | ~20-30% |
| 优化1 | 16 | 4 | 否 | ~40-55% |
| 优化2 | 16 | 4 | 是 | ~60-75% |
| 优化3 (推荐 8×16) | 16 per process | 2 | 是 | **70-90%** |

## 依赖

- Python 3.8+
- PyTorch 1.10+
- `psutil` (可选，用于CPU绑定)
- `mpi4py` (可选，用于多进程并行)

## 更新日志

- **v1.6.4**: 初始版本，包含本文所述优化
- 优化内容:
  - 添加性能优化参数到 `train_pretrained.py`
  - 创建MPI并行启动脚本 `run_parallel_train.py`
  - 优化 `MHFSpectralConv` 和 `MHFSpectralConvWithAttention` 的内存分配
  - 编写优化说明文档

## 作者

天渊团队 (Tianyuan Team) - MHF-FNO 项目
