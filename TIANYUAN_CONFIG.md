# 天渊团队配置详解

> 更新时间: 2026-03-26
> 版本: v1.2.0

---

## 团队基本信息

| 项目 | 信息 |
|------|------|
| **团队名称** | 天渊团队 (Tianyuan Team) |
| **Team ID** | `team_tianyuan_fft` |
| **项目** | MHF-FNO 神经网络优化 |
| **GitHub** | https://github.com/xuefenghao5121/mhf-fno |

---

## 团队成员

| 角色 | 名称 | 职责 |
|------|------|------|
| 🏛️ 架构师 | 天渊 | 整体架构设计、技术选型 |
| 🔬 研究员 | 天池 | 理论分析、适用性研究 |
| 💻 工程师 | 天渠 | 代码实现、测试验证 |
| 🧪 测试 | 天井 | 测试执行、结果分析 |

---

## 核心模型配置

### 1. MHFFNO (基础版)

```python
from mhf_fno import MHFFNO

# 最佳配置 (推荐)
model = MHFFNO.best_config(
    n_modes=(12, 12),      # 频率模式数
    hidden_channels=32,    # 隐藏通道数
    in_channels=1,         # 输入通道
    out_channels=1         # 输出通道
)

# 实际配置
# mhf_layers=[0, 2]  # 首尾层使用 MHF
# n_layers=3          # 总层数
# 参数量: ~232,177 (减少 48.8%)
```

---

### 2. MHFFNOWithAttention (推荐 ⭐)

```python
from mhf_fno import MHFFNOWithAttention

# 最佳配置 (推荐)
model = MHFFNOWithAttention.best_config(
    n_modes=(12, 12),      # 频率模式数
    hidden_channels=32,    # 隐藏通道数
    in_channels=1,
    out_channels=1,
    n_heads=4,             # 注意力头数
    attn_dropout=0.0,      # Dropout
    positional_embedding='grid'  # 位置嵌入
)

# 实际配置
# mhf_layers=[0, 2]        # MHF 层
# attention_layers=[0, 2]   # 注意力层
# bottleneck=4              # CoDA 瓶颈大小
# gate_init=0.1             # 门控初始化
# 参数量: ~232,923 (减少 48.6%)
```

---

### 3. 自定义配置

```python
from mhf_fno import create_mhf_fno_with_attention

model = create_mhf_fno_with_attention(
    # 基础参数
    n_modes=(16, 16),
    hidden_channels=64,
    in_channels=1,
    out_channels=1,
    n_layers=4,
    
    # MHF 参数
    mhf_layers=[0, 2, 3],   # 哪些层使用 MHF
    n_heads=8,               # MHF 头数
    head_dim=16,             # 每个头的维度
    
    # 注意力参数
    attention_layers=[0, 2, 3],
    bottleneck=8,            # CoDA 瓶颈
    gate_init=0.05,          # 门控初始化
    
    # 位置编码
    positional_embedding='grid'
)
```

---

## 数据集配置

### Darcy Flow 2D (椭圆型 PDE)

```python
# 数据配置
dataset = 'darcy'
resolution = 32           # 32×32 网格
n_train = 500            # 训练样本 (推荐 1000+)
n_test = 100             # 测试样本

# 模型配置
n_modes = (16, 16)       # resolution // 2
hidden_channels = 32

# 结果
# FNO:       453,361 参数, Loss=0.3935
# MHF-FNO:   232,177 参数, Loss=0.3645 (+7.36%)
# MHF+CoDA:  232,923 参数, Loss=0.3613 (+8.17%)
```

---

### Burgers 1D (抛物型 PDE)

```python
# 数据配置
dataset = 'burgers'
resolution = 256         # 1D 网格
n_train = 500
n_test = 100

# 模型配置
n_modes = (128,)         # 1D 用元组
hidden_channels = 32

# 结果
# FNO:       210,545 参数, Loss=0.0413
# MHF-FNO:   142,961 参数, Loss=0.0369 (+10.83%)
# MHF+CoDA:  143,707 参数, Loss=0.0281 (+32.12%) ⭐
```

---

### Navier-Stokes 2D (双曲型 PDE)

```python
# 数据配置
dataset = 'navier_stokes'
resolution = 32
n_train = 1000           # ⚠️ 需要更多数据
n_test = 200
viscosity = 1e-3
n_steps = 100

# 模型配置
n_modes = (16, 16)
hidden_channels = 32
mhf_layers = [0]         # ⚠️ 只用第一层 (保守)

# 结果
# FNO:       453,361 参数, Loss=0.3801
# MHF-FNO:   232,177 参数, Loss=0.3849 (-1.27%)
# MHF+CoDA:  232,923 参数, Loss=0.3844 (-1.15%)

# ⚠️ NS 需要优化：
# 1. 增加数据量 (n_train=1000+)
# 2. 减少 MHF 层数
# 3. 使用 PINO 物理约束
```

---

## 训练配置

### 方案 A: 快速测试 (推荐)

```python
config = {
    'epochs': 20,
    'n_train': 200,
    'n_test': 50,
    'batch_size': 32,
    'learning_rate': 1e-3,
    'seed': 42,
    'device': 'cpu'  # 或 'cuda'
}

# 预计时间: ~4 分钟
```

---

### 方案 B: 标准测试

```python
config = {
    'epochs': 50,
    'n_train': 500,
    'n_test': 100,
    'batch_size': 32,
    'learning_rate': 5e-4,    # 降低学习率
    'weight_decay': 1e-4,     # 正则化
    'seed': 42,
    'device': 'cpu'
}

# 预计时间: ~10 分钟
```

---

### 方案 C: 完整测试 (1000 样本)

```python
config = {
    'epochs': 50,
    'n_train': 1000,
    'n_test': 200,
    'batch_size': 32,
    'learning_rate': 5e-4,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,        # 梯度裁剪
    'early_stopping': True,   # 早停
    'patience': 10,
    'seed': 42,
    'device': 'cpu'
}

# 预计时间: ~50 分钟
```

---

## PINO 物理约束配置

```python
from mhf_fno import PINOLoss

# PINO 损失函数
criterion = PINOLoss(
    lambda_physics=0.01,    # 物理约束权重
    viscosity=1e-3,         # NS 粘性系数
    dt=0.01                 # 时间步长
)

# 训练循环
for epoch in range(epochs):
    for x, y in dataloader:
        # 前向传播
        y_pred = model(x)
        
        # PINO 损失 = 数据损失 + 物理损失
        loss = criterion(y_pred, y, x)
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

---

## 优化器配置

### Adam (默认)

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-5
)
```

---

### OneCycleLR (推荐)

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(dataloader),
    pct_start=0.3
)
```

---

## 性能基准

### 测试结果汇总

| 数据集 | PDE 类型 | 模型 | 参数减少 | vs FNO | 推荐 |
|--------|----------|------|----------|--------|------|
| **Darcy 2D** | 椭圆型 | MHF+CoDA | -48.6% | **+8.17%** | ✅ |
| **Burgers 1D** | 抛物型 | MHF+CoDA | -31.7% | **+32.12%** | ✅✅ |
| **NS 2D** | 双曲型 | MHF+CoDA | -48.6% | **-1.15%** | ⚠️ |

---

### 最佳实践

| 数据集 | 推荐配置 | 原因 |
|--------|----------|------|
| **Darcy 2D** | `mhf_layers=[0,2]`, `n_heads=4` | 频率解耦，效果最佳 |
| **Burgers 1D** | `mhf_layers=[0,2]`, `n_heads=4` | 1D 特性匹配 |
| **NS 2D** | `mhf_layers=[0]`, `n_heads=2`, **PINO** | 强耦合，需保守 |

---

## 运行命令

### 快速测试

```bash
cd /root/.openclaw/workspace/memory/projects/tianyuan-fft/benchmark

# 单数据集
python run_benchmarks.py --dataset darcy --epochs 20 --n_train 200

# 全部数据集
python run_benchmarks.py --dataset all --epochs 20 --n_train 200
```

---

### 数据生成

```bash
cd /root/.openclaw/workspace/memory/projects/tianyuan-fft/benchmark

# Darcy Flow
python generate_data.py --dataset darcy --n_train 500 --n_test 100 --resolution 32

# Burgers
python generate_data.py --dataset burgers --n_train 500 --n_test 100 --resolution 256

# Navier-Stokes
python generate_data.py --dataset navier_stokes --n_train 1000 --n_test 200 --resolution 32
```

---

## 文件结构

```
tianyuan-fft/
├── mhf_fno/              # 核心代码
│   ├── __init__.py       # 导出接口
│   ├── mhf_fno.py        # MHF-FNO 实现
│   ├── mhf_1d.py         # 1D 版本
│   ├── mhf_2d.py         # 2D 版本
│   └── mhf_attention.py  # CoDA 注意力
│
├── benchmark/            # 基准测试
│   ├── generate_data.py  # 数据生成
│   ├── run_benchmarks.py # 标准测试
│   ├── test_ns_pino.py   # PINO 测试
│   └── data/             # 数据目录
│
├── README.md             # 项目文档
├── requirements.txt      # 依赖
└── setup.py              # 安装配置
```

---

此配置文档由西西整理
