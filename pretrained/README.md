# MHF-FNO 预训练模型使用指南 (v1.6.4)

## 概述

本目录包含 MHF-FNO (Multi-Head Fourier Neural Operator) 预训练模型的完整工具链：
- **预训练模型**: Darcy Flow 32x32 上训练的推理就绪模型
- **训练脚本**: 在本地数据集上训练新模型
- **推理脚本**: 加载模型进行预测
- **数据加载器**: 支持多种格式的本地数据集加载

## 文件结构

```
pretrained/
├── models/                              # 预训练模型输出目录
│   └── mhf_fno_darcy_pretrained.pth     # Darcy Flow 预训练模型
├── train_pretrained.py                  # 训练脚本
├── inference.py                         # 推理脚本
├── local_data_loader.py                 # 本地数据加载器
└── README.md                            # 本文档
```

## 快速开始

### 1. 加载预训练模型进行推理

```python
import torch
import sys
sys.path.insert(0, '/path/to/tianyuan-mhf-fno')
from mhf_fno import MHFFNOWithAttention

# 加载模型
checkpoint = torch.load('pretrained/models/mhf_fno_darcy_pretrained.pth', 
                         map_location='cpu', weights_only=False)
config = checkpoint['config']

# 重建模型
model = MHFFNOWithAttention.best_config(**config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
x = torch.randn(1, 1, 32, 32)  # [B, C, H, W]
with torch.no_grad():
    y = model(x)
print(f"输入: {x.shape} -> 输出: {y.shape}")
```

### 2. 命令行推理

```bash
cd tianyuan-mhf-fno

# 使用预训练模型推理
python3 pretrained/inference.py \
    --model pretrained/models/mhf_fno_darcy_pretrained.pth \
    --input /home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/darcy_test_32.pt \
    --output predictions.pt \
    --n_samples 10
```

### 3. 在本地数据集上训练新模型

```bash
# Darcy Flow (默认, 30 epochs)
python3 pretrained/train_pretrained.py --epochs 30

# Navier-Stokes
python3 pretrained/train_pretrained.py --dataset navier_stokes --epochs 20

# 自定义数据集
python3 pretrained/train_pretrained.py \
    --dataset custom \
    --train_path ./data/train.pt \
    --test_path ./data/test.pt \
    --epochs 50

# 快速验证
python3 pretrained/train_pretrained.py --epochs 5 --batch_size 32
```

### 4. 使用数据加载器

```python
from pretrained.local_data_loader import LocalDataLoader

loader = LocalDataLoader()

# 加载 Darcy 数据
train_x, train_y, test_x, test_y = loader.load_darcy(n_train=1000, n_test=200)

# 加载自定义数据
train_x, train_y, test_x, test_y = loader.load_custom(
    train_path='./data/train.pt',
    test_path='./data/test.pt',
)

# 一站式获取 DataLoader
from torch.utils.data import DataLoader
train_loader, test_loader = loader.get_dataloaders(
    dataset='darcy', batch_size=64
)
```

## 预训练模型详情

| 属性 | 值 |
|------|------|
| 模型架构 | MHF-FNO + CODA (Cross-Head Attention) |
| 数据集 | Darcy Flow 32x32 |
| 训练样本 | 5,000 |
| 测试样本 | 1,000 |
| 训练轮数 | 30 |
| 最佳测试损失 | 0.064968 |
| 参数量 | 308,411 |
| 输入形状 | [B, 1, 32, 32] |
| 输出形状 | [B, 1, 32, 32] |
| 优化器 | AdamW (lr=1e-3) |
| 学习率调度 | CosineAnnealing |
| 损失函数 | L2 相对误差 |

### 模型配置

```json
{
    "n_modes": [16, 16],
    "hidden_channels": 32,
    "in_channels": 1,
    "out_channels": 1,
    "n_heads": 2,
    "mhf_layers": [0, 2],
    "attention_layers": [0, 2]
}
```

## 支持的数据集

| 数据集 | 默认分辨率 | 默认路径 | 状态 |
|--------|-----------|----------|------|
| Darcy Flow | 32x32 | `mhf-data/darcy_train_32.pt` | ✅ 可用 |
| Navier-Stokes | 128x128 | `mhf-data/nsforcing_train_128.pt` | ✅ 可用 |
| Burgers | 1D | `mhf-data/rand_burgers_data_R10.pt` | ✅ 可用 |
| 自定义 | 任意 | 用户指定 | ✅ 支持 |

### 数据格式支持

- **PT/PTH**: PyTorch 格式，支持 dict (x/y)、tuple (x,y)、single tensor
- **NPY**: NumPy 数组
- **NPZ**: NumPy 压缩格式
- **H5/HDF5**: HDF5 格式 (需安装 h5py)

## .pth 模型文件结构

预训练模型保存为标准 PyTorch checkpoint，包含以下字段：

```python
{
    'model_state_dict': {...},      # 模型权重
    'config': {...},                 # 模型配置参数
    'dataset': 'darcy',             # 训练数据集名称
    'version': '1.6.4',            # MHF-FNO 版本
    'train_history': {...},         # 训练历史 (train_loss, test_loss, lr)
    'best_test_loss': 0.064968,     # 最佳测试损失
    'total_epochs': 30,             # 总训练轮数
    'timestamp': '2026-04-02T...',  # 训练时间戳
    'input_shape': [1, 32, 32],    # 输入形状
    'output_shape': [1, 32, 32],   # 输出形状
}
```

## 自定义训练

### 完整参数列表

```
python3 pretrained/train_pretrained.py \
    --dataset darcy|navier_stokes|burgers|custom \
    --train_path ./train.pt \          # 自定义数据集必需
    --test_path ./test.pt \            # 自定义数据集必需
    --n_train 1000 \                   # 限制训练样本数
    --n_test 200 \                     # 限制测试样本数
    --epochs 50 \                      # 训练轮数
    --batch_size 64 \                  # 批大小
    --lr 1e-3 \                        # 学习率
    --hidden_channels 32 \             # 隐藏通道数
    --n_heads 4 \                      # MHF 头数
    --output_dir ./models \            # 输出目录
    --log_interval 5                   # 日志打印间隔
```

### 推理完整参数

```
python3 pretrained/inference.py \
    --model ./models/mhf_fno_darcy_pretrained.pth \  # 模型路径 (必需)
    --input ./data/test.pt \                          # 输入数据 (必需)
    --input_key x \                                   # 字典中的 key
    --output predictions.pt \                         # 输出路径
    --n_samples 100 \                                 # 限制样本数
    --batch_size 64 \                                 # 推理批大小
    --device auto                                     # auto|cpu|cuda
```

## 依赖

```
torch >= 2.0
neuralop >= 2.0
numpy
```

可选:
```
h5py    # H5 格式支持
scipy   # MAT 格式支持
```
