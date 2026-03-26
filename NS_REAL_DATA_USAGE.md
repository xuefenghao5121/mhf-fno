# NS 真实数据使用用例

## 天渊团队 (Tianyuan Team)
**仓库**: https://github.com/xuefenghao5121/mhf-fno
**版本**: v1.2.0
**日期**: 2026-03-26

---

## 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/xuefenghao5121/mhf-fno.git
cd mhf-fno

# 安装依赖
pip install -r requirements.txt
pip install -e .
```

### 2. 准备 NS 真实数据

#### 数据格式要求
```python
# 数据格式 (PyTorch tensor)
train_data = {
    'x': torch.Tensor,  # [N_train, C, H, W] - 输入场
    'y': torch.Tensor   # [N_train, C, H, W] - 输出场
}

test_data = {
    'x': torch.Tensor,  # [N_test, C, H, W]
    'y': torch.Tensor   # [N_test, C, H, W]
}
```

#### 保存数据
```python
import torch

# 假设你的数据
train_x = ...  # [N_train, C, H, W]
train_y = ...  # [N_train, C, H, W]
test_x = ...   # [N_test, C, H, W]
test_y = ...   # [N_test, C, H, W]

# 保存为 .pt 文件
torch.save({
    'x': train_x,
    'y': train_y
}, 'data/ns_train_real.pt')

torch.save({
    'x': test_x,
    'y': test_y
}, 'data/ns_test_real.pt')
```

---

## 使用方式

### 方式 A: 基础 MHF-FNO（推荐）

```python
import torch
from mhf_fno import MHFFNO

# 加载数据
train_data = torch.load('data/ns_train_real.pt')
test_data = torch.load('data/ns_test_real.pt')

train_x, train_y = train_data['x'], train_data['y']
test_x, test_y = test_data['x'], test_data['y']

# 创建模型（最佳配置）
model = MHFFNO.best_config(
    n_modes=(16, 16),      # 频率模式数
    hidden_channels=32,    # 隐藏通道数
)

# 训练
import torch.nn as nn
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.MSELoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    pred = model(train_x)
    loss = loss_fn(pred, train_y)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = loss_fn(test_pred, test_y)
        print(f"Epoch {epoch+1}: Train {loss.item():.4f}, Test {test_loss.item():.4f}")
```

### 方式 B: 带 PINO 物理约束（实验性）

```python
import torch
from mhf_fno import MHFFNO
from mhf_fno.pino_high_freq import HighFreqPINOLoss

# 创建模型
model = MHFFNO.best_config(n_modes=(16, 16), hidden_channels=32)

# 创建 PINO 损失函数
pino_loss_fn = HighFreqPINOLoss(
    lambda_physics=0.0001,  # 物理约束权重
    freq_threshold=0.5      # 高频阈值
)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    pred = model(train_x)
    
    # 使用 PINO 损失
    loss = pino_loss_fn(pred, train_y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss {loss.item():.4f}")
```

### 方式 C: 自定义配置

```python
from mhf_fno import create_hybrid_fno

# 自定义 MHF 层配置
model = create_hybrid_fno(
    n_modes=(16, 16),
    hidden_channels=64,
    n_layers=4,
    n_heads=4,              # 注意力头数
    mhf_layers=[0, 2, 3],   # 哪些层使用 MHF
    use_attention=True      # 是否使用跨头注意力
)
```

---

## 测试脚本

### 完整测试脚本
```bash
# 使用你的真实数据
python benchmark/test_ns_with_existing_data.py \
    --train_data data/ns_train_real.pt \
    --test_data data/ns_test_real.pt \
    --epochs 50 \
    --batch_size 32
```

### 快速测试（10 epochs）
```bash
python benchmark/test_ns_quick_conservative.py \
    --train_data data/ns_train_real.pt \
    --test_data data/ns_test_real.pt \
    --epochs 10
```

---

## 预期结果

### 基线对比（NS 32×32，500 train，100 test）

| 模型 | 参数量 | 参数减少 | Test Loss | vs FNO |
|------|--------|----------|-----------|--------|
| FNO | 453,361 | - | 0.3779 | 基准 |
| **MHF-FNO** | **232,177** | **-48.8%** | **0.3769** | **+0.00%** ✅ |
| MHF-PINO | 232,177 | -48.8% | 0.3771 | +0.05% ❌ |

### 关键结论
- **MHF-FNO** 已达最优：参数减少 48.8%，精度持平
- **PINO** 在当前数据上无效（3 轮测试均失败）
- **建议**：使用 MHF-FNO，不使用 PINO

---

## 配置建议

### 数据规模
| 数据规模 | 推荐配置 |
|----------|----------|
| < 500 样本 | `n_modes=(8,8), hidden=32, n_heads=2` |
| 500-2000 样本 | `n_modes=(16,16), hidden=32, n_heads=2` |
| > 2000 样本 | `n_modes=(32,32), hidden=64, n_heads=4` |

### 分辨率
| 分辨率 | 推荐配置 |
|--------|----------|
| 16×16 | `n_modes=(8,8)` |
| 32×32 | `n_modes=(16,16)` |
| 64×64 | `n_modes=(32,32)` |
| 128×128 | `n_modes=(64,64)` |

---

## PINO 使用说明

### 当前状态
- ✅ 代码已实现
- ❌ 在 NS 数据上无效
- ⚠️ 保留方向，等待更好的数据

### PINO 策略
1. **Round 1**: 高频噪声惩罚
2. **Round 2**: 自适应 lambda
3. **Round 3**: 梯度异常惩罚

### 失败原因
- 当前数据是标量场，缺少速度场和时间序列
- 无法计算真正的 NS 方程 PDE 残差
- MHF-FNO 已达最优，PINO 无法再提升

---

## 完整示例

```python
#!/usr/bin/env python3
"""
NS 真实数据测试完整示例
"""
import torch
import torch.nn as nn
from mhf_fno import MHFFNO

def main():
    # 1. 加载数据
    print("加载数据...")
    train_data = torch.load('data/ns_train_real.pt')
    test_data = torch.load('data/ns_test_real.pt')
    
    train_x, train_y = train_data['x'], train_data['y']
    test_x, test_y = test_data['x'], test_data['y']
    
    print(f"训练集: {train_x.shape}")
    print(f"测试集: {test_x.shape}")
    
    # 2. 创建模型
    print("\n创建模型...")
    model = MHFFNO.best_config(
        n_modes=(16, 16),
        hidden_channels=32
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")
    
    # 3. 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loss_fn = nn.MSELoss()
    
    # 4. 训练
    print("\n开始训练...")
    best_test_loss = float('inf')
    
    for epoch in range(50):
        # 训练
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        loss = loss_fn(pred, train_y)
        loss.backward()
        optimizer.step()
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_pred = model(test_x)
            test_loss = loss_fn(test_pred, test_y).item()
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}: Train {loss.item():.4f}, Test {test_loss:.4f}")
    
    print(f"\n✅ 最佳测试损失: {best_test_loss:.4f}")

if __name__ == "__main__":
    main()
```

---

## 常见问题

### Q1: 如何调整模型大小？
```python
# 更大的模型
model = MHFFNO.best_config(
    n_modes=(32, 32),      # 增加频率模式
    hidden_channels=64     # 增加隐藏通道
)

# 更小的模型
model = MHFFNO.best_config(
    n_modes=(8, 8),        # 减少频率模式
    hidden_channels=16     # 减少隐藏通道
)
```

### Q2: 如何使用 GPU？
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
train_x, train_y = train_x.to(device), train_y.to(device)
test_x, test_y = test_x.to(device), test_y.to(device)
```

### Q3: PINO 什么时候有效？
- ✅ 有速度场 + 时间序列数据
- ✅ 基线模型性能较差
- ❌ 标量场数据（当前情况）
- ❌ 基线已优（当前情况）

---

## 支持

- **GitHub Issues**: https://github.com/xuefenghao5121/mhf-fno/issues
- **文档**: `README.md`, `QUICK_START.md`
- **测试报告**: `FINAL_REPORT.md`, `PINO_OPTIMIZATION_FINAL.md`

---

**天渊团队**
**2026-03-26**
