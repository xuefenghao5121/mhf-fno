# Examples 示例代码

本目录提供了 MHF-FNO 的简洁使用示例，分别对应三种数据集的最佳配置。

## 📁 文件说明

| 文件名 | 功能 | 数据集 | 关键特性 |
|--------|------|--------|----------|
| **example_darcy.py** | 🎯 Darcy Flow 示例 | Darcy Flow 32x32 | MHF + CoDA 最佳配置 |
| **example_ns.py** | 🌊 Navier-Stokes 示例 | Navier-Stokes 128x128 | MHF + CoDA + PINO 最佳配置 |
| **example_burgers.py** | 🔥 Burgers 方程示例 | Burgers 1D | MHF + CoDA 最佳配置 |

## 🚀 快速上手

### 1. 运行 Darcy Flow 示例
```bash
python example_darcy.py
```
**最佳配置**:
- 模型: MHF-FNO with CoDA
- 参数: hidden_channels=32, n_modes=(16,16), mhf_layers=[0,2]
- 数据: `/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/darcy_train_32.pt`

### 2. 运行 Navier-Stokes 示例
```bash
python example_ns.py
```
**最佳配置**:
- 模型: MHF-FNO with CoDA + PINO
- 参数: hidden_channels=64, n_modes=(64,64), mhf_layers=[0,2,4]
- 数据: `/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/nsforcing_train_128.pt`

### 3. 运行 Burgers 方程示例
```bash
python example_burgers.py
```
**最佳配置**:
- 模型: MHF-FNO with CoDA
- 参数: hidden_channels=32, n_modes=(n_x//2,), mhf_layers=[0,2]
- 数据: `/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/burgers/*.mat`

## 📊 最佳配置说明

### Darcy Flow (32x32)
```python
model = MHFFNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_modes=(16, 16),  # 分辨率的一半
    n_layers=4,
    mhf_layers=[0, 2],  # 在第1和第3层使用MHF
    n_heads=2,
    use_coda=True,      # 启用 Cross-Head Attention
    use_pino=False,
)
```

### Navier-Stokes (128x128)
```python
model = MHFFNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    n_modes=(64, 64),   # 分辨率的一半
    n_layers=6,
    mhf_layers=[0, 2, 4],  # 在第1、3、5层使用MHF
    n_heads=4,
    use_coda=True,       # 启用 Cross-Head Attention
    use_pino=True,       # 启用物理约束
    pino_weight=0.1,
)
```

### Burgers 方程 (1D)
```python
model = MHFFNO(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    n_modes=(n_x // 2,),  # 1D 分辨率的一半
    n_layers=4,
    mhf_layers=[0, 2],
    n_heads=2,
    use_coda=True,
    use_pino=False,
)
```

## 🔧 数据路径配置

所有示例使用统一的真实数据路径:
- **真实数据根目录**: `/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data/`
- **Darcy Flow**: `darcy_train_32.pt` / `darcy_test_32.pt`
- **Navier-Stokes**: `nsforcing_train_128.pt` / `nsforcing_test_128.pt`
- **Burgers**: `burgers/` 目录下的 `.mat` 文件

## 📝 使用说明

1. **确保数据存在**: 检查上述数据路径是否存在对应文件
2. **运行示例**: 直接运行对应的示例文件
3. **查看结果**: 每个示例会输出训练损失和测试损失
4. **保存模型**: 训练完成后会保存最佳模型到当前目录

## 🎯 特点

- **简洁易懂**: 每个示例文件约100行代码
- **最佳配置**: 使用经过验证的最佳参数
- **真实数据**: 直接加载本地真实数据文件
- **即用性**: 无需复杂配置，直接运行
- **可扩展**: 可修改配置适应不同需求