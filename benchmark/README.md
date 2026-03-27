# Benchmark 基准测试

本目录包含 MHF-FNO 的基准测试脚本，用于复现论文结果，也支持客户自定义数据集。

## 📁 目录结构

\`\`\`
benchmark/
├── README.md              # 本文件
├── data_loader.py         # ✨ 通用数据加载器 (v1.6.1)
├── run_benchmarks.py      # 完整基准测试入口
├── generate_data.py       # 统一数据生成器 (Darcy/Burgers/NS)
├── quick_test.py          # 快速测试最佳配置 (MHF+CoDA+PINO)
└── data/                  # 数据存储目录
\`\`\`

## 🚀 快速开始

### 方式1: NeuralOperator 官方数据集（论文复现）
```bash
# 1. 手动下载数据集（推荐 - 避免网络问题）
mkdir -p benchmark/data

# Navier-Stokes 数据集
wget https://zenodo.org/records/12825163/files/nsforcing_v1.0_128.tar.gz
tar -xzf nsforcing_v1.0_128.tar.gz -C benchmark/data/

# Darcy Flow 数据集
wget https://zenodo.org/records/12784353/files/darcy_poisson_v1.0_64.tar.gz
tar -xzf darcy_poisson_v1.0_64.tar.gz -C benchmark/data/

# 2. 运行基准测试
cd benchmark
python run_benchmarks.py \\
    --dataset navier_stokes \\
    --n_train 1000 \\
    --n_test 200 \\
    --resolution 128
```

### 方式2: 客户提供的 H5 文件（推荐用于商业项目）⭐
```bash
# 客户已经提供了 train.h5 和 test.h5 文件
# 放到 benchmark/data/ 目录下:
#   benchmark/data/customer_train.h5
#   benchmark/data/customer_test.h5

# 使用 Python 脚本加载
python -c "
from data_loader import load_dataset
train_x, train_y, test_x, test_y, info = load_dataset(
    dataset_name='custom',
    data_format='h5',
    train_path='./data/customer_train.h5',
    test_path='./data/customer_test.h5',
    n_train=1000,
    n_test=200,
    resolution=64,
)
print('✅ 客户数据集加载成功')
print(f'训练集: {train_x.shape}, 测试集: {test_x.shape}')
"
```

### 方式3: 客户提供的 PT/Torch 文件
```bash
# 客户已经提供了 train.pt 和 test.pt 文件
# 放到 benchmark/data/ 目录下:
#   benchmark/data/customer_train.pt
#   benchmark/data/customer_test.pt

python -c "
from data_loader import load_dataset
train_x, train_y, test_x, test_y, info = load_dataset(
    dataset_name='custom',
    data_format='pt',
    train_path='./data/customer_train.pt',
    test_path='./data/customer_test.pt',
    n_train=1000,
    n_test=200,
)
print('✅ 客户数据集加载成功')
"
```

### 方式4: 本地生成数据集（无需下载）
```bash
# 生成所有数据集
python generate_data.py --dataset all
```

## 📊 数据加载器说明 (v1.6.1)

从 v1.6.1 开始，数据加载器支持**多种数据源**：

| 数据源类型 | 适用场景 | 是否需要网络 | 配置参数 |
|-----------|----------|--------------|----------|
| **NeuralOperator官方** | 论文复现、基准测试 | 可选（可手动下载） | `dataset_name='navier_stokes'` |
| **客户H5文件** | 商业项目、客户数据集 | ❌ 不需要 | `dataset_name='custom', data_format='h5', train_path=..., test_path=...` |
| **客户PT文件** | 商业项目、客户数据集 | ❌ 不需要 | `dataset_name='custom', data_format='pt', train_path=..., test_path=...` |

## 📥 完整使用示例

### 1. NeuralOperator 官方数据集

```python
from data_loader import load_dataset

# 加载 Navier-Stokes 数据集 (NeuralOperator官方)
train_x, train_y, test_x, test_y, info = load_dataset(
    dataset_name='navier_stokes',
    n_train=1000,
    n_test=200,
    resolution=128,
    download=False,  # 手动下载模式（默认）
)

# 返回格式:
# train_x: torch.Size([1000, 1, 128, 128])
# train_y: torch.Size([1000, 1, 128, 128])
# test_x: torch.Size([200, 1, 128, 128])
# test_y: torch.Size([200, 1, 128, 128])
# info: dict with metadata
```

### 2. 客户提供的 H5 文件 ⭐

```python
from data_loader import load_dataset

# H5 文件格式:
# - 方式1: 标准 x/y 数据集
#   train.h5 包含:
#     - x: [N, H, W] 或 [N, C, H, W]
#     - y: [N, H, W] 或 [N, C, H, W]
#
# - 方式2: PDEBench 时间序列格式
#   train.h5 包含:
#     - u: [N, T, H, W] (时间序列)
#   会自动提取: x = u[:, 0, :, :], y = u[:, -1, :, :]
#
# - 方式3: 任意键名（自动检测）
#   train.h5 包含任意键名，会自动使用第一个和最后一个

train_x, train_y, test_x, test_y, info = load_dataset(
    dataset_name='custom',       # 自定义数据集
    data_format='h5',            # H5 格式
    train_path='./data/customer_train.h5',
    test_path='./data/customer_test.h5',
    n_train=1000,
    n_test=200,
    resolution=64,
)
```

### 3. 客户提供的 PT 文件 ⭐

```python
from data_loader import load_dataset

# PT 文件格式:
# - 方式1: 字典格式
#   train.pt = {'x': ..., 'y': ...}
#   或: train.pt = {'input': ..., 'output': ...}
#
# - 方式2: 元组格式
#   train.pt = (x, y)
#
# - 方式3: 直接 tensor
#   train.pt = x (y = x)

train_x, train_y, test_x, test_y, info = load_dataset(
    dataset_name='custom',
    data_format='pt',             # PT 格式
    train_path='./data/customer_train.pt',
    test_path='./data/customer_test.pt',
    n_train=1000,
    n_test=200,
)
```

## 🛠 命令行参数

### 数据集参数
| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| \`--dataset\` | 数据集名称 (navier_stokes/darcy/custom) | \`darcy\` | ✅ |
| \`--data_format\` | 数据格式 (h5/pt/pth) - custom模式必需 | \`None\` | custom模式必需 |
| \`--train_path\` | 训练数据路径 - custom模式必需 | \`None\` | custom模式必需 |
| \`--test_path\` | 测试数据路径 - custom模式必需 | \`None\` | custom模式必需 |
| \`--resolution\` | 数据分辨率 | \`None\` | ✅ |
| \`--n_train\` | 训练样本数 | \`1000\` | - |
| \`--n_test\` | 测试样本数 | \`200\` | - |

### 训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| \`--epochs\` | 训练轮数 | \`50\` |
| \`--batch_size\` | 批次大小 | \`32\` |
| \`--lr\` | 学习率 | \`0.001\` |
| \`--seed\` | 随机种子 | \`42\` |

### 物理方程参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| \`--viscosity\` | 黏性系数 | \`1e-3\` |
| \`--n_steps\` | 时间步数 | \`100\` |

## 🔧 关键脚本说明

| 脚本 | 功能 | 状态 |
|------|------|------|
| **data_loader.py** | 通用数据加载器（NeuralOperator + 客户数据集） | ✅ v1.6.1 |
| **run_benchmarks.py** | 完整的基准测试，支持各种配置 | ✅ 已更新 |
| **generate_data.py** | 统一生成所有数据集，无需下载 | ✅ 本地生成 |
| **quick_test.py** | 快速测试 MHF+CoDA+PINO 最佳组合 | ✅ 测试最佳配置 |

## 📦 版本更新说明

### v1.6.1 (2026-03-27) - 数据加载器通用化

#### 新增功能
- ✅ **支持客户自定义数据集** (H5/PT文件，train/test分离）
  - 新增 \`dataset_name='custom'\` 模式
  - 支持 \`data_format='h5'\` 和 \`data_format='pt'\`
  - 支持多种文件格式（字典、元组、直接tensor）

#### 修复的问题
- ✅ **网络访问限制**: 默认 \`download=False\` 避免 Zenodo 访问问题
  - 不是所有环境都能访问 Zenodo（内网、代理、受限环境）
  - 用户可以手动下载数据集后加载
  - 自动下载仅在有网络环境启用（需显式指定）

#### 文档更新
- ✅ **README.md**: 完全重写，支持多种数据源
  - 新增客户数据集使用指南
  - 新增数据源对比表
  - 新增网络环境说明

### v1.6.0 (2026-03-27) - 重大重构

#### 数据加载重构
- **替换为 NeuralOperator 2.0.0 数据加载器**
  - 删除 ~1000 行自定义数据加载代码
  - 新增 ~100 行封装代码
  - 代码量减少 90%
  - 稳定性和兼容性大幅提升

## 🎯 迁移指南

### 从 v1.5.x 迁移到 v1.6.0+

\`\`\`bash
# 旧版本 (v1.5.x)
python run_benchmarks.py \\
    --dataset navier_stokes \\
    --format h5 \\
    --train_path ./data/NS_Train.h5 \\
    --test_path ./data/NS_Test.h5 \\
    --n_train 1000 \\
    --n_test 200

# 新版本 (v1.6.0+)
python run_benchmarks.py \\
    --dataset navier_stokes \\
    --resolution 128 \\
    --n_train 1000 \\
    --n_test 200
\`\`\`

## 📊 预期输出示例

\`\`\`
============================================================
MHF-FNO 基准测试
============================================================

📊 加载数据集: custom
   配置: n_train=1000, n_test=200, resolution=64
   数据格式: H5
   训练数据: ./data/customer_train.h5
   测试数据: ./data/customer_test.h5
   训练文件结构: ['x', 'y']
   测试文件结构: ['x', 'y']
   ✅ 数据加载成功
      train_x: torch.Size([1000, 1, 64, 64])
      train_y: torch.Size([1000, 1, 64, 64])
      test_x: torch.Size([200, 1, 64, 64])
      test_y: torch.Size([200, 1, 64, 64])

数据集信息:
  名称: 客户数据集 (H5)
  分辨率: 64x64
  训练集: 1000
  测试集: 200

============================================================
测试 FNO (基准)
============================================================
参数量: 3,200,433
  Epoch 10/50: Train 0.0234, Test 0.0215, Time 1.2s
  ...
\`\`\`

## 🔗 参考链接

-   [NeuralOperator GitHub](https://github.com/neuraloperator/neuraloperator)
-   [Zenodo Navier-Stokes Dataset](https://zenodo.org/records/12825163)
-   [Zenodo Darcy Flow Dataset](https://zenodo.org/records/12784353)
-   [MHF-FNO GitHub](https://github.com/xuefenghao5121/mhf-fno)

## 🌐 网络环境说明

⭐ **重要：网络限制**

不是所有环境都能访问 Zenodo：
-   ❌ **内网环境**（公司内网、HPC 集群）
-   ❌ **代理服务器**（需要特殊配置）
-   ❌ **受限环境**（防火墙、白名单限制）

**解决方案**：
1.  ✅ **推荐**：在有网络的环境手动下载，然后传输到目标环境
2.  ✅ **备选**：使用客户提供的本地数据集（H5/PT文件）
3.  ✅ **备选**：使用 \`generate_data.py\` 本地生成数据集
4.  ⚠️ **自动**：仅在有直接网络的环境启用 \`download=True\`

**手动下载命令**：
\`\`\`bash
# 在有网络的环境
wget https://zenodo.org/records/12825163/files/nsforcing_v1.0_128.tar.gz

# 通过 scp/sftp/ftp 传输到目标环境
scp nsforcing_v1.0_128.tar.gz user@target-server:/path/to/benchmark/data/

# 在目标环境解压
tar -xzf nsforcing_v1.0_128.tar.gz -C benchmark/data/
\`\`\`
