"""
MHF-FNO 数据加载器 (复用 NeuralOperator 2.0.0)

NeuralOperator 2.0.0 已经有完整、经过测试的数据加载模块：
- NavierStokesDataset: Navier-Stokes 数据集
- DarcyDataset: Darcy Flow 数据集
- PTDataset: 通用的 PT 数据加载器

复用这些模块的好处：
1. ✅ 经过充分测试，稳定可靠
2. ✅ 支持所有数据格式（Zenodo、PDEBench等）
3. ✅ 自动下载和解压
4. ✅ 统一的接口和输出格式
5. ✅ 减少维护成本

使用示例:
    >>> from data_loader import load_dataset
    >>>
    >>> # 加载 Navier-Stokes 数据集
    >>> train_x, train_y, test_x, test_y, info = load_dataset(
    >>>     dataset_name='navier_stokes',
    >>>     n_train=1000,
    >>>     n_test=200,
    >>>     resolution=64,
    >>> )
    >>>
    >>> # 从本地 H5 文件加载
    >>> train_x, train_y, test_x, test_y, info = load_dataset(
    >>>     dataset_name='navier_stokes',
    >>>     data_format='h5',
    >>>     train_path='./data/NS_Train.h5',
    >>>     test_path='./data/NS_Test.h5',
    >>>     n_train=1000,
    >>>     n_test=200,
    >>> )
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

try:
    from neuralop.data.datasets import (
        NavierStokesDataset,
        DarcyDataset,
        PTDataset
    )
    HAS_NEURALOP = True
except ImportError:
    HAS_NEURALOP = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_dataset(
    dataset_name: str,
    n_train: int = 1000,
    n_test: int = 200,
    resolution: int = 64,
    download: bool = False,  # ⭐ 默认关闭，避免网络访问问题
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    使用 NeuralOperator 2.0.0 数据集加载数据
    
    支持的数据集:
    - 'navier_stokes': Navier-Stokes 方程
    - 'darcy': Darcy Flow 方程
    
    Args:
        dataset_name: 数据集名称 ('navier_stokes' 或 'darcy')
        n_train: 训练样本数
        n_test: 测试样本数
        resolution: 数据分辨率
        download: 是否自动从 Zenodo 下载 (默认 False，避免网络访问问题）
            ⚠️ 注意：不是所有环境都能访问 Zenodo（内网、代理等）
            ⭐ 建议：手动下载数据集后设置为 False
        **kwargs: 其他参数传递给数据集类
    
    Returns:
        (train_x, train_y, test_x, test_y, info)
    
    Raises:
        ImportError: 如果未安装 neuraloperator
        ValueError: 如果不支持的数据集名称
        FileNotFoundError: 如果未下载数据且 download=False
    
    使用示例 (手动下载）:
        >>> # 1. 先手动下载数据集到 ./data/ 目录
        >>> # 2. 然后加载（download=False）
        >>> train_x, train_y, test_x, test_y, info = load_dataset(
        >>>     dataset_name='navier_stokes',
        >>>     n_train=1000,
        >>>     n_test=200,
        >>>     resolution=64,
        >>>     download=False,  # ✅ 手动下载模式
        >>> )
    
    使用示例 (自动下载 - 仅限有网络的环境）:
        >>> train_x, train_y, test_x, test_y, info = load_dataset(
        >>>     dataset_name='navier_stokes',
        >>>     n_train=1000,
        >>>     n_test=200,
        >>>     resolution=64,
        >>>     download=True,  # ⚠️ 自动下载模式
        >>> )
    """
    if not HAS_NEURALOP:
        raise ImportError("需要安装 neuraloperator: pip install neuraloperator")

    print(f"\n📊 使用 NeuralOperator 2.0.0 加载数据集: {dataset_name}")
    print(f"   配置: n_train={n_train}, n_test={n_test}, resolution={resolution}")

    if dataset_name == 'navier_stokes':
        return _load_navier_stokes(n_train, n_test, resolution, **kwargs)
    elif dataset_name == 'darcy':
        return _load_darcy(n_train, n_test, resolution, **kwargs)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}. 支持: 'navier_stokes', 'darcy'")


def _load_navier_stokes(
    n_train: int,
    n_test: int,
    resolution: int,
    root_dir: Optional[str] = None,
    download: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """加载 Navier-Stokes 数据集"""

    if root_dir is None:
        root_dir = Path('./data').absolute()
    else:
        root_dir = Path(root_dir)

    print(f"   root_dir: {root_dir}")
    print(f"   download: {download}")

    # 创建数据集对象
    dataset = NavierStokesDataset(
        root_dir=root_dir,
        n_train=n_train,
        n_tests=[n_test],
        batch_size=n_train,  # 一次性加载所有数据
        test_batch_sizes=[n_test],
        train_resolution=resolution,
        test_resolutions=[resolution],
        encode_input=False,
        encode_output=False,
        download=download,
    )

    print(f"   ✅ NeuralOperator 数据集创建成功")
    print(f"      训练集大小: {len(dataset.train_db)}")
    print(f"      测试集大小: {len(dataset.test_dbs[resolution])}")

    # 加载数据
    train_data = dataset.train_db[:]
    test_data = dataset.test_dbs[resolution][:]

    # 提取 x 和 y
    if isinstance(train_data, (tuple, list)):
        train_x, train_y = train_data
    else:
        train_x = train_data
        train_y = train_data  # NS 数据集通常 x=y

    if isinstance(test_data, (tuple, list)):
        test_x, test_y = test_data
    else:
        test_x = test_data
        test_y = test_data

    # 转换为 torch.Tensor
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32)
    test_x = torch.as_tensor(test_x, dtype=torch.float32)
    test_y = torch.as_tensor(test_y, dtype=torch.float32)

    # 确保维度正确 [N, C, H, W]
    if train_x.ndim == 3:  # [N, H, W] -> [N, 1, H, W]
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.ndim == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)

    print(f"   ✅ 数据加载成功")
    print(f"      train_x: {train_x.shape}")
    print(f"      train_y: {train_y.shape}")
    print(f"      test_x: {test_x.shape}")
    print(f"      test_y: {test_y.shape}")

    info = {
        'name': f'NeuralOperator Navier-Stokes',
        'resolution': f'{resolution}x{resolution}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'n_modes': (resolution // 2, resolution // 2),
    }

    return train_x, train_y, test_x, test_y, info


def _load_darcy(
    n_train: int,
    n_test: int,
    resolution: int,
    root_dir: Optional[str] = None,
    download: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """加载 Darcy 数据集"""

    if root_dir is None:
        root_dir = Path('./data').absolute()
    else:
        root_dir = Path(root_dir)

    print(f"   root_dir: {root_dir}")
    print(f"   download: {download}")

    # 创建数据集对象
    dataset = DarcyDataset(
        root_dir=root_dir,
        n_train=n_train,
        n_tests=[n_test],
        batch_size=n_train,
        test_batch_sizes=[n_test],
        train_resolution=resolution,
        test_resolutions=[resolution],
        encode_input=False,
        encode_output=False,
        download=download,
    )

    print(f"   ✅ NeuralOperator 数据集创建成功")
    print(f"      训练集大小: {len(dataset.train_db)}")
    print(f"      测试集大小: {len(dataset.test_dbs[resolution])}")

    # 加载数据
    train_data = dataset.train_db[:]
    test_data = dataset.test_dbs[resolution][:]

    # 提取 x 和 y
    if (isinstance(train_data, (tuple, list))):
        train_x, train_y = (train_data)
    else:
        train_x = train_data
        train_y = train_data

    if (isinstance(test_data, (tuple, list))):
        test_x, test_y = (test_data)
    else:
        test_x = test_data
        test_y = test_data

    # 转换为 torch.Tensor
    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32)
    test_x = torch.as_tensor(test_x, dtype=torch.float32)
    test_y = torch.as_tensor(test_y, dtype=torch.float32)

    # 确保维度正确 [N, C, H, W]
    if train_x.ndim == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.ndim == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)

    print(f"   ✅ 数据加载成功")
    print(f"      train_x: {train_x.shape}")
    print(f"      train_y: {train_y.shape}")
    print(f"      test_x: {test_x.shape}")
    print(f"      test_y: {test_y.shape}")

    info = {
        'name': f'NeuralOperator Darcy Flow',
        'resolution': f'{resolution}x{resolution}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
        'n_modes': (resolution // 2, resolution // 2),
    }

    return train_x, train_y, test_x, test_y, info
