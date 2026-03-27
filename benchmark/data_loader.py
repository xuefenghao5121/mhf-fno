"""
MHF-FNO 数据加载器 (v1.6.1)

支持多种数据源：
1. NeuralOperator 官方数据集 (NavierStokes, Darcy)
2. 客户提供的本地 H5 文件 (train/test分离)
3. 客户提供的本地 PT/Torch 文件

数据格式支持：
- H5格式：Zenodo格式、PDEBench格式、自定义H5
- PT格式：.pt, .pth (PyTorch格式）
- 支持多种维度：[N, L], [N, C, L], [N, H, W], [N, C, H, W]

使用示例:
    >>> from data_loader import load_dataset
    >>>
    >>> # 1. 加载 NeuralOperator 官方数据集
    >>> train_x, train_y, test_x, test_y, info = load_dataset(
    >>>     dataset_name='navier_stokes',
    >>>     n_train=1000, n_test=200, resolution=64,
    >>>     download=False,
    >>> )
    >>>
    >>> # 2. 加载客户提供的 H5 文件
    >>> train_x, train_y, test_x, test_y, info = load_dataset(
    >>>     dataset_name='custom',
    >>>     data_format='h5',
    >>>     train_path='./data/client_train.h5',
    >>>     test_path='./data/client_test.h5',
    >>>     n_train=1000, n_test=200, resolution=64,
    >>> )
    >>>
    >>> # 3. 加载客户提供的 PT 文件
    >>> train_x, train_y, test_x, test_y, info = load_dataset(
    >>>     dataset_name='custom',
    >>>     data_format='pt',
    >>>     train_path='./data/client_train.pt',
    >>>     test_path='./data/client_test.pt',
    >>>     n_train=1000, n_test=200,
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
    data_format: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    通用数据加载器，支持多种数据源和格式

    支持的数据集:
    - 'navier_stokes': Navier-Stokes 方程 (NeuralOperator官方)
    - 'darcy': Darcy Flow 方程 (NeuralOperator官方)
    - 'custom': 客户自定义数据集 (需要指定train_path, test_path)

    Args:
        dataset_name: 数据集名称 ('navier_stokes', 'darcy', 或 'custom')
        n_train: 训练样本数
        n_test: 测试样本数
        resolution: 数据分辨率
        data_format: 数据格式 ('h5', 'pt', 'pth') - 客户数据集必需
        train_path: 训练数据路径 - 客户数据集必需
        test_path: 测试数据路径 - 客户数据集必需
        download: 是否自动从 Zenodo 下载 (默认 False)
        **kwargs: 其他参数传递给数据集类

    Returns:
        (train_x, train_y, test_x, test_y, info)

    使用示例 (NeuralOperator官方数据集）:
        >>> train_x, train_y, test_x, test_y, info = load_dataset(
        >>>     dataset_name='navier_stokes',
        >>>     n_train=1000, n_test=200, resolution=64,
        >>>     download=False,
        >>> )

    使用示例 (客户提供的H5文件）:
        >>> train_x, train_y, test_x, test_y, info = load_dataset(
        >>>     dataset_name='custom',
        >>>     data_format='h5',
        >>>     train_path='./data/client_train.h5',
        >>>     test_path='./data/client_test.h5',
        >>>     n_train=1000, n_test=200, resolution=64,
        >>> )

    使用示例 (客户提供的PT文件）:
        >>> train_x, train_y, test_x, test_y, info = load_dataset(
        >>>     dataset_name='custom',
        >>>     data_format='pt',
        >>>     train_path='./data/client_train.pt',
        >>>     test_path='./data/client_test.pt',
        >>>     n_train=1000, n_test=200,
        >>> )
    """
    print(f"\\n📊 加载数据集: {dataset_name}")
    print(f"   配置: n_train={n_train}, n_test={n_test}, resolution={resolution}")

    # 客户自定义数据集
    if dataset_name == 'custom':
        if data_format is None:
            raise ValueError("客户数据集需要指定 data_format ('h5', 'pt', 'pth')")
        if train_path is None or test_path is None:
            raise ValueError("客户数据集需要指定 train_path 和 test_path")
        return _load_custom(data_format, train_path, test_path, n_train, n_test, resolution, **kwargs)

    # NeuralOperator 官方数据集
    if not HAS_NEURALOP:
        raise ImportError("需要安装 neuraloperator: pip install neuraloperator")

    if dataset_name == 'navier_stokes':
        return _load_navier_stokes(n_train, n_test, resolution, download=download, **kwargs)
    elif dataset_name == 'darcy':
        return _load_darcy(n_train, n_test, resolution, download=download, **kwargs)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}. 支持: 'navier_stokes', 'darcy', 'custom'")


def _load_custom(
    data_format: str,
    train_path: str,
    test_path: str,
    n_train: int,
    n_test: int,
    resolution: int,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """加载客户自定义数据集 (H5/PT格式)"""
    print(f"   数据格式: {data_format.upper()}")
    print(f"   训练数据: {train_path}")
    print(f"   测试数据: {test_path}")

    # H5 格式
    if data_format.lower() in ['h5', 'hdf5']:
        return _load_h5_custom(train_path, test_path, n_train, n_test, resolution, **kwargs)

    # PT 格式
    elif data_format.lower() in ['pt', 'pth']:
        return _load_pt_custom(train_path, test_path, n_train, n_test, resolution, **kwargs)

    else:
        raise ValueError(f"不支持的数据格式: {data_format}. 支持: 'h5', 'pt', 'pth'")


def _load_h5_custom(
    train_path: str,
    test_path: str,
    n_train: int,
    n_test: int,
    resolution: int,
    is_2d: Optional[bool] = None,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """从 H5 文件加载客户数据集"""
    if not HAS_H5PY:
        raise ImportError("需要安装 h5py: pip install h5py")

    print(f"   使用 h5py 加载 H5 文件...")

    # 加载训练数据
    with h5py.File(train_path, 'r') as f:
        print(f"   训练文件结构: {list(f.keys())}")

        if 'x' in f and 'y' in f:
            train_x = f['x'][:]
            train_y = f['y'][:]
        elif 'u' in f:
            data = f['u'][:]
            if data.ndim == 4:
                train_x = data[:, 0, :, :]
                train_y = data[:, -1, :, :]
            else:
                train_x = data
                train_y = data
        else:
            keys = list(f.keys())
            train_x = f[keys[0]][:]
            train_y = f[keys[-1]][:]

    # 加载测试数据
    with h5py.File(test_path, 'r') as f:
        print(f"   测试文件结构: {list(f.keys())}")

        if 'x' in f and 'y' in f:
            test_x = f['x'][:]
            test_y = f['y'][:]
        elif 'u' in f:
            data = f['u'][:]
            if data.ndim == 4:
                test_x = data[:, 0, :, :]
                test_y = data[:, -1, :, :]
            else:
                test_x = data
                test_y = data
        else:
            keys = list(f.keys())
            test_x = f[keys[0]][:]
            test_y = f[keys[-1]][:]

    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32)
    test_x = torch.as_tensor(test_x, dtype=torch.float32)
    test_y = torch.as_tensor(test_y, dtype=torch.float32)

    if is_2d is None:
        is_2d = train_x.ndim >= 3 and train_x.shape[-1] == train_x.shape[-2]

    if is_2d:
        if train_x.ndim == 3:
            train_x = train_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        if test_x.ndim == 3:
            test_x = test_x.unsqueeze(1)
            test_y = test_y.unsqueeze(1)
    else:
        if train_x.ndim == 2:
            train_x = train_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        if test_x.ndim == 2:
            test_x = test_x.unsqueeze(1)
            test_y = test_y.unsqueeze(1)

    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = test_x[:n_test]
    test_y = test_y[:n_test]

    print(f"   ✅ 数据加载成功")
    print(f"      train_x: {train_x.shape}")
    print(f"      train_y: {train_y.shape}")
    print(f"      test_x: {test_x.shape}")
    print(f"      test_y: {test_y.shape}")

    info = {
        'name': f'客户数据集 (H5)',
        'resolution': f'{train_x.shape[-2]}x{train_x.shape[-1]}' if is_2d else f'{train_x.shape[-1]}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
    }

    return train_x: train_y, test_x, test_y, info


def _load_pt_custom(
    train_path: str,
    test_path: str,
    n_train: int,
    n_test: int,
    resolution: int,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """从 PT 文件加载客户数据集"""
    print(f"   使用 torch.load 加载 PT 文件...")

    train_data = torch.load(train_path, map_location='cpu')

    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('input'))
        train_y = train_data.get('y', train_data.get('output', train_x))
    elif isinstance(train_data, (tuple, list)):
        train_x, train_y = train_data
    else:
        train_x = train_data
        train_y = train_data

    test_data = torch.load(test_path, map_location='cpu')

    if isinstance(test_data, dict):
        test_x = test_data.get('x', test_data.get('input'))
        test_y = test_data.get('y', test_data.get('output', test_x))
    elif isinstance(test_data, (tuple, list)):
        test_x, test_y = test_data
    else:
        test_x = test_data
        test_y = test_data

    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()

    is_2d = train_x.ndim >= 3 and train_x.shape[-1] == train_x.shape[-2]

    if is_2d:
        if train_x.ndim == 3:
            train_x = train_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        if test_x.ndim == 3:
            test_x = test_x.unsqueeze(1)
            test_y = test_y.unsqueeze(1)
    else:
        if train_x.ndim == 2:
            train_x = train_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        if test_x.ndim == 2:
            test_x = test_x.unsqueeze(1)
            test_y = test_y.unsqueeze(1)

    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = test_x[:n_test]
    test_y = test_y[:n_test]

    print(f"   ✅ 数据加载成功")
    print(f"      train_x: {train_x.shape}")
    print(f"      train_y: {train_y.shape}")
    print(f"      test_x: {test_x.shape}")
    print(f"      test_y: {test_y.shape}")

    info = {
        'name': f'客户数据集 (PT)',
        'resolution': f'{train_x.shape[-2]}x{train_x.shape[-1]}' if is_2d else f'{train_x.shape[-1]}',
        'n_train': train_x.shape[0],
        'n_test': test_x.shape[0],
        'input_channels': train_x.shape[1],
        'output_channels': train_y.shape[1],
    }

    return train_x, train_y, test_x, test_y, info


def _load_navier_stokes(
    n_train: int,
    n_test: int,
    resolution: int,
    root_dir: Optional[str] = None,
    download: bool = False,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """加载 Navier-Stokes 数据集 (NeuralOperator官方）"""

    if root_dir is None:
        root_dir = Path('./data').absolute()
    else:
        root_dir = Path(root_dir)

    print(f"   使用 NeuralOperator 官方数据集")
    print(f"   root_dir: {root_dir}")
    print(f"   download: {download}")

    dataset = NavierStokesDataset(
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

    train_data = dataset.train_db[:]
    test_data = dataset.test_dbs[resolution][:]

    if isinstance(train_data, (tuple, list)):
        train_x, train_y = train_data
    else:
        train_x = train_data
        train_y = train_data

    if isinstance(test_data, (tuple, list)):
        test_x, test_y = test_data
    else:
        test_x = test_data
        test_y = test_data

    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32)
    test_x = torch.as_tensor(test_data, dtype=torch.float32)
    test_y = torch.as_tensor(test_y, dtype=torch.float32)

    if train_x.ndim == 3:
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.ndim == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)

    print(f"   ✅ 数据加载成功")
    print(f"      train_x: {train_x.shape}")
    print(f"      train_y: {train{y.shape}")
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
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """加载 Darcy 数据集 (NeuralOperator官方）"""

    if root_dir is None:
        root_dir = Path('./data').absolute()
    else:
        root_dir = Path(root_dir)

    print(f"   使用 NeuralOperator 官方数据集")
    print(f"   root_dir: {root_dir}")
    print(f"   download: {download}")

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

    train_data = dataset.train_db[:]
    test_data = dataset.test_dbs[resolution][:]

    if isinstance(train_data, (tuple, list)):
        train_x, train_y = train_data
    else:
        train_x = train_data
        train_y = train_data

    if isinstance(test_data, (tuple, list)):
        test_x, test_y = test_data
    else:
        test_x = test_data
        test_y = test_data

    train_x = torch.as_tensor(train_x, dtype=torch.float32)
    train_y = torch.as_tensor(train_y, dtype=torch.float32)
    test_x = torch.as_tensor(test_data, dtype=torch.float32)
    test_y = torch.as_tensor(test_y, dtype=torch.float32)

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
