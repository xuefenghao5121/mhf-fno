#!/usr/bin/env python3
"""
MHF-FNO 本地数据集加载器 (v1.6.4)

独立的数据加载模块，不依赖 benchmark/data_loader.py。

支持:
- PT/PTH 格式 (PyTorch)
- NPY/NPZ 格式 (NumPy)
- H5/HDF5 格式
- 自动维度检测和通道添加
- 训练/测试分割

使用方法:
    from local_data_loader import LocalDataLoader

    # 加载 Darcy 数据
    loader = LocalDataLoader()
    train_x, train_y, test_x, test_y = loader.load_darcy()

    # 加载自定义数据
    train_x, train_y, test_x, test_y = loader.load_custom(
        train_path='./data/train.pt',
        test_path='./data/test.pt',
    )

    # 一站式: 加载 + 创建 DataLoader
    train_loader, test_loader = loader.get_dataloaders(
        dataset='darcy', batch_size=64
    )
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from torch.utils.data import DataLoader, TensorDataset

# 默认数据路径
DEFAULT_DATA_PATH = Path("/home/huawei/Desktop/home/xuefenghao/workspace/mhf-data")

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class LocalDataLoader:
    """
    本地数据集加载器
    
    支持从本地文件加载 PDE 数据集，自动处理各种格式和维度。
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Args:
            data_path: 数据根目录，默认使用内置路径
        """
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
    
    def load_file(self, file_path: str, key: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从文件加载数据，自动检测格式
        
        Args:
            file_path: 文件路径
            key: 字典/h5 中的 key，默认自动检测
        
        Returns:
            (x, y) 元组
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix in ['.pt', '.pth']:
            return self._load_pt(file_path, key)
        elif suffix in ['.npy']:
            data = torch.from_numpy(np.load(file_path)).float()
            return data, data
        elif suffix in ['.npz']:
            npz = np.load(file_path)
            k = key or list(npz.keys())[0]
            data = torch.from_numpy(npz[k]).float()
            return data, data
        elif suffix in ['.h5', '.hdf5'] and HAS_H5PY:
            return self._load_h5(file_path, key)
        else:
            raise ValueError(f"不支持的格式: {suffix}")
    
    def _load_pt(self, path: str, key: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载 PT 文件"""
        data = torch.load(path, map_location='cpu', weights_only=False)
        
        if isinstance(data, dict):
            x = data.get(key or 'x', data.get('input', list(data.values())[0]))
            y = data.get('y', data.get('output', x))
        elif isinstance(data, (tuple, list)):
            x, y = data[0], data[1]
        else:
            x, y = data, data
        
        return x.float(), y.float()
    
    def _load_h5(self, path: str, key: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载 H5 文件"""
        import h5py
        with h5py.File(path, 'r') as f:
            if 'x' in f and 'y' in f:
                x = torch.from_numpy(f['x'][:]).float()
                y = torch.from_numpy(f['y'][:]).float()
            elif 'u' in f:
                data = torch.from_numpy(f['u'][:]).float()
                if data.ndim == 4:
                    x, y = data[:, 0:1], data[:, -1:]
                else:
                    x, y = data, data
            else:
                keys = list(f.keys())
                x = torch.from_numpy(f[keys[0]][:]).float()
                y = torch.from_numpy(f[keys[-1]][:]).float()
        return x, y
    
    def _ensure_channel_dim(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """确保有通道维度"""
        if x.dim() == 2:
            x, y = x.unsqueeze(1), y.unsqueeze(1)
        if x.dim() == 3 and x.shape[1] > 4:
            # 时间序列
            x, y = x[:, 0:1], x[:, -1:]
        return x, y
    
    def load_darcy(self, n_train: Optional[int] = None, n_test: Optional[int] = None,
                   resolution: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载 Darcy Flow 数据
        
        Args:
            n_train: 训练样本数限制
            n_test: 测试样本数限制
            resolution: 分辨率 (目前仅支持 32)
        
        Returns:
            (train_x, train_y, test_x, test_y)
        """
        train_path = self.data_path / f"darcy_train_{resolution}.pt"
        test_path = self.data_path / f"darcy_test_{resolution}.pt"
        
        train_x, train_y = self.load_file(train_path)
        test_x, test_y = self.load_file(test_path)
        
        train_x, train_y = self._ensure_channel_dim(train_x, train_y)
        test_x, test_y = self._ensure_channel_dim(test_x, test_y)
        
        if n_train:
            train_x, train_y = train_x[:n_train], train_y[:n_train]
        if n_test:
            test_x, test_y = test_x[:n_test], test_y[:n_test]
        
        return train_x, train_y, test_x, test_y
    
    def load_navier_stokes(self, n_train: Optional[int] = None,
                           n_test: Optional[int] = None,
                           resolution: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """加载 Navier-Stokes 数据"""
        train_path = self.data_path / f"nsforcing_train_{resolution}.pt"
        test_path = self.data_path / f"nsforcing_test_{resolution}.pt"
        
        train_x, train_y = self.load_file(train_path)
        test_x, test_y = self.load_file(test_path)
        
        train_x, train_y = self._ensure_channel_dim(train_x, train_y)
        test_x, test_y = self._ensure_channel_dim(test_x, test_y)
        
        if n_train:
            train_x, train_y = train_x[:n_train], train_y[:n_train]
        if n_test:
            test_x, test_y = test_x[:n_test], test_y[:n_test]
        
        return train_x, train_y, test_x, test_y
    
    def load_burgers(self, n_train: Optional[int] = None,
                     n_test: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """加载 Burgers 数据"""
        for name in ['rand_burgers_data_R10.pt', 'uniform_burgers_data_R10.pt']:
            p = self.data_path / name
            if p.exists():
                x, y = self.load_file(p)
                x, y = self._ensure_channel_dim(x, y)
                n = len(x)
                split = int(0.8 * n)
                if n_train:
                    split = n_train
                end = (split + n_test) if n_test else n
                return x[:split], y[:split], x[split:end], y[split:end]
        raise FileNotFoundError(f"Burgers 数据不存在于 {self.data_path}")
    
    def load_custom(self, train_path: str, test_path: str,
                    n_train: Optional[int] = None,
                    n_test: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        加载自定义数据集
        
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径
            n_train: 训练样本数限制
            n_test: 测试样本数限制
        
        Returns:
            (train_x, train_y, test_x, test_y)
        """
        train_x, train_y = self.load_file(train_path)
        test_x, test_y = self.load_file(test_path)
        
        train_x, train_y = self._ensure_channel_dim(train_x, train_y)
        test_x, test_y = self._ensure_channel_dim(test_x, test_y)
        
        if n_train:
            train_x, train_y = train_x[:n_train], train_y[:n_train]
        if n_test:
            test_x, test_y = test_x[:n_test], test_y[:n_test]
        
        return train_x, train_y, test_x, test_y
    
    def get_dataloaders(self, dataset: str = 'darcy', batch_size: int = 64,
                        n_train: Optional[int] = None, n_test: Optional[int] = None,
                        num_workers: int = 0, **kwargs) -> Tuple[DataLoader, DataLoader]:
        """
        一站式: 加载数据 + 创建 DataLoader
        
        Args:
            dataset: 数据集名称 ('darcy', 'navier_stokes', 'burgers', 'custom')
            batch_size: 批大小
            n_train: 训练样本数
            n_test: 测试样本数
            num_workers: DataLoader worker 数
            **kwargs: 传递给对应加载方法的参数
        
        Returns:
            (train_loader, test_loader) 元组
        """
        if dataset == 'darcy':
            train_x, train_y, test_x, test_y = self.load_darcy(n_train, n_test)
        elif dataset == 'navier_stokes':
            train_x, train_y, test_x, test_y = self.load_navier_stokes(n_train, n_test)
        elif dataset == 'burgers':
            train_x, train_y, test_x, test_y = self.load_burgers(n_train, n_test)
        elif dataset == 'custom':
            train_x, train_y, test_x, test_y = self.load_custom(**kwargs)
        else:
            raise ValueError(f"不支持的数据集: {dataset}")
        
        train_dataset = TensorDataset(train_x, train_y)
        test_dataset = TensorDataset(test_x, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        return train_loader, test_loader


# 便捷函数
def load_darcy(**kwargs):
    return LocalDataLoader().load_darcy(**kwargs)

def load_navier_stokes(**kwargs):
    return LocalDataLoader().load_navier_stokes(**kwargs)

def load_burgers(**kwargs):
    return LocalDataLoader().load_burgers(**kwargs)

def load_custom(train_path, test_path, **kwargs):
    return LocalDataLoader().load_custom(train_path, test_path, **kwargs)
