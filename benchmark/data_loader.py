"""
MHF-FNO 通用数据加载器

支持多种数据格式:
1. PT 格式:
   - 单文件: 一个文件包含 train+test
   - 双文件: train.pt + test.pt (两个独立文件)
2. H5 格式 (PDEBench / Zenodo):
   - 单文件: 一个文件包含 train+test
   - 双文件: train.h5 + test.h5 (Zenodo 下载格式，训练集和测试集分开)

使用方法:
    from data_loader import load_dataset
    train_x, train_y, test_x, test_y, info = load_dataset(
        dataset_name='darcy',
        data_format='h5',
        train_path='./data/2D_DarcyFlow_Train.h5',
        test_path='./data/2D_DarcyFlow_Test.h5',
        n_train=1000,
        n_test=200,
    )

支持的数据集 (Zenodo https://zenodo.org/records/13355846:
- Burgers 1D:
  - 1D_Burgers_Re1000_Train.h5
  - 1D_Burgers_Re1000_Test.h5
- Navier-Stokes 2D:
  - 2D_NS_Re100_Train.h5
  - 2D_NS_Re100_Test.h5
- Darcy Flow 2D:
  - 2D_DarcyFlow_Train.h5
  - 2D_DarcyFlow_Test.h5
"""

import re
import numpy as np
import torch

# H5 支持
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("⚠️  h5py not installed, H5 format not available")
    print("   Install: pip install h5py")


# ============================================================================
# 工具函数
# ============================================================================

def parse_resolution_from_filename(filename: str) -> int:
    """
    从文件名解析分辨率。
    
    支持的命名格式:
    - darcy_train_16.pt -> 16
    - ns_train_64.pt -> 64
    - burgers_train_1024.pt -> 1024
    - 1D_Burgers_Re1000_Train.h5 -> 1024 (Re后面是分辨率)
    - 2D_NS_Re100_Train.h5 -> 64 (默认)
    - 2D_DarcyFlow_64_Train.h5 -> 64
    
    Args:
        filename: 文件名 (不含路径)
    
    Returns:
        int: 解析出的分辨率，如果无法解析返回默认值
    """
    # 尝试匹配任意数字
    # 格式: _数字_Train 或 _数字.pt 或 _数字_ 或 数字_Train
    match = re.search(r'_(\d+)', filename, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        # 过滤掉不合理的数字 (年份，如 1D, 2D)
        if num in [1, 2]:
            # 继续找下一个数字
            remaining = filename[match.end():]
            match2 = re.search(r'_(\d+)', remaining)
            if match2:
                return int(match2.group(1))
        return num
    
    # 尝试匹配 Re 后面的数字 (Burgers/NS: Re1000, Re100)
    match_re = re.search(r'Re(\d+)', filename, re.IGNORECASE)
    if match_re:
        # Re 后面的数字是雷诺数，对于 Burgers 这就是分辨率相关
        # Burgers: Re1000 -> 1024, Re100 -> 128
        re_num = int(match_re.group(1))
        if re_num >= 1000:
            return 1024
        elif re_num >= 100:
            return 64
    
    # 默认返回
    print(f"⚠️  无法从文件名 '{filename}' 解析分辨率，使用默认值 64")
    return 64


def adjust_resolution(data, target_resolution, is_2d=True):
    """
    调整数据到目标分辨率。
    
    Args:
        data: torch.Tensor [..., H, W] 或 [..., L]
        target_resolution: 目标分辨率
    
    Returns:
        调整后的张量
    """
    if target_resolution is None:
        return data
    
    if data.ndim >= 3:  # 至少有 batch, channel, ...
        current_h = data.shape[-2] if is_2d else data.shape[-1]
        if is_2d and current_h == target_resolution:
            return data
        if not is_2d and current_h == target_resolution:
            return data
    
    # 使用插值调整
    if is_2d:
        # [..., H, W] -> [..., target_resolution, target_resolution]
        if data.ndim == 4:  # [N, C, H, W]
            data = torch.nn.functional.interpolate(
                data, size=(target_resolution, target_resolution),
                mode='bilinear',
                align_corners=False
            )
        elif data.ndim == 3:  # [N, H, W]
            data = data.unsqueeze(1)
            data = torch.nn.functional.interpolate(
                data, size=(target_resolution, target_resolution),
                mode='bilinear',
                align_corners=False
            )
            data = data.squeeze(1)
    else:
        # 1D 插值
        if data.ndim == 3:  # [N, C, L]
            data = torch.nn.functional.interpolate(
                data, size=(target_resolution,),
                mode='linear',
                align_corners=False
            )
        elif data.ndim == 2:  # [N, L]
            data = data.unsqueeze(1)
            data = torch.nn.functional.interpolate(
                data, size=(target_resolution,),
                mode='linear',
                align_corners=False
            )
            data = data.squeeze(1)
    
    return data


# ============================================================================
# H5 加载函数
# ============================================================================

def load_h5_single_file(h5_path, n_train=1000, n_test=200, resolution=None, is_2d=True):
    """
    从单个 H5 文件加载数据 (PDEBench 原格式)。
    
    格式说明:
    - 单个 H5 文件包含所有数据，前一半是输入，后一半是输出
    
    Args:
        h5_path: H5 文件路径
        n_train: 训练样本数
        n_test: 测试样本数
        resolution: 目标分辨率，如果为 None 不调整
        is_2d: 是否是 2D 数据
    
    Returns:
        train_x, train_y, test_x, test_y, info
    """
    if not HAS_H5PY:
        raise ImportError("需要安装 h5py: pip install h5py")
    
    print(f"\n📊 从单文件 H5 加载: {h5_path}")
    
    x_data = None
    y_data = None
    
    with h5py.File(h5_path, 'r') as f:
        # 尝试不同数据键
        if 'tensor' in f:
            data = f['tensor'][:]
        elif 'data' in f:
            data = f['data'][:]
        elif 'x' in f and 'y' in f:
            # PDEBench 有些分开存储
            x_data = f['x'][:]
            y_data = f['y'][:]
            data = None
        else:
            # 尝试第一个键
            keys = list(f.keys())
            data = f[keys[0]][:]
    
    if data is not None:
        # 单个文件包含输入输出，分割
        n_samples = data.shape[0]
        split = n_samples // 2
        x_data = data[:split]
        y_data = data[split:]
    
    # 转换为 PyTorch 张量
    x_data = torch.from_numpy(x_data).float()
    y_data = torch.from_numpy(y_data).float()
    
    # 添加 channel 维度
    if is_2d:
        # [N, H, W] -> [N, 1, H, W]
        if x_data.ndim == 3:
            x_data = x_data.unsqueeze(1)
            y_data = y_data.unsqueeze(1)
    else:
        # [N, L] -> [N, 1, L]
        if x_data.ndim == 2:
            x_data = x_data.unsqueeze(1)
            y_data = y_data.unsqueeze(1)
    
    # 分割训练测试
    train_x = x_data[:n_train]
    train_y = y_data[:n_train]
    test_x = x_data[n_train:n_train+n_test]
    test_y = y_data[n_train:n_train+n_test]
    
    # 获取实际分辨率 (信任数据shape，覆盖文件名解析)
    if is_2d:
        actual_res = train_x.shape[-1]
    else:
        actual_res = train_x.shape[-1]
    
    # 如果文件名解析分辨率和实际不匹配，使用实际分辨率
    if resolution is not None and resolution != actual_res:
        print(f"⚠️  文件名解析分辨率 {resolution}, 实际数据分辨率 {actual_res}, 使用实际分辨率")
        resolution = actual_res
    
    # 调整分辨率
    if resolution is not None:
        train_x = adjust_resolution(train_x, resolution, is_2d=is_2d)
        train_y = adjust_resolution(train_y, resolution, is_2d=is_2d)
        test_x = adjust_resolution(test_x, resolution, is_2d=is_2d)
        test_y = adjust_resolution(test_y, resolution, is_2d=is_2d)
    
    # 收集信息
    if is_2d:
        res = train_x.shape[-1]
        info = {
            'name': f'H5 2D ({h5_path.split("/")[-1]})',
            'resolution': f'{res}x{res}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (res // 2, res // 2) if is_2d else (res // 2,),
        }
    else:
        res = train_x.shape[-1]
        info = {
            'name': f'H5 1D ({h5_path.split("/")[-1]})',
            'resolution': f'{res}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (res // 2,),
        }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}, 分辨率 {info['resolution']}")
    return train_x, train_y, test_x, test_y, info


def load_h5_two_files(train_h5_path, test_h5_path, n_train=1000, n_test=200, resolution=None, is_2d=True):
    """
    从两个独立 H5 文件加载数据 (Zenodo 下载格式)。
    
    格式说明:
    - 训练集在 train 文件，包含输入和输出
    - 测试集在 test 文件，包含输入和输出
    
    Zenodo 数据集格式 (https://zenodo.org/records/13355846):
    - 每个 H5 文件中 'x' 是输入，'y' 是输出
    - 格式: [N, ...]
    
    Args:
        train_h5_path: 训练集 H5 文件路径
        test_h5_path: 测试集 H5 文件路径
        n_train: 训练样本数 (如果文件更多，截取前 n_train)
        n_test: 测试样本数 (如果文件更多，截取前 n_test)
        resolution: 目标分辨率，如果为 None 从文件名推断
        is_2d: 是否是 2D 数据
    
    Returns:
        train_x, train_y, test_x, test_y, info
    """
    if not HAS_H5PY:
        raise ImportError("需要安装 h5py: pip install h5py")
    
    print(f"\n📊 从双文件 H5 加载:")
    print(f"   训练集: {train_h5_path}")
    print(f"   测试集: {test_h5_path}")
    
    # 自动推断分辨率从文件名
    if resolution is None:
        fname = train_h5_path.split('/')[-1]
        resolution = parse_resolution_from_filename(fname)
        print(f"   从文件名推断分辨率: {resolution}")
    
    # 加载训练集
    with h5py.File(train_h5_path, 'r') as f:
        if 'x' in f:
            train_x_np = f['x'][:]
        elif 'input' in f:
            train_x_np = f['input'][:]
        else:
            keys = list(f.keys())
            train_x_np = f[keys[0]][:]
        
        if 'y' in f:
            train_y_np = f['y'][:]
        elif 'output' in f:
            train_y_np = f['output'][:]
        else:
            # 假设第二个键
            keys = list(f.keys())
            if len(keys) >= 2:
                train_y_np = f[keys[1]][:]
            else:
                # 分割
                n = train_x_np.shape[0] // 2
                train_y_np = train_x_np[n:]
                train_x_np = train_x_np[:n]
    
    # 加载测试集
    with h5py.File(test_h5_path, 'r') as f:
        if 'x' in f:
            test_x_np = f['x'][:]
        elif 'input' in f:
            test_x_np = f['input'][:]
        else:
            keys = list(f.keys())
            test_x_np = f[keys[0]][:]
        
        if 'y' in f:
            test_y_np = f['y'][:]
        elif 'output' in f:
            test_y_np = f['output'][:]
        else:
            keys = list(f.keys())
            if len(keys) >= 2:
                test_y_np = f[keys[1]][:]
            else:
                n = test_x_np.shape[0] // 2
                test_y_np = test_x_np[n:]
                test_x_np = test_x_np[:n]
    
    # 转换为 PyTorch 张量
    train_x = torch.from_numpy(train_x_np).float()
    train_y = torch.from_numpy(train_y_np).float()
    test_x = torch.from_numpy(test_x_np).float()
    test_y = torch.from_numpy(test_y_np).float()
    
    # 添加 channel 维度
    if is_2d:
        # [N, H, W] -> [N, 1, H, W]
        if train_x.ndim == 3:
            train_x = train_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        if test_x.ndim == 3:
            test_x = test_x.unsqueeze(1)
            test_y = test_y.unsqueeze(1)
    else:
        # [N, L] -> [N, 1, L]
        if train_x.ndim == 2:
            train_x = train_x.unsqueeze(1)
            train_y = train_y.unsqueeze(1)
        if test_x.ndim == 2:
            test_x = test_x.unsqueeze(1)
            test_y = test_y.unsqueeze(1)
    
    # 获取实际分辨率 (信任数据shape，覆盖文件名解析的)
    if is_2d:
        actual_res = train_x.shape[-1]
    else:
        actual_res = train_x.shape[-1]
    
    # 如果文件名解析的分辨率和实际不匹配，使用实际分辨率
    if resolution is not None and resolution != actual_res:
        print(f"⚠️  文件名解析分辨率 {resolution}, 实际数据分辨率 {actual_res}, 使用实际分辨率")
        resolution = actual_res
    
    # 调整分辨率
    if resolution is not None:
        train_x = adjust_resolution(train_x, resolution, is_2d=is_2d)
        train_y = adjust_resolution(train_y, resolution, is_2d=is_2d)
        test_x = adjust_resolution(test_x, resolution, is_2d=is_2d)
        test_y = adjust_resolution(test_y, resolution, is_2d=is_2d)
    
    # 收集信息
    if is_2d:
        res = train_x.shape[-1]
        info = {
            'name': f'H5 2D (双文件)',
            'resolution': f'{res}x{res}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (res // 2, res // 2),
        }
    else:
        res = train_x.shape[-1]
        info = {
            'name': f'H5 1D (双文件)',
            'resolution': f'{res}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (res // 2,),
        }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}, 分辨率 {info['resolution']}")
    return train_x, train_y, test_x, test_y, info


# ============================================================================
# PT 加载函数
# ============================================================================

def load_pt_two_files(train_pt_path, test_pt_path, n_train=1000, n_test=200, resolution=None):
    """
    从两个 PT 文件加载 PT 格式数据。
    
    Args:
        train_pt_path: 训练集文件路径
        test_pt_path: 测试集文件路径
        n_train: 训练样本数
        n_test: 测试样本数
        resolution: 目标分辨率
    
    Returns:
        train_x, train_y, test_x, test_y, info
    """
    print(f"\n📊 从双文件 PT 加载:")
    print(f"   训练集: {train_pt_path}")
    print(f"   测试集: {test_pt_path}")
    
    # 自动推断分辨率
    if resolution is None:
        fname = train_pt_path.split('/')[-1]
        resolution = parse_resolution_from_filename(fname)
        print(f"   从文件名推断分辨率: {resolution}")
    
    # 加载训练集
    train_data = torch.load(train_pt_path, weights_only=False)
    if isinstance(train_data, dict):
        train_x = train_data.get('x', train_data.get('train_x', train_data.get('input')))
        train_y = train_data.get('y', train_data.get('train_y', train_data.get('output')))
    else:
        # 假设是元组 (x, y)
        train_x, train_y = train_data[0], train_data[1]
    
    # 加载测试集
    test_data = torch.load(test_pt_path, weights_only=False)
    if isinstance(test_data, dict):
        test_x = test_data.get('x', test_data.get('test_x', test_data.get('input')))
        test_y = test_data.get('y', test_data.get('test_y', test_data.get('output')))
    else:
        test_x, test_y = test_data[0], test_data[1]
    
    # 转换为 float
    train_x = train_x.float()
    train_y = train_y.float()
    test_x = test_x.float()
    test_y = test_y.float()
    
    # 添加 channel 维度如果需要
    is_2d = train_x.ndim == 3 or train_x.ndim == 4
    if train_x.ndim == 3:  # [N, H, W] -> 2D 数据
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.ndim == 3:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    if train_x.ndim == 2:  # [N, L] -> 1D 数据，需要添加 channel 维度
        train_x = train_x.unsqueeze(1)
        train_y = train_y.unsqueeze(1)
    if test_x.ndim == 2:
        test_x = test_x.unsqueeze(1)
        test_y = test_y.unsqueeze(1)
    
    # 截取
    train_x = train_x[:n_train]
    train_y = train_y[:n_train]
    test_x = test_x[:n_test]
    test_y = test_y[:n_test]
    
    # 获取实际分辨率
    actual_resolution = train_x.shape[-1]
    if resolution is None:
        resolution = actual_resolution
    
    # 信息
    if is_2d:
        info = {
            'name': f'PT 2D (双文件)',
            'resolution': f'{resolution}x{resolution}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (resolution // 2, resolution // 2),
        }
    else:
        info = {
            'name': 'PT 1D (双文件)',
            'resolution': f'{resolution}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (resolution // 2,),
        }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}, 分辨率 {info['resolution']}")
    return train_x, train_y, test_x, test_y, info


def load_pt_single_file(pt_path, n_train=1000, n_test=200, resolution=None):
    """
    从单个 PT 文件加载数据。
    
    Args:
        pt_path: PT 文件路径
        n_train: 训练样本数
        n_test: 测试样本数
        resolution: 目标分辨率
    
    Returns:
        train_x, train_y, test_x, test_y, info
    """
    print(f"\n📊 从单文件 PT 加载: {pt_path}")
    
    data = torch.load(pt_path, weights_only=False)
    if isinstance(data, dict):
        x = data.get('x')
        y = data.get('y')
    else:
        x, y = data[0], data[1]
    
    x = x.float()
    y = y.float()
    
    # 添加 channel 维度
    if x.ndim == 3:  # [N, H, W] -> 2D 数据
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
    if x.ndim == 2:  # [N, L] -> 1D 数据
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
    
    # 分割
    train_x = x[:n_train]
    train_y = y[:n_train]
    test_x = x[n_train:n_train+n_test]
    test_y = y[n_train:n_train+n_test]
    
    # 推断分辨率
    if resolution is None:
        resolution = parse_resolution_from_filename(pt_path.split('/')[-1])
    
    is_2d = train_x.ndim == 4
    if is_2d:
        actual_res = train_x.shape[-1]
    else:
        actual_res = train_x.shape[-1]
    
    if resolution != actual_res:
        resolution = actual_res
    
    # 信息
    if is_2d:
        info = {
            'name': 'PT 2D (单文件)',
            'resolution': f'{resolution}x{resolution}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (resolution // 2, resolution // 2),
        }
    else:
        info = {
            'name': 'PT 1D (单文件)',
            'resolution': f'{resolution}',
            'n_train': train_x.shape[0],
            'n_test': test_x.shape[0],
            'input_channels': train_x.shape[1],
            'output_channels': train_y.shape[1],
            'n_modes': (resolution // 2,),
        }
    
    print(f"✅ 加载成功: 训练 {train_x.shape[0]}, 测试 {test_x.shape[0]}, 分辨率 {info['resolution']}")
    return train_x, train_y, test_x, test_y, info


# ============================================================================
# 主入口函数
# ============================================================================

def load_dataset(
    dataset_name: str,
    data_format: str = 'pt',
    train_path: str = None,
    test_path: str = None,
    data_path: str = None,
    n_train: int = 1000,
    n_test: int = 200,
    resolution: int = None,
):
    """
    通用数据集加载入口函数。
    
    支持所有格式:
    - PT 单文件
    - PT 双文件 (train + test)
    - H5 单文件
    - H5 双文件 (train + test) ← 这就是你从 Zenodo 下载的格式
    
    参数说明:
    - 如果只有 data_path，就是单文件模式
    - 如果同时提供 train_path 和 test_path，就是双文件模式
    
    Args:
        dataset_name: 'darcy', 'burgers', 'navier_stokes'
        data_format: 'pt' 或 'h5'
        train_path: 训练集文件路径 (双文件模式必需)
        test_path: 测试集文件路径 (双文件模式必需)
        data_path: 数据文件路径 (单文件模式必需)
        n_train: 训练样本数
        n_test: 测试样本数
        resolution: 目标分辨率，如果为 None 从文件名推断
    
    Returns:
        (train_x, train_y, test_x, test_y, info)
    
    使用示例 (Zenodo 双文件 H5):
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name='navier_stokes',
            data_format='h5',
            train_path='./data/2D_NS_Re100_Train.h5',
            test_path='./data/2D_NS_Re100_Test.h5',
            n_train=1000,
            n_test=200,
        )
    
    使用示例 (本地生成 PT):
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name='darcy',
            data_format='pt',
            data_path='./data/darcy_train_16.pt',
            n_train=1000,
            n_test=200,
        )
    """
    # 判断是 2D 还是 1D
    is_2d = dataset_name in ['darcy', 'navier_stokes']
    
    # 双文件模式优先级高
    if train_path is not None and test_path is not None:
        if data_format == 'h5':
            return load_h5_two_files(
            train_h5_path=train_path,
            test_h5_path=test_path,
            n_train=n_train,
            n_test=n_test,
            resolution=resolution,
            is_2d=is_2d,
        )
        elif data_format == 'pt':
            return load_pt_two_files(
                train_pt_path=train_path,
                test_pt_path=test_path,
                n_train=n_train,
                n_test=n_test,
                resolution=resolution,
            )
        else:
            raise ValueError(f"不支持的数据格式: {data_format}")
    
    # 单文件模式
    elif data_path is not None:
        if data_format == 'h5':
            return load_h5_single_file(
                h5_path=data_path,
                n_train=n_train,
                n_test=n_test,
                resolution=resolution,
                is_2d=is_2d,
            )
        elif data_format == 'pt':
            return load_pt_single_file(
                pt_path=data_path,
                n_train=n_train,
                n_test=n_test,
                resolution=resolution,
            )
        else:
            raise ValueError(f"不支持的数据格式: {data_format}")
    
    else:
        raise ValueError("必须提供 data_path 或者 train_path + test_path")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MHF-FNO 通用数据加载测试')
    
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['darcy', 'burgers', 'navier_stokes'],
                       help='数据集名称')
    parser.add_argument('--format', type=str, default='h5',
                       choices=['pt', 'h5'], help='数据格式')
    parser.add_argument('--train_path', type=str, default=None, help='训练集路径 (双文件)')
    parser.add_argument('--test_path', type=str, default=None, help='测试集路径 (双文件)')
    parser.add_argument('--data_path', type=str, default=None, help='数据路径 (单文件)')
    parser.add_argument('--n_train', type=int, default=1000, help='训练样本数')
    parser.add_argument('--n_test', type=int, default=200, help='测试样本数')
    parser.add_argument('--resolution', type=int, default=None, help='目标分辨率')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MHF-FNO 通用数据加载测试")
    print("=" * 60)
    
    try:
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name=args.dataset,
            data_format=args.format,
            train_path=args.train_path,
            test_path=args.test_path,
            data_path=args.data_path,
            n_train=args.n_train,
            n_test=args.n_test,
            resolution=args.resolution,
        )
        
        print("\n加载结果:")
        print(f"  名称: {info['name']}")
        print(f"  分辨率: {info['resolution']}")
        print(f"  训练集: {train_x.shape}")
        print(f"  测试集: {test_x.shape}")
        print(f"  输入通道: {info['input_channels']}")
        print(f"  输出通道: {info['output_channels']}")
        print(f"\n✅ 测试通过！")
        
    except Exception as e:
            print(f"\n❌ 加载失败: {e}")
            import traceback
            traceback.print_exc()
