"""
Pytest 配置文件

定义通用 fixtures 和配置
"""

import pytest
import torch


@pytest.fixture
def device():
    """返回可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@pytest.fixture
def small_2d_input():
    """小尺寸 2D 输入张量"""
    return torch.randn(2, 1, 16, 16)


@pytest.fixture
def small_1d_input():
    """小尺寸 1D 输入张量"""
    return torch.randn(2, 1, 64)


@pytest.fixture
def default_config():
    """默认模型配置"""
    return {
        'n_modes': (8, 8),
        'hidden_channels': 32,
        'in_channels': 1,
        'out_channels': 1,
        'n_layers': 3,
        'n_heads': 4,
    }