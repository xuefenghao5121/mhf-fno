"""
MHF-FNO 基础使用示例
"""

import torch
from mhf_fno import MHFSpectralConv, create_hybrid_fno, MHFFNO

def example_basic():
    """基础使用示例"""
    print("=" * 50)
    print("示例 1: 基础使用")
    print("=" * 50)
    
    # 方法 1: 使用预设的最佳配置
    model = MHFFNO.best_config()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    x = torch.randn(10, 1, 16, 16)
    y = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")


def example_custom_config():
    """自定义配置示例"""
    print("\n" + "=" * 50)
    print("示例 2: 自定义配置")
    print("=" * 50)
    
    # 自定义配置
    model = create_hybrid_fno(
        n_modes=(16, 16),      # 更多的频率模式
        hidden_channels=64,     # 更大的隐藏层
        n_layers=4,             # 更多层
        mhf_layers=[0, 2, 3],   # 第1、3、4层使用 MHF
        n_heads=8               # 更多的头
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"自定义模型参数量: {params:,}")


def example_compare_params():
    """参数量对比"""
    print("\n" + "=" * 50)
    print("示例 3: 参数量对比")
    print("=" * 50)
    
    from neuralop.models import FNO
    
    # 标准 FNO
    fno = FNO(n_modes=(8, 8), hidden_channels=32, n_layers=3)
    fno_params = sum(p.numel() for p in fno.parameters())
    
    # MHF-FNO (最佳配置)
    mhf = MHFFNO.best_config()
    mhf_params = sum(p.numel() for p in mhf.parameters())
    
    print(f"标准 FNO-32: {fno_params:,} 参数")
    print(f"MHF-FNO:     {mhf_params:,} 参数")
    print(f"参数减少:    {(fno_params - mhf_params) / fno_params * 100:.1f}%")


if __name__ == "__main__":
    example_basic()
    example_custom_config()
    example_compare_params()