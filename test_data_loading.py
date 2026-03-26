#!/usr/bin/env python3
"""
完整数据加载测试
测试所有格式和维度组合:
- 1D PT 双文件
- 2D PT 双文件  
- 1D PT 单文件
- 2D PT 单文件
"""

import torch
from benchmark.data_loader import load_dataset

def test_1d_pt_double():
    """测试1D PT双文件加载"""
    print("\n" + "="*60)
    print("TEST 1: 1D PT 双文件加载 (burgers)")
    print("="*60)
    
    train_x, train_y, test_x, test_y, info = load_dataset(
        dataset_name='burgers',
        data_format='pt',
        train_path='./data/burgers_train_256.pt',
        test_path='./data/burgers_test_256.pt',
        n_train=100,
        n_test=20,
    )
    
    print(f"train_x shape = {train_x.shape}")
    print(f"train_y shape = {train_y.shape}")
    print(f"test_x shape = {test_x.shape}")
    print(f"info: {info}")
    
    # 1D数据期望形状: [N, C, L] = [100, 1, 256]
    assert train_x.ndim == 3, f"Expected 3 dimensions, got {train_x.ndim}"
    assert train_x.shape[0] == 100, f"Expected 100 train samples, got {train_x.shape[0]}"
    assert train_x.shape[1] == 1, f"Expected 1 input channel, got {train_x.shape[1]}"
    assert train_x.shape[2] == 256, f"Expected resolution 256, got {train_x.shape[2]}"
    assert info['input_channels'] == 1, f"Expected 1 input channel in info, got {info['input_channels']}"
    
    print("\n✅ 1D PT 双文件测试通过!")
    return True

def test_2d_pt_double():
    """测试2D PT双文件加载"""
    print("\n" + "="*60)
    print("TEST 2: 2D PT 双文件加载 (darcy)")
    print("="*60)
    
    train_x, train_y, test_x, test_y, info = load_dataset(
        dataset_name='darcy',
        data_format='pt',
        train_path='./data/darcy_train_32.pt',
        test_path='./data/darcy_test_32.pt',
        n_train=100,
        n_test=20,
    )
    
    print(f"train_x shape = {train_x.shape}")
    print(f"train_y shape = {train_y.shape}")
    print(f"test_x shape = {test_x.shape}")
    print(f"info: {info}")
    
    # 2D数据期望形状: [N, C, H, W] = [100, 1, 32, 32]
    assert train_x.ndim == 4, f"Expected 4 dimensions, got {train_x.ndim}"
    assert train_x.shape[0] == 100, f"Expected 100 train samples, got {train_x.shape[0]}"
    assert train_x.shape[1] == 1, f"Expected 1 input channel, got {train_x.shape[1]}"
    assert train_x.shape[2] == 32, f"Expected height 32, got {train_x.shape[2]}"
    assert train_x.shape[3] == 32, f"Expected width 32, got {train_x.shape[3]}"
    assert info['input_channels'] == 1, f"Expected 1 input channel in info, got {info['input_channels']}"
    
    print("\n✅ 2D PT 双文件测试通过!")
    return True

def test_2d_ns_pt_double():
    """测试2D Navier-Stokes PT双文件加载"""
    print("\n" + "="*60)
    print("TEST 3: 2D Navier-Stokes PT 双文件加载")
    print("="*60)
    
    train_x, train_y, test_x, test_y, info = load_dataset(
        dataset_name='navier_stokes',
        data_format='pt',
        train_path='./data/ns_train_32.pt',
        test_path='./data/ns_test_32.pt',
        n_train=100,
        n_test=20,
    )
    
    print(f"train_x shape = {train_x.shape}")
    print(f"train_y shape = {train_y.shape}")
    print(f"test_x shape = {test_x.shape}")
    print(f"info: {info}")
    
    assert train_x.ndim == 4, f"Expected 4 dimensions, got {train_x.ndim}"
    assert train_x.shape[1] == 1, f"Expected 1 input channel, got {train_x.shape[1]}"
    
    print("\n✅ 2D Navier-Stokes PT 双文件测试通过!")
    return True

def test_1d_pt_single():
    """测试1D PT单文件加载"""
    print("\n" + "="*60)
    print("TEST 4: 1D PT 单文件加载")
    print("="*60)
    
    # 创建临时测试文件
    n_samples = 1000
    x = torch.randn(n_samples, 256)
    y = torch.randn(n_samples, 256)
    data = {'x': x, 'y': y}
    torch.save(data, './data/_test_1d_single.pt')
    
    try:
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name='burgers',
            data_format='pt',
            data_path='./data/_test_1d_single.pt',
            n_train=800,
            n_test=200,
            resolution=256,
        )
        
        print(f"train_x shape = {train_x.shape}")
        print(f"train_y shape = {train_y.shape}")
        print(f"test_x shape = {test_x.shape}")
        
        assert train_x.ndim == 3, f"Expected 3 dimensions, got {train_x.ndim}"
        assert train_x.shape[1] == 1, f"Expected 1 input channel, got {train_x.shape[1]}"
        assert train_x.shape[0] == 800, f"Expected 800 train samples, got {train_x.shape[0]}"
        
        print("\n✅ 1D PT 单文件测试通过!")
        return True
    finally:
        import os
        os.remove('./data/_test_1d_single.pt')

def test_2d_pt_single():
    """测试2D PT单文件加载"""
    print("\n" + "="*60)
    print("TEST 5: 2D PT 单文件加载")
    print("="*60)
    
    # 创建临时测试文件
    n_samples = 1200
    x = torch.randn(n_samples, 32, 32)
    y = torch.randn(n_samples, 32, 32)
    data = {'x': x, 'y': y}
    torch.save(data, './data/_test_2d_single.pt')
    
    try:
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name='darcy',
            data_format='pt',
            data_path='./data/_test_2d_single.pt',
            n_train=1000,
            n_test=200,
            resolution=32,
        )
        
        print(f"train_x shape = {train_x.shape}")
        print(f"train_y shape = {train_y.shape}")
        print(f"test_x shape = {test_x.shape}")
        
        assert train_x.ndim == 4, f"Expected 4 dimensions, got {train_x.ndim}"
        assert train_x.shape[1] == 1, f"Expected 1 input channel, got {train_x.shape[1]}"
        assert train_x.shape[0] == 1000, f"Expected 1000 train samples, got {train_x.shape[0]}"
        
        print("\n✅ 2D PT 单文件测试通过!")
        return True
    finally:
        import os
        os.remove('./data/_test_2d_single.pt')

def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("MHF-FNO 数据加载完整测试套件")
    print("="*60)
    
    tests = [
        test_1d_pt_double,
        test_2d_pt_double,
        test_2d_ns_pt_double, 
        test_1d_pt_single,
        test_2d_pt_single,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print(f"测试完成: 通过 {passed}, 失败 {failed}")
    print("="*60)
    
    if failed == 0:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print("\n💔 有测试失败，请修复问题!")
        return 1

if __name__ == '__main__':
    exit(main())
