"""
集成测试 - 模拟真实使用场景从 load_dataset 入口调用
验证通过 dataset_name 正确传递 is_2d 给 PT 加载函数
"""

import torch
import tempfile
import os
from data_loader import load_dataset

def test_integration_1d_pt_single():
    """集成测试: 1D burgers PT单文件"""
    print("\n🧪 集成测试: 1D burgers PT单文件 [N, L]")
    
    N = 100
    L = 128
    x = torch.randn(N, L)
    y = torch.randn(N, L)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save((x, y), f.name)
        temp_path = f.name
    
    try:
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name='burgers',  # 这会设置 is_2d=False
            data_format='pt',
            data_path=temp_path,
            n_train=80,
            n_test=20,
        )
        
        print(f"  train_x shape: {train_x.shape}")
        print(f"  input_channels: {info['input_channels']}")
        
        assert train_x.shape == (80, 1, 128), f"Expected (80, 1, 128), got {train_x.shape}"
        assert info['input_channels'] == 1, f"Expected 1, got {info['input_channels']}"
        
        print("✅ 通过")
        return True
    finally:
        os.unlink(temp_path)

def test_integration_1d_pt_double():
    """集成测试: 1D burgers PT双文件"""
    print("\n🧪 集成测试: 1D burgers PT双文件 [N, L]")
    
    N_train = 80
    N_test = 20
    L = 128
    train_x = torch.randn(N_train, L)
    train_y = torch.randn(N_train, L)
    test_x = torch.randn(N_test, L)
    test_y = torch.randn(N_test, L)
    
    with tempfile.NamedTemporaryFile(suffix='_train.pt', delete=False) as f1:
        torch.save((train_x, train_y), f1.name)
        train_path = f1.name
    
    with tempfile.NamedTemporaryFile(suffix='_test.pt', delete=False) as f2:
        torch.save((test_x, test_y), f2.name)
        test_path = f2.name
    
    try:
        train_x_out, train_y_out, test_x_out, test_y_out, info = load_dataset(
            dataset_name='burgers',  # is_2d=False
            data_format='pt',
            train_path=train_path,
            test_path=test_path,
            n_train=80,
            n_test=20,
        )
        
        print(f"  train_x shape: {train_x_out.shape}")
        print(f"  input_channels: {info['input_channels']}")
        
        assert train_x_out.shape == (80, 1, 128), f"Expected (80, 1, 128), got {train_x_out.shape}"
        assert info['input_channels'] == 1, f"Expected 1, got {info['input_channels']}"
        
        print("✅ 通过")
        return True
    finally:
        os.unlink(train_path)
        os.unlink(test_path)

def test_integration_1d_already_has_channel():
    """集成测试: 1D 已经有通道维度 [N, 1, L]"""
    print("\n🧪 集成测试: 1D burgers PT单文件 [N, 1, L] (已有通道)")
    
    N = 100
    L = 128
    x = torch.randn(N, 1, L)
    y = torch.randn(N, 1, L)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save((x, y), f.name)
        temp_path = f.name
    
    try:
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name='burgers',  # is_2d=False
            data_format='pt',
            data_path=temp_path,
            n_train=80,
            n_test=20,
        )
        
        print(f"  Original: {x.shape}")
        print(f"  Output: {train_x.shape}")
        print(f"  input_channels: {info['input_channels']}")
        
        # 不应该重复添加通道，形状保持不变
        assert train_x.shape == (80, 1, 128), f"Expected (80, 1, 128), got {train_x.shape}"
        assert info['input_channels'] == 1, f"Expected 1, got {info['input_channels']}"
        
        print("✅ 通过 (不会重复添加)")
        return True
    finally:
        os.unlink(temp_path)

def test_integration_2d_pt_single():
    """集成测试: 2D darcy PT单文件"""
    print("\n🧪 集成测试: 2D darcy PT单文件 [N, H, W]")
    
    N = 100
    H = W = 32
    x = torch.randn(N, H, W)
    y = torch.randn(N, H, W)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save((x, y), f.name)
        temp_path = f.name
    
    try:
        train_x, train_y, test_x, test_y, info = load_dataset(
            dataset_name='darcy',  # is_2d=True
            data_format='pt',
            data_path=temp_path,
            n_train=80,
            n_test=20,
        )
        
        print(f"  train_x shape: {train_x.shape}")
        print(f"  input_channels: {info['input_channels']}")
        
        assert train_x.shape == (80, 1, 32, 32), f"Expected (80, 1, 32, 32), got {train_x.shape}"
        assert info['input_channels'] == 1, f"Expected 1, got {info['input_channels']}"
        
        print("✅ 通过")
        return True
    finally:
        os.unlink(temp_path)

def main():
    tests = [
        test_integration_1d_pt_single,
        test_integration_1d_pt_double,
        test_integration_1d_already_has_channel,
        test_integration_2d_pt_single,
    ]
    
    print("\n🚀 运行集成测试...")
    passed = []
    failed = []
    
    for test in tests:
        try:
            if test():
                passed.append(test.__name__)
            else:
                failed.append(test.__name__)
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n📊 集成测试总结:")
    print(f"✅ 通过: {len(passed)}/{len(tests)}")
    print(f"❌ 失败: {len(failed)}/{len(tests)}")
    
    if len(failed) == 0:
        print("\n🎉 所有集成测试通过！修复完整正确")
        return True
    else:
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
