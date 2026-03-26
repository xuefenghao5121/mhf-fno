"""
测试所有维度处理的边界情况
验证所有数据格式都正确添加了通道维度
"""

import torch
import tempfile
import os
from data_loader import (
    load_pt_single_file,
    load_pt_two_files,
    load_h5_single_file,
    load_h5_two_files,
    parse_resolution_from_filename,
)

def test_1d_pt_single_file_without_channel():
    """测试 1D PT 单文件 - 原始数据没有通道维度 [N, L]"""
    print("\n" + "="*60)
    print("🧪 测试 1D PT 单文件 - 原始数据 [N, L] (缺少通道维度)")
    print("="*60)
    
    # 创建测试数据
    N = 100
    L = 128
    x = torch.randn(N, L)
    y = torch.randn(N, L)
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save((x, y), f.name)
        temp_path = f.name
    
    try:
        train_x, train_y, test_x, test_y, info = load_pt_single_file(
            temp_path,
            n_train=80,
            n_test=20,
            resolution=128,
            is_2d=False,  # 1D data
        )
        
        print(f"✓ train_x shape: {train_x.shape}")
        print(f"✓ train_y shape: {train_y.shape}")
        print(f"✓ test_x shape: {test_x.shape}")
        print(f"✓ test_x shape: {test_y.shape}")
        print(f"✓ input_channels: {info['input_channels']}")
        
        # 验证形状正确 [N, 1, L]
        assert train_x.ndim == 3, f"train_x should be 3D, got {train_x.ndim}D"
        assert train_x.shape[1] == 1, f"channel dimension should be 1, got {train_x.shape[1]}"
        assert train_x.shape[2] == 128, f"resolution should be 128, got {train_x.shape[2]}"
        assert info['input_channels'] == 1, f"input_channels should be 1, got {info['input_channels']}"
        
        print("\n✅ 1D PT 单文件测试通过！通道维度正确添加")
        return True
    finally:
        os.unlink(temp_path)

def test_1d_pt_two_files_without_channel():
    """测试 1D PT 双文件 - 原始数据没有通道维度 [N, L] - 这就是刚刚修复的bug!"""
    print("\n" + "="*60)
    print("🧪 测试 1D PT 双文件 - 原始数据 [N, L] (缺少通道维度) - BUG FIX VERIFICATION")
    print("="*60)
    
    # 创建测试数据
    N_train = 80
    N_test = 20
    L = 128
    train_x = torch.randn(N_train, L)
    train_y = torch.randn(N_train, L)
    test_x = torch.randn(N_test, L)
    test_y = torch.randn(N_test, L)
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(suffix='_train.pt', delete=False) as f1:
        torch.save((train_x, train_y), f1.name)
        train_path = f1.name
    
    with tempfile.NamedTemporaryFile(suffix='_test.pt', delete=False) as f2:
        torch.save((test_x, test_y), f2.name)
        test_path = f2.name
    
    try:
        train_x_out, train_y_out, test_x_out, test_y_out, info = load_pt_two_files(
            train_path,
            test_path,
            n_train=80,
            n_test=20,
            resolution=128,
            is_2d=False,  # 1D data
        )
        
        print(f"✓ train_x shape: {train_x_out.shape}")
        print(f"✓ train_y shape: {train_y_out.shape}")
        print(f"✓ test_x shape: {test_x_out.shape}")
        print(f"✓ test_y shape: {test_y_out.shape}")
        print(f"✓ input_channels: {info['input_channels']}")
        
        # 验证形状正确 [N, 1, L]
        assert train_x_out.ndim == 3, f"train_x should be 3D [N, 1, L], got {train_x_out.ndim}D shape {train_x_out.shape}"
        assert train_x_out.shape[1] == 1, f"channel dimension should be 1, got {train_x_out.shape[1]}"
        assert train_x_out.shape[2] == 128, f"resolution should be 128, got {train_x_out.shape[2]}"
        assert info['input_channels'] == 1, f"input_channels should be 1, got {info['input_channels']}"
        
        # 验证测试集也正确
        assert test_x_out.ndim == 3, f"test_x should be 3D [N, 1, L], got {test_x_out.ndim}D shape {test_x_out.shape}"
        assert test_x_out.shape[1] == 1, f"test_x channel should be 1, got {test_x_out.shape[1]}"
        
        print("\n✅ 1D PT 双文件测试通过！通道维度正确添加 (BUG 已修复)")
        return True
    finally:
        os.unlink(train_path)
        os.unlink(test_path)

def test_2d_pt_single_file_without_channel():
    """测试 2D PT 单文件 - 原始数据没有通道维度 [N, H, W]"""
    print("\n" + "="*60)
    print("🧪 测试 2D PT 单文件 - 原始数据 [N, H, W] (缺少通道维度)")
    print("="*60)
    
    N = 100
    H = W = 32
    x = torch.randn(N, H, W)
    y = torch.randn(N, H, W)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save((x, y), f.name)
        temp_path = f.name
    
    try:
        train_x, train_y, test_x, test_y, info = load_pt_single_file(
            temp_path,
            n_train=80,
            n_test=20,
            resolution=32,
        )
        
        print(f"✓ train_x shape: {train_x.shape}")
        print(f"✓ train_y shape: {train_y.shape}")
        print(f"✓ input_channels: {info['input_channels']}")
        
        assert train_x.ndim == 4, f"train_x should be 4D [N, 1, H, W], got {train_x.ndim}D"
        assert train_x.shape[1] == 1, f"channel should be 1, got {train_x.shape[1]}"
        assert train_x.shape[2] == 32, f"H should be 32, got {train_x.shape[2]}"
        assert train_x.shape[3] == 32, f"W should be 32, got {train_x.shape[3]}"
        assert info['input_channels'] == 1, f"input_channels should be 1, got {info['input_channels']}"
        
        print("\n✅ 2D PT 单文件测试通过！")
        return True
    finally:
        os.unlink(temp_path)

def test_2d_pt_two_files_without_channel():
    """测试 2D PT 双文件 - 原始数据没有通道维度 [N, H, W]"""
    print("\n" + "="*60)
    print("🧪 测试 2D PT 双文件 - 原始数据 [N, H, W] (缺少通道维度)")
    print("="*60)
    
    N_train = 80
    N_test = 20
    H = W = 32
    train_x = torch.randn(N_train, H, W)
    train_y = torch.randn(N_train, H, W)
    test_x = torch.randn(N_test, H, W)
    test_y = torch.randn(N_test, H, W)
    
    with tempfile.NamedTemporaryFile(suffix='_train.pt', delete=False) as f1:
        torch.save((train_x, train_y), f1.name)
        train_path = f1.name
    
    with tempfile.NamedTemporaryFile(suffix='_test.pt', delete=False) as f2:
        torch.save((test_x, test_y), f2.name)
        test_path = f2.name
    
    try:
        train_x_out, train_y_out, test_x_out, test_y_out, info = load_pt_two_files(
            train_path, test_path,
            n_train=80, n_test=20,
            resolution=32,
        )
        
        print(f"✓ train_x shape: {train_x_out.shape}")
        print(f"✓ input_channels: {info['input_channels']}")
        
        assert train_x_out.ndim == 4, f"train_x should be 4D, got {train_x_out.ndim}D"
        assert train_x_out.shape[1] == 1, f"channel should be 1, got {train_x_out.shape[1]}"
        assert info['input_channels'] == 1, f"input_channels should be 1, got {info['input_channels']}"
        
        print("\n✅ 2D PT 双文件测试通过！")
        return True
    finally:
        os.unlink(train_path)
        os.unlink(test_path)

def test_already_has_channel():
    """测试数据已经有通道维度的情况"""
    print("\n" + "="*60)
    print("🧪 测试数据已经包含通道维度 - 不应该重复添加")
    print("="*60)
    
    # 1D 已经有通道 [N, 1, L]
    N = 100
    L = 128
    x = torch.randn(N, 1, L)
    y = torch.randn(N, 1, L)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save((x, y), f.name)
        temp_path = f.name
    
    try:
        train_x, train_y, test_x, test_y, info = load_pt_single_file(
            temp_path, n_train=80, n_test=20, resolution=128,
            is_2d=False,  # 1D data - [N, 1, L] 已经有通道
        )
        
        print(f"✓ Original x shape: {x.shape}")
        print(f"✓ Output train_x shape: {train_x.shape}")
        print(f"✓ input_channels: {info['input_channels']}")
        
        # 不应该重复添加通道，形状保持 [N, 1, L]
        assert train_x.shape == (80, 1, 128), f"shape should remain (80, 1, 128), got {train_x.shape}"
        assert info['input_channels'] == 1, f"input_channels should be 1, got {info['input_channels']}"
        
        print("\n✅ 已有通道维度测试通过！不会重复添加")
        return True
    finally:
        os.unlink(temp_path)

def test_2d_already_has_channel():
    """测试 2D 已经有通道维度"""
    print("\n" + "="*60)
    print("🧪 测试 2D 数据已经包含通道维度")
    print("="*60)
    
    N = 100
    H = W = 32
    x = torch.randn(N, 1, H, W)
    y = torch.randn(N, 1, H, W)
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save((x, y), f.name)
        temp_path = f.name
    
    try:
        train_x, train_y, test_x, test_y, info = load_pt_single_file(
            temp_path, n_train=80, n_test=20, resolution=32,
        )
        
        print(f"✓ Original x shape: {x.shape}")
        print(f"✓ Output train_x shape: {train_x.shape}")
        print(f"✓ input_channels: {info['input_channels']}")
        
        assert train_x.shape == (80, 1, 32, 32), f"shape should remain (80, 1, 32, 32), got {train_x.shape}"
        assert info['input_channels'] == 1, f"input_channels should be 1, got {info['input_channels']}"
        
        print("\n✅ 2D 已有通道维度测试通过！")
        return True
    finally:
        os.unlink(temp_path)

def test_resolution_parsing():
    """测试文件名分辨率解析"""
    print("\n" + "="*60)
    print("🧪 测试文件名分辨率解析各种格式")
    print("="*60)
    
    test_cases = [
        ("burgers_train_128.pt", 128),
        ("darcy_train_64.pt", 64),
        ("1D_Burgers_Re1000_Train.h5", 1024),  # Re1000 -> 1024
        ("2D_NS_Re100_Train.h5", 128),         # Re100 -> 128
        ("64x64_darcy.h5", 64),
        ("1024x1024_darcy.h5", 1024),
        ("darcy-256-train.h5", 256),  # 连字符格式
        ("128-burgers-train.h5", 128), # 开头数字连字符
        ("burgers-train-2048.h5", 2048), # 结尾数字连字符
        ("file-128.pt", 128),
        ("128-file.pt", 128),
        ("ns_train_1024.pt", 1024),
        ("NavierStokes2d_256.h5", 256),
    ]
    
    all_ok = True
    for filename, expected in test_cases:
        result = parse_resolution_from_filename(filename)
        ok = result == expected
        all_ok = all_ok and ok
        status = "✅" if ok else "❌"
        print(f"{status} {filename:<40} -> {result}, expected {expected}")
    
    if all_ok:
        print("\n✅ 所有分辨率解析测试通过！")
    else:
        print("\n❌ 某些分辨率解析测试失败！")
    
    return all_ok

def main():
    """运行所有测试"""
    print("\n🚀 开始运行所有维度处理测试...")
    
    tests = [
        test_1d_pt_single_file_without_channel,
        test_1d_pt_two_files_without_channel,  # <- 这个就是修复的关键测试
        test_2d_pt_single_file_without_channel,
        test_2d_pt_two_files_without_channel,
        test_already_has_channel,
        test_2d_already_has_channel,
        test_resolution_parsing,
    ]
    
    passed = []
    failed = []
    
    for test in tests:
        try:
            if test():
                passed.append(test.__name__)
            else:
                failed.append(test.__name__)
        except Exception as e:
            print(f"\n❌ {test.__name__} 抛出异常: {e}")
            import traceback
            traceback.print_exc()
            failed.append(test.__name__)
    
    print("\n" + "="*60)
    print("📊 测试总结")
    print("="*60)
    print(f"✅ 通过: {len(passed)}/{len(tests)}")
    print(f"❌ 失败: {len(failed)}/{len(tests)}")
    
    if len(failed) == 0:
        print("\n🎉 所有测试通过！所有维度处理bug都已修复")
        return True
    else:
        print(f"\n⚠️  有 {len(failed)} 个测试失败")
        for f in failed:
            print(f"  - {f}")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
