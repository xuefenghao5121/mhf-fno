#!/usr/bin/env python3
"""测试修复后的分辨率解析"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import parse_resolution_from_filename

# 测试用例
test_cases = [
    # (输入文件名, 期望输出, 描述)
    ("2D_DarcyFlow_64x64_Train.h5", 64, "2D 带 NxN 格式"),
    ("NavierStokes2d_128_Train.h5", 128, "下划线后直接跟分辨率"),
    ("2D_NS_Re100_Train.h5", 128, "Re100 → 映射到 128"),
    ("1D_Burgers_Re1000_Train.h5", 1024, "Re1000 → 映射到 1024"),
    ("darcy_train_16.pt", 16, "简单下划线格式"),
    ("ns_train_64.pt", 64, "简单下划线格式"),
    ("burgers_train_1024.pt", 1024, "简单下划线格式"),
    ("2D_DarcyFlow_64_Train.h5", 64, "没有 x 格式"),
    ("2D_DarcyFlow_128x128_Test.h5", 128, "NxN 格式"),
    ("2D_DarcyFlow_2048x2048_Train.h5", 2048, "高分辨率 NxN"),
    ("4096x4096_Darcy_Train.h5", 4096, "开头就是 NxN"),
    ("2D_DarcyFlow-256x256-Train.h5", 256, "连字符分隔 NxN"),
    ("darcy-512-train.pt", 512, "连字符分隔纯数字"),
    ("1D_Re10000_Burgers.h5", 2048, "Re10000 → 映射到 2048"),
    ("2D_Re10_NS.h5", 64, "Re10 → 映射到 64"),
    ("Re1000_Burgers_Train.h5", 1024, "开头就是 Re"),
    ("1024_Darcy_Train.h5", 1024, "开头纯数字"),
    ("2D_DarcyFlow_Train.h5", 64, "没有数字 → 默认值 64"),
    ("DarcyFlow.h5", 64, "完全没有数字 → 默认值 64"),
    ("2D_DarcyFlow_PDE_1024_Train.h5", 1024, "多个下划线后分辨率"),
    ("NavierStokes_2d_256x256_Train.h5", 256, "2d 后面 NxN"),
    ("dataset_1024_darcy_train.h5", 1024, "中间出现分辨率"),
    ("1_2_1024_4096.h5", 1024, "多个数字，第一个大于 8 的是 1024"),
]

print("=" * 70)
print("测试修复后的 parse_resolution_from_filename")
print("=" * 70)
print()

passed = 0
failed = 0

for filename, expected, description in test_cases:
    result = parse_resolution_from_filename(filename)
    if result == expected:
        print(f"✅ PASS: '{filename:<40}' → {result:<4} ({description})")
        passed += 1
    else:
        print(f"❌ FAIL: '{filename:<40}' → {result:<4} 期望 {expected} ({description})")
        failed += 1

print()
print("=" * 70)
print(f"总结: 通过 {passed}/{passed+failed}")

if failed > 0:
    print(f"失败: {failed} 个测试")
    sys.exit(1)
else:
    print("🎉 所有测试通过！")
    sys.exit(0)
