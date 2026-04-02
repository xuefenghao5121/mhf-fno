#!/usr/bin/env python3
"""
MHF-FNO 推理脚本 (v1.6.4)

加载预训练模型进行推理预测。

使用方法:
    # 基本推理
    python inference.py --model pretrained/models/mhf_fno_darcy_pretrained.pth \
        --input ./data/sample.pt

    # 批量推理
    python inference.py --model pretrained/models/mhf_fno_darcy_pretrained.pth \
        --input ./data/test_samples.pt --batch_size 32

    # 从数据目录加载
    python inference.py --model pretrained/models/mhf_fno_darcy_pretrained.pth \
        --input ./data/train.pt --input_key x --n_samples 10
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from mhf_fno import MHFFNOWithAttention, get_device


def load_pretrained_model(model_path, device='cpu'):
    """
    加载预训练模型
    
    Args:
        model_path: .pth 文件路径
        device: 推理设备
    
    Returns:
        (model, config) 元组
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    version = checkpoint.get('version', 'unknown')
    dataset = checkpoint.get('dataset', 'unknown')
    
    print(f"📦 加载预训练模型: {model_path}")
    print(f"   版本: {version}")
    print(f"   数据集: {dataset}")
    print(f"   配置: {json.dumps(config, indent=4)}")
    
    model = MHFFNOWithAttention.best_config(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {n_params:,}")
    print(f"   输入形状: {checkpoint.get('input_shape', 'N/A')}")
    print(f"   输出形状: {checkpoint.get('output_shape', 'N/A')}")
    print(f"   最佳测试损失: {checkpoint.get('best_test_loss', 'N/A')}")
    
    return model, config


def load_input(input_path, input_key=None, n_samples=None, device='cpu'):
    """
    加载输入数据
    
    Args:
        input_path: 数据文件路径 (.pt, .npy, .npz)
        input_key: 字典中的 key (默认 'x' 或第一个)
        n_samples: 限制样本数
        device: 设备
    
    Returns:
        torch.Tensor
    """
    path = Path(input_path)
    
    if path.suffix in ['.pt', '.pth']:
        data = torch.load(input_path, map_location=device, weights_only=False)
    elif path.suffix in ['.npy']:
        data = torch.from_numpy(np.load(input_path))
    elif path.suffix in ['.npz']:
        npz = np.load(input_path)
        key = input_key or list(npz.keys())[0]
        data = torch.from_numpy(npz[key])
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    # 提取
    if isinstance(data, dict):
        key = input_key or 'x'
        data = data.get(key, data.get('input', list(data.values())[0]))
    elif isinstance(data, (tuple, list)):
        data = data[0]
    
    data = data.float()
    
    # 自动添加通道维度
    if data.dim() == 2:
        data = data.unsqueeze(1)
    elif data.dim() == 3 and data.shape[1] > 4:
        # 可能是时间序列
        data = data[:, 0:1]
    
    if n_samples:
        data = data[:n_samples]
    
    return data


def inference(args):
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    
    # 加载模型
    model, config = load_pretrained_model(args.model, device)
    
    # 加载输入
    print(f"\n📊 加载输入数据: {args.input}")
    input_tensor = load_input(args.input, args.input_key, args.n_samples, device)
    print(f"   输入形状: {input_tensor.shape}")
    
    # 推理
    print(f"\n🔮 开始推理...")
    model.eval()
    
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(input_tensor), args.batch_size):
            batch = input_tensor[i:i+args.batch_size].to(device)
            pred = model(batch)
            all_preds.append(pred.cpu())
    
    predictions = torch.cat(all_preds, dim=0)
    print(f"   输出形状: {predictions.shape}")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix in ['.pt', '.pth']:
            torch.save(predictions, output_path)
        elif output_path.suffix in ['.npy']:
            np.save(output_path, predictions.numpy())
        else:
            torch.save(predictions, output_path)
        
        print(f"\n💾 预测结果已保存: {output_path}")
        print(f"   文件大小: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        print(f"\n   预测统计:")
        print(f"   均值: {predictions.mean():.6f}")
        print(f"   标准差: {predictions.std():.6f}")
        print(f"   最小值: {predictions.min():.6f}")
        print(f"   最大值: {predictions.max():.6f}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='MHF-FNO 推理')
    parser.add_argument('--model', type=str, required=True, help='预训练模型路径 (.pth)')
    parser.add_argument('--input', type=str, required=True, help='输入数据路径')
    parser.add_argument('--input_key', type=str, default=None, help='输入数据字典中的 key')
    parser.add_argument('--output', type=str, default=None, help='输出结果路径')
    parser.add_argument('--n_samples', type=int, default=None, help='推理样本数')
    parser.add_argument('--batch_size', type=int, default=64, help='推理批大小')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()
    
    inference(args)


if __name__ == '__main__':
    main()
