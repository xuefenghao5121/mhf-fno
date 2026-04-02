#!/usr/bin/env python3
"""
MHF-FNO 并行预训练启动脚本 (v1.6.4)

用于在多核CPU系统上启动并行训练，优化单核利用率。
支持MPI多进程并行，每个进程绑定到指定CPU核心范围。

使用方法:
    # 使用8个进程，每个进程使用16个线程 (总共128核)
    mpirun -np 8 python run_parallel_train.py --omp_num_threads 16 --bind_cpu --dataset darcy
    
    # 自定义配置
    mpirun -np 4 python run_parallel_train.py \
        --omp_num_threads 32 \
        --bind_cpu \
        --dataset navier_stokes \
        --epochs 100 \
        --batch_size 64
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='MHF-FNO 并行预训练启动')
    parser.add_argument('--dataset', type=str, default='darcy',
                        choices=['darcy', 'navier_stokes', 'burgers', 'custom'])
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--n_train', type=int, default=None)
    parser.add_argument('--n_test', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=5)
    
    # 性能优化参数
    parser.add_argument('--omp_num_threads', type=int, default=16,
                        help='每个MPI进程使用的OpenMP线程数')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='每个进程的数据加载worker数')
    parser.add_argument('--bind_cpu', action='store_true',
                        help='启用CPU绑定')
    parser.add_argument('--pin_memory', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    
    # MPI环境检测
    args = parser.parse_args()
    
    # 检测MPI rank
    mpi_rank = int(os.environ.get('PMI_RANK', os.environ.get('OMPI_COMM_WORLD_RANK', 0)))
    mpi_size = int(os.environ.get('PMI_SIZE', os.environ.get('OMPI_COMM_WORLD_SIZE', 1)))
    
    print(f"🔧 MPI 进程信息: rank={mpi_rank}, size={mpi_size}")
    
    # 计算本进程起始核心
    # 每个进程独占 args.omp_num_threads 个核心
    start_core = mpi_rank * args.omp_num_threads
    
    # 构建训练命令
    script_path = Path(__file__).parent / 'train_pretrained.py'
    
    cmd = [sys.executable, str(script_path)]
    cmd.extend(['--dataset', args.dataset])
    if args.train_path:
        cmd.extend(['--train_path', args.train_path])
    if args.test_path:
        cmd.extend(['--test_path', args.test_path])
    if args.n_train:
        cmd.extend(['--n_train', str(args.n_train)])
    if args.n_test:
        cmd.extend(['--n_test', str(args.n_test)])
    cmd.extend(['--epochs', str(args.epochs)])
    cmd.extend(['--batch_size', str(args.batch_size)])
    cmd.extend(['--lr', str(args.lr)])
    cmd.extend(['--hidden_channels', str(args.hidden_channels)])
    cmd.extend(['--n_heads', str(args.n_heads)])
    if args.output_dir:
        cmd.extend(['--output_dir', args.output_dir])
    cmd.extend(['--log_interval', str(args.log_interval)])
    
    # 性能优化参数
    cmd.extend(['--omp_num_threads', str(args.omp_num_threads)])
    cmd.extend(['--num_workers', str(args.num_workers)])
    cmd.extend(['--prefetch_factor', str(args.prefetch_factor)])
    if args.bind_cpu:
        cmd.append('--bind_cpu')
    if args.pin_memory:
        cmd.append('--pin_memory')
    cmd.extend(['--start_core', str(start_core)])
    
    # 显示命令
    if mpi_rank == 0:
        print(f"🚀 启动训练命令: {' '.join(cmd)}")
        print(f"📊 并行配置: {mpi_size} 进程 x {args.omp_num_threads} 线程 = {mpi_size * args.omp_num_threads} 核心")
    
    # 执行训练
    os.execv(sys.executable, cmd)

if __name__ == '__main__':
    main()
