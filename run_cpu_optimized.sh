#!/bin/bash
# CPU 优化启动脚本
# 用法: ./run_cpu_optimized.sh python benchmark/run_benchmarks.py --dataset darcy

# 获取 CPU 核心数
N_CORES=$(nproc)

echo "=============================================="
echo "CPU 优化启动"
echo "=============================================="
echo "CPU 核心数: $N_CORES"

# 设置所有并行化相关的环境变量
export OMP_NUM_THREADS=$N_CORES
export MKL_NUM_THREADS=$N_CORES
export OPENBLAS_NUM_THREADS=$N_CORES
export VECLIB_MAXIMUM_THREADS=$N_CORES
export NUMEXPR_NUM_THREADS=$N_CORES

# KMP (Intel OpenMP) 设置
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

echo "环境变量:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  KMP_AFFINITY=$KMP_AFFINITY"
echo "=============================================="

# 检查是否有参数
if [ $# -eq 0 ]; then
    echo "用法: $0 <command>"
    echo "示例: $0 python benchmark/run_benchmarks.py --dataset darcy"
    exit 1
fi

# 执行命令
echo "执行: $@"
echo ""

# 使用 taskset 绑定所有核心 (可选，取消注释以启用)
# taskset -c 0-$((N_CORES-1)) "$@"

"$@"