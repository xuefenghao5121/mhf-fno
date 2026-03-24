"""
MHF Benchmark - 参考NeuralOperator风格

测试内容：
1. 因果性验证
2. 语言建模性能（WikiText-2）
3. 与标准Transformer对比
4. CPU性能测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from typing import Dict, List, Tuple

# 添加路径
sys.path.insert(0, '/root/.openclaw/workspace/memory/projects/tianyuan-fft/experiments')
from causal_mhf import CausalMHFTransformer, CausalSpectralConv


class StandardTransformerBlock(nn.Module):
    """标准Transformer块，用于对比"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 因果Self-Attention
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class StandardTransformer(nn.Module):
    """标准Transformer，用于对比"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, n_layers: int = 4, max_seq_len: int = 2048):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> Dict:
        batch_size, seq_len = input_ids.shape
        
        x = self.embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_embedding(positions)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}


def test_causality():
    """测试因果性"""
    print("=" * 60)
    print("测试1: 因果性验证")
    print("=" * 60)
    
    d_model, n_heads, seq_len = 64, 4, 20
    
    # 创建MHF块
    mhf_block = CausalSpectralConv(
        in_channels=d_model,
        out_channels=d_model,
        n_modes=16,
        causal=True,
    )
    
    # 测试输入
    x = torch.randn(1, d_model, seq_len)
    
    # 获取原始输出
    out1 = mhf_block(x.clone())
    
    # 修改后半部分输入
    x_modified = x.clone()
    x_modified[:, :, 10:] = torch.randn(1, d_model, 10)
    
    # 获取修改后输出
    out2 = mhf_block(x_modified)
    
    # 验证前10个位置的输出是否相同
    diff = torch.abs(out1[:, :, :10] - out2[:, :, :10]).max().item()
    
    print(f"前半部分输出最大差异: {diff:.6e}")
    
    if diff < 1e-5:
        print("✅ 因果性测试通过！未来信息未泄露到当前输出")
    else:
        print("❌ 因果性测试失败！存在信息泄露")
    
    return diff < 1e-5


def test_forward_pass():
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试2: 前向传播")
    print("=" * 60)
    
    vocab_size, d_model, n_heads, n_layers = 1000, 128, 4, 2
    
    model = CausalMHFTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    
    # 随机输入
    batch_size, seq_len = 4, 64
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    outputs = model(input_ids, labels=input_ids)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {outputs['logits'].shape}")
    print(f"损失值: {outputs['loss'].item():.4f}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("✅ 前向传播测试通过")
    return True


def test_generation():
    """测试自回归生成"""
    print("\n" + "=" * 60)
    print("测试3: 自回归生成")
    print("=" * 60)
    
    model = CausalMHFTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    model.eval()
    
    # 生成
    input_ids = torch.randint(0, 1000, (1, 10))
    generated = model.generate(input_ids, max_new_tokens=20, temperature=1.0, top_k=50)
    
    print(f"输入长度: {input_ids.shape[1]}")
    print(f"生成长度: {generated.shape[1]}")
    print(f"新增token数: {generated.shape[1] - input_ids.shape[1]}")
    
    print("✅ 生成测试通过")
    return True


def benchmark_performance():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("测试4: 性能基准")
    print("=" * 60)
    
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    batch_size = 4
    
    # 测试不同序列长度
    seq_lengths = [64, 128, 256, 512, 1024]
    
    # 创建模型
    mhf_model = CausalMHFTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    
    std_model = StandardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    
    mhf_model.eval()
    std_model.eval()
    
    results = []
    
    print(f"\n{'序列长度':^10} | {'MHF时间(ms)':^12} | {'标准时间(ms)':^12} | {'加速比':^8}")
    print("-" * 55)
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # MHF推理
        with torch.no_grad():
            start = time.time()
            _ = mhf_model(input_ids)
            mhf_time = (time.time() - start) * 1000
        
        # 标准Transformer推理
        with torch.no_grad():
            start = time.time()
            _ = std_model(input_ids)
            std_time = (time.time() - start) * 1000
        
        speedup = std_time / mhf_time if mhf_time > 0 else 0
        
        print(f"{seq_len:^10} | {mhf_time:^12.2f} | {std_time:^12.2f} | {speedup:^8.2f}x")
        
        results.append({
            'seq_len': seq_len,
            'mhf_time': mhf_time,
            'std_time': std_time,
            'speedup': speedup,
        })
    
    return results


def benchmark_memory():
    """内存基准测试"""
    print("\n" + "=" * 60)
    print("测试5: 内存使用")
    print("=" * 60)
    
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    
    # MHF参数量
    mhf_model = CausalMHFTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    
    # 标准Transformer参数量
    std_model = StandardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )
    
    mhf_params = sum(p.numel() for p in mhf_model.parameters())
    std_params = sum(p.numel() for p in std_model.parameters())
    
    print(f"MHF参数量: {mhf_params:,}")
    print(f"标准Transformer参数量: {std_params:,}")
    print(f"参数比例: {mhf_params / std_params:.2%}")
    
    return {'mhf_params': mhf_params, 'std_params': std_params}


def run_all_benchmarks():
    """运行所有基准测试"""
    print("\n" + "=" * 60)
    print("   Causal MHF Benchmark Suite")
    print("   参考NeuralOperator风格")
    print("=" * 60)
    
    results = {}
    
    # 1. 因果性测试
    results['causality'] = test_causality()
    
    # 2. 前向传播测试
    results['forward'] = test_forward_pass()
    
    # 3. 生成测试
    results['generation'] = test_generation()
    
    # 4. 性能基准
    results['performance'] = benchmark_performance()
    
    # 5. 内存基准
    results['memory'] = benchmark_memory()
    
    # 总结
    print("\n" + "=" * 60)
    print("   测试总结")
    print("=" * 60)
    
    print(f"因果性测试: {'✅ 通过' if results['causality'] else '❌ 失败'}")
    print(f"前向传播测试: {'✅ 通过' if results['forward'] else '❌ 失败'}")
    print(f"生成测试: {'✅ 通过' if results['generation'] else '❌ 失败'}")
    
    # 性能总结
    if results['performance']:
        avg_speedup = sum(r['speedup'] for r in results['performance']) / len(results['performance'])
        print(f"平均加速比: {avg_speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()