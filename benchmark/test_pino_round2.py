#!/usr/bin/env python3
"""
PINO Round 2 测试: 自适应 lambda

策略: 从 0.0001 开始，每 10 epoch 增加 1.5 倍
目标: 自动找到最佳物理约束强度
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNO
from mhf_fno.pino_high_freq import AdaptiveHighFreqPINOLoss
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss

print("="*70)
print("PINO Round 2 测试: 自适应 lambda (10 epochs)")
print("="*70)

# 配置
config = {
    'n_train': 400,
    'n_test': 100,
    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'seed': 42,
}

torch.manual_seed(config['seed'])
np.random.seed(config['seed'])

# 加载数据
data_path = Path(__file__).parent.parent / 'data' / 'ns_train_32.pt'
data = torch.load(data_path, weights_only=False)

if isinstance(data, dict):
    full_x = data.get('x', data.get('train_x'))
    full_y = data.get('y', data.get('train_y'))
else:
    full_x, full_y = data[0], data[1]

if full_x.dim() == 3:
    full_x = full_x.unsqueeze(1).float()
    full_y = full_y.unsqueeze(1).float()

train_x = full_x[:400]
train_y = full_y[:400]
test_x = full_x[400:500]
test_y = full_y[400:500]

print(f"\n数据: train {train_x.shape}, test {test_x.shape}")

# 测试函数
def train_adaptive(model, train_x, train_y, test_x, test_y, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 自适应 PINO loss
    pino_loss_fn = AdaptiveHighFreqPINOLoss(
        initial_lambda=0.0001,
        growth_factor=1.5,
        growth_interval=10,  # 每10 epoch增加
        max_lambda=0.01,
        freq_threshold=0.5
    )
    
    test_loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    print(f"  初始 lambda: {pino_loss_fn.lambda_phy:.6f}")
    
    for epoch in range(config['epochs']):
        model.train()
        perm = torch.randperm(len(train_x))
        train_loss = 0
        batch_count = 0
        
        for i in range(0, len(train_x), config['batch_size']):
            bx = train_x[perm[i:i+config['batch_size']]]
            by = train_y[perm[i:i+config['batch_size']]]
            
            optimizer.zero_grad()
            pred = model(bx)
            loss = pino_loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        # 更新 lambda
        pino_loss_fn.step_epoch()
        
        model.eval()
        with torch.no_grad():
            test_loss = test_loss_fn(model(test_x), test_y).item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: "
                  f"Train {train_loss/batch_count:.4f}, "
                  f"Test {test_loss:.4f}, "
                  f"λ={pino_loss_fn.lambda_phy:.6f}")
    
    return test_loss

# 测试MHF-PINO (自适应)
print(f"\n{'='*60}")
print("MHF-PINO (自适应 lambda)")
print(f"{'='*60}")

torch.manual_seed(config['seed'])
model_pino = MHFFNO.best_config(n_modes=(16,16), hidden_channels=32, in_channels=1, out_channels=1)
pino_loss = train_adaptive(model_pino, train_x, train_y, test_x, test_y, config)

# 对比 Round 1 结果
print(f"\n{'='*70}")
print("结果对比")
print(f"{'='*70}")

# 从 Round 1 快速测试加载结果
round1_file = Path(__file__).parent.parent / 'pino_round1_quick_results.json'
if round1_file.exists():
    with open(round1_file) as f:
        round1_results = json.load(f)
    mhf_baseline = round1_results['results']['MHF-FNO']
    round1_pino = round1_results['results']['MHF-PINO']
    
    print(f"\n{'模型':<20} {'Test Loss':>12} {'vs MHF-FNO':>12}")
    print(f"{'-'*45}")
    print(f"{'MHF-FNO (baseline)':<20} {mhf_baseline:>12.4f} {'baseline':>12}")
    print(f"{'MHF-PINO (Round 1)':<20} {round1_pino:>12.4f} {(round1_pino-mhf_baseline)/mhf_baseline*100:>11.2f}%")
    print(f"{'MHF-PINO (Round 2)':<20} {pino_loss:>12.4f} {(pino_loss-mhf_baseline)/mhf_baseline*100:>11.2f}%")
    
    # 成功判断
    success = pino_loss < mhf_baseline
    improvement = (mhf_baseline - pino_loss) / mhf_baseline * 100
    
    print(f"\n{'='*70}")
    if success:
        print(f"✅ Round 2成功! MHF-PINO ({pino_loss:.4f}) < MHF-FNO ({mhf_baseline:.4f})")
        print(f"   改进: {improvement:.2f}%")
    else:
        print(f"❌ Round 2失败! MHF-PINO ({pino_loss:.4f}) >= MHF-FNO ({mhf_baseline:.4f})")
        print(f"   恶化: {-improvement:.2f}%")
        print(f"\n建议: 尝试 Round 3 (梯度异常惩罚)")
    
    # 保存结果
    results = {
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'results': {
            'MHF-FNO': mhf_baseline,
            'MHF-PINO-Round1': round1_pino,
            'MHF-PINO-Round2': pino_loss,
        },
        'success': success,
        'improvement': improvement if success else None,
    }
    
    output_file = Path(__file__).parent.parent / 'pino_round2_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    sys.exit(0 if success else 1)
else:
    print(f"\n⚠️ 未找到 Round 1 结果文件")
    sys.exit(1)
