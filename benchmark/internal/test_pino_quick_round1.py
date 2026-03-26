#!/usr/bin/env python3
"""
PINO Round 1 快速测试 (10 epochs)

用于快速验证高频噪声惩罚的效果
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from mhf_fno import MHFFNO
from mhf_fno.pino_high_freq import HighFreqPINOLoss
from neuralop.models import FNO
from neuralop.losses.data_losses import LpLoss

print("="*70)
print("PINO Round 1 快速测试 (10 epochs)")
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
    train_x = data.get('x', data.get('train_x'))
    train_y = data.get('y', data.get('train_y'))
else:
    train_x, train_y = data[0], data[1]

if train_x.dim() == 3:
    train_x = train_x.unsqueeze(1).float()
    train_y = train_y.unsqueeze(1).float()

# 分割 (总共500个样本)
# 保存原始完整数据
full_x = train_x.clone()
full_y = train_y.clone()

# 分割
train_x = full_x[:400]
train_y = full_y[:400]
test_x = full_x[400:500]
test_y = full_y[400:500]

print(f"\n数据: train {train_x.shape}, test {test_x.shape}")

# 测试函数
def quick_test(model, train_x, train_y, test_x, test_y, name, loss_fn=None):
    print(f"\n测试 {name}...")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    if loss_fn is None:
        loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    test_loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
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
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        model.eval()
        with torch.no_grad():
            test_loss = test_loss_fn(model(test_x), test_y).item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']}: Train {train_loss/batch_count:.4f}, Test {test_loss:.4f}")
    
    return test_loss

# 测试1: FNO
print(f"\n{'='*60}")
print("1. FNO (baseline)")
print(f"{'='*60}")
torch.manual_seed(config['seed'])
model_fno = FNO(n_modes=(16,16), hidden_channels=32, in_channels=1, out_channels=1, n_layers=3)
fno_loss = quick_test(model_fno, train_x, train_y, test_x, test_y, "FNO")

# 测试2: MHF-FNO
print(f"\n{'='*60}")
print("2. MHF-FNO (无PINO)")
print(f"{'='*60}")
torch.manual_seed(config['seed'])
model_mhf = MHFFNO.best_config(n_modes=(16,16), hidden_channels=32, in_channels=1, out_channels=1)
mhf_loss = quick_test(model_mhf, train_x, train_y, test_x, test_y, "MHF-FNO")

# 测试3: MHF-PINO
print(f"\n{'='*60}")
print("3. MHF-PINO (高频噪声惩罚)")
print(f"{'='*60}")
torch.manual_seed(config['seed'])
model_pino = MHFFNO.best_config(n_modes=(16,16), hidden_channels=32, in_channels=1, out_channels=1)
pino_loss_fn = HighFreqPINOLoss(lambda_physics=0.0001, freq_threshold=0.5)
pino_loss = quick_test(model_pino, train_x, train_y, test_x, test_y, "MHF-PINO", loss_fn=pino_loss_fn)

# 结果
print(f"\n{'='*70}")
print("结果对比")
print(f"{'='*70}")
print(f"\n{'模型':<15} {'Test Loss':>12} {'vs MHF-FNO':>12}")
print(f"{'-'*40}")
print(f"{'FNO':<15} {fno_loss:>12.4f} {'baseline':>12}")
print(f"{'MHF-FNO':<15} {mhf_loss:>12.4f} {(mhf_loss-fno_loss)/fno_loss*100:>11.2f}%")
print(f"{'MHF-PINO':<15} {pino_loss:>12.4f} {(pino_loss-mhf_loss)/mhf_loss*100:>11.2f}%")

# 成功判断
success = pino_loss < mhf_loss
print(f"\n{'='*70}")
if success:
    print(f"✅ 成功! MHF-PINO ({pino_loss:.4f}) < MHF-FNO ({mhf_loss:.4f})")
    print(f"   改进: {(mhf_loss - pino_loss) / mhf_loss * 100:.2f}%")
else:
    print(f"❌ Round 1失败! MHF-PINO ({pino_loss:.4f}) >= MHF-FNO ({mhf_loss:.4f})")
    print(f"   恶化: {(pino_loss - mhf_loss) / mhf_loss * 100:.2f}%")
    print(f"\n建议: 尝试 Round 2 (自适应 lambda)")

# 保存结果
results = {
    'config': config,
    'timestamp': datetime.now().isoformat(),
    'results': {
        'FNO': fno_loss,
        'MHF-FNO': mhf_loss,
        'MHF-PINO': pino_loss,
    },
    'success': success,
    'improvement': (mhf_loss - pino_loss) / mhf_loss * 100 if success else None,
}

output_file = Path(__file__).parent.parent / 'pino_round1_quick_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 结果已保存: {output_file}")

sys.exit(0 if success else 1)
