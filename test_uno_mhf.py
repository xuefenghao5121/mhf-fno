"""
UNO + MHF 验证 (单独测试)
"""

import torch
import torch.nn as nn
import time
import gc

from neuralop.models import UNO
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.losses.data_losses import LpLoss


class MHFSpectralConv(SpectralConv):
    """MHF SpectralConv"""
    
    def __init__(self, in_channels, out_channels, n_modes, n_heads=4, bias=True, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, n_modes=n_modes, bias=bias, **kwargs)
        
        del self.weight
        self.n_heads = n_heads
        self.use_mhf = (in_channels % n_heads == 0 and out_channels % n_heads == 0)
        
        if self.use_mhf:
            self.head_in = in_channels // n_heads
            self.head_out = out_channels // n_heads
            
            if len(n_modes) == 1:
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0])
            else:
                modes_y = n_modes[1] // 2 + 1
                weight_shape = (n_heads, self.head_in, self.head_out, n_modes[0], modes_y)
            
            init_std = (2 / (in_channels + out_channels)) ** 0.5
            self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.cfloat) * init_std)
        else:
            self.weight = nn.Parameter(torch.randn(in_channels, out_channels, *n_modes, dtype=torch.cfloat) * 0.01)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels, *((1,) * len(n_modes))))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x, output_shape=None, *args, **kwargs):
        if not self.use_mhf:
            return self._forward_standard(x)
        if x.dim() == 3:
            return self._forward_1d(x)
        return self._forward_2d(x)
    
    def _forward_1d(self, x):
        B, C, L = x.shape
        x_freq = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.weight.shape[-1], x_freq.shape[-1])
        x_freq = x_freq.view(B, self.n_heads, self.head_in, -1)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, x_freq.shape[-1], dtype=x_freq.dtype, device=x.device)
        out_freq[..., :n_modes] = torch.einsum('bhif,hiof->bhof', x_freq[..., :n_modes], self.weight[..., :n_modes])
        out_freq = out_freq.reshape(B, self.out_channels, -1)
        x_out = torch.fft.irfft(out_freq, n=L, dim=-1)
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out
    
    def _forward_2d(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.weight.shape[-3] if self.weight.dim() > 3 else self.weight.shape[-2], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        x_freq = x_freq.view(B, self.n_heads, self.head_in, freq_H, freq_W)
        out_freq = torch.zeros(B, self.n_heads, self.head_out, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :, :m_x, :m_y] = torch.einsum('bhiXY,hioXY->bhoXY', x_freq[:, :, :, :m_x, :m_y], self.weight[:, :, :, :m_x, :m_y])
        out_freq = out_freq.reshape(B, self.out_channels, freq_H, freq_W)
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out
    
    def _forward_standard(self, x):
        B, C, H, W = x.shape
        x_freq = torch.fft.rfft2(x, dim=(-2, -1))
        freq_H, freq_W = x_freq.shape[-2], x_freq.shape[-1]
        m_x = min(self.weight.shape[-2], freq_H)
        m_y = min(self.weight.shape[-1], freq_W)
        out_freq = torch.zeros(B, self.out_channels, freq_H, freq_W, dtype=x_freq.dtype, device=x.device)
        out_freq[:, :, :m_x, :m_y] = torch.einsum('biXY,ioXY->boXY', x_freq[:, :, :m_x, :m_y], self.weight[:, :, :m_x, :m_y])
        x_out = torch.fft.irfft2(out_freq, s=(H, W), dim=(-2, -1))
        if self.bias is not None:
            x_out = x_out + self.bias
        return x_out


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(model, train_x, train_y, test_x, test_y, epochs=100, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = LpLoss(d=2, p=2, reduction='mean')
    
    n_train = train_x.shape[0]
    start = time.time()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        
        for i in range(0, n_train, batch_size):
            bx = train_x[perm[i:i+batch_size]]
            by = train_y[perm[i:i+batch_size]]
            
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(test_x), test_y).item()
        if test_loss < best_loss:
            best_loss = test_loss
    
    return best_loss, time.time() - start


def main():
    print("\n" + "=" * 70)
    print(" 算子 3: UNO + MHF 验证")
    print("=" * 70)
    
    # 加载数据
    data_path = '/usr/local/lib/python3.11/site-packages/neuralop/data/datasets/data/'
    train_data = torch.load(f'{data_path}/darcy_train_16.pt', weights_only=False)
    test_data = torch.load(f'{data_path}/darcy_test_16.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1).float()
    train_y = train_data['y'].unsqueeze(1).float()
    test_x = test_data['x'].unsqueeze(1).float()
    test_y = test_data['y'].unsqueeze(1).float()
    
    print(f"数据: {train_x.shape}")
    results = []
    
    # 检查 UNO 结构
    print("\n[检查] UNO 内部结构...")
    try:
        torch.manual_seed(42)
        model_uno = UNO(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=5,
            uno_out_channels=[32, 64, 64, 64, 32],
            uno_n_modes=[[8,8], [8,8], [8,8], [8,8], [8,8]],
            uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [1, 1], [2, 2]]
        )
        
        # 检查结构
        if hasattr(model_uno, 'fno_blocks'):
            print("  ✅ UNO 有 fno_blocks 属性")
            if hasattr(model_uno.fno_blocks, 'convs'):
                n_convs = len(model_uno.fno_blocks.convs)
                print(f"  ✅ 有 {n_convs} 个 conv 层")
                for i, conv in enumerate(model_uno.fno_blocks.convs):
                    if hasattr(conv, 'in_channels'):
                        print(f"     层{i}: in={conv.in_channels}, out={conv.out_channels}")
        
        # 基准 UNO
        print("\n[1/2] 标准 UNO...")
        params_uno = count_params(model_uno)
        loss_uno, time_uno = train_and_evaluate(model_uno, train_x, train_y, test_x, test_y, epochs=100)
        print(f"  参数: {params_uno:,}, Loss: {loss_uno:.4f}, 时间: {time_uno:.1f}s")
        results.append({"name": "UNO-基准", "params": params_uno, "loss": loss_uno, "time": time_uno})
        
        del model_uno
        gc.collect()
        
        # UNO + MHF (边缘层)
        print("\n[2/2] UNO + MHF (边缘层)...")
        torch.manual_seed(42)
        model_uno_mhf = UNO(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            n_layers=5,
            uno_out_channels=[32, 64, 64, 64, 32],
            uno_n_modes=[[8,8], [8,8], [8,8], [8,8], [8,8]],
            uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [1, 1], [2, 2]]
        )
        
        n_layers = len(model_uno_mhf.fno_blocks.convs)
        replaced = []
        
        try:
            for idx in [0, n_layers-1]:
                conv = model_uno_mhf.fno_blocks.convs[idx]
                if hasattr(conv, 'in_channels') and conv.in_channels % 4 == 0:
                    model_uno_mhf.fno_blocks.convs[idx] = MHFSpectralConv(
                        conv.in_channels, conv.out_channels, (8, 8), n_heads=4
                    )
                    replaced.append(idx)
                    print(f"  ✅ 替换层 {idx}: in={conv.in_channels}, out={conv.out_channels}")
            
            params_mhf = count_params(model_uno_mhf)
            loss_mhf, time_mhf = train_and_evaluate(model_uno_mhf, train_x, train_y, test_x, test_y, epochs=100)
            print(f"  参数: {params_mhf:,}, Loss: {loss_mhf:.4f}, 时间: {time_mhf:.1f}s")
            results.append({"name": "UNO+MHF-边缘", "params": params_mhf, "loss": loss_mhf, "time": time_mhf, "replaced": replaced})
        except Exception as e:
            print(f"  ❌ MHF 替换失败: {e}")
            results.append({"name": "UNO+MHF-边缘", "error": str(e)})
        
    except Exception as e:
        print(f"  ❌ UNO 初始化失败: {e}")
        results.append({"name": "UNO", "error": str(e)})
    
    # 汇总
    if results and 'error' not in results[0]:
        print("\n" + "-" * 70)
        print("UNO 验证结果:")
        print(f"{'配置':<20} {'参数量':<12} {'L2误差':<10} {'改进':<10}")
        print("-" * 55)
        baseline = results[0]['loss']
        for r in results:
            if 'error' not in r:
                improve = (r['loss'] - baseline) / baseline * 100
                print(f"{r['name']:<20} {r['params']:<12,} {r['loss']:<10.4f} {improve:+.1f}%")
            else:
                print(f"{r['name']:<20} ERROR: {r['error']}")
    
    return results


if __name__ == "__main__":
    main()