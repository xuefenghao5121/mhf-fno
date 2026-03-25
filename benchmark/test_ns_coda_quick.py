#!/usr/bin/env python3
"""NS + CoDA Quick Test with Built-in Data"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import numpy as np
from datetime import datetime
from neuralop.losses.data_losses import LpLoss
from neuralop.models import FNO

from mhf_fno import create_hybrid_fno, MHFSpectralConvWithAttention, create_mhf_fno_with_attention

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train_epoch(model, train_x, train_y, optimizer, loss_fn, batch_size=32):
    model.train()
    perm = torch.randperm(len(train_x))
    total_loss = 0
    count = 0
    for i in range(0, len(train_x), batch_size):
        bx = train_x[perm[i:i+batch_size]]
        by = train_y[perm[i:i+batch_size]]
        optimizer.zero_grad()
        loss = loss_fn(model(bx), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / count

def evaluate(model, test_x, test_y, loss_fn):
    model.eval()
    with torch.no_grad():
        return loss_fn(model(test_x), test_y).item()

def generate_ns_data(n_samples=1000, resolution=32, viscosity=1e-3, dt=1e-3, T=1.0):
    """Generate simple Navier-Stokes data using spectral method"""
    print(f'  Generating NS data: {n_samples} samples, {resolution}x{resolution}', flush=True)
    
    import torch.fft as fft
    
    # Grid
    L = 1.0
    x = torch.linspace(0, L - L/resolution, resolution)
    y = torch.linspace(0, L - L/resolution, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Wavenumbers
    kx = torch.fft.fftfreq(resolution, L/resolution) * 2 * np.pi
    ky = torch.fft.fftfreq(resolution, L/resolution) * 2 * np.pi
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    
    # Prevent division by zero
    K2[0, 0] = 1.0
    
    # Generate initial conditions
    all_x = []
    all_y = []
    
    n_steps = int(T / dt)
    
    for i in range(n_samples):
        # Random initial vorticity field
        torch.manual_seed(i + 42)
        w0 = torch.randn(resolution, resolution) * 0.1
        
        # Add some structure
        for _ in range(3):
            cx, cy = torch.rand(2) * L
            sigma = 0.1 + torch.rand(1) * 0.1
            w0 += torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)) * (torch.rand(1) - 0.5) * 0.5
        
        # Evolve using spectral method (simplified)
        w_hat = fft.fft2(w0)
        
        for _ in range(n_steps):
            # Get velocity from vorticity
            psi_hat = -w_hat / K2
            psi_hat[0, 0] = 0
            
            # Velocity in Fourier space
            u_hat = 1j * KY * psi_hat
            v_hat = -1j * KX * psi_hat
            
            # Nonlinear term
            u = fft.ifft2(u_hat).real
            v = fft.ifft2(v_hat).real
            w = fft.ifft2(w_hat).real
            
            # Advection
            dw_dx = fft.ifft2(1j * KX * w_hat).real
            dw_dy = fft.ifft2(1j * KY * w_hat).real
            
            # Diffusion
            lap_w = fft.ifft2(-K2 * w_hat).real
            
            # RHS
            rhs = -(u * dw_dx + v * dw_dy) + viscosity * lap_w
            
            # Time stepping (forward Euler)
            w_hat = w_hat + dt * fft.fft2(rhs)
            
            # Anti-aliasing
            w_hat[K2 > (resolution/3)**2 * 4] = 0
        
        # Final state
        w_final = fft.ifft2(w_hat).real
        
        all_x.append(w0.unsqueeze(0))
        all_y.append(w_final.unsqueeze(0))
        
        if (i + 1) % 200 == 0:
            print(f'    Generated {i+1}/{n_samples}', flush=True)
    
    x = torch.stack(all_x)
    y = torch.stack(all_y)
    
    return x, y

print('='*60, flush=True)
print('NS + CoDA Detailed Test', flush=True)
print('='*60, flush=True)

# Generate or load data
data_dir = Path(__file__).parent / 'data'
data_file = data_dir / 'ns_train_32_large.pt'

if not data_file.exists():
    print('\nGenerating NS data...', flush=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_x, train_y = generate_ns_data(n_samples=1000, resolution=32)
    test_x, test_y = generate_ns_data(n_samples=200, resolution=32)
    
    # Save for future use
    torch.save({'x': train_x.squeeze(1), 'y': train_y.squeeze(1)}, data_dir / 'ns_train_32_large.pt')
    torch.save({'x': test_x.squeeze(1), 'y': test_y.squeeze(1)}, data_dir / 'ns_test_32_large.pt')
    print('  Data saved', flush=True)
else:
    print('\nLoading data...', flush=True)
    train_data = torch.load(data_dir / 'ns_train_32_large.pt', weights_only=False)
    test_data = torch.load(data_dir / 'ns_test_32_large.pt', weights_only=False)
    
    train_x = train_data['x'].unsqueeze(1) if train_data['x'].dim() == 3 else train_data['x']
    train_y = train_data['y'].unsqueeze(1) if train_data['y'].dim() == 3 else train_data['y']
    test_x = test_data['x'].unsqueeze(1) if test_data['x'].dim() == 3 else test_data['x']
    test_y = test_data['y'].unsqueeze(1) if test_data['y'].dim() == 3 else test_data['y']

# Ensure float
train_x, train_y = train_x.float(), train_y.float()
test_x, test_y = test_x.float(), test_y.float()

print(f'  Train: {train_x.shape}', flush=True)
print(f'  Test: {test_x.shape}', flush=True)

# Config
resolution = train_x.shape[-1]
n_modes = (resolution // 2, resolution // 2)
hidden_channels = 32
n_layers = 3
n_heads = 4
epochs = 50
batch_size = 32
lr = 1e-3

print(f'  n_modes={n_modes}, hidden={hidden_channels}, epochs={epochs}', flush=True)

results = {
    'config': {
        'n_modes': n_modes,
        'hidden_channels': hidden_channels,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'n_train': len(train_x),
        'n_test': len(test_x),
    },
    'timestamp': datetime.now().isoformat(),
}

# -------------------------------------------------------------------------
# Test 1: FNO baseline
# -------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print('Test 1: FNO (Baseline)', flush=True)
print('='*60, flush=True)

torch.manual_seed(42)
model_fno = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3)
params_fno = count_params(model_fno)
print(f'  Parameters: {params_fno:,}', flush=True)

optimizer = torch.optim.Adam(model_fno.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss_fn = LpLoss(d=2, p=2, reduction='mean')

fno_train_losses = []
fno_test_losses = []
best_fno_loss = float('inf')
best_fno_epoch = 0

print('  Training...', flush=True)
for epoch in range(epochs):
    train_loss = train_epoch(model_fno, train_x, train_y, optimizer, loss_fn, batch_size)
    test_loss = evaluate(model_fno, test_x, test_y, loss_fn)
    scheduler.step()
    
    fno_train_losses.append(train_loss)
    fno_test_losses.append(test_loss)
    
    if test_loss < best_fno_loss:
        best_fno_loss = test_loss
        best_fno_epoch = epoch + 1
    
    if (epoch + 1) % 10 == 0:
        print(f'    Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}', flush=True)

print(f'  Best Loss: {best_fno_loss:.4f} @ epoch {best_fno_epoch}', flush=True)

results['FNO'] = {
    'params': params_fno,
    'train_losses': fno_train_losses,
    'test_losses': fno_test_losses,
    'best_test_loss': best_fno_loss,
    'best_epoch': best_fno_epoch,
}

# -------------------------------------------------------------------------
# Test 2: MHF-FNO (no CoDA)
# -------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print('Test 2: MHF-FNO (no CoDA)', flush=True)
print('='*60, flush=True)

torch.manual_seed(42)
model_mhf = create_hybrid_fno(
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    in_channels=1,
    out_channels=1,
    n_layers=n_layers,
    n_heads=n_heads,
    mhf_layers=[0, 2]
)
params_mhf = count_params(model_mhf)
print(f'  Parameters: {params_mhf:,} ({(1-params_mhf/params_fno)*100:.1f}% reduction)', flush=True)

optimizer = torch.optim.Adam(model_mhf.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

mhf_train_losses = []
mhf_test_losses = []
best_mhf_loss = float('inf')
best_mhf_epoch = 0

print('  Training...', flush=True)
for epoch in range(epochs):
    train_loss = train_epoch(model_mhf, train_x, train_y, optimizer, loss_fn, batch_size)
    test_loss = evaluate(model_mhf, test_x, test_y, loss_fn)
    scheduler.step()
    
    mhf_train_losses.append(train_loss)
    mhf_test_losses.append(test_loss)
    
    if test_loss < best_mhf_loss:
        best_mhf_loss = test_loss
        best_mhf_epoch = epoch + 1
    
    if (epoch + 1) % 10 == 0:
        print(f'    Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}', flush=True)

print(f'  Best Loss: {best_mhf_loss:.4f} @ epoch {best_mhf_epoch}', flush=True)

results['MHF-FNO'] = {
    'params': params_mhf,
    'train_losses': mhf_train_losses,
    'test_losses': mhf_test_losses,
    'best_test_loss': best_mhf_loss,
    'best_epoch': best_mhf_epoch,
}

# -------------------------------------------------------------------------
# Test 3: MHF+CoDA (best config)
# -------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print('Test 3: MHF+CoDA (best config)', flush=True)
print('='*60, flush=True)
print('  Config: mhf_layers=[0, 2], bottleneck=4, gate_init=0.1', flush=True)

torch.manual_seed(42)
model_coda = create_mhf_fno_with_attention(
    n_modes=n_modes,
    hidden_channels=hidden_channels,
    in_channels=1,
    out_channels=1,
    n_layers=n_layers,
    n_heads=n_heads,
    mhf_layers=[0, 2]
)
params_coda = count_params(model_coda)

# Count attention params
attn_params = 0
for name, module in model_coda.named_modules():
    if 'cross_head_attn' in name:
        attn_params += sum(p.numel() for p in module.parameters())

print(f'  Parameters: {params_coda:,} ({(1-params_coda/params_fno)*100:.1f}% reduction)', flush=True)
print(f'  Attention params: {attn_params:,} ({attn_params/params_coda*100:.2f}%)', flush=True)

optimizer = torch.optim.Adam(model_coda.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

coda_train_losses = []
coda_test_losses = []
best_coda_loss = float('inf')
best_coda_epoch = 0

print('  Training...', flush=True)
for epoch in range(epochs):
    train_loss = train_epoch(model_coda, train_x, train_y, optimizer, loss_fn, batch_size)
    test_loss = evaluate(model_coda, test_x, test_y, loss_fn)
    scheduler.step()
    
    coda_train_losses.append(train_loss)
    coda_test_losses.append(test_loss)
    
    if test_loss < best_coda_loss:
        best_coda_loss = test_loss
        best_coda_epoch = epoch + 1
    
    if (epoch + 1) % 10 == 0:
        print(f'    Epoch {epoch+1}: Train={train_loss:.4f}, Test={test_loss:.4f}', flush=True)

print(f'  Best Loss: {best_coda_loss:.4f} @ epoch {best_coda_epoch}', flush=True)

results['MHF+CoDA'] = {
    'params': params_coda,
    'attention_params': attn_params,
    'train_losses': coda_train_losses,
    'test_losses': coda_test_losses,
    'best_test_loss': best_coda_loss,
    'best_epoch': best_coda_epoch,
}

# -------------------------------------------------------------------------
# Test 4: Bottleneck sensitivity
# -------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print('Test 4: Bottleneck Sensitivity Analysis', flush=True)
print('='*60, flush=True)

bottleneck_values = [2, 4, 6, 8]
bn_results = {}

for bn in bottleneck_values:
    print(f'\n  Bottleneck = {bn}', flush=True)
    torch.manual_seed(42)
    
    # Create model with custom bottleneck
    model = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3)
    
    # Replace MHF layers
    for layer_idx in [0, 2]:
        mhf_conv = MHFSpectralConvWithAttention(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_heads=n_heads,
            use_attention=True,
            attn_reduction=bn  # bottleneck
        )
        model.fno_blocks.convs[layer_idx] = mhf_conv
    
    params = count_params(model)
    
    # Count attention params
    attn_p = 0
    for name, module in model.named_modules():
        if 'cross_head_attn' in name:
            attn_p += sum(p.numel() for p in module.parameters())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    best_ep = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_x, train_y, optimizer, loss_fn, batch_size)
        test_loss = evaluate(model, test_x, test_y, loss_fn)
        scheduler.step()
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_ep = epoch + 1
    
    bn_results[bn] = {
        'params': params,
        'attention_params': attn_p,
        'best_test_loss': best_loss,
        'best_epoch': best_ep,
    }
    
    print(f'    Params: {params:,}, Attention: {attn_p:,}, Best Loss: {best_loss:.4f} @ epoch {best_ep}', flush=True)

results['bottleneck_sensitivity'] = bn_results

# -------------------------------------------------------------------------
# Test 5: Gate_init sensitivity
# -------------------------------------------------------------------------
print(f"\n{'='*60}", flush=True)
print('Test 5: Gate_init Sensitivity Analysis', flush=True)
print('='*60, flush=True)

gate_init_values = [0.05, 0.1, 0.2, 0.5]
gi_results = {}

for gi in gate_init_values:
    print(f'\n  Gate_init = {gi}', flush=True)
    torch.manual_seed(42)
    
    # Create model with custom gate_init
    model = FNO(n_modes=n_modes, hidden_channels=hidden_channels, in_channels=1, out_channels=1, n_layers=3)
    
    # Replace MHF layers
    for layer_idx in [0, 2]:
        mhf_conv = MHFSpectralConvWithAttention(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=n_modes,
            n_heads=n_heads,
            use_attention=True,
            attn_reduction=4
        )
        # Set gate init
        with torch.no_grad():
            mhf_conv.cross_head_attn.gate.fill_(gi)
        
        model.fno_blocks.convs[layer_idx] = mhf_conv
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    best_ep = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_x, train_y, optimizer, loss_fn, batch_size)
        test_loss = evaluate(model, test_x, test_y, loss_fn)
        scheduler.step()
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_ep = epoch + 1
    
    gi_results[str(gi)] = {
        'best_test_loss': best_loss,
        'best_epoch': best_ep,
    }
    
    print(f'    Best Loss: {best_loss:.4f} @ epoch {best_ep}', flush=True)

results['gate_init_sensitivity'] = gi_results

# -------------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------------
output_path = 'ns_coda_detailed_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}", flush=True)

# -------------------------------------------------------------------------
# Final summary
# -------------------------------------------------------------------------
print(f"\n{'='*70}", flush=True)
print('Final Summary', flush=True)
print(f"{'='*70}", flush=True)

print(f"\nModel Performance Comparison:", flush=True)
print(f"{'Model':<15} {'Params':<12} {'Param Red':<12} {'Best Loss':<12} {'vs FNO':<10}", flush=True)
print(f"{'-'*60}", flush=True)

for name in ['FNO', 'MHF-FNO', 'MHF+CoDA']:
    r = results[name]
    param_red = (1 - r['params'] / params_fno) * 100
    improvement = (best_fno_loss - r['best_test_loss']) / best_fno_loss * 100
    print(f"{name:<15} {r['params']:<12,} {param_red:<12.1f}% {r['best_test_loss']:<12.4f} {improvement:+.2f}%", flush=True)

print(f"\nBottleneck Sensitivity:", flush=True)
print(f"{'Bottleneck':<12} {'Attn Params':<15} {'Best Loss':<12} {'Rel Change':<10}", flush=True)
print(f"{'-'*50}", flush=True)
bn_base = bn_results[4]['best_test_loss']
for bn, r in bn_results.items():
    rel_change = (r['best_test_loss'] - bn_base) / bn_base * 100
    print(f"{bn:<12} {r['attention_params']:<15,} {r['best_test_loss']:<12.4f} {rel_change:+.2f}%", flush=True)

print(f"\nGate_init Sensitivity:", flush=True)
print(f"{'Gate_init':<12} {'Best Loss':<12} {'Rel Change':<10}", flush=True)
print(f"{'-'*35}", flush=True)
gi_base = gi_results['0.1']['best_test_loss']
for gi, r in gi_results.items():
    rel_change = (r['best_test_loss'] - gi_base) / gi_base * 100
    print(f"{gi:<12} {r['best_test_loss']:<12.4f} {rel_change:+.2f}%", flush=True)