# MHF-FNO

Multi-Head Fourier Neural Operator based on TransFourier.

## Quick Start

```python
from mhf_1d import MHFFNO1D

model = MHFFNO1D(
    in_channels=2,
    out_channels=1,
    hidden_channels=64,
    n_modes=16,
    n_layers=4,
    n_heads=4,
)
```

## Results

| Model | Parameters | L2 Error | Time |
|-------|------------|----------|------|
| FNO | 49,345 | 0.4800 | 25.10s |
| MHF-FNO | 12,481 | 0.4279 | 21.54s |

**Improvement: -74.7% params, -10.86% error, -14.17% time**