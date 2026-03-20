# MHF-FNO: Multi-Head Fourier Neural Operator

> TransFourier-style Multi-Head Fourier module for NeuralOperator 2.0.0

## Overview

This project implements a **Multi-Head Fourier (MHF)** module based on the TransFourier paper, designed as a drop-in replacement for standard SpectralConv in NeuralOperator's FNO.

### Key Results on Darcy Flow (NeuralOperator Official Benchmark)

#### 2D Darcy Flow (32x32 resolution)

| Metric | FNO (Official) | MHF-FNO | Improvement |
|--------|----------------|---------|-------------|
| Parameters | 133,873 | 24,769 | **-81.5%** |
| L2 Error | 0.3090 | 0.3077 | **-0.41%** |
| Training Time | 154.75s | 39.46s | **-74.5%** |

#### 1D Darcy Flow (64 resolution)

| Metric | Standard FNO | MHF-FNO | Improvement |
|--------|-------------|---------|-------------|
| Parameters | 49,345 | 12,481 | **-74.7%** |
| L2 Error | 0.4800 | 0.4279 | **-10.86%** |
| Training Time | 25.10s | 21.54s | **-14.17%** |

## Features

- ✅ **Multi-Head Architecture**: Each head learns independent frequency domain weights
- ✅ **Parameter Efficient**: 75% fewer parameters than standard FNO
- ✅ **Better Generalization**: Lower overfitting risk
- ✅ **CPU Friendly**: Pure PyTorch implementation, no CUDA dependencies
- ✅ **NeuralOperator Compatible**: Drop-in replacement for SpectralConv

## Installation

```bash
pip install torch numpy
pip install neuraloperator  # optional, for comparison
```

## Quick Start

```python
from mhf_1d import MHFFNO1D

# Create MHF-FNO model
model = MHFFNO1D(
    in_channels=2,
    out_channels=1,
    hidden_channels=64,
    n_modes=16,
    n_layers=4,
    n_heads=4,
)

# Forward pass
# x: (batch, in_channels, seq_len)
output = model(x)  # (batch, out_channels, seq_len)
```

## Theory

### Multi-Head Fourier (MHF)

Standard FNO uses a single large weight matrix for frequency domain mixing:

```
Standard SpectralConv:
  Weight: (in_channels, out_channels, n_modes)
  Output = IFFT(FFT(x) × Weight)
```

MHF splits channels into multiple heads, each with independent weights:

```
MHF SpectralConv:
  Weight: (n_heads, head_in, head_out, n_modes)
  Each head learns different frequency patterns
  Output = IFFT(Concat(FFT(x_head_i) × Weight_i))
```

### Advantages

1. **Parameter Sharing**: Heads share structure, reducing parameters
2. **Diverse Frequency Learning**: Each head captures different frequency patterns
3. **Better Regularization**: Implicit regularization through head structure
4. **Parallelizable**: Heads can be processed in parallel

## Experiments

### Darcy Flow 1D

```bash
cd experiments
python darcy_flow_test.py
```

Results:
- Training: 800 samples
- Validation: 100 samples  
- Test: 100 samples
- Resolution: 64

### Performance Comparison

```
Model          Parameters    L2 Error    Training Time
-------------------------------------------------------
FNO            49,345       0.4800       25.10s
MHF-FNO        12,481       0.4279       21.54s
```

## References

1. **TransFourier**: FFT Is All You Need
   - https://openreview.net/forum?id=TSHMAEItPc

2. **Fourier Neural Operators Explained**
   - https://arxiv.org/abs/2512.01421

3. **NeuralOperator Library**
   - https://github.com/neuraloperator/neuraloperator

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code, please cite:

```bibtex
@misc{mhf-fno2026,
  title={MHF-FNO: Multi-Head Fourier Neural Operator},
  author={Tianyuan Team},
  year={2026},
  url={https://github.com/yourusername/mhf-fno}
}
```

## Acknowledgments

- NeuralOperator team for the excellent library
- TransFourier authors for the MHF concept
- DeepSeek for Engram architecture inspiration