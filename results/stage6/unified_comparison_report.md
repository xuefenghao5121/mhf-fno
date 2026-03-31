# Stage 6: Unified Comparison Report

**Generated**: 2026-03-31 00:27:40

## Unified Experimental Conditions
- **Seed**: 42
- **Data**: NS 128x128, train=50, test=20
- **Training**: 5 epochs, batch_size=32, lr=0.001, Adam
- **Model**: MHF-FNO, n_modes=(16, 16), hidden=64, n_heads=2, mhf_layers=[0]
- **Device**: cpu
- **PINO λ**: 0.01

## Unified Comparison Table

| Rank | Configuration | Test MSE | Lp Error | Δ MSE vs Baseline | Latency (ms) | Total Params |
|------|--------------|----------|----------|-------------------|--------------|-------------|
| 1 | Baseline (no PINO) | 0.494642 | 19.673018 | +0.00% | 34.5 | 1,517,025 |
| 2 | FDPA (physics attention) | 0.494642 | 19.673018 | +0.00% | 34.7 | 1,517,025 |
| 3 | Uniform PINO (λ=0.01) | 0.494642 | 19.673018 | +0.00% | 35.2 | 1,517,025 |
| 4 | PSPT (progressive) | 0.494642 | 19.673018 | +0.00% | 35.5 | 1,517,025 |
| 5 | FA-PINO + PSPT | 0.494642 | 19.673018 | +0.00% | 34.3 | 1,517,025 |
| 6 | AFP-PINO (adaptive) | 0.495205 | 19.685766 | +0.11% | 34.0 | 1,517,027 |
| 7 | FA-PINO (frequency-aware) | 0.510142 | 19.985203 | +3.13% ⚠️ | 34.7 | 1,517,025 |
| 8 | SP-JR (spectral-physics joint) | 0.510928 | 20.000828 | +3.29% ⚠️ | 34.3 | 1,517,025 |

**Baseline**: MSE=0.494642, Lp=19.673018

## 🏆 Best Configuration: 1_Baseline
- Baseline (no PINO)
- MSE=0.494642 (vs baseline 0.494642)
- Improvement: 0.00%

## Analysis

### ➖ Similar to Baseline (6 configs)
- **Baseline (no PINO)**: comparable
- **FDPA (physics attention)**: comparable
- **Uniform PINO (λ=0.01)**: comparable
- **PSPT (progressive)**: comparable
- **FA-PINO + PSPT**: comparable
- **AFP-PINO (adaptive)**: comparable

### ⚠️ Degraded vs Baseline (2 configs)
- **FA-PINO (frequency-aware)**: 3.13% degradation
- **SP-JR (spectral-physics joint)**: 3.29% degradation

## Training Convergence
- Baseline (no PINO): 0.4621 → 0.4583 → 0.4538 → 0.4502 → 0.4481
- Uniform PINO (λ=0.01): 0.4621 → 0.4583 → 0.4538 → 0.4502 → 0.4481
- FA-PINO (frequency-aware): 0.6814 → 0.6379 → 0.4755 → 0.5306 → 0.5168
- PSPT (progressive): 0.4621 → 0.4583 → 0.4538 → 0.4502 → 0.4481
- FA-PINO + PSPT: 0.4621 → 0.4583 → 0.4538 → 0.4502 → 0.4481
- AFP-PINO (adaptive): 0.4623 → 0.4585 → 0.4542 → 0.4507 → 0.4488
- SP-JR (spectral-physics joint): 13.0923 → 42.9627 → 0.5589 → 2.0239 → 4.7109
- FDPA (physics attention): 0.4621 → 0.4583 → 0.4538 → 0.4502 → 0.4481

## Method-Specific Notes

### 1. Baseline (no PINO)
- Pure data-driven MHF-FNO, no physics constraints
- Serves as the reference point for all comparisons

### 2. Uniform PINO
- Constant λ=0.01 laplacian smoothness penalty
- Simplest physics-informed approach
- Pro: Easy to tune, stable training
- Con: Same weight for all frequencies, may over-constrain high-freq

### 3. FA-PINO (Frequency-Aware)
- Different λ per frequency band (low-freq: 2λ, high-freq: 0.5λ)
- Pro: Respects frequency-dependent physics importance
- Con: Static weights, not adaptive during training

### 4. PSPT (Progressive Spectral Physics)
- Gradually increases physics weight during training
- Low-freq physics activated early, high-freq later
- Pro: Prevents early-stage physics interference
- Con: Requires tuning of schedule parameters

### 5. FA-PINO + PSPT (Combined)
- Frequency-aware weights + progressive schedule
- Pro: Combines benefits of both approaches
- Con: More hyperparameters to tune

### 6. AFP-PINO (Adaptive Frequency Physics)
- Learnable λ per frequency band (softmax-normalized)
- Pro: Automatically finds optimal frequency weighting
- Con: Additional learnable parameters, may overfit on small data

### 7. SP-JR (Spectral-Physics Joint)
- PINO + spectral uniformity (KL divergence) + cross-band consistency
- Pro: Prevents frequency collapse, encourages balanced representation
- Con: Multiple loss terms may compete

### 8. FDPA (Physics-Informed Attention)
- Physics residual gates attention to frequency bands
- Bands with smaller residual get higher attention
- Pro: Self-regulating, focuses physics where it helps most
- Con: Attention mechanism adds complexity


## Reproducibility Verification
- All experiments use seed=42
- DataLoader shuffle=False, num_workers=0
- Model re-created from scratch for each experiment
- Baseline should be identical across runs