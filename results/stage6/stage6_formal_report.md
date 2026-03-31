# Stage 6: Formal Comparison Report

**Generated**: 2026-03-31 04:32:58

## Unified Experimental Conditions (正式参数)
- **Seed**: 42
- **Data**: NS 128x128, train=50, test=20
- **Training**: 50 epochs, batch_size=32, lr=0.001, Adam
- **Model**: MHF-FNO, n_modes=(16, 16), hidden=32, n_heads=2, mhf_layers=[0]
- **Device**: cpu
- **PINO λ**: 0.1
- **PINO config**: viscosity=1e-3, dt=1.0, dx=1.0

## Unified Comparison Table

| Rank | Configuration | Test MSE | Lp Error | Δ MSE vs Baseline | Latency (ms) | Total Params |
|------|--------------|----------|----------|-------------------|--------------|-------------|
| 1 | AFP-PINO (adaptive) | 0.138025 | 10.250827 | -3.12% ✅ | 17.5 | 379,635 |
| 2 | PSPT (progressive) | 0.142436 | 10.374921 | -0.02% | 18.7 | 379,633 |
| 3 | FA-PINO + PSPT | 0.142436 | 10.374921 | -0.02% | 13.9 | 379,633 |
| 4 | Uniform PINO (λ=0.1) | 0.142453 | 10.375444 | -0.01% | 15.1 | 379,633 |
| 5 | Baseline (no PINO) | 0.142471 | 10.376029 | +0.00% | 16.4 | 379,633 |
| 6 | FDPA (physics attention) | 0.142471 | 10.376029 | +0.00% | 16.5 | 379,633 |
| 7 | SP-JR (spectral-physics joint) | 0.509868 | 19.980658 | +257.88% ⚠️ | 17.5 | 379,633 |
| 8 | FA-PINO (frequency-aware) | 0.510297 | 19.988415 | +258.18% ⚠️ | 16.2 | 379,633 |

**Baseline**: MSE=0.142471, Lp=10.376029

## 🏆 Best Configuration: 6_AFP_PINO
- AFP-PINO (adaptive)
- MSE=0.138025 (vs baseline 0.142471)
- Improvement: 3.12%

## Analysis

### ✅ Improved over Baseline (1 configs)
- **AFP-PINO (adaptive)**: 3.12% improvement

### ➖ Similar to Baseline (5 configs)
- **PSPT (progressive)**: comparable
- **FA-PINO + PSPT**: comparable
- **Uniform PINO (λ=0.1)**: comparable
- **Baseline (no PINO)**: comparable
- **FDPA (physics attention)**: comparable

### ⚠️ Degraded vs Baseline (2 configs)
- **SP-JR (spectral-physics joint)**: 257.88% degradation
- **FA-PINO (frequency-aware)**: 258.18% degradation

## Training Convergence
- Baseline (no PINO): 0.4622 → 0.4606 → 0.4593 → 0.4575 → 0.4555 → 0.4531 → 0.4499 → 0.4456 → 0.4400 → 0.4326 → 0.4226 → 0.4093 → 0.3916 → 0.3684 → 0.3385 → 0.3014 → 0.2585 → 0.2148 → 0.1819 → 0.1724 → 0.1730 → 0.1609 → 0.1460 → 0.1388 → 0.1382 → 0.1391 → 0.1388 → 0.1369 → 0.1339 → 0.1308 → 0.1284 → 0.1268 → 0.1259 → 0.1253 → 0.1247 → 0.1242 → 0.1237 → 0.1233 → 0.1230 → 0.1228 → 0.1226 → 0.1225 → 0.1224 → 0.1223 → 0.1223 → 0.1222 → 0.1222 → 0.1222 → 0.1222 → 0.1222
- Uniform PINO (λ=0.1): 0.4622 → 0.4606 → 0.4593 → 0.4575 → 0.4555 → 0.4531 → 0.4499 → 0.4456 → 0.4400 → 0.4326 → 0.4226 → 0.4093 → 0.3916 → 0.3684 → 0.3385 → 0.3014 → 0.2585 → 0.2149 → 0.1819 → 0.1724 → 0.1729 → 0.1609 → 0.1460 → 0.1389 → 0.1382 → 0.1391 → 0.1389 → 0.1369 → 0.1339 → 0.1308 → 0.1284 → 0.1268 → 0.1258 → 0.1252 → 0.1247 → 0.1242 → 0.1237 → 0.1233 → 0.1230 → 0.1228 → 0.1226 → 0.1225 → 0.1224 → 0.1223 → 0.1223 → 0.1222 → 0.1222 → 0.1222 → 0.1222 → 0.1222
- FA-PINO (frequency-aware): 1.9327 → 1.1938 → 1.2583 → 0.5173 → 0.7800 → 0.8890 → 0.5709 → 0.4961 → 0.6441 → 0.6159 → 0.4870 → 0.4859 → 0.5445 → 0.5243 → 0.4717 → 0.4722 → 0.4965 → 0.4892 → 0.4672 → 0.4653 → 0.4752 → 0.4747 → 0.4661 → 0.4628 → 0.4659 → 0.4676 → 0.4654 → 0.4629 → 0.4627 → 0.4637 → 0.4640 → 0.4634 → 0.4627 → 0.4624 → 0.4625 → 0.4627 → 0.4628 → 0.4627 → 0.4626 → 0.4625 → 0.4624 → 0.4624 → 0.4624 → 0.4624 → 0.4624 → 0.4624 → 0.4624 → 0.4624 → 0.4624 → 0.4624
- PSPT (progressive): 0.4622 → 0.4606 → 0.4593 → 0.4575 → 0.4555 → 0.4531 → 0.4499 → 0.4456 → 0.4400 → 0.4326 → 0.4226 → 0.4093 → 0.3916 → 0.3684 → 0.3385 → 0.3015 → 0.2586 → 0.2149 → 0.1820 → 0.1724 → 0.1728 → 0.1608 → 0.1460 → 0.1389 → 0.1382 → 0.1392 → 0.1389 → 0.1369 → 0.1339 → 0.1308 → 0.1284 → 0.1268 → 0.1258 → 0.1252 → 0.1247 → 0.1242 → 0.1237 → 0.1233 → 0.1230 → 0.1228 → 0.1226 → 0.1225 → 0.1224 → 0.1223 → 0.1223 → 0.1222 → 0.1222 → 0.1222 → 0.1222 → 0.1222
- FA-PINO + PSPT: 0.4622 → 0.4606 → 0.4593 → 0.4575 → 0.4555 → 0.4531 → 0.4499 → 0.4456 → 0.4400 → 0.4326 → 0.4226 → 0.4093 → 0.3916 → 0.3684 → 0.3385 → 0.3015 → 0.2586 → 0.2149 → 0.1820 → 0.1724 → 0.1728 → 0.1608 → 0.1460 → 0.1389 → 0.1382 → 0.1392 → 0.1389 → 0.1369 → 0.1339 → 0.1308 → 0.1284 → 0.1268 → 0.1258 → 0.1252 → 0.1247 → 0.1242 → 0.1237 → 0.1233 → 0.1230 → 0.1228 → 0.1226 → 0.1225 → 0.1224 → 0.1223 → 0.1223 → 0.1222 → 0.1222 → 0.1222 → 0.1222 → 0.1222
- AFP-PINO (adaptive): 0.4622 → 0.4607 → 0.4595 → 0.4579 → 0.4561 → 0.4540 → 0.4511 → 0.4473 → 0.4424 → 0.4358 → 0.4269 → 0.4151 → 0.3993 → 0.3784 → 0.3513 → 0.3172 → 0.2770 → 0.2349 → 0.2024 → 0.1951 → 0.1999 → 0.1913 → 0.1743 → 0.1627 → 0.1582 → 0.1568 → 0.1554 → 0.1530 → 0.1499 → 0.1469 → 0.1445 → 0.1429 → 0.1420 → 0.1413 → 0.1406 → 0.1399 → 0.1392 → 0.1385 → 0.1380 → 0.1375 → 0.1372 → 0.1370 → 0.1368 → 0.1367 → 0.1366 → 0.1365 → 0.1365 → 0.1365 → 0.1364 → 0.1364
- SP-JR (spectral-physics joint): 7.8837 → 1.5995 → 6.2092 → 1.6668 → 0.5534 → 0.5687 → 0.8928 → 1.4868 → 1.4423 → 0.9254 → 0.6295 → 0.5544 → 0.5301 → 0.5304 → 0.5453 → 0.5564 → 0.5698 → 0.5821 → 0.5886 → 0.5876 → 0.5808 → 0.5719 → 0.5633 → 0.5564 → 0.5510 → 0.5467 → 0.5433 → 0.5402 → 0.5374 → 0.5347 → 0.5321 → 0.5296 → 0.5274 → 0.5257 → 0.5244 → 0.5235 → 0.5230 → 0.5227 → 0.5226 → 0.5224 → 0.5224 → 0.5223 → 0.5222 → 0.5222 → 0.5221 → 0.5221 → 0.5220 → 0.5220 → 0.5220 → 0.5220
- FDPA (physics attention): 0.4622 → 0.4606 → 0.4593 → 0.4575 → 0.4555 → 0.4531 → 0.4499 → 0.4456 → 0.4400 → 0.4326 → 0.4226 → 0.4093 → 0.3916 → 0.3684 → 0.3385 → 0.3014 → 0.2585 → 0.2148 → 0.1819 → 0.1724 → 0.1730 → 0.1609 → 0.1460 → 0.1388 → 0.1382 → 0.1391 → 0.1388 → 0.1369 → 0.1339 → 0.1308 → 0.1284 → 0.1268 → 0.1259 → 0.1253 → 0.1247 → 0.1242 → 0.1237 → 0.1233 → 0.1230 → 0.1228 → 0.1226 → 0.1225 → 0.1224 → 0.1223 → 0.1223 → 0.1222 → 0.1222 → 0.1222 → 0.1222 → 0.1222

## Method-Specific Notes

### 1. Baseline (no PINO)
- Pure data-driven MHF-FNO, no physics constraints
- Serves as the reference point for all comparisons

### 2. Uniform PINO
- Constant λ=0.1 laplacian smoothness penalty
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