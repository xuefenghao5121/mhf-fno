# 阶段5修正版实验报告

**时间**: 2026-03-31 00:04:07

**设备**: cpu


## 修正说明

1. **Burgers数据集**: 之前错误地将uniform当训练集、rand当测试集。
   修正: uniform(前100样本,下采样到256点)训练，rand和conexp分别作为独立测试集。
2. **NS PINO约束**: 所有NS变体训练时加入VorticityPINO物理约束
   (Laplacian平滑正则 + 高频噪声惩罚)。


## Part 1: Burgers 双测试集结果

| 模型 | 测试集(rand) MSE | 测试集(rand) LP | 测试集(conexp) MSE | 测试集(conexp) LP |
|------|-----------------|---------------|-------------------|-----------------|
| FNO | 0.0962 | 28.0968 | 0.0710 | 24.0595 |
| MHF-FNO | 0.0715 | 23.4205 | 0.0579 | 21.3078 |

## Part 2: NS 128x128 + PINO 结果

| 模型 | No PINO MSE | No PINO LP | With PINO MSE | With PINO LP | PINO改进 |
|------|------------|-----------|--------------|-------------|---------|
| FNO | 0.5088 | 2555.1660 | ERR | ERR | - |
| MHF-FNO | 0.5007 | 2533.7454 | ERR | ERR | - |
| MHF+CoDA | 0.4557 | 2419.5591 | ERR | ERR | - |
| MHF+AFNO | 0.4918 | 2511.2671 | ERR | ERR | - |
| MHF+CoDA+AFNO | 0.4865 | 2499.6279 | ERR | ERR | - |

## 对比分析

### Burgers数据集

- uniform(高分辨率8192点)训练后，在rand和conexp(低分辨率64点)上测试
- 由于分辨率差异大(256 vs 64)，结果反映模型的跨分辨率泛化能力

### NS PINO约束

- VorticityPINO通过Laplacian平滑正则和高频惩罚施加物理约束
- 正PINO改进表示物理约束帮助了泛化，负值表示可能需要调整权重