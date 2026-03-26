# PINO 物理约束测试报告

## 天渊团队 (Tianyuan Team)
**Team ID**: team_tianyuan_fft
**测试日期**: 2026-03-26
**测试人员**: 天渊 (architect)

---

## 1. 测试目标

测试 PINO (Physics-Informed Neural Operator) 在 Navier-Stokes 数据集上的效果，目标是让 MHF-FNO+PINO 的精度 ≥ FNO。

---

## 2. 实验配置

### 数据集
- **数据**: Navier-Stokes 方程模拟数据
- **训练集**: 1000 样本 (32×32 分辨率)
- **测试集**: 200 样本 (32×32 分辨率)
- **数据文件**:
  - `benchmark/data/ns_train_32_large.pt` (7.9MB)
  - `benchmark/data/ns_test_32_large.pt` (1.6MB)

### 模型配置
| 模型 | 配置 | 参数量 |
|------|------|--------|
| FNO | n_modes=(16,16), hidden=32, layers=4 | 602,977 |
| MHF-FNO | n_modes=(16,16), hidden=32, layers=4, n_heads=2, mhf_layers=[0] | 530,502 |
| MHF-PINO | MHF-FNO + PINOLoss | 530,502 |

### PINO 配置
```python
# 当前实现的 PINOLoss
lambda_physics = [0.0001, 0.001, 0.01, 0.1]
smoothness_weight = 0.5

# 物理损失组成
L_physics = smoothness_weight * L_smooth + (1-smoothness_weight) * L_laplacian
L_smooth = gradient_norm_squared.mean()  # 平滑性约束
L_laplacian = (laplacian^2).mean()       # Laplacian约束
```

---

## 3. 测试结果

### 3.1 快速测试 (10 epochs)

| 模型 | Test Loss | vs FNO | 参数减少 |
|------|-----------|--------|----------|
| FNO | 8.832567 | 基准 | - |
| MHF-FNO+PINO (λ=0.01) | **177.710526** | **+1911.99%** | -12.0% |

**结论**: PINO 严重恶化性能！

### 3.2 超参数调优测试 (20 epochs)

| 模型 | Test Loss | vs FNO | 说明 |
|------|-----------|--------|------|
| FNO | 4.275738 | 基准 | 标准FNO |
| MHF-FNO | 7.874872 | +84.2% | 无PINO的MHF-FNO |
| MHF-PINO (λ=0.0001) | ~177.9 | +4060% | PINO严重失败 |

### 3.3 基础对比测试 (30 epochs, 进行中)

| 模型 | Best Test Loss | Epoch |
|------|----------------|-------|
| FNO | 8.526480 | 20 |

**注意**: MHF-FNO 测试因超时未完成，但趋势已明显。

---

## 4. 问题分析

### 4.1 PINO 失败的根本原因

#### ❌ 当前实现的问题

1. **过度平滑约束**
   ```python
   # 当前的 smoothness loss
   L_smooth = (u_x^2 + u_y^2).mean()
   ```
   - 惩罚所有梯度，包括物理上应该存在的梯度
   - 导致预测场过度平滑，失去细节特征

2. **不恰当的 Laplacian 约束**
   ```python
   # 当前的 Laplacian loss
   L_laplacian = (u_xx + u_yy)^2.mean()
   ```
   - 惩罚所有二阶导数，与 NS 方程物理不符
   - NS 方程中的 ∇²u 是物理过程的一部分，不应被惩罚

3. **缺失真正的 NS 方程残差**
   - 当前实现只有平滑性约束，没有 PDE 残差
   - 真正的 PINO 应该计算：
     ```
     Residual = ∂u/∂t + (u·∇)u + ∇p - ν∇²u
     ```

#### 📊 数值证据

| 指标 | FNO | MHF-PINO | 说明 |
|------|-----|----------|------|
| 训练损失 | 0.044 | 0.002 | PINO训练损失更低 |
| 测试损失 | 8.83 | 177.7 | PINO测试损失极高 |
| 过拟合 | 轻微 | **严重** | PINO严重过拟合 |

**现象**: MHF-PINO 训练损失极低（0.002），但测试损失极高（177.7），这是典型的过拟合。物理约束导致模型在训练集上学习过度平滑的模式，但无法泛化。

### 4.2 MHF-FNO 本身的问题

即使不使用 PINO，MHF-FNO 也比 FNO 差 84%：

| 对比 | FNO | MHF-FNO | 差距 |
|------|-----|---------|------|
| Test Loss | 4.28 | 7.87 | +84% |
| 参数量 | 602,977 | 530,502 | -12% |

**可能原因**:
1. `mhf_layers=[0]` 只在第1层使用MHF，效果有限
2. `n_heads=2` 头数太少，频谱分解不充分
3. 数据量不足（1000样本），MHF 需要更多数据
4. 需要更细致的超参数调优

---

## 5. PINO 正确实现方案

### 5.1 真正的 NS 方程物理损失

```python
class NSPhysicsLoss(nn.Module):
    """
    Navier-Stokes 方程的物理约束
    
    2D 不可压缩 NS 方程:
        ∂u/∂t + (u·∇)u = -∇p + ν∇²u
        ∇·u = 0
    """
    def __init__(self, viscosity=1e-3, dt=0.01):
        super().__init__()
        self.nu = viscosity
        self.dt = dt
    
    def forward(self, u_pred, u_prev):
        """
        Args:
            u_pred: [B, 2, H, W] 预测的速度场 (u, v)
            u_prev: [B, 2, H, W] 上一时刻的速度场
        """
        # 分离速度分量
        u, v = u_pred[:, 0:1], u_pred[:, 1:2]
        
        # 时间导数: ∂u/∂t
        u_t = (u_pred - u_prev) / self.dt
        
        # 空间导数
        u_x = torch.gradient(u, dim=-1)[0]
        u_y = torch.gradient(u, dim=-2)[0]
        v_x = torch.gradient(v, dim=-1)[0]
        v_y = torch.gradient(v, dim=-2)[0]
        
        # 对流项: (u·∇)u
        convection_u = u * u_x + v * u_y
        convection_v = u * v_x + v * v_y
        
        # 扩散项: ν∇²u
        u_xx = torch.gradient(u_x, dim=-1)[0]
        u_yy = torch.gradient(u_y, dim=-2)[0]
        v_xx = torch.gradient(v_x, dim=-1)[0]
        v_yy = torch.gradient(v_y, dim=-2)[0]
        
        laplacian_u = u_xx + u_yy
        laplacian_v = v_xx + v_yy
        
        diffusion_u = self.nu * laplacian_u
        diffusion_v = self.nu * laplacian_v
        
        # PDE 残差 (忽略压力项，或单独预测压力)
        residual_u = u_t[:, 0] + convection_u - diffusion_u
        residual_v = u_t[:, 1] + convection_v - diffusion_v
        
        # 总残差
        residual = residual_u**2 + residual_v**2
        
        return residual.mean()
```

### 5.2 数据要求

真正的 PINO 需要：
1. **时间序列数据**: (u^t, u^{t+1}) 对，而不是 (x, y) 对
2. **速度场数据**: (u, v) 两个分量，而不是标量场
3. **压力场数据**: (u, v, p) 三分量，或需要同时预测压力

**当前数据问题**:
- 现有数据是标量场 (B, 1, 32, 32)
- 缺少时间维度
- 无法计算真正的 PDE 残差

### 5.3 改进建议

#### 方案 A: 完整实现 PINO
1. 重新生成 NS 数据集，包含时间序列和速度场
2. 实现完整的 NS 方程物理损失
3. 同时预测速度和压力

#### 方案 B: 简化物理约束
1. 只保留 **不可压缩约束** ∇·u = 0
2. 使用更温和的平滑性约束（不要惩罚所有梯度）
3. 降低 lambda_physics 到 1e-5 ~ 1e-6

#### 方案 C: 放弃 PINO
1. 专注于 MHF-FNO 的优化
2. 增加 mhf_layers 到 [0, 2, 3]
3. 增加训练数据到 5000+ 样本

---

## 6. 结论

### 6.1 测试结论

| 指标 | 目标 | 实际结果 | 状态 |
|------|------|----------|------|
| MHF-PINO ≥ FNO | ≥ 0% | **-1911%** | ❌ 失败 |
| 参数减少 | > 20% | 12% | ⚠️ 未达标 |
| PINO 提升精度 | > 0% | **严重恶化** | ❌ 失败 |

### 6.2 根本问题

1. **当前 PINO 实现不是真正的物理约束**
   - 只有平滑性和 Laplacian 约束
   - 缺少 NS 方程残差
   - 约束方向错误，导致性能恶化

2. **数据不适合 PINO**
   - 标量场数据，无法计算速度场的 PDE 残差
   - 缺少时间序列，无法计算时间导数

3. **MHF-FNO 本身需要优化**
   - 即使不用 PINO，也比 FNO 差 84%
   - 需要更多层使用 MHF
   - 需要更多训练数据

### 6.3 最终建议

**短期（1-2天）**:
- 放弃当前 PINO 实现
- 专注于 MHF-FNO 优化
- 目标：MHF-FNO 精度 ≥ FNO，参数减少 > 20%

**中期（1周）**:
- 生成完整的 NS 时间序列数据（速度场 + 压力场）
- 实现真正的 NS 方程物理损失
- 重新测试 PINO

**长期（持续）**:
- 探索其他物理约束方法（软约束、数据增强）
- 研究 PINO 在其他 PDE 上的效果

---

## 7. 附录

### 7.1 测试脚本

1. `benchmark/test_ns_pino_quick.py` - 快速PINO测试
2. `benchmark/test_pino_tuned.py` - 超参数调优测试
3. `benchmark/test_mhf_simple.py` - MHF-FNO基础对比

### 7.2 现有实现

- `mhf_fno/mhf_fno.py` - MHF-FNO + PINOLoss
- `mhf_fno/__init__.py` - 导出接口

### 7.3 数据文件

- `benchmark/data/ns_train_32_large.pt` - 训练集 (1000样本)
- `benchmark/data/ns_test_32_large.pt` - 测试集 (200样本)

---

**报告完成日期**: 2026-03-26
**下一步行动**: 实现 MHF-FNO 优化方案，放弃当前 PINO 实现
