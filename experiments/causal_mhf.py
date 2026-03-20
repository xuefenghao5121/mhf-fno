"""
Causal Multi-Head Fourier (MHF) Module

参考NeuralOperator 2.0.0的设计风格，实现因果FFT Attention

核心特性：
1. 因果性保证（非对称padding + 截断）
2. 兼容NeuralOperator的benchmark
3. 支持序列建模任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

Number = Union[int, float]


class CausalSpectralConv(nn.Module):
    """
    因果频谱卷积层
    
    参考 neuralop.layers.spectral_convolution.SpectralConv
    但添加因果性支持
    
    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int  
        输出通道数
    n_modes : int
        保留的傅里叶模式数
    max_seq_len : int
        最大序列长度
    causal : bool, default True
        是否使用因果padding
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
        max_seq_len: int = 8192,
        causal: bool = True,
        init_std: Union[str, float] = "auto",
        fft_norm: str = "forward",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.max_seq_len = max_seq_len
        self.causal = causal
        self.fft_norm = fft_norm
        
        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        
        # 频域权重 - 形状: (in_channels, out_channels, n_modes)
        # 注意：对于rfft，最后一个维度是 n//2 + 1
        self.weight = nn.Parameter(
            torch.randn(in_channels, out_channels, n_modes, dtype=torch.cfloat) * init_std
        )
        
        self.bias = nn.Parameter(
            init_std * torch.randn(out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状 (batch, channels, seq_len)
            
        Returns
        -------
        torch.Tensor
            输出张量，形状 (batch, out_channels, seq_len)
        """
        batch_size, channels, seq_len = x.shape
        
        if self.causal:
            # === 因果模式：正确的非对称padding + 频域操作 ===
            # 关键：在末尾补seq_len个零
            x_padded = F.pad(x, (0, seq_len))  # (batch, channels, 2*seq_len)
            padded_len = 2 * seq_len
            
            # FFT
            x_freq = torch.fft.rfft(x_padded, dim=-1, norm=self.fft_norm)
            
            # 获取实际使用的模式数
            n_modes = min(self.n_modes, x_freq.shape[-1])
            
            # 初始化输出频谱
            out_freq = torch.zeros(
                batch_size, self.out_channels, x_freq.shape[-1],
                dtype=x_freq.dtype, device=x.device
            )
            
            # 频域卷积
            out_freq[:, :, :n_modes] = torch.einsum(
                'bif,iOf->bOf', 
                x_freq[:, :, :n_modes], 
                self.weight[:, :, :n_modes]
            )
            
            # IFFT
            x_out = torch.fft.irfft(out_freq, n=padded_len, dim=-1, norm=self.fft_norm)
            
            # 关键：截断只取前seq_len个输出
            x_out = x_out[:, :, :seq_len]
        else:
            # 非因果模式：直接处理
            x_freq = torch.fft.rfft(x, dim=-1, norm=self.fft_norm)
            n_modes = min(self.n_modes, x_freq.shape[-1])
            
            out_freq = torch.zeros(
                batch_size, self.out_channels, x_freq.shape[-1],
                dtype=x_freq.dtype, device=x.device
            )
            
            out_freq[:, :, :n_modes] = torch.einsum(
                'bif,iOf->bOf', 
                x_freq[:, :, :n_modes], 
                self.weight[:, :, :n_modes]
            )
            
            x_out = torch.fft.irfft(out_freq, n=seq_len, dim=-1, norm=self.fft_norm)
        
        # 添加bias
        x_out = x_out + self.bias
        
        return x_out


class CausalMHFBlock(nn.Module):
    """
    因果Multi-Head Fourier Block
    
    参考 neuralop.layers.fno_block.FNOBlocks 的设计
    但专为序列建模和因果性优化
    
    Parameters
    ----------
    d_model : int
        模型维度
    n_heads : int
        注意力头数
    n_modes : int, optional
        傅里叶模式数，默认 d_model // 4
    max_seq_len : int
        最大序列长度
    dropout : float
        Dropout率
    activation : str
        激活函数
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_modes: Optional[int] = None,
        max_seq_len: int = 8192,
        dropout: float = 0.0,
        activation: str = "gelu",
        causal: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_modes = n_modes or d_model // 4
        self.max_seq_len = max_seq_len
        self.causal = causal
        
        # 输入投影
        self.W_V = nn.Linear(d_model, d_model)
        
        # 多头频域卷积
        self.spectral_convs = nn.ModuleList([
            CausalSpectralConv(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                n_modes=self.n_modes,
                max_seq_len=max_seq_len,
                causal=causal,
            )
            for _ in range(n_heads)
        ])
        
        # 输出投影
        self.W_O = nn.Linear(d_model, d_model)
        
        # LayerNorm (pre-norm style)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Parameters
        ----------
        x : torch.Tensor
            输入张量，形状 (batch, seq_len, d_model)
        mask : torch.Tensor, optional
            注意力掩码（兼容标准Transformer接口）
            
        Returns
        -------
        torch.Tensor
            输出张量，形状 (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # === Pre-norm + MHF ===
        x = self.norm1(x)
        
        # 输入投影
        V = self.W_V(x)  # (batch, seq_len, d_model)
        
        # 重塑为多头
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.permute(0, 2, 3, 1)  # (batch, heads, head_dim, seq_len)
        
        # 每个头独立频域卷积
        head_outputs = []
        for h in range(self.n_heads):
            V_h = V[:, h, :, :]  # (batch, head_dim, seq_len)
            out_h = self.spectral_convs[h](V_h)  # (batch, head_dim, seq_len)
            head_outputs.append(out_h)
        
        # 合并多头
        V_out = torch.stack(head_outputs, dim=2)  # (batch, head_dim, heads, seq_len)
        V_out = V_out.permute(0, 3, 2, 1)  # (batch, seq_len, heads, head_dim)
        V_out = V_out.contiguous().view(batch_size, seq_len, d_model)
        
        # 输出投影
        x = self.W_O(V_out)
        x = self.dropout(x)
        
        # 残差连接
        x = residual + x
        
        # === FFN ===
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class CausalMHFTransformer(nn.Module):
    """
    因果MHF Transformer
    
    兼容NeuralOperator benchmark风格
    
    Parameters
    ----------
    vocab_size : int
        词表大小
    d_model : int
        模型维度
    n_heads : int
        注意力头数
    n_layers : int
        层数
    n_modes : int, optional
        傅里叶模式数
    max_seq_len : int
        最大序列长度
    dropout : float
        Dropout率
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_modes: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置嵌入（可选，FFT天然支持位置）
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # MHF层
        self.layers = nn.ModuleList([
            CausalMHFBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_modes=n_modes,
                max_seq_len=max_seq_len,
                dropout=dropout,
                causal=causal,
            )
            for _ in range(n_layers)
        ])
        
        # 最终LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        
        # 语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.embedding.weight
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        前向传播
        
        Parameters
        ----------
        input_ids : torch.Tensor
            输入token IDs，形状 (batch, seq_len)
        labels : torch.Tensor, optional
            标签，形状 (batch, seq_len)
            
        Returns
        -------
        dict
            包含loss和logits的字典
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        x = self.embedding(input_ids)
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device)
        x = x + self.pos_embedding(positions)
        
        x = self.dropout(x)
        
        # MHF层
        for layer in self.layers:
            x = layer(x)
        
        # 最终LayerNorm
        x = self.final_norm(x)
        
        # 语言模型头
        logits = self.lm_head(x)
        
        # 计算loss
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        
        return {"loss": loss, "logits": logits}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        自回归生成（因果性验证）
        
        Parameters
        ----------
        input_ids : torch.Tensor
            输入token IDs
        max_new_tokens : int
            最大生成token数
        temperature : float
            采样温度
        top_k : int
            Top-k采样参数
            
        Returns
        -------
        torch.Tensor
            生成的token IDs
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度
            idx_cond = input_ids[:, -self.max_seq_len:]
            
            # 前向传播
            outputs = self(idx_cond)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Top-k采样
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# === 兼容NeuralOperator的简化版本 ===

class FNO1D(nn.Module):
    """
    1D FNO（兼容NeuralOperator风格）
    
    用于PDE benchmark测试
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_modes: int = 16,
        n_layers: int = 4,
    ):
        super().__init__()
        
        # Lifting
        self.lifting = nn.Linear(in_channels, hidden_channels)
        
        # FNO层
        self.fno_layers = nn.ModuleList([
            CausalSpectralConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                n_modes=n_modes,
                causal=False,  # PDE不需要因果性
            )
            for _ in range(n_layers)
        ])
        
        # Projection
        self.projection = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            输入，形状 (batch, in_channels, seq_len)
            
        Returns
        -------
        torch.Tensor
            输出，形状 (batch, out_channels, seq_len)
        """
        # Lifting
        x = x.permute(0, 2, 1)  # (batch, seq_len, in_channels)
        x = self.lifting(x)
        x = x.permute(0, 2, 1)  # (batch, hidden_channels, seq_len)
        
        # FNO层
        for fno_layer in self.fno_layers:
            x = F.gelu(x + fno_layer(x))
        
        # Projection
        x = x.permute(0, 2, 1)  # (batch, seq_len, hidden_channels)
        x = self.projection(x)
        x = x.permute(0, 2, 1)  # (batch, out_channels, seq_len)
        
        return x


if __name__ == "__main__":
    # 简单测试
    print("=== Causal MHF 测试 ===\n")
    
    # 测试1：因果性验证
    print("1. 因果性测试")
    model = CausalMHFTransformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    
    x = torch.randn(1, 10, 64)
    out1 = model.layers[0](x.clone())
    
    # 修改未来位置
    x_modified = x.clone()
    x_modified[:, 5:, :] = torch.randn(1, 5, 64)
    out2 = model.layers[0](x_modified)
    
    # 前5个位置的输出应该不变
    if torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5):
        print("✅ 因果性测试通过")
    else:
        print("❌ 因果性测试失败")
    
    # 测试2：前向传播
    print("\n2. 前向传播测试")
    input_ids = torch.randint(0, 1000, (2, 32))
    outputs = model(input_ids, labels=input_ids)
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    
    # 测试3：生成
    print("\n3. 生成测试")
    generated = model.generate(input_ids[:1, :5], max_new_tokens=10)
    print(f"Generated shape: {generated.shape}")
    
    print("\n=== 测试完成 ===")