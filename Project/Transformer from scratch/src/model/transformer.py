import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int

from .RMSNorm import RMSNorm
from .attention import MultiHeadSelfAttention
from .FFN import SwiGLU
from .Embedding import RotaryPositionalEmbedding, Embedding
from .Linear import Linear

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding,
        device: torch.device | str | None=None,
        dtype: torch.dtype | None=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.ln1 = RMSNorm(d_model, **factory_kwargs)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
    
    def forward(
        self,
        x: Float[Tensor, "... batch seq_len d_model"],
        token_positions: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        residual = x
        x_norm = self.ln(x)
        attn_out = self.attn(x_norm, token_positions)
        x = residual + attn_out

        residual = x
        x_norm = self.ln(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out

        return x
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None=None,
        dtype: torch.dtype | None=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        d_head = d_model // num_heads
        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)
        rope = RotaryPositionalEmbedding(d_head, context_length, rope_theta, device=device)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope, **factory_kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, **factory_kwargs)
        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self,
        in_indices: Int[Tensor, "batch seq_len"],
    ) -> Float[Tensor, "batch seq_len vocab_size"]:
        batch_size, seq_len = in_indices.shape
        device = in_indices.device
        token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits