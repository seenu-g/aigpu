import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Hyperparameters for our Small Language Model (SLM).
    Optimized for training on a single local GPU.
    """
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dim: int = 512            # Embedding dimension
    n_layers: int = 8         # Number of transformer blocks
    n_heads: int = 8          # Number of attention heads
    multiple_of: int = 256    # For SwiGLU feedforward dimension alignment
    norm_eps: float = 1e-5

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization. Stabilizes training like LayerNorm 
    but skips calculating the mean, saving compute.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class SwiGLU(nn.Module):
    """
    SwiGLU Activation FeedForward Network (FFN). 
    Used by Llama/Mistral instead of standard ReLU/GELU for superior learning capacity.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(2 * (config.dim * 4) / 3)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(x * W1) * (x * W3) -> W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    Forces tokens to only attend to past tokens (no looking ahead).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.n_heads != 0:
            raise ValueError(
                f"config.dim ({config.dim}) must be divisible by config.n_heads ({config.n_heads})"
            )
        if config.max_seq_len < 1:
            raise ValueError("config.max_seq_len must be positive")

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.max_seq_len = config.max_seq_len

        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wk = nn.Linear(config.dim, config.dim, bias=False)
        self.wv = nn.Linear(config.dim, config.dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        if t > self.max_seq_len:
            raise ValueError(
                f"Sequence length {t} exceeds max_seq_len {self.max_seq_len} for this attention block."
            )

        q = self.wq(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal_mask = self.mask[:, :, :t, :t]
        att = att.masked_fill(~causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.wo(y)

class TransformerBlock(nn.Module):
    """
    A single standard Transformer Layer combining Attention, Norms, and Feed-Forward.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Residual connection strategy
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class CustomLLM(nn.Module):
    """
    The Core Decoder-Only Language Model Wrapper.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Tie token embeddings to output layer weights to save memory (Weight Tying)
        self.tok_embeddings.weight = self.output.weight

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        _b, t = tokens.size()
        
        # 1. Look up token vectors
        h = self.tok_embeddings(tokens)
        
        # 2. Process vectors through consecutive transformer blocks
        for layer in self.layers:
            h = layer(h)
            
        # 3. Apply final structural normalization
        h = self.norm(h)
        
        # 4. Map final hidden states back to target vocabulary logits
        logits = self.output(h)

        loss = None
        if targets is not None:
            # Reshape tensors for PyTorch's cross_entropy loss implementation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
