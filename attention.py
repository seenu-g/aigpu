import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttentionHead(nn.Module):
    """One head of scaled dot-product attention with a causal mask."""

    def __init__(self, embedding_dim, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        _, sequence_length, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        scores = scores.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        return attention_weights @ v, attention_weights


class MultiHeadAttention(nn.Module):
    """Multiple causal self-attention heads running in parallel."""

    def __init__(self, num_heads, embedding_dim, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            CausalAttentionHead(embedding_dim, head_size, block_size)
            for _ in range(num_heads)
        )
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)

    def forward(self, x):
        head_outputs = [head(x)[0] for head in self.heads]
        return self.proj(torch.cat(head_outputs, dim=-1))


class FeedForward(nn.Module):
    """Per-token feed-forward network."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)
