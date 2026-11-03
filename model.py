import torch.nn as nn
import torch.nn.functional as F

from attention import TransformerBlock
from embeddings import TokenPositionEmbedding


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, block_size):
        super().__init__()
        self.block_size = block_size
        self.embeddings = TokenPositionEmbedding(vocab_size, embedding_dim, block_size)
        self.block = TransformerBlock(embedding_dim, num_heads, block_size)
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, token_ids, targets=None):
        _, sequence_length = token_ids.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds block size {self.block_size}"
            )

        _, _, x = self.embeddings(token_ids)
        x = self.block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )

        return logits, loss
