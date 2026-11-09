import torch.nn as nn


class TokenPositionEmbedding(nn.Module):
    """Add learned token and position embeddings for a batch of token IDs."""

    def __init__(self, vocab_size, embedding_dim, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)

    def forward(self, token_ids):
        _, sequence_length = token_ids.shape
        token_embeddings = self.token_embedding(token_ids)
        positions = self.position_embedding.weight[:sequence_length]
        return token_embeddings, positions, token_embeddings + positions
