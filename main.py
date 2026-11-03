from pathlib import Path

import torch

from attention import CausalAttentionHead, FeedForward, MultiHeadAttention
from data_loader import load_dataset
from embeddings import TokenPositionEmbedding
from tokenizers import build_character_tokenizer, load_or_build_bpe_tokenizer


def main():
    data_folder = Path(__file__).resolve().parent / "literature"
    raw_data = load_dataset(data_folder)
    test_phrase = "Whether 'tis nobler in the mind"

    print("\n=== Instantiating Character Tokenizer ===")
    char_encode, char_decode, char_vocab_size = build_character_tokenizer(raw_data)
    char_ids = char_encode(test_phrase)
    print(f"Character Vocab Size : {char_vocab_size}")
    print(f"Encoded Sequence ({len(char_ids)} tokens): {char_ids}")
    print(f"Decoded Phrase       : '{char_decode(char_ids)}'")

    print("\n=== Instantiating Subword BPE Tokenizer ===")
    bpe_encode, bpe_decode, bpe_vocab_size = load_or_build_bpe_tokenizer(
        raw_data, target_vocab_size=512
    )
    bpe_ids = bpe_encode(test_phrase)
    print(f"BPE Vocab Size       : {bpe_vocab_size}")
    print(f"Encoded Sequence ({len(bpe_ids)} tokens): {bpe_ids}")
    print(f"Decoded Phrase       : '{bpe_decode(bpe_ids)}'")

    print("\n=== Processing Embedding Layers ===")
    batch_size = 4
    block_size = 8
    embedding_dim = 64
    num_heads = 4
    head_size = embedding_dim // num_heads

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected Device Environment: {device}")

    repeats = (block_size + len(bpe_ids) - 1) // len(bpe_ids)
    sample_seq = (bpe_ids * repeats)[:block_size]
    mock_x_batch = torch.tensor([sample_seq for _ in range(batch_size)], dtype=torch.long, device=device)

    embedding_layer = TokenPositionEmbedding(bpe_vocab_size, embedding_dim, block_size).to(device)
    token_embeddings, position_embeddings, input_embeddings = embedding_layer(mock_x_batch)

    print(f"Token Input Integers Matrix (Batch, Context): {mock_x_batch.shape}")
    print(f"Token Vector Space Representation Layout     : {token_embeddings.shape}")
    print(f"Position Sequence Context Footprint Layout    : {position_embeddings.shape}")
    print(f"Final Integrated Embedding Output Matrix (B,T,C): {input_embeddings.shape}")

    attention_layer = CausalAttentionHead(embedding_dim, head_size, block_size).to(device)
    attention_output, weights = attention_layer(input_embeddings)

    print("\n=== Processing Self-Attention Step ===")
    print(f"Integrated Embedded Input Tensor Shape (B, T, C) : {input_embeddings.shape}")
    print(f"Attention Relationship Matrix Layout  (B, T, T) : {weights.shape}")
    print(f"Attention Layer final Output Tensor Layout (B, T, H): {attention_output.shape}")

    mha_layer = MultiHeadAttention(num_heads, embedding_dim, head_size, block_size).to(device)
    ffwd_layer = FeedForward(embedding_dim).to(device)

    attention_output = mha_layer(input_embeddings)
    final_block_output = ffwd_layer(attention_output)

    print("\n=== Processing Multi-Head & Computation Pipeline ===")
    print(f"Initial Vector Space Input Shape  (B, T, C) : {input_embeddings.shape}")
    print(f"Multi-Head Context Output Layout  (B, T, C) : {attention_output.shape}")
    print(f"Post Feed-Forward Computation Layout (B, T, C) : {final_block_output.shape}")


if __name__ == "__main__":
    main()
