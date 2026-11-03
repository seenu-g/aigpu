import argparse
from pathlib import Path

import torch

from data_loader import load_dataset
from model import LanguageModel
from tokenizers import load_or_build_bpe_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small BPE language model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--target-vocab-size", type=int, default=512)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "literature",
    )
    return parser.parse_args()


def train(args):
    if args.batch_size < 1 or args.block_size < 1 or args.max_iters < 1:
        raise ValueError("batch size, block size, and max iterations must be positive")
    if args.eval_interval < 1:
        raise ValueError("evaluation interval must be positive")
    if args.embedding_dim < 1 or args.num_heads < 1:
        raise ValueError("embedding dimension and number of heads must be positive")
    if args.embedding_dim % args.num_heads:
        raise ValueError("embedding dimension must be divisible by the number of heads")
    if args.target_vocab_size < 256:
        raise ValueError("target vocabulary size must be at least 256")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using architecture framework running on: {device}")

    raw_text = load_dataset(args.data_dir)
    bpe_encode, _, vocab_size = load_or_build_bpe_tokenizer(
        raw_text, target_vocab_size=args.target_vocab_size
    )

    print("Tokenizing master dataset string...")
    data = torch.tensor(bpe_encode(raw_text), dtype=torch.long)
    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]
    if min(len(train_data), len(val_data)) <= args.block_size:
        raise ValueError(
            "The training and validation splits must each contain more tokens "
            "than the configured block size"
        )

    def get_batch(split):
        split_data = train_data if split == "train" else val_data
        starts = torch.randint(
            len(split_data) - args.block_size,
            (args.batch_size,),
        )
        inputs = torch.stack([split_data[start : start + args.block_size] for start in starts])
        targets = torch.stack(
            [split_data[start + 1 : start + args.block_size + 1] for start in starts]
        )
        return inputs.to(device), targets.to(device)

    model = LanguageModel(
        vocab_size,
        args.embedding_dim,
        args.num_heads,
        args.block_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = {}
        for split in ("train", "val"):
            batch_losses = torch.empty(10)
            for index in range(len(batch_losses)):
                inputs, targets = get_batch(split)
                _, loss = model(inputs, targets)
                batch_losses[index] = loss.item()
            losses[split] = batch_losses.mean().item()
        model.train()
        return losses

    print("\nStarting core model training optimization loop...")
    for step in range(args.max_iters):
        if step % args.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"Step {step:4d} | Train Loss: {losses['train']:.4f} "
                f"| Val Loss: {losses['val']:.4f}"
            )

        inputs, targets = get_batch("train")
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Final Optimized Loss: {loss.item():.4f}")
    return model


if __name__ == "__main__":
    train(parse_args())
