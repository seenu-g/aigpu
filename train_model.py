import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from checkpoint import load_checkpoint, save_checkpoint
from data_loader import load_dataset
from evaluation import evaluate_generations
from model import LanguageModel
from tokenizers import load_or_build_bpe_tokenizer


def _start_timing(label):
    started_at = datetime.now().astimezone()
    started_counter = perf_counter()
    print(f"{label} started at {started_at.isoformat(timespec='seconds')}")
    return started_counter


def _finish_timing(label, started_counter):
    finished_at = datetime.now().astimezone()
    elapsed = perf_counter() - started_counter
    print(f"{label} finished at {finished_at.isoformat(timespec='seconds')} ({elapsed:.2f}s)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small BPE language model.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--sample-interval", type=int, default=500)
    parser.add_argument("--sample-tokens", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--target-vocab-size", type=int, default=512)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "literature",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "language_model.pt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model and optimizer state from --checkpoint-path",
    )
    return parser.parse_args()


def train(args):
    run_started = _start_timing("Training run")
    if args.batch_size < 1 or args.block_size < 1 or args.max_iters < 1:
        raise ValueError("batch size, block size, and max iterations must be positive")
    if args.eval_interval < 1:
        raise ValueError("evaluation interval must be positive")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be positive")
    if args.sample_interval < 1 or args.sample_tokens < 0:
        raise ValueError("sample interval must be positive and sample tokens non-negative")
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

    data_load_started = _start_timing("Corpus loading")
    raw_text = load_dataset(args.data_dir)
    _finish_timing("Corpus loading", data_load_started)
    dataset_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    bpe_encode, bpe_decode, vocab_size = load_or_build_bpe_tokenizer(
        raw_text, target_vocab_size=args.target_vocab_size
    )

    tokenization_started = _start_timing("Dataset tokenization")
    data = torch.tensor(bpe_encode(raw_text), dtype=torch.long)
    _finish_timing("Dataset tokenization", tokenization_started)
    split_index = int(0.9 * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]
    if min(len(train_data), len(val_data)) <= args.block_size:
        raise ValueError(
            "The training and validation splits must each contain more tokens "
            "than the configured block size"
        )
    training_text_started = _start_timing(
        "Training split decode for source-overlap checks"
    )
    training_text = bpe_decode(train_data.tolist())
    _finish_timing(
        "Training split decode for source-overlap checks", training_text_started
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
    start_iteration = 0
    if args.resume:
        start_iteration = load_checkpoint(
            model,
            optimizer,
            args,
            vocab_size,
            dataset_hash,
            device,
        )
    if args.max_iters < start_iteration:
        raise ValueError(
            f"--max-iters ({args.max_iters}) is less than the checkpoint's "
            f"completed iteration ({start_iteration})"
        )

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

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = _start_timing("Model training")
    final_losses = None
    print(
        f"Training to {args.max_iters} total iterations, starting after "
        f"{start_iteration}; reporting every {args.eval_interval} and saving "
        f"every {args.checkpoint_interval}; evaluating generated samples every "
        f"{args.sample_interval}."
    )
    for step in range(start_iteration + 1, args.max_iters + 1):
        inputs, targets = get_batch("train")
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_iters:
            losses = estimate_loss()
            final_losses = losses
            progress = 100 * step / args.max_iters
            print(
                f"Completed {step}/{args.max_iters} iterations ({progress:.1f}%) "
                f"| Train Loss: {losses['train']:.4f} "
                f"| Val Loss: {losses['val']:.4f}"
            )

        if (
            step % args.checkpoint_interval == 0
            or step % args.sample_interval == 0
            or step == args.max_iters
        ):
            save_checkpoint(
                model, optimizer, args, vocab_size, step, dataset_hash, device
            )

            if step % args.sample_interval == 0 or step == args.max_iters:
                evaluate_generations(
                    model,
                    bpe_encode,
                    bpe_decode,
                    training_text,
                    device,
                    max_new_tokens=args.sample_tokens,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _finish_timing("Model training", training_started)

    if final_losses is None:
        final_losses = estimate_loss()

    if args.max_iters == start_iteration:
        save_checkpoint(
            model,
            optimizer,
            args,
            vocab_size,
            args.max_iters,
            dataset_hash,
            device,
        )
        evaluate_generations(
            model,
            bpe_encode,
            bpe_decode,
            training_text,
            device,
            max_new_tokens=args.sample_tokens,
        )

    print_model_summary(
        model,
        args,
        vocab_size,
        len(data),
        args.max_iters,
        final_losses,
        device,
    )
    _finish_timing("Training run", run_started)
    return model


def print_model_summary(model, args, vocab_size, dataset_tokens, completed_iterations, losses, device):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_size_mb = parameter_count * 4 / (1024 ** 2)

    print("\n=== Trained Model Summary ===")
    print(f"Architecture       : Transformer language model")
    print(f"Vocabulary size    : {vocab_size} (target: {args.target_vocab_size})")
    print(f"Embedding dimension: {args.embedding_dim}")
    print(f"Attention heads    : {args.num_heads}")
    print(f"Context length     : {args.block_size}")
    print(f"Trainable params   : {parameter_count:,} (~{parameter_size_mb:.2f} MiB in float32)")
    print(f"Dataset tokens     : {dataset_tokens:,}")
    print(f"Training device    : {device}")
    print(f"Iterations complete: {completed_iterations}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Learning rate      : {args.learning_rate:g}")
    print(f"Final train loss   : {losses['train']:.4f}")
    print(f"Final validation loss: {losses['val']:.4f}")
    print(f"Checkpoint         : {args.checkpoint_path}")


if __name__ == "__main__":
    train(parse_args())
