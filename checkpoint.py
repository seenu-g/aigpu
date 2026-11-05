from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch


def _start_timing(label):
    started_at = datetime.now().astimezone()
    started_counter = perf_counter()
    print(f"{label} started at {started_at.isoformat(timespec='seconds')}")
    return started_counter


def _finish_timing(label, started_counter):
    finished_at = datetime.now().astimezone()
    elapsed = perf_counter() - started_counter
    print(f"{label} finished at {finished_at.isoformat(timespec='seconds')} ({elapsed:.2f}s)")


def save_checkpoint(
    model, optimizer, args, vocab_size, completed_iterations, dataset_hash, device
):
    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_started = _start_timing("Model checkpoint save")
    checkpoint = {
        "format_version": 2,
        "model_state_dict": {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_iterations": completed_iterations,
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "model_config": {
            "vocab_size": vocab_size,
            "embedding_dim": args.embedding_dim,
            "num_heads": args.num_heads,
            "block_size": args.block_size,
        },
        "training_config": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "target_vocab_size": args.target_vocab_size,
            "train_fraction": args.train_fraction,
        },
        "dataset_sha256": dataset_hash,
    }
    temporary_path = args.checkpoint_path.with_suffix(
        args.checkpoint_path.suffix + ".tmp"
    )
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(args.checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    _finish_timing("Model checkpoint save", checkpoint_started)
    print(f"Saved checkpoint at iteration {completed_iterations} to {args.checkpoint_path}")


def load_checkpoint(model, optimizer, args, vocab_size, dataset_hash, device):
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    checkpoint_started = _start_timing("Model checkpoint load")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError(
            "This checkpoint does not contain resumable optimizer and iteration "
            "state; start a new run to create a resumable checkpoint."
        )

    expected_model_config = {
        "vocab_size": vocab_size,
        "embedding_dim": args.embedding_dim,
        "num_heads": args.num_heads,
        "block_size": args.block_size,
    }
    if checkpoint.get("model_config") != expected_model_config:
        raise ValueError("Checkpoint model configuration does not match this run")

    expected_training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "target_vocab_size": args.target_vocab_size,
        "train_fraction": args.train_fraction,
    }
    checkpoint_training_config = checkpoint.get("training_config")
    if checkpoint_training_config == {
        key: value
        for key, value in expected_training_config.items()
        if key != "train_fraction"
    } and args.train_fraction == 0.9:
        checkpoint_training_config = expected_training_config
    if checkpoint_training_config != expected_training_config:
        raise ValueError("Checkpoint training configuration does not match this run")
    if checkpoint.get("dataset_sha256") != dataset_hash:
        raise ValueError("Checkpoint was created with a different training corpus")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    cuda_rng_states = checkpoint.get("cuda_rng_states", [])
    if device.type == "cuda" and cuda_rng_states:
        if len(cuda_rng_states) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA device count does not match this run")
        torch.cuda.set_rng_state_all(cuda_rng_states)

    completed_iterations = checkpoint["completed_iterations"]
    _finish_timing("Model checkpoint load", checkpoint_started)
    print(f"Resumed checkpoint from iteration {completed_iterations}")
    return completed_iterations
