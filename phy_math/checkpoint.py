import csv
import hashlib
from pathlib import Path

import torch


def build_checkpoint(
    model,
    optimizer,
    config,
    step,
    loss,
    validation_loss,
    subject,
    tokenizer_path,
    split_folders,
    keywords,
    max_seq_len,
    batch_size,
    learning_rate,
    device,
):
    """Build a resumable checkpoint, including compatible-run metadata."""
    return {
        "format_version": 1,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "validation_loss": validation_loss,
        "model_config": config.__dict__,
        "training_config": {
            "subject": subject,
            "tokenizer_sha256": file_sha256(tokenizer_path),
            "split_folders": split_folders,
            "keywords": keywords,
            "max_seq_len": max_seq_len,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else []
        ),
    }


def save_checkpoint(path, checkpoint):
    """Atomically save a checkpoint at the requested path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    except OSError as exc:
        raise OSError(f"Could not save checkpoint to {path}: {exc}") from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    config,
    subject,
    tokenizer_path,
    split_folders,
    keywords,
    device,
    max_seq_len,
    batch_size,
    learning_rate,
):
    """Restore a checkpoint after verifying it matches this training run."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc

    expected_training_config = {
        "subject": subject,
        "tokenizer_sha256": file_sha256(tokenizer_path),
        "split_folders": split_folders,
        "keywords": keywords,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported or non-resumable checkpoint format")
    if checkpoint.get("model_config") != config.__dict__:
        raise ValueError("Checkpoint model configuration does not match this run")
    if checkpoint.get("training_config") != expected_training_config:
        raise ValueError(
            "Checkpoint training settings or tokenizer do not match this run"
        )

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        cuda_rng_states = checkpoint.get("cuda_rng_states", [])
        if device.type == "cuda" and cuda_rng_states:
            if len(cuda_rng_states) != torch.cuda.device_count():
                raise ValueError("Checkpoint CUDA device count does not match this run")
            torch.cuda.set_rng_state_all(cuda_rng_states)
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Checkpoint state is incompatible: {exc}") from exc

    return (
        checkpoint["step"],
        checkpoint.get("loss"),
        checkpoint.get("validation_loss"),
    )


def find_latest_checkpoint(checkpoint_dir, subject):
    """Select the most recently written resumable checkpoint for a subject."""
    checkpoint_dir = Path(checkpoint_dir)
    candidates = list(
        checkpoint_dir.glob(f"checkpoint_{subject}_step_*.pt")
    )
    final_checkpoint = checkpoint_dir / f"final_llm_{subject}.pt"
    if final_checkpoint.is_file():
        candidates.append(final_checkpoint)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for subject '{subject}' in {checkpoint_dir}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def file_sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OSError(f"Could not read tokenizer file {path}: {exc}") from exc
    return digest.hexdigest()


def append_training_metric(path, row):
    fieldnames = [
        "run_id",
        "timestamp",
        "subject",
        "step",
        "event",
        "train_loss",
        "train_perplexity",
        "validation_loss",
        "step_duration_seconds",
    ]
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.is_file() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:
        raise OSError(f"Could not write training metrics to {path}: {exc}") from exc


def save_model_summary(
    path,
    model,
    config,
    subject,
    split_folders,
    tokenizer_path,
    device,
    completed_steps,
    batch_size,
    learning_rate,
    train_loss,
    validation_loss,
    checkpoint_path,
    started_at,
    completed_at,
    elapsed_seconds,
):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lines = [
        "=== Trained Model Summary ===",
        f"Subject            : {subject}",
        f"Data sources       : {', '.join(split_folders)}",
        f"Tokenizer          : {tokenizer_path}",
        f"Vocabulary size    : {config.vocab_size}",
        f"Embedding dimension: {config.dim}",
        f"Transformer layers : {config.n_layers}",
        f"Attention heads    : {config.n_heads}",
        f"Context length     : {config.max_seq_len}",
        f"Trainable params   : {parameter_count:,}",
        f"Training device    : {device}",
        f"Iterations complete: {completed_steps}",
        f"Batch size         : {batch_size}",
        f"Learning rate      : {learning_rate:g}",
        f"Final train loss   : {train_loss:.4f}",
        f"Final validation loss: {validation_loss:.4f}",
        f"Checkpoint         : {checkpoint_path}",
        f"Started at         : {started_at.isoformat(timespec='seconds')}",
        f"Completed at       : {completed_at.isoformat(timespec='seconds')}",
        f"Elapsed time       : {elapsed_seconds:.2f} seconds",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        separator = "\n" if path.is_file() and path.stat().st_size else ""
        with path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(separator + "\n".join(lines) + "\n")
    except OSError as exc:
        raise OSError(f"Could not save model summary to {path}: {exc}") from exc
