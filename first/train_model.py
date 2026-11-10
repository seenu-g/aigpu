from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from checkpoint import load_checkpoint, save_checkpoint
from evaluation import estimate_loss, evaluate_generations
from model import LanguageModel
from train_config import parse_args, save_settings_snapshot
from training_data import prepare_training_data
from training_stats import (
    plot_loss_curve,
    prepare_loss_history,
    record_losses,
    report_loss_diagnostics,
)


def _start_timing(label):
    started_at = datetime.now().astimezone()
    started_counter = perf_counter()
    print(f"{label} started at {started_at.isoformat(timespec='seconds')}")
    return started_counter


def _finish_timing(label, started_counter):
    finished_at = datetime.now().astimezone()
    elapsed = perf_counter() - started_counter
    print(f"{label} finished at {finished_at.isoformat(timespec='seconds')} ({elapsed:.2f}s)")


def train(settings):
    run_started = _start_timing("Training run")
    if settings.batch_size < 1 or settings.block_size < 1 or settings.max_iters < 1:
        raise ValueError("batch size, block size, and max iterations must be positive")
    if settings.eval_interval < 1:
        raise ValueError("evaluation interval must be positive")
    if settings.checkpoint_interval < 1:
        raise ValueError("checkpoint interval must be positive")
    if settings.sample_interval < 1 or settings.sample_tokens < 0:
        raise ValueError("sample interval must be positive and sample tokens non-negative")
    if settings.embedding_dim < 1 or settings.num_heads < 1:
        raise ValueError("embedding dimension and number of heads must be positive")
    if settings.embedding_dim % settings.num_heads:
        raise ValueError("embedding dimension must be divisible by the number of heads")
    if settings.target_vocab_size < 256:
        raise ValueError("target vocabulary size must be at least 256")
    if settings.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using architecture framework running on: {device}")

    training_data = prepare_training_data(
        settings.data_dir,
        settings.target_vocab_size,
        settings.batch_size,
        settings.block_size,
        device,
        train_fraction=settings.train_fraction,
    )

    model = LanguageModel(
        training_data.vocab_size,
        settings.embedding_dim,
        settings.num_heads,
        settings.block_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    start_iteration = 0
    if settings.resume:
        start_iteration = load_checkpoint(
            model,
            optimizer,
            settings,
            training_data.vocab_size,
            training_data.dataset_hash,
            device,
        )
    if settings.max_iters < start_iteration:
        raise ValueError(
            f"--max-iters ({settings.max_iters}) is less than the checkpoint's "
            f"completed iteration ({start_iteration})"
        )
    save_settings_snapshot(settings)
    last_loss_iteration = prepare_loss_history(
        settings.loss_history_path, settings.resume, start_iteration
    )
    if last_loss_iteration is not None:
        plot_loss_curve(settings.loss_history_path, settings.loss_plot_path)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = _start_timing("Model training")
    final_losses = None
    print(
        f"Training to {settings.max_iters} total iterations, starting after "
        f"{start_iteration}; reporting every {settings.eval_interval} and saving "
        f"every {settings.checkpoint_interval}; evaluating generated samples every "
        f"{settings.sample_interval}."
    )
    for step in range(start_iteration + 1, settings.max_iters + 1):
        inputs, targets = training_data.get_batch("train")
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % settings.eval_interval == 0 or step == settings.max_iters:
            losses = estimate_loss(model, training_data.get_batch)
            final_losses = losses
            last_loss_iteration = record_losses(
                settings.loss_history_path, step, losses, last_loss_iteration
            )
            plot_loss_curve(settings.loss_history_path, settings.loss_plot_path)
            progress = 100 * step / settings.max_iters
            print(
                f"Completed {step}/{settings.max_iters} iterations ({progress:.1f}%) "
                f"| Train Loss: {losses['train']:.4f} "
                f"| Val Loss: {losses['val']:.4f}"
            )

        if (
            step % settings.checkpoint_interval == 0
            or step % settings.sample_interval == 0
            or step == settings.max_iters
        ):
            save_checkpoint(
                model,
                optimizer,
                settings,
                training_data.vocab_size,
                step,
                training_data.dataset_hash,
                device,
            )

            if step % settings.sample_interval == 0 or step == settings.max_iters:
                evaluate_generations(
                    model,
                    training_data.encode,
                    training_data.decode,
                    training_data.training_text,
                    device,
                    max_new_tokens=settings.sample_tokens,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _finish_timing("Model training", training_started)

    if final_losses is None:
        final_losses = estimate_loss(model, training_data.get_batch)
        last_loss_iteration = record_losses(
            settings.loss_history_path,
            settings.max_iters,
            final_losses,
            last_loss_iteration,
        )
        plot_loss_curve(settings.loss_history_path, settings.loss_plot_path)

    if settings.max_iters == start_iteration:
        save_checkpoint(
            model,
            optimizer,
            settings,
            training_data.vocab_size,
            settings.max_iters,
            training_data.dataset_hash,
            device,
        )
        evaluate_generations(
            model,
            training_data.encode,
            training_data.decode,
            training_data.training_text,
            device,
            max_new_tokens=settings.sample_tokens,
        )

    print_model_summary(
        model,
        settings,
        training_data.vocab_size,
        training_data.token_count,
        settings.max_iters,
        final_losses,
        device,
        perf_counter() - run_started,
    )
    report_loss_diagnostics(settings.loss_history_path)
    return model


def print_model_summary(
    model,
    settings,
    vocab_size,
    dataset_tokens,
    completed_iterations,
    losses,
    device,
    run_duration_seconds,
):
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_size_mb = parameter_count * 4 / (1024 ** 2)
    summary_lines = [
        "\n=== Trained Model Summary ===",
        f"Architecture       : Transformer language model",
        f"Vocabulary size    : {vocab_size} (target: {settings.target_vocab_size})",
        f"Embedding dimension: {settings.embedding_dim}",
        f"Attention heads    : {settings.num_heads}",
        f"Context length     : {settings.block_size}",
        f"Trainable params   : {parameter_count:,} (~{parameter_size_mb:.2f} MiB in float32)",
        f"Dataset tokens     : {dataset_tokens:,}",
        f"Training device    : {device}",
        f"Iterations complete: {completed_iterations}",
        f"Batch size         : {settings.batch_size}",
        f"Learning rate      : {settings.learning_rate:g}",
        f"Final train loss   : {losses['train']:.4f}",
        f"Final validation loss: {losses['val']:.4f}",
        f"Checkpoint         : {settings.checkpoint_path}",
        f"Run time           : {run_duration_seconds:.2f} seconds",
    ]

    summary_text = "\n".join(summary_lines) + "\n"
    for line in summary_lines:
        print(line)

    summary_path = Path(__file__).resolve().parent / "model_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    train(parse_args())
