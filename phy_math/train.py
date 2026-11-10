import argparse
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch
from tokenizers import Tokenizer

from checkpoint import (
    append_training_metric,
    build_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_model_summary,
)
from load_data import TrainingDataset
from model import CustomLLM, ModelConfig


def get_subject_data_config(subject, project_dir):
    data_dir = Path("/home/guru/data/cosmopedia_raw/data")

    if subject == "math":
        return (
            data_dir,
            ["auto_math_text"],
            project_dir / "math_tokenizer.json",
            None,
        )
    elif subject == "physics":
        return (
            data_dir,
            ["stanford", "openstax", "khanacademy"],
            project_dir / "physics_tokenizer.json",
            [
                "physics",
                "quantum",
                "mechanics",
                "thermodynamics",
                "electromagnetism",
                "kinematics",
                "relativity",
            ],
        )
    raise ValueError("subject must be either 'math' or 'physics'")


def run_pretraining(subject="math", max_steps=10000, resume_from=None):
    run_started = perf_counter()
    run_started_at = datetime.now().astimezone()
    print(f"\n=== Initializing Pre-training Loop for Domain: [{subject.upper()}] ===")

    project_dir = Path(__file__).resolve().parent
    data_dir, split_folders, tokenizer_path, keywords = get_subject_data_config(
        subject, project_dir
    )
    print(f" -> Data sources: {', '.join(split_folders)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Computing Unit Target: {device}")

    max_seq_len = 1024
    batch_size = 8
    learning_rate = 3e-4
    save_interval = 1000
    eval_interval = 500
    validation_batches = 20
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    train_dataset = TrainingDataset(
        data_dir=data_dir,
        split_folders=split_folders,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        keywords=keywords,
        split="train",
    )
    validation_dataset = TrainingDataset(
        data_dir=data_dir,
        split_folders=split_folders,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        keywords=keywords,
        split="validation",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
    )

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load tokenizer from {tokenizer_path}: {exc}") from exc

    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=max_seq_len,
        dim=512,
        n_layers=8,
        n_heads=8,
    )
    model = CustomLLM(config).to(device)
    print(
        f" -> Model structure constructed. Total weights: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    completed_steps = 0
    last_loss = None
    last_validation_loss = None
    if resume_from is not None:
        completed_steps, last_loss, last_validation_loss = load_checkpoint(
            resume_from,
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
        )
        if completed_steps > max_steps:
            raise ValueError(
                f"Checkpoint step {completed_steps} exceeds requested max_steps "
                f"{max_steps}."
            )
        print(f" -> Resuming from step {completed_steps}: {resume_from}")

    model.train()
    print(" -> Commencing pre-training loop execution blocks...")
    metrics_path = project_dir / "checkpoints" / f"training_metrics_{subject}.csv"
    run_id = run_started_at.isoformat(timespec="microseconds")

    train_iterator = iter(train_loader)
    for _ in range(completed_steps):
        _, train_iterator = _next_training_batch(train_loader, train_iterator)

    last_validation_step = completed_steps
    while completed_steps < max_steps:
        (x, y), train_iterator = _next_training_batch(
            train_loader, train_iterator
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_started = perf_counter()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_duration = perf_counter() - step_started

        completed_steps += 1
        last_loss = loss.item()

        perplexity = math.exp(last_loss) if last_loss < 100 else float("inf")
        print(
            f"Step {completed_steps}/{max_steps} | Loss: {last_loss:.4f} "
            f"| Perplexity: {perplexity:.2f} "
            f"| Train step time: {step_duration:.3f}s"
        )
        append_training_metric(
            metrics_path,
            {
                "run_id": run_id,
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="microseconds"
                ),
                "subject": subject,
                "step": completed_steps,
                "event": "training",
                "train_loss": last_loss,
                "train_perplexity": perplexity,
                "validation_loss": "",
                "step_duration_seconds": step_duration,
            },
        )

        if completed_steps % eval_interval == 0:
            last_validation_loss = _estimate_validation_loss(
                model, validation_loader, device, validation_batches
            )
            last_validation_step = completed_steps
            print(
                f"Validation at step {completed_steps}: "
                f"loss {last_validation_loss:.4f}"
            )
            append_training_metric(
                metrics_path,
                {
                    "run_id": run_id,
                    "timestamp": datetime.now().astimezone().isoformat(
                        timespec="microseconds"
                    ),
                    "subject": subject,
                    "step": completed_steps,
                    "event": "validation",
                    "train_loss": "",
                    "train_perplexity": "",
                    "validation_loss": last_validation_loss,
                    "step_duration_seconds": "",
                },
            )

        if completed_steps % save_interval == 0:
            checkpoint_path = project_dir / "checkpoints" / (
                f"checkpoint_{subject}_step_{completed_steps}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                build_checkpoint(
                    model,
                    optimizer,
                    config,
                    completed_steps,
                    last_loss,
                    last_validation_loss,
                    subject,
                    tokenizer_path,
                    split_folders,
                    keywords,
                    max_seq_len,
                    batch_size,
                    learning_rate,
                    device,
                ),
            )
            print(f" [*] Checkpoint saved: {checkpoint_path}")

    if last_validation_step != completed_steps:
        last_validation_loss = _estimate_validation_loss(
            model, validation_loader, device, validation_batches
        )
        print(
            f"Final validation at step {completed_steps}: "
            f"loss {last_validation_loss:.4f}"
        )
        append_training_metric(
            metrics_path,
            {
                "run_id": run_id,
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="microseconds"
                ),
                "subject": subject,
                "step": completed_steps,
                "event": "validation",
                "train_loss": "",
                "train_perplexity": "",
                "validation_loss": last_validation_loss,
                "step_duration_seconds": "",
            },
        )

    final_path = project_dir / "checkpoints" / f"final_llm_{subject}.pt"
    save_checkpoint(
        final_path,
        build_checkpoint(
            model,
            optimizer,
            config,
            completed_steps,
            last_loss,
            last_validation_loss,
            subject,
            tokenizer_path,
            split_folders,
            keywords,
            max_seq_len,
            batch_size,
            learning_rate,
            device,
        ),
    )
    print(
        f" === Training finished after {completed_steps} steps. "
        f"Checkpoint saved to: {final_path} ==="
    )
    summary_path = project_dir / "checkpoints" / "model_summary.txt"
    run_completed_at = datetime.now().astimezone()
    save_model_summary(
        summary_path,
        model,
        config,
        subject,
        split_folders,
        tokenizer_path,
        device,
        completed_steps,
        batch_size,
        learning_rate,
        last_loss,
        last_validation_loss,
        final_path,
        run_started_at,
        run_completed_at,
        perf_counter() - run_started,
    )
    print(f"Model summary saved to: {summary_path}"    )


@torch.no_grad()
def _estimate_validation_loss(model, validation_loader, device, max_batches):
    was_training = model.training
    model.eval()
    losses = []
    try:
        for batch_index, (inputs, targets) in enumerate(validation_loader):
            if batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            losses.append(loss.item())
    finally:
        model.train(was_training)

    if not losses:
        raise RuntimeError(
            "Validation dataset produced no batches; check its content and split."
        )
    return sum(losses) / len(losses)


def _next_training_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        try:
            return next(iterator), iterator
        except StopIteration as exc:
            raise RuntimeError("Training dataset produced no batches.") from exc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train the math or physics model.")
    parser.add_argument("--subject", choices=("math", "physics"), default="math")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint for the selected subject.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from a specific checkpoint file.",
    )
    args = parser.parse_args()

    try:
        if args.resume and args.resume_from is not None:
            parser.error("--resume and --resume-from cannot be used together")

        resume_from = args.resume_from
        if args.resume:
            project_dir = Path(__file__).resolve().parent
            resume_from = find_latest_checkpoint(
                project_dir / "checkpoints", args.subject
            )

        run_pretraining(
            subject=args.subject,
            max_steps=args.max_steps,
            resume_from=resume_from,
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Pre-training failed: {exc}") from exc
