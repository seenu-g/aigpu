import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

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


def run_pretraining(subject="math"):
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
    max_steps = 10000
    save_interval = 1000
    eval_interval = 500
    validation_batches = 20

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
    model.train()
    completed_steps = 0
    last_loss = None
    last_validation_loss = None
    last_validation_step = 0
    print(" -> Commencing pre-training loop execution blocks...")

    for x, y in train_loader:
        if completed_steps >= max_steps:
            break

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        completed_steps += 1
        last_loss = loss.item()

        if completed_steps == 1 or completed_steps % 50 == 0:
            perplexity = math.exp(last_loss) if last_loss < 100 else float("inf")
            print(
                f"Step {completed_steps:5d} | Loss: {last_loss:.4f} "
                f"| Perplexity: {perplexity:.2f}"
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

        if completed_steps % save_interval == 0:
            checkpoint_path = project_dir / "checkpoints" / (
                f"checkpoint_{subject}_step_{completed_steps}.pt"
            )
            _save_checkpoint(
                checkpoint_path,
                _checkpoint_state(
                    model,
                    optimizer,
                    config,
                    completed_steps,
                    last_loss,
                    last_validation_loss,
                ),
            )
            print(f" [*] Checkpoint saved: {checkpoint_path}")

    if completed_steps == 0:
        raise RuntimeError("Training dataset produced no batches; no model was trained.")

    if last_validation_step != completed_steps:
        last_validation_loss = _estimate_validation_loss(
            model, validation_loader, device, validation_batches
        )
        print(
            f"Final validation at step {completed_steps}: "
            f"loss {last_validation_loss:.4f}"
        )

    final_path = project_dir / f"final_llm_{subject}.pt"
    _save_checkpoint(
        final_path,
        _checkpoint_state(
            model,
            optimizer,
            config,
            completed_steps,
            last_loss,
            last_validation_loss,
        ),
    )
    print(
        f" === Training finished after {completed_steps} steps. "
        f"Checkpoint saved to: {final_path} ==="
    )


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


def _checkpoint_state(
    model, optimizer, config, step, loss, validation_loss
):
    return {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "validation_loss": validation_loss,
        "model_config": config.__dict__,
    }


def _save_checkpoint(path, checkpoint):
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


if __name__ == "__main__":
    try:
        run_pretraining(subject="math")
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Pre-training failed: {exc}") from exc
