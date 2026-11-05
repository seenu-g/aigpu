import argparse
import csv
import math
from itertools import chain
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt

from tokenizers import load_or_build_bpe_tokenizer


def prepare_loss_history(path, resume, start_iteration):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not resume or not path.exists():
        with path.open("w", encoding="utf-8", newline="") as history_file:
            csv.writer(history_file).writerow(
                ("iteration", "train_loss", "val_loss")
            )
        return None

    last_iteration = None
    with path.open("r", encoding="utf-8", newline="") as history_file:
        reader = csv.DictReader(history_file)
        if reader.fieldnames != ["iteration", "train_loss", "val_loss"]:
            raise ValueError(f"Invalid loss history CSV header in '{path}'")
        for row_number, row in enumerate(reader, start=2):
            try:
                iteration = int(row["iteration"])
                train_loss = float(row["train_loss"])
                val_loss = float(row["val_loss"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid loss history values on row {row_number} in '{path}'"
                ) from error
            if (
                iteration < 1
                or not math.isfinite(train_loss)
                or not math.isfinite(val_loss)
                or (last_iteration is not None and iteration <= last_iteration)
            ):
                raise ValueError(
                    f"Invalid or out-of-order loss history on row {row_number} in '{path}'"
                )
            last_iteration = iteration

    if last_iteration is not None and last_iteration > start_iteration:
        raise ValueError(
            f"Loss history in '{path}' extends beyond checkpoint iteration "
            f"{start_iteration}; use a matching loss history file."
        )
    return last_iteration


def record_losses(path, iteration, losses, last_iteration):
    path = Path(path)
    if last_iteration == iteration:
        return last_iteration
    if last_iteration is not None and iteration < last_iteration:
        raise ValueError(
            f"Cannot append iteration {iteration} after loss history iteration "
            f"{last_iteration} in '{path}'"
        )
    with path.open("a", encoding="utf-8", newline="") as history_file:
        csv.writer(history_file).writerow(
            (iteration, losses["train"], losses["val"])
        )
    return iteration


def report_loss_diagnostics(path, window_size=10):
    path = Path(path)
    history = []
    with path.open("r", encoding="utf-8", newline="") as history_file:
        reader = csv.reader(history_file)
        first_row = next(reader, None)
        if first_row == ["iteration", "train_loss", "val_loss"]:
            first_row = None
            first_row_number = 2
        else:
            first_row_number = 1
        rows = reader if first_row is None else chain([first_row], reader)
        for row_number, row in enumerate(rows, start=first_row_number):
            try:
                if len(row) != 3:
                    raise ValueError
                iteration = int(row[0])
                train_loss = float(row[1])
                val_loss = float(row[2])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid loss history values on row {row_number} in '{path}'"
                ) from error
            if (
                iteration < 1
                or not math.isfinite(train_loss)
                or not math.isfinite(val_loss)
                or (history and iteration <= history[-1][0])
            ):
                raise ValueError(
                    f"Invalid or out-of-order loss history on row {row_number} "
                    f"in '{path}'"
                )
            history.append((iteration, train_loss, val_loss))

    if not history:
        raise ValueError(f"No loss measurements found in '{path}'")

    best = min(history, key=lambda row: row[2])
    latest = history[-1]
    gap = latest[2] - latest[1]
    gap_percent = 100 * gap / latest[2] if latest[2] else 0

    print("\n=== Loss Trend Diagnostics ===")
    print(
        f"Best validation loss: {best[2]:.4f} at iteration {best[0]:,} "
        f"(latest: {latest[2]:.4f} at {latest[0]:,})"
    )
    print(f"Latest train/validation gap: {gap:.4f} ({gap_percent:.1f}% of validation loss)")

    comparison_size = min(window_size, len(history) // 2)
    if comparison_size < 2:
        print("Not enough loss measurements yet to assess recent trends.")
        return

    previous = history[-2 * comparison_size:-comparison_size]
    recent = history[-comparison_size:]
    previous_train = sum(row[1] for row in previous) / comparison_size
    recent_train = sum(row[1] for row in recent) / comparison_size
    previous_val = sum(row[2] for row in previous) / comparison_size
    recent_val = sum(row[2] for row in recent) / comparison_size
    train_change = 100 * (recent_train - previous_train) / max(abs(previous_train), 1e-12)
    val_change = 100 * (recent_val - previous_val) / max(abs(previous_val), 1e-12)
    recent_gap_percent = (
        100 * (recent_val - recent_train) / recent_val if recent_val else 0
    )

    print(
        f"Recent averages ({comparison_size} evaluations each): "
        f"train {previous_train:.4f} -> {recent_train:.4f} ({train_change:+.1f}%), "
        f"validation {previous_val:.4f} -> {recent_val:.4f} ({val_change:+.1f}%)"
    )
    if train_change <= -1 and val_change >= 1:
        print(
            "Signal: possible overfitting (training loss is falling while "
            "validation loss is rising)."
        )
    elif train_change <= -1 and val_change <= -1:
        print(
            "Signal: still learning (both training and validation losses are "
            "decreasing)."
        )
    elif (
        abs(train_change) < 1
        and abs(val_change) < 1
        and abs(recent_gap_percent) < 5
    ):
        print(
            "Signal: losses are stable and close, consistent with a possible "
            "converged fit. Whether the fit is good depends on the task; loss "
            "history alone cannot establish that."
        )
    elif train_change >= 1 and val_change >= 1:
        print(
            "Signal: both losses are rising; check for training instability "
            "or a data/evaluation issue."
        )
    elif abs(train_change) < 1 and abs(val_change) < 1:
        print(
            "Signal: losses have plateaued but remain separated; check the "
            "validation trend for possible overfitting."
        )
    else:
        print(
            "Signal: mixed loss trend; no clear diagnosis from these windows. "
            "Loss-based signals are heuristic, not definitive."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot dataset token lengths and training versus validation loss."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "literature",
        help="Folder containing the dataset's .txt files.",
    )
    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=512,
        help="BPE vocabulary size used to tokenize the dataset (default: 512).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="Number of histogram bins (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "sequence_length_histogram.png",
        help="Path to save the histogram image.",
    )
    parser.add_argument(
        "--loss-history",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "training_loss.csv",
        help="CSV loss history written by train_model.py.",
    )
    parser.add_argument(
        "--loss-output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "training_validation_loss.png",
        help="Path to save the training versus validation loss curve.",
    )
    parser.add_argument(
        "--analyze-loss-only",
        action="store_true",
        help="Print loss-trend diagnostics without reading the dataset or making plots.",
    )
    return parser.parse_args()


def plot_loss_curve(history_path, output_path):
    history_path = Path(history_path)
    output_path = Path(output_path)
    iterations = []
    train_losses = []
    val_losses = []
    with history_path.open("r", encoding="utf-8", newline="") as history_file:
        reader = csv.DictReader(history_file)
        if reader.fieldnames != ["iteration", "train_loss", "val_loss"]:
            raise ValueError(f"Invalid loss history CSV header in '{history_path}'")
        for row_number, row in enumerate(reader, start=2):
            try:
                iteration = int(row["iteration"])
                train_loss = float(row["train_loss"])
                val_loss = float(row["val_loss"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid loss history values on row {row_number} in '{history_path}'"
                ) from error
            if (
                iteration < 1
                or not math.isfinite(train_loss)
                or not math.isfinite(val_loss)
                or (iterations and iteration <= iterations[-1])
            ):
                raise ValueError(
                    f"Invalid or out-of-order loss history on row {row_number} "
                    f"in '{history_path}'"
                )
            iterations.append(iteration)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    if not iterations:
        raise ValueError(f"No loss measurements found in '{history_path}'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, marker="o", label="Training loss")
    plt.plot(iterations, val_losses, marker="o", label="Validation loss")
    plt.title("LLM Training vs Validation Loss Curve")
    plt.xlabel("Training iteration")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {output_path.resolve()}")


def main():
    args = parse_args()
    if args.analyze_loss_only:
        report_loss_diagnostics(args.loss_history)
        return

    if args.target_vocab_size < 256:
        raise ValueError("target vocabulary size must be at least 256")
    if args.bins < 1:
        raise ValueError("number of histogram bins must be positive")

    files = sorted(args.data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt dataset files found in '{args.data_dir}'.")

    documents = [path.read_text(encoding="utf-8") for path in files]
    raw_text = "\n".join(documents)
    encode, _, _ = load_or_build_bpe_tokenizer(
        raw_text, target_vocab_size=args.target_vocab_size
    )
    sequence_lengths = [len(encode(document)) for document in documents]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=args.bins, edgecolor="black")
    plt.title(
        f"Sequence Lengths Across Dataset (BPE vocabulary: {args.target_vocab_size})"
    )
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Number of text files")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()

    print(f"Dataset files: {len(sequence_lengths)}")
    print(
        f"Token lengths: min={min(sequence_lengths):,}, "
        f"median={median(sequence_lengths):,.0f}, "
        f"mean={mean(sequence_lengths):,.0f}, "
        f"max={max(sequence_lengths):,}"
    )
    print(f"Histogram saved to: {args.output.resolve()}")
    if args.loss_history.is_file():
        plot_loss_curve(args.loss_history, args.loss_output)
    else:
        print(
            f"Loss history not found at '{args.loss_history}'; run train_model.py "
            "to generate it. No loss curve was created."
        )


if __name__ == "__main__":
    main()
