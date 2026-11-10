import argparse
import csv
import math
from pathlib import Path
from statistics import mean, median


def read_training_metrics(path, subject):
    path = Path(path)
    required_fields = {
        "run_id",
        "subject",
        "step",
        "event",
        "train_loss",
        "validation_loss",
    }
    runs = {}
    with path.open("r", encoding="utf-8", newline="") as metrics_file:
        reader = csv.DictReader(metrics_file)
        if reader.fieldnames is None or not required_fields.issubset(
            reader.fieldnames
        ):
            raise ValueError(f"Invalid training metrics CSV header in '{path}'")

        for row_number, row in enumerate(reader, start=2):
            if row["subject"] != subject:
                continue
            event = row["event"]
            if event not in {"training", "validation"}:
                continue
            try:
                step = int(row["step"])
                value_field = (
                    "train_loss" if event == "training" else "validation_loss"
                )
                loss = float(row[value_field])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {event} metric on row {row_number} in '{path}'"
                ) from error
            run_id = row["run_id"]
            if step < 1 or not math.isfinite(loss) or not run_id:
                raise ValueError(
                    f"Invalid {event} metric on row {row_number} in '{path}'"
                )

            run = runs.setdefault(
                run_id,
                {"training": {}, "validation": {}},
            )
            run[event][step] = loss

    if not runs:
        raise ValueError(f"No {subject} loss measurements found in '{path}'")
    return runs


def paired_loss_history(runs):
    history_by_step = {}
    for run in runs.values():
        if not run["training"]:
            continue
        paired = [
            (step, run["training"][step], validation_loss)
            for step, validation_loss in sorted(run["validation"].items())
            if step in run["training"]
        ]
        if not paired:
            continue
        first_step = min(run["training"])
        if history_by_step and first_step <= max(history_by_step):
            history_by_step = {
                step: losses
                for step, losses in history_by_step.items()
                if step < first_step
            }
        for step, train_loss, validation_loss in paired:
            history_by_step[step] = (train_loss, validation_loss)
    return [
        (step, losses[0], losses[1])
        for step, losses in sorted(history_by_step.items())
    ]


def analyze_loss_history(history, window_size=10):
    if not history:
        raise ValueError("No matching training and validation losses to analyze")
    if window_size < 1:
        raise ValueError("window size must be positive")

    best = min(history, key=lambda row: row[2])
    latest = history[-1]
    gap = latest[2] - latest[1]
    gap_percent = 100 * gap / latest[2] if latest[2] else 0
    comparison_size = min(window_size, len(history) // 2)
    if comparison_size < 2:
        return {
            "best": best,
            "latest": latest,
            "gap": gap,
            "gap_percent": gap_percent,
            "comparison_size": comparison_size,
            "signal": "Not enough loss measurements yet to assess recent trends.",
        }

    previous = history[-2 * comparison_size : -comparison_size]
    recent = history[-comparison_size:]
    previous_train = sum(row[1] for row in previous) / comparison_size
    recent_train = sum(row[1] for row in recent) / comparison_size
    previous_val = sum(row[2] for row in previous) / comparison_size
    recent_val = sum(row[2] for row in recent) / comparison_size
    train_change = (
        100 * (recent_train - previous_train) / max(abs(previous_train), 1e-12)
    )
    val_change = (
        100 * (recent_val - previous_val) / max(abs(previous_val), 1e-12)
    )
    recent_gap_percent = (
        100 * (recent_val - recent_train) / recent_val if recent_val else 0
    )

    if train_change <= -1 and val_change >= 1:
        signal = (
            "Possible overfitting (training loss is falling while "
            "validation loss is rising)."
        )
    elif train_change <= -1 and val_change <= -1:
        signal = (
            "Still learning (both training and validation losses are decreasing)."
        )
    elif (
        abs(train_change) < 1
        and abs(val_change) < 1
        and abs(recent_gap_percent) < 5
    ):
        signal = (
            "Losses are stable and close, consistent with a possible "
            "converged fit. Whether the fit is good depends on the task; "
            "loss history alone cannot establish that."
        )
    elif train_change >= 1 and val_change >= 1:
        signal = (
            "Both losses are rising; check for training instability "
            "or a data/evaluation issue."
        )
    elif abs(train_change) < 1 and abs(val_change) < 1:
        signal = (
            "Losses have plateaued but remain separated; check the "
            "validation trend for possible overfitting."
        )
    else:
        signal = (
            "Mixed loss trend; loss-based signals are heuristic, not definitive."
        )

    return {
        "best": best,
        "latest": latest,
        "gap": gap,
        "gap_percent": gap_percent,
        "comparison_size": comparison_size,
        "previous_train": previous_train,
        "recent_train": recent_train,
        "previous_val": previous_val,
        "recent_val": recent_val,
        "train_change": train_change,
        "val_change": val_change,
        "signal": signal,
    }


def report_loss_diagnostics(path, subject):
    history = paired_loss_history(read_training_metrics(path, subject))
    analysis = analyze_loss_history(history)
    best = analysis["best"]
    latest = analysis["latest"]

    print("\n=== Loss Trend Diagnostics ===")
    print(
        f"Best validation loss: {best[2]:.4f} at step {best[0]:,} "
        f"(latest: {latest[2]:.4f} at {latest[0]:,})"
    )
    print(
        f"Latest train/validation gap: {analysis['gap']:.4f} "
        f"({analysis['gap_percent']:.1f}% of validation loss)"
    )
    if analysis["comparison_size"] < 2:
        print(analysis["signal"])
        return

    print(
        f"Recent averages ({analysis['comparison_size']} evaluations each): "
        f"train {analysis['previous_train']:.4f} -> "
        f"{analysis['recent_train']:.4f} ({analysis['train_change']:+.1f}%), "
        f"validation {analysis['previous_val']:.4f} -> "
        f"{analysis['recent_val']:.4f} ({analysis['val_change']:+.1f}%)"
    )
    print(f"Signal: {analysis['signal']}")


def plot_loss_curve(runs, output_path):
    import matplotlib.pyplot as plt

    plotted = False
    plt.figure(figsize=(10, 6))
    for run_number, run in enumerate(runs.values(), start=1):
        if run["training"]:
            steps, losses = zip(*sorted(run["training"].items()))
            plt.plot(steps, losses, linewidth=1, label=f"Training (run {run_number})")
            plotted = True
        if run["validation"]:
            steps, losses = zip(*sorted(run["validation"].items()))
            plt.plot(
                steps,
                losses,
                marker="o",
                linestyle="none",
                label=f"Validation (run {run_number})",
            )
            plotted = True

    if not plotted:
        plt.close()
        raise ValueError("No loss measurements to plot")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.title("Training and Validation Loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to: {output_path.resolve()}")


def collect_sequence_lengths(data_dir, tokenizer_path, split_folders, keywords):
    from tokenizers import Tokenizer

    from load_data import get_parquet_files, read_parquet_file_data

    files = get_parquet_files(data_dir, split_folders)
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not load tokenizer from {tokenizer_path}: {error}") from error

    sequence_lengths = [
        len(tokenizer.encode(text).ids)
        for text in read_parquet_file_data(files, keywords=keywords, split="all")
    ]
    return sequence_lengths, tokenizer.get_vocab_size()


def plot_sequence_lengths(sequence_lengths, output_path, bins, subject, vocab_size):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=bins, edgecolor="black")
    plt.title(
        f"{subject.title()} Document Token Lengths "
        f"(BPE vocabulary: {vocab_size:,})"
    )
    plt.xlabel("Document length (tokens)")
    plt.ylabel("Number of documents")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot dataset token lengths and training loss from the metrics CSV."
    )
    parser.add_argument("--subject", choices=("math", "physics"), default="math")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Cosmopedia data directory (defaults to the path configured in train.py).",
    )
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loss-history", type=Path)
    parser.add_argument("--loss-output", type=Path)
    parser.add_argument(
        "--analyze-loss-only",
        action="store_true",
        help="Print loss-trend diagnostics without reading the dataset or making plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parent
    metrics_path = args.loss_history or (
        project_dir / "checkpoints" / f"training_metrics_{args.subject}.csv"
    )
    loss_output = args.loss_output or (
        project_dir / "output" / f"training_validation_loss_{args.subject}.png"
    )
    if args.analyze_loss_only:
        report_loss_diagnostics(metrics_path, args.subject)
        return

    if args.bins < 1:
        raise ValueError("number of histogram bins must be positive")
    from train import get_subject_data_config

    data_dir, split_folders, tokenizer_path, keywords = get_subject_data_config(
        args.subject, project_dir
    )
    if args.data_dir is not None:
        data_dir = args.data_dir
    sequence_lengths, vocab_size = collect_sequence_lengths(
        data_dir,
        tokenizer_path,
        split_folders,
        keywords,
    )
    if not sequence_lengths:
        raise ValueError("No documents found for the selected subject")

    output_path = args.output or (
        project_dir / "output" / f"sequence_length_histogram_{args.subject}.png"
    )
    plot_sequence_lengths(
        sequence_lengths,
        output_path,
        args.bins,
        args.subject,
        vocab_size,
    )
    print(f"Documents: {len(sequence_lengths):,}")
    print(
        f"Token lengths: min={min(sequence_lengths):,}, "
        f"median={median(sequence_lengths):,.0f}, "
        f"mean={mean(sequence_lengths):,.0f}, "
        f"max={max(sequence_lengths):,}"
    )
    print(f"Histogram saved to: {Path(output_path).resolve()}")

    if metrics_path.is_file():
        runs = read_training_metrics(metrics_path, args.subject)
        plot_loss_curve(runs, loss_output)
    else:
        print(
            f"Loss history not found at '{metrics_path}'; "
            "train the model to generate it. No loss curve was created."
        )


if __name__ == "__main__":
    main()
