import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def save_probability_report(
    output_dir,
    prompt,
    temperature,
    top_k,
    seed,
    decode,
    probability_history,
):
    if not probability_history:
        print("No generated tokens; probability report was not created.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    report_name = f"token_generation_probabilities_{run_timestamp}"
    csv_path = output_dir / f"{report_name}.csv"
    plot_path = output_dir / f"{report_name}.png"
    generated_prefixes = []
    prefix_ids = []
    for item in probability_history:
        prefix_ids.append(item["token_id"])
        generated_prefixes.append(decode(prefix_ids))

    with csv_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.writer(report_file)
        writer.writerow(
            (
                "prompt",
                "temperature",
                "top_k",
                "seed",
                "step",
                "token_id",
                "generated_prefix",
                "model_probability",
                "sampling_probability",
                "top_model_probability",
                "entropy_bits",
            )
        )
        for step, (item, prefix) in enumerate(
            zip(probability_history, generated_prefixes), start=1
        ):
            writer.writerow(
                (
                    prompt,
                    temperature,
                    top_k if top_k is not None else "",
                    seed,
                    step,
                    item["token_id"],
                    prefix,
                    item["model_probability"],
                    item["sampling_probability"],
                    item["top_model_probability"],
                    item["entropy_bits"],
                )
            )

    steps = list(range(1, len(probability_history) + 1))
    plt.figure(figsize=(11, 8))
    probability_axis = plt.subplot(2, 1, 1)
    probability_axis.plot(
        steps,
        [item["model_probability"] for item in probability_history],
        marker=".",
        label="Generated token: model probability",
    )
    probability_axis.plot(
        steps,
        [item["sampling_probability"] for item in probability_history],
        marker=".",
        label="Generated token: sampling probability",
    )
    probability_axis.plot(
        steps,
        [item["top_model_probability"] for item in probability_history],
        linestyle="--",
        label="Most likely token: model probability",
    )
    probability_axis.set_ylabel("Probability")
    probability_axis.set_ylim(0, 1)
    probability_axis.grid(alpha=0.3)
    probability_axis.legend()

    entropy_axis = plt.subplot(2, 1, 2)
    entropy_axis.plot(
        steps,
        [item["entropy_bits"] for item in probability_history],
        color="tab:purple",
    )
    entropy_axis.set_xlabel("Generated token step")
    entropy_axis.set_ylabel("Model entropy (bits)")
    entropy_axis.grid(alpha=0.3)

    sampling_description = f"temperature={temperature}, top_k={top_k or 'all'}"
    plt.suptitle(
        "Token Generation Probabilities (Inference Output)\n"
        f"{sampling_description}"
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Probability data saved to: {csv_path.resolve()}")
    print(f"Probability graph saved to: {plot_path.resolve()}")
