import argparse
from datetime import datetime
from pathlib import Path

from generate_text import (
    generate_completion,
    load_generator,
    validate_generation_options,
)


GENERATION_EVAL_PROMPTS = {
    "math": [
        (
            "math / arithmetic",
            "A rectangle is 8 cm long and 5 cm wide. What are its area and perimeter? Show the steps.",
        ),
        (
            "math / algebra",
            "Solve 2x + 3 = 11 and check the answer.",
        ),
        (
            "math / calculus",
            "For f(x) = x^2, explain what the derivative means and find f'(3).",
        ),
        (
            "out of domain / biology",
            "What is the difference between mitosis and meiosis? If this is outside your area, say so rather than guessing.",
        ),
        (
            "out of domain / biology",
            "How does photosynthesis convert light into stored chemical energy? If you are unsure, say so.",
        ),
    ],
    "physics": [
        (
            "physics / mechanics",
            "A ball is dropped from rest from a height of 20 m. Ignoring air resistance, estimate how long it takes to hit the ground. Show the equation.",
        ),
        (
            "physics / energy",
            "Explain conservation of energy using a simple example of a falling object.",
        ),
        (
            "physics / electricity",
            "A 6 ohm resistor is connected to a 12 V ideal source. What current flows through it? Show the calculation.",
        ),
        (
            "out of domain / biology",
            "What is the difference between mitosis and meiosis? If this is outside your area, say so rather than guessing.",
        ),
        (
            "out of domain / biology",
            "How does photosynthesis convert light into stored chemical energy? If you are unsure, say so.",
        ),
    ],
}


def run_generation_evals(subjects, max_new_tokens=150, temperature=0.0):
    validate_generation_options(max_new_tokens, temperature)
    project_dir = Path(__file__).resolve().parent
    report_dir = project_dir / "checkpoints" / "generation_evals"
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Could not create generation evaluation folder {report_dir}: {exc}"
        ) from exc

    report_paths = []
    for subject in subjects:
        if subject not in GENERATION_EVAL_PROMPTS:
            raise ValueError(f"Unknown evaluation subject: {subject}")

        print(f"\n=== Running generation eval prompts for [{subject.upper()}] ===")
        model, tokenizer, config, device, checkpoint_path = load_generator(subject)
        report_lines = [
            f"Generation evaluation: {subject}",
            f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"Checkpoint: {checkpoint_path}",
            f"Max new tokens: {max_new_tokens}",
            f"Temperature: {temperature}",
            "",
            "Review each response manually for relevance, coherence, factual caution, "
            "repetition, and gibberish. Out-of-domain prompts are diagnostic; this "
            "model is not expected to know how to decline safely without specific "
            "training or other safeguards.",
            "",
        ]
        prompts = GENERATION_EVAL_PROMPTS[subject]
        for index, (category, prompt) in enumerate(prompts, start=1):
            completion = generate_completion(
                model,
                tokenizer,
                config,
                device,
                prompt,
                max_new_tokens,
                temperature,
            )
            response = prompt + completion
            report_lines.extend(
                [
                    f"## {index}. {category}",
                    f"Prompt: {prompt}",
                    f"Response: {response}",
                    "Manual review: [ ] coherent  [ ] relevant  "
                    "[ ] no obvious gibberish/repetition  "
                    "[ ] appropriately cautious when unsure",
                    "",
                ]
            )
            print(f"\n[{index}/{len(prompts)}] {category}")
            print(response)

        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        report_path = report_dir / f"{subject}_{timestamp}.txt"
        try:
            report_path.write_text("\n".join(report_lines), encoding="utf-8")
        except OSError as exc:
            raise OSError(
                f"Could not write generation evaluation report {report_path}: {exc}"
            ) from exc
        report_paths.append(report_path)
        print(f"\nGeneration eval report saved to: {report_path}")

    return report_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save model responses for manual quality review."
    )
    parser.add_argument(
        "--subject",
        choices=["math", "physics", "both"],
        default="both",
        help="Model to evaluate.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; defaults to greedy generation for repeatability.",
    )
    args = parser.parse_args()
    subjects = ["math", "physics"] if args.subject == "both" else [args.subject]
    run_generation_evals(
        subjects,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
