from pathlib import Path

from datasets import load_dataset


def get_dataset():
    print("Streaming InfoBayAI English STEM Dataset...")
    try:
        return load_dataset(
            "InfoBayAI/English-STEM-Textbook-Dataset",
            split="train",
            streaming=True,
        )
    except (ConnectionError, TimeoutError, OSError, ValueError) as error:
        print(f"Failed to load the dataset: {error}")
        raise


def inspect_dataset_columns(dataset):
    first_row = next(iter(dataset), None)
    if first_row is None:
        raise ValueError("Cannot inspect columns: the dataset is empty")

    columns = list(first_row.keys())
    print("Available metadata columns in this dataset:")
    print(columns)
    return columns


def save_dataset(dataset, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    math_count = 0
    physics_count = 0
    processed_count = 0

    try:
        with open(output_dir / "infobay_math.txt", "w", encoding="utf-8") as f_math, \
             open(output_dir / "infobay_physics.txt", "w", encoding="utf-8") as f_phys:
            for row in dataset:
                processed_count += 1
                try:
                    row_text = row["text"]
                    subject = row["subject"].strip().casefold()
                    if not isinstance(row_text, str) or not isinstance(row["subject"], str):
                        raise TypeError
                except (KeyError, AttributeError, TypeError) as error:
                    raise ValueError(
                        f"Invalid dataset row {processed_count}: expected string 'subject' and 'text' fields"
                    ) from error

                if subject in {"math", "mathematics"}:
                    f_math.write(row_text + "\n\n")
                    math_count += 1

                elif subject == "physics":
                    f_phys.write(row_text + "\n\n")
                    physics_count += 1

                if processed_count % 10000 == 0:
                    print(f"Processed {processed_count} rows...")
    except OSError as error:
        print(f"Failed while reading the dataset or writing output files: {error}")
        raise
    except ValueError as error:
        print(f"Dataset processing failed: {error}")
        raise

    print(
        f"Data separation successful! Saved {math_count} Math entries "
        f"and {physics_count} Physics entries."
    )


def main():
    output_dir = Path(__file__).resolve().parent / "data"
    dataset = get_dataset()
    save_dataset(dataset, output_dir)


if __name__ == "__main__":
    main()
