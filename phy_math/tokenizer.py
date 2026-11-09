import argparse
import glob
import os
import sys

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def get_local_text_iterator(data_dir, split_folder, keywords=None):
    """Find local parquet files in a split and stream their text values."""
    search_path = os.path.join(data_dir, split_folder, "*.parquet")
    local_files = sorted(glob.glob(search_path))

    if not local_files:
        print(f"Warning: No parquet files found in {os.path.join(data_dir, split_folder)}")
        return

    try:
        dataset = load_dataset(
            "parquet",
            data_files=local_files,
            split="train",
            streaming=True,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        print(f"Error loading parquet dataset from {split_folder}: {exc}")
        return

    for row in dataset:
        try:
            text_content = row.get("text", "") or ""
            prompt_content = row.get("prompt", "") or ""
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            print(f"Skipping malformed row in {split_folder}: {exc}")
            continue

        if keywords:
            combined_meta = (
                (prompt_content + " " + row.get("audience", "") + " " + row.get("format", ""))
                .lower()
            )
            if any(keyword in combined_meta for keyword in keywords):
                yield text_content
        else:
            yield text_content


def main():
    parser = argparse.ArgumentParser(
        description="Train custom BPE Tokenizers on local Cosmopedia dataset chunks."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/guru/data/cosmopedia_raw/data",
        help="Absolute or relative path to cosmopedia_raw/data (default: /home/guru/data/cosmopedia_raw/data)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for the BPE model",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Directory to save the resulting tokenizer JSON files",
    )
    args = parser.parse_args()

    try:
        data_dir = os.path.abspath(args.data_dir)
        out_dir = os.path.abspath(args.out_dir)

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"The path '{data_dir}' does not exist or is not a directory."
            )

        os.makedirs(out_dir, exist_ok=True)
        special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "[INST]", "[/INST]"]

        print("=== Starting Math Tokenizer Training ===")
        math_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        math_tokenizer.pre_tokenizer = Whitespace()
        math_trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            continuing_subword_prefix="##",
        )

        math_iterator = get_local_text_iterator(data_dir, "auto_math_text")
        try:
            math_tokenizer.train_from_iterator(math_iterator, trainer=math_trainer)
        except Exception as exc:
            print(f"Error training math tokenizer: {exc}")
            raise

        math_out_path = os.path.join(out_dir, "math_tokenizer.json")
        math_tokenizer.save(math_out_path)
        print(f"Successfully created and saved target: {math_out_path}\n")

        print("=== Starting Physics Tokenizer Training ===")
        physics_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        physics_tokenizer.pre_tokenizer = Whitespace()
        physics_trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            continuing_subword_prefix="##",
        )

        physics_keywords = [
            "physics",
            "quantum",
            "mechanics",
            "thermodynamics",
            "electromagnetism",
            "kinematics",
            "relativity",
        ]

        def combined_physics_iterator():
            for path in [
                (data_dir, "stanford", physics_keywords),
                (data_dir, "openstax", physics_keywords),
                (data_dir, "khanacademy", physics_keywords),
            ]:
                for text in get_local_text_iterator(*path):
                    yield text

        try:
            physics_tokenizer.train_from_iterator(
                combined_physics_iterator(), trainer=physics_trainer
            )
        except Exception as exc:
            print(f"Error training physics tokenizer: {exc}")
            raise

        physics_out_path = os.path.join(out_dir, "physics_tokenizer.json")
        physics_tokenizer.save(physics_out_path)
        print(f"Successfully created and saved target: {physics_out_path}")

    except Exception as exc:
        print(f"Tokenizer training failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
