import argparse
import os
import sys
from time import perf_counter

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from load_data import get_parquet_files, read_parquet_file_data


SPECIAL_TOKENS = ["<s>", "</s>", "<unk>", "<pad>", "[INST]", "[/INST]"]
PHYSICS_KEYWORDS = [
    "physics",
    "quantum",
    "mechanics",
    "thermodynamics",
    "electromagnetism",
    "kinematics",
    "relativity",
]


def _train_and_save_tokenizer(
    data_dir,
    split_folders,
    out_dir,
    filename,
    vocab_size,
    keywords=None,
):
    tokenizer_started = perf_counter()
    files = get_parquet_files(data_dir, split_folders)
    text_iterator = read_parquet_file_data(
        files,
        keywords=keywords,
        split="train",
    )

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
    )
    training_started = perf_counter()
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    training_duration = perf_counter() - training_started

    output_path = os.path.join(out_dir, filename)
    save_started = perf_counter()
    tokenizer.save(output_path)
    save_duration = perf_counter() - save_started
    total_duration = perf_counter() - tokenizer_started
    print(f"Successfully created and saved tokenizer: {output_path}")
    print(f"  BPE training time: {training_duration:.2f}s")
    print(f"  Save time:         {save_duration:.2f}s")
    print(f"  Total time:        {total_duration:.2f}s")
    return output_path


def save_math_tokenizer(data_dir, out_dir, vocab_size=32000):
    """Train and save the math tokenizer from the training portion of its corpus."""
    print("=== Starting Math Tokenizer Training ===")
    return _train_and_save_tokenizer(
        data_dir=data_dir,
        split_folders=["auto_math_text"],
        out_dir=out_dir,
        filename="math_tokenizer.json",
        vocab_size=vocab_size,
    )


def save_physics_tokenizer(data_dir, out_dir, vocab_size=32000):
    """Train and save the physics tokenizer from the filtered training corpus."""
    print("=== Starting Physics Tokenizer Training ===")
    return _train_and_save_tokenizer(
        data_dir=data_dir,
        split_folders=["stanford", "openstax", "khanacademy"],
        out_dir=out_dir,
        filename="physics_tokenizer.json",
        vocab_size=vocab_size,
        keywords=PHYSICS_KEYWORDS,
    )


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
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"),
        help="Directory to save tokenizer JSON files (default: checkpoints next to this script)",
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
        save_math_tokenizer(data_dir, out_dir, args.vocab_size)
        save_physics_tokenizer(data_dir, out_dir, args.vocab_size)

    except Exception as exc:
        print(f"Tokenizer training failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
