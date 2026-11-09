import argparse
import json
from pathlib import Path

from tokenizers import restore_bpe_learned_data


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect learned pieces in a cached BPE tokenizer."
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=PROJECT_DIR / "cache" / "bpe_512.json",
        help="Tokenizer cache JSON file (default: cache/bpe_512.json)",
    )
    parser.add_argument(
        "--all-tokens",
        action="store_true",
        help="Show every learned piece instead of only pieces containing whitespace",
    )
    return parser.parse_args()


def inspect_tokenizer(cache_path, all_tokens=False):
    if not cache_path.is_file():
        raise FileNotFoundError(f"Tokenizer cache not found: {cache_path}")

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("format_version") != 1:
        raise ValueError(f"Unsupported tokenizer cache format: {cache.get('format_version')}")

    merges, vocab = restore_bpe_learned_data(cache["merges"])
    learned_ids = sorted(vocab)
    learned_ids = [token_id for token_id in learned_ids if token_id >= 256]
    whitespace_ids = [
        token_id
        for token_id in learned_ids
        if any(
            character.isspace()
            for character in vocab[token_id].decode(
                "utf-8", errors="backslashreplace"
            )
        )
    ]

    print(f"Tokenizer cache: {cache_path}")
    print(f"Target vocabulary size: {cache.get('target_vocab_size', 'unknown')}")
    print(f"Learned merges: {len(merges)}")
    print(f"Learned pieces containing whitespace: {len(whitespace_ids)}")

    token_ids = learned_ids if all_tokens else whitespace_ids
    if not token_ids:
        if all_tokens:
            print("No learned pieces found.")
        else:
            print("No learned pieces contain whitespace. Use --all-tokens to list all pieces.")
        return

    print("\nLearned pieces:")
    for token_id in token_ids:
        piece = vocab[token_id].decode("utf-8", errors="backslashreplace")
        marker = " [contains whitespace]" if token_id in whitespace_ids else ""
        print(f"{token_id:>5}  {piece!r}{marker}")

    print(
        "\nWhitespace-containing pieces cross a whitespace boundary; "
        "they do not identify which language a word belongs to."
    )


if __name__ == "__main__":
    args = parse_args()
    inspect_tokenizer(args.cache_path, args.all_tokens)
