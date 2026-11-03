import argparse
from pathlib import Path

import torch

from data_loader import load_dataset
from inference import generate_tokens
from model import LanguageModel
from tokenizers import load_or_build_bpe_tokenizer


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a text continuation from the trained language model."
    )
    parser.add_argument("prompt", help="Text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optionally sample only from the top K next-token choices",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_DIR / "checkpoints" / "language_model.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR / "literature",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError(f"Unsupported model checkpoint format: {checkpoint.get('format_version')}")

    model_config = checkpoint["model_config"]
    model = LanguageModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def generate(args):
    if not args.prompt.strip():
        raise ValueError("prompt must contain at least one non-whitespace character")
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.top_k is not None and args.top_k < 1:
        raise ValueError("--top-k must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(args.checkpoint_path, device)
    raw_text = load_dataset(args.data_dir)
    training_config = checkpoint["training_config"]
    encode, decode, vocab_size = load_or_build_bpe_tokenizer(
        raw_text,
        target_vocab_size=training_config["target_vocab_size"],
    )
    if vocab_size != checkpoint["model_config"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary size does not match the model checkpoint")

    prompt_tokens = encode(args.prompt)
    if not prompt_tokens:
        raise ValueError("prompt did not produce any tokens")
    input_tokens = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    output_tokens = generate_tokens(
        model,
        input_tokens,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print(decode(output_tokens[0].tolist()))


if __name__ == "__main__":
    generate(parse_args())
