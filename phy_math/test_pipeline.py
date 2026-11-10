import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import CustomLLM, ModelConfig


def test_bridge(tokenizer_path=None):
    print("=== Testing Tokenizer to Model Architecture Bridge ===")

    try:
        if tokenizer_path is None:
            tokenizer_path = (
                Path(__file__).resolve().parent
                / "checkpoints"
                / "math_tokenizer.json"
            )
        else:
            tokenizer_path = Path(tokenizer_path).expanduser().resolve()

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f" -> Tokenizer loaded successfully from {tokenizer_path}.")
    except Exception as exc:
        print(f" -> Error loading tokenizer: {exc}")
        print("Please ensure you ran the tokenizer training script first and that the JSON file exists.")
        return

    try:
        sample_text = (
            "Let f(x) = 3x^2 + 2x. Find the derivative df/dx using the chain rule."
        )
        encoded = tokenizer.encode(sample_text)
        token_ids = encoded.ids

        print(f" -> Sample Text: '{sample_text}'")
        print(f" -> Tokenized IDs (Length {len(token_ids)}): {token_ids[:10]}...")

        config = ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            dim=512,
            n_layers=4,
            n_heads=8,
        )
        model = CustomLLM(config)
        print(f" -> PyTorch Model initialized with Vocab Size: {config.vocab_size}")

        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(input_tensor)

        print(" -> Forward pass through architecture successful!")
        print(f" -> Output Tensor (Logits) Shape: {logits.shape}")
        print("    [Expected Shape: (Batch=1, SeqLen, VocabSize)]")
        print(
            "\n=== Bridge Verification complete! Step 1 and Step 2 are perfectly aligned. ==="
        )
    except Exception as exc:
        print(f" -> Error during model bridge validation: {exc}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a trained tokenizer and verify it fits the toy model architecture."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Optional path to a tokenizer JSON file; defaults to checkpoints/math_tokenizer.json.",
    )
    args = parser.parse_args()
    test_bridge(args.tokenizer_path)
