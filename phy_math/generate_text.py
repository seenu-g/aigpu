import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from checkpoint import file_sha256
from model import CustomLLM, ModelConfig


def validate_generation_options(max_new_tokens, temperature):
    if type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a non-negative integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be a finite, non-negative number")


def load_generator(subject):
    if subject not in {"math", "physics"}:
        raise ValueError("Subject choice must be 'math' or 'physics'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_dir = Path(__file__).resolve().parent
    tokenizer_path = project_dir / f"{subject}_tokenizer.json"
    checkpoint_path = project_dir / "checkpoints" / f"final_llm_{subject}.pt"

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load tokenizer from {tokenizer_path}: {exc}") from exc

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load checkpoint {checkpoint_path}: {exc}") from exc

    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported or invalid checkpoint format: {checkpoint_path}")

    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError(f"Checkpoint has no training configuration: {checkpoint_path}")
    if training_config.get("subject") != subject:
        raise ValueError(
            f"Checkpoint subject does not match requested subject '{subject}'"
        )
    if training_config.get("tokenizer_sha256") != file_sha256(tokenizer_path):
        raise ValueError(
            f"Tokenizer at {tokenizer_path} does not match the checkpoint. "
            "Use the tokenizer that was used during training."
        )

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"Checkpoint has no model configuration: {checkpoint_path}")
    try:
        config = ModelConfig(**model_config)
    except TypeError as exc:
        raise ValueError(f"Checkpoint model configuration is invalid: {exc}") from exc
    if config.vocab_size != tokenizer.get_vocab_size():
        raise ValueError(
            "Tokenizer vocabulary size does not match the checkpoint model configuration"
        )

    model = CustomLLM(config)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError, TypeError) as exc:
        raise ValueError(f"Checkpoint model state is incompatible: {exc}") from exc

    model = model.to(device)
    model.eval()
    return model, tokenizer, config, device, checkpoint_path


def generate_completion(
    model,
    tokenizer,
    config,
    device,
    prompt,
    max_new_tokens,
    temperature,
):
    validate_generation_options(max_new_tokens, temperature)
    bos_id = tokenizer.token_to_id("<s>")
    if bos_id is None:
        raise ValueError("Tokenizer does not define the required '<s>' token")
    input_ids = tokenizer.encode(prompt).ids
    if not input_ids or input_ids[0] != bos_id:
        input_ids = [bos_id] + input_ids

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("</s>")
    generated_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = (
                x
                if x.size(1) <= config.max_seq_len
                else x[:, -config.max_seq_len:]
            )
            logits, _ = model(x_cond)
            next_token_logits = logits[:, -1, :]
            if temperature == 0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token_id = next_token.item()
            if next_token_id == eos_id:
                break
            generated_ids.append(next_token_id)
            x = torch.cat((x, next_token), dim=1)
    return tokenizer.decode(generated_ids)


def generate_text(
    subject="math",
    prompt="Let x = 2. Then x^2 =",
    max_new_tokens=100,
    temperature=0.7,
):
    validate_generation_options(max_new_tokens, temperature)
    print(f"\n=== Initializing Generator for [{subject.upper()}] ===")
    model, tokenizer, config, device, _ = load_generator(subject)
    completion = generate_completion(
        model, tokenizer, config, device, prompt, max_new_tokens, temperature
    )
    print(f" -> Prompt received: '{prompt}'")
    print(" -> Generating output tokens autoregressively...\n")
    print(prompt + completion)
    print("\n=== Text Generation Run Complete! ===")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained math or physics model."
    )
    parser.add_argument(
        "--subject",
        choices=["math", "physics"],
        default="math",
        help="Model to run.",
    )
    parser.add_argument(
        "--prompt",
        default="Theorem: If a function f is differentiable at point c, then",
        help="Text prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args()
    generate_text(
        subject=args.subject,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
