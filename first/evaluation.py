import torch

from inference import generate_tokens


FIXED_PROMPTS = (
    "To be, or not to be",
    "Now is the winter of our discontent",
    "What's in a name?",
)
REPETITION_NGRAM_SIZE = 4
SOURCE_MATCH_WINDOW_SIZE = 40
SOURCE_MATCH_STRIDE = 20


@torch.no_grad()
def estimate_loss(model, get_batch, num_batches=10):
    """Estimate mean training and validation loss over sampled batches."""
    if num_batches < 1:
        raise ValueError("num_batches must be positive")

    was_training = model.training
    model.eval()
    try:
        losses = {}
        for split in ("train", "val"):
            batch_losses = torch.empty(num_batches)
            for index in range(num_batches):
                inputs, targets = get_batch(split)
                _, loss = model(inputs, targets)
                batch_losses[index] = loss.item()
            losses[split] = batch_losses.mean().item()
        return losses
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate_generations(
    model,
    encode,
    decode,
    training_text,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
):
    """Generate fixed-prompt samples and report repetition/source overlap."""
    was_training = model.training
    model.eval()
    print("\n=== Fixed-prompt generation evaluation ===")
    try:
        for prompt_index, prompt in enumerate(FIXED_PROMPTS):
            prompt_ids = encode(prompt)
            if not prompt_ids:
                raise ValueError(f"Fixed evaluation prompt produced no tokens: {prompt!r}")

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generator = torch.Generator(device=device).manual_seed(2026 + prompt_index)
            output_ids = generate_tokens(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                generator=generator,
            )
            continuation_ids = output_ids[0, len(prompt_ids) :].tolist()
            continuation = decode(continuation_ids)

            repeated_ngram_rate = _repeated_ngram_rate(
                continuation_ids, REPETITION_NGRAM_SIZE
            )
            matching_windows, total_windows = _source_overlap(
                continuation,
                training_text,
                SOURCE_MATCH_WINDOW_SIZE,
                SOURCE_MATCH_STRIDE,
            )

            print(f"\nPrompt: {prompt}")
            print(f"Continuation: {continuation}")
            print(
                f"Repeated {REPETITION_NGRAM_SIZE}-gram rate: "
                f"{repeated_ngram_rate:.1%}"
            )
            if total_windows:
                print(
                    f"Exact training-text overlap "
                    f"({SOURCE_MATCH_WINDOW_SIZE}-character windows): "
                    f"{matching_windows}/{total_windows}"
                )
            else:
                print("Exact training-text overlap: not enough generated text to check")
    finally:
        model.train(was_training)

    print(
        "\nThese are diagnostic signals, not proof of quality or memorization. "
        "Exact overlap is expected for common phrases in a small source corpus."
    )


def _repeated_ngram_rate(token_ids, ngram_size):
    ngram_count = len(token_ids) - ngram_size + 1
    if ngram_count <= 0:
        return 0.0
    ngrams = [
        tuple(token_ids[index : index + ngram_size])
        for index in range(ngram_count)
    ]
    return (ngram_count - len(set(ngrams))) / ngram_count


def _source_overlap(text, training_text, window_size, stride):
    if len(text) < window_size:
        return 0, 0

    windows = [
        text[start : start + window_size]
        for start in range(0, len(text) - window_size + 1, stride)
    ]
    matches = sum(training_text.find(window) >= 0 for window in windows)
    return matches, len(windows)
