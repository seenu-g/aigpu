import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_tokens(
    model, token_ids, max_new_tokens, temperature=1.0, top_k=None, generator=None
):
    """Generate a continuation from a batch containing one prompt."""
    output_tokens, _ = _generate_tokens(
        model, token_ids, max_new_tokens, temperature, top_k, generator
    )
    return output_tokens


@torch.no_grad()
def generate_tokens_with_probabilities(
    model, token_ids, max_new_tokens, temperature=1.0, top_k=None, generator=None
):
    """Generate tokens and collect model and sampling probabilities per token."""
    return _generate_tokens(
        model,
        token_ids,
        max_new_tokens,
        temperature,
        top_k,
        generator,
        collect_probabilities=True,
    )


def _generate_tokens(
    model,
    token_ids,
    max_new_tokens,
    temperature,
    top_k,
    generator,
    collect_probabilities=False,
):
    """Generate tokens, optionally retaining per-step probability diagnostics."""
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise ValueError("generation requires a non-empty prompt with shape (1, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when specified")

    probability_history = []
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.block_size :]
        logits, _ = model(context)
        raw_next_token_logits = logits[:, -1, :]
        model_probabilities = F.softmax(raw_next_token_logits, dim=-1)
        next_token_logits = raw_next_token_logits / temperature

        if top_k is not None:
            count = min(top_k, next_token_logits.shape[-1])
            threshold = torch.topk(next_token_logits, count).values[:, -1, None]
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < threshold,
                float("-inf"),
            )

        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        if collect_probabilities:
            token_id = next_token_id.item()
            probability_history.append(
                {
                    "token_id": token_id,
                    "model_probability": model_probabilities[0, token_id].item(),
                    "sampling_probability": probabilities[0, token_id].item(),
                    "top_model_probability": model_probabilities.max().item(),
                    "entropy_bits": (
                        -model_probabilities
                        * torch.log2(model_probabilities.clamp_min(1e-12))
                    )
                    .sum()
                    .item(),
                }
            )
        token_ids = torch.cat((token_ids, next_token_id), dim=1)

    if collect_probabilities:
        return token_ids, probability_history
    return token_ids, None
