import torch
import torch.nn.functional as F


@torch.no_grad()
def generate_tokens(
    model, token_ids, max_new_tokens, temperature=1.0, top_k=None, generator=None
):
    """Generate a continuation from a batch containing one prompt."""
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise ValueError("generation requires a non-empty prompt with shape (1, sequence)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive when specified")

    for _ in range(max_new_tokens):
        context = token_ids[:, -model.block_size :]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            count = min(top_k, next_token_logits.shape[-1])
            threshold = torch.topk(next_token_logits, count).values[:, -1, None]
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < threshold,
                float("-inf"),
            )

        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        token_ids = torch.cat((token_ids, next_token), dim=1)

    return token_ids
