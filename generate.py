import math

import torch


def generate(
    prompt, num_new_tokens, model, tokenizer, context_size=32, temperature=1.0,
):
    """Sample BPE IDs, then decode together; invalid UTF-8 displays as U+FFFD."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")
    if not 1 <= context_size <= model.position_embedding.num_embeddings:
        raise ValueError("context_size exceeds the model's positional capacity")
    if not isinstance(num_new_tokens, int) or num_new_tokens < 0:
        raise ValueError("num_new_tokens must be a non-negative integer")
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt must contain at least one token")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for _ in range(num_new_tokens):
            context = torch.tensor(
                [token_ids[-context_size:]], dtype=torch.long, device=device
            )
            logits = model(context)
            probabilities = torch.softmax(logits / temperature, dim=-1)
            token_ids.append(torch.multinomial(probabilities, 1).item())
    model.train(was_training)
    return tokenizer.decode(token_ids, errors="replace")
