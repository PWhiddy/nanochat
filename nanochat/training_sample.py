"""Fixed-prompt samples for watching a base model learn, without compiling inference."""

import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache, sample_next_token


class TokenSampleSchedule:
    """Sample at initialization, then at the first update crossing each token boundary."""

    def __init__(self, every_tokens, start_tokens=0):
        self.every_tokens = every_tokens
        # A resumed run starts at the next boundary, without replaying earlier samples.
        self.next_tokens = 0 if start_tokens == 0 or every_tokens <= 0 else (
            (start_tokens // every_tokens + 1) * every_tokens
        )

    def due(self, tokens_seen):
        if self.every_tokens <= 0 or tokens_seen < self.next_tokens:
            return False
        # Anchor to multiples of the interval, so rounding to optimizer steps won't drift.
        # If one update crosses several boundaries, sample its weights just once.
        self.next_tokens = (tokens_seen // self.every_tokens + 1) * self.every_tokens
        return True


@torch.inference_mode()
def generate_training_sample(model, tokenizer, prompt, max_new_tokens=512, seed=42):
    """Return exactly max_new_tokens using a batch-one KV cache and a private RNG.

    Pass the original, uncompiled model with FP8 temporarily disabled. Unlike the
    chat engine, this is raw text continuation: special tokens neither stop it
    early nor invoke tools. Only the completed token buffer is copied to the CPU.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    device = model.get_device()
    tokens = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    config = model.config
    cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=len(tokens) + max_new_tokens,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device=device,
        dtype=COMPUTE_DTYPE,
    )
    rng = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([tokens], dtype=torch.long, device=device)
    generated = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        for i in range(max_new_tokens):
            logits = model(ids, kv_cache=cache)[:, -1, :]
            ids = sample_next_token(logits, rng, temperature=0.8, top_k=50)
            generated[i] = ids[0, 0]
    finally:
        model.train(was_training)
    return tokenizer.decode(generated.tolist())
