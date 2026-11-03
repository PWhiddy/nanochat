"""Cheap numerical checks at optimizer boundaries, outside compiled training graphs."""

import torch
import torch.distributed as dist


@torch.no_grad()
def check_finite_tensors(named_tensors, *, step, phase):
    """Stop all ranks on NaN/Inf, reporting tensor names without copying their data.

    Infinity norms avoid overflowing on large but finite tensors. The foreach
    reduction only reads tensors, and copies one boolean to the CPU on success.
    """
    entries = [(name, tensor) for name, tensor in named_tensors if tensor is not None]
    if not entries:
        return
    names, tensors = zip(*entries)
    norms = torch._foreach_norm(tensors, float("inf"))
    finite = torch.isfinite(torch.stack([norm.float() for norm in norms]))
    all_finite = finite.all().to(dtype=torch.int32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
    if not all_finite.item():
        bad_names = [name for name, ok in zip(names, finite.tolist()) if not ok]
        details = ", ".join(bad_names[:8]) if bad_names else "tensor(s) on another rank"
        if len(bad_names) > 8:
            details += f", and {len(bad_names) - 8} more"
        raise FloatingPointError(
            f"NaN/Inf {phase} at training step {step}: {details}. "
            "Training stopped; do not resume from non-finite weights. "
            "Keep this error and the run settings to identify the first numerical failure."
        )
