"""Small, auditable objective components; NOT a full GRPO trainer.

Shape convention: rewards [P,G], token logprobs/mask [P,G,T].
The old log probabilities and rewards are frozen rollout observations.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ObjectiveConfig:
    advantage: str = "std"  # "std" or "center"
    aggregation: str = "response"  # "response", "token", "fixed"
    clip_low: float = 0.2
    clip_high: float = 0.2
    max_completion_length: int = 512
    eps: float = 1e-8


def group_advantages(rewards: torch.Tensor, mode: str = "std", eps: float = 1e-8) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [prompts, group_size]")
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    # Rounding in the mean leaves centered=-delta, std=|delta| > eps for some
    # identical groups (e.g. float32 0.7 x 8), which std mode would turn into +-1.
    identical = (rewards == rewards[:, :1]).all(dim=1, keepdim=True)
    centered = centered.masked_fill(identical, 0.0)
    if mode == "center":
        return centered
    if mode == "std":
        std = rewards.std(dim=1, keepdim=True, unbiased=False)
        return centered / std.clamp_min(eps)
    raise ValueError(f"unsupported advantage mode: {mode}")


def policy_objective(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    config: ObjectiveConfig,
) -> torch.Tensor:
    """Return per-token PPO clipped surrogate (to MAXIMIZE).

    advantages: [P,G], broadcast over T. Caller supplies frozen rollout
    logprobs in old_logp. For textbook on-policy first update, new==old.
    Padding values must be finite here; compute_loss neutralizes them.
    """
    if new_logp.shape != old_logp.shape or new_logp.ndim != 3:
        raise ValueError("log probabilities must have matching [P,G,T] shapes")
    if advantages.shape != new_logp.shape[:2]:
        raise ValueError("advantages must have shape [P,G]")
    ratio = (new_logp - old_logp.detach()).exp()
    a = advantages.detach().unsqueeze(-1)
    lower = 1 - config.clip_low
    upper = 1 + config.clip_high
    clipped = ratio.clamp(lower, upper)
    return torch.minimum(ratio * a, clipped * a)


def policy_loss(
    surrogate: torch.Tensor,
    mask: torch.Tensor,
    config: ObjectiveConfig,
) -> torch.Tensor:
    """Aggregate token surrogate into scalar loss to MINIMIZE.

    Mask must be 0 for padding; empty sequences are rejected. All modes
    use the same batch and rewards to isolate the aggregation mechanism.
    """
    if surrogate.shape != mask.shape or surrogate.ndim != 3:
        raise ValueError("surrogate and mask must have matching [P,G,T] shapes")
    if config.max_completion_length <= 0:
        raise ValueError("max_completion_length must be positive")
    mask = mask.to(dtype=surrogate.dtype)
    lengths = mask.sum(dim=-1)
    if (lengths <= 0).any():
        raise ValueError("every response must contain at least one valid token")
    sequence_sums = (surrogate * mask).sum(dim=-1)
    if config.aggregation == "response":
        result = (sequence_sums / lengths).mean()
    elif config.aggregation == "token":
        result = sequence_sums.sum() / lengths.sum()
    elif config.aggregation == "fixed":
        result = sequence_sums.mean() / config.max_completion_length
    else:
        raise ValueError(f"unsupported aggregation: {config.aggregation}")
    return -result


def compute_loss(new_logp, old_logp, rewards, mask, config: ObjectiveConfig):
    if not new_logp.shape == old_logp.shape == mask.shape:
        raise ValueError("log probabilities and mask must have matching [P,G,T] shapes")
    # Masking the surrogate alone is not enough: an inf ratio at a padded
    # position still backpropagates 0 * inf = NaN, so pin padding to ratio 1.
    padding = ~mask.bool()
    new_logp = new_logp.masked_fill(padding, 0.0)
    old_logp = old_logp.masked_fill(padding, 0.0)
    advantage = group_advantages(rewards, config.advantage, config.eps)
    surrogate = policy_objective(new_logp, old_logp, advantage, config)
    return policy_loss(surrogate, mask, config)
