"""DAPO overlong handling helpers (soft punishment and token counting)."""
import torch

# DAPO uses a 4,096-token buffer under a 20,480-token cap (20%); 40 is ~20% of our 192 cap.
DEFAULT_CACHE = 40


def soft_overlong_penalty(length: int, max_length: int = 192, cache: int = DEFAULT_CACHE) -> float:
    """0 up to max_length-cache, then linear to -1 at max_length (DAPO Eq. 13)."""
    if cache <= 0 or cache > max_length:
        raise ValueError("cache must be in (0, max_length]")
    threshold = max_length - cache
    if length <= threshold:
        return 0.0
    if length <= max_length:
        return (threshold - length) / cache
    return -1.0


def completion_token_counts(completion_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Generated tokens per row of a padded [B,T] tensor, counting the first EOS.

    Rows without EOS hit the length cap and count as T. Unlike completion_mask.sum(),
    this is unaffected by mask_truncated_completions, so rejected or masked
    completions still count toward the sampling budget.
    """
    is_eos = completion_ids == eos_id
    first = is_eos.int().argmax(dim=1)  # index of the first EOS (0 if none)
    width = torch.full_like(first, completion_ids.shape[1])
    return torch.where(is_eos.any(dim=1), first + 1, width)


class OverlongShapedReward:
    """Wraps a correctness reward: returns correctness + soft overlong penalty.

    last_correct keeps the unshaped 0/1 rewards of the latest call so that dynamic
    sampling can judge a group by correctness instead of by the shaped reward.
    """

    def __init__(self, base_reward, max_length: int = 192, cache: int = DEFAULT_CACHE, penalty: bool = True):
        self.base_reward, self.max_length, self.cache, self.penalty = base_reward, max_length, cache, penalty
        self.last_correct = []
        self.__name__ = "overlong_shaped_reward"  # TRL reads reward_func.__name__

    def __call__(self, completions, ground_truth, completion_ids=None, **kwargs):
        correct = list(self.base_reward(completions, ground_truth=ground_truth))
        self.last_correct = correct
        if not self.penalty:
            return correct
        if completion_ids is None:
            raise ValueError("completion_ids are required for the overlong penalty")
        return [c + soft_overlong_penalty(len(ids), self.max_length, self.cache)
                for c, ids in zip(correct, completion_ids, strict=True)]
