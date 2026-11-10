import pytest
import torch

from grpo_ablation.overlong import OverlongShapedReward, completion_token_counts, soft_overlong_penalty

EOS = 7


@pytest.mark.parametrize("length,expected", [(1, 0.0), (152, 0.0), (172, -0.5), (192, -1.0), (193, -1.0)])
def test_soft_penalty_values(length, expected):
    assert soft_overlong_penalty(length, 192, 40) == pytest.approx(expected)


def test_penalty_is_monotone():
    values = [soft_overlong_penalty(n, 192, 40) for n in range(0, 200)]
    assert values == sorted(values, reverse=True)


def test_token_counts_include_first_eos_and_cap():
    ids = torch.tensor([[1, 2, EOS, EOS, EOS],   # ended after 3 tokens, padded with EOS
                        [1, 2, 3, 4, 5],         # hit the cap, no EOS
                        [EOS, EOS, EOS, EOS, EOS]])
    assert completion_token_counts(ids, EOS).tolist() == [3, 5, 1]


def test_token_counts_match_trl_completion_mask():
    ids = torch.tensor([[1, 2, EOS, EOS], [1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    assert completion_token_counts(ids, EOS).sum().item() == mask.sum().item()


def test_shaped_reward_keeps_correctness_and_adds_penalty():
    base = lambda completions, ground_truth: [1.0, 1.0, 0.0, 1.0]
    r = OverlongShapedReward(base, 192, 40)
    out = r(["a"] * 4, ground_truth=["1"] * 4, completion_ids=[[0] * 100, [0] * 172, [0] * 100, [0] * 192])
    assert out == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert r.last_correct == [1.0, 1.0, 0.0, 1.0]       # unshaped, for dynamic sampling
    assert r.__name__ == "overlong_shaped_reward"


def test_penalty_requires_lengths():
    r = OverlongShapedReward(lambda completions, ground_truth: [1.0], 192, 40)
    with pytest.raises(ValueError):
        r(["a"], ground_truth=["1"])
