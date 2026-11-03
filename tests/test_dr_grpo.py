"""Numerical verification of the E0-E3 normalization factorial.

Expected values are literals derived by hand or come from the pure-Python
reference below, which never calls grpo_ablation.objective.
"""
import dataclasses
import math

import pytest
import torch

from grpo_ablation.configs import PRESETS
from grpo_ablation.objective import (
    ObjectiveConfig,
    compute_loss,
    group_advantages,
    policy_loss,
    policy_objective,
)

# id -> (preset, advantage, aggregation), from docs/EXPERIMENT_PLAN.md section 3.
EXPERIMENTS = {
    "E0": ("grpo_core", "std", "response"),
    "E1": ("length_only", "std", "fixed"),
    "E2": ("std_only", "center", "response"),
    "E3": ("dr_grpo_core", "center", "fixed"),
}
ALL_EXPERIMENTS = list(EXPERIMENTS.values())


def preset(name, length):
    return dataclasses.replace(PRESETS[name], max_completion_length=length)


def ref_advantages(group, mode):
    mean = sum(group) / len(group)
    centered = [r - mean for r in group]
    if mode == "center":
        return centered
    std = math.sqrt(sum(c * c for c in centered) / len(group))
    return [c / std if std > 0 else 0.0 for c in centered]


def ref_token(log_ratio, adv, clip_low=0.2, clip_high=0.2):
    """Return the PPO surrogate and its derivative with respect to new_logp."""
    ratio = math.exp(log_ratio)
    clipped = min(max(ratio, 1 - clip_low), 1 + clip_high)
    frozen = (adv > 0 and ratio > 1 + clip_high) or (adv < 0 and ratio < 1 - clip_low)
    return min(ratio * adv, clipped * adv), 0.0 if frozen else adv * ratio


def ref_loss_and_grads(rewards, log_ratios, lengths, advantage, aggregation, max_length):
    """Evaluate -J and d(-J)/d(new_logp) token by token; grads holds valid tokens only."""
    batch = sum(len(group) for group in rewards)
    objective, grads = 0.0, {}
    for p, group in enumerate(rewards):
        advantages = ref_advantages(group, advantage)
        for g, adv in enumerate(advantages):
            denominator = lengths[p][g] if aggregation == "response" else max_length
            weight = 1 / (batch * denominator)
            for t in range(lengths[p][g]):
                value, slope = ref_token(log_ratios[p][g][t], adv)
                objective += weight * value
                grads[p, g, t] = -weight * slope
    return -objective, grads


# Two distinct reward stds (0.5 and sqrt(3)/4) keep E2 from being a rescaled E0;
# the last group is homogeneous. T=7 > max(T_i)=6 and L=8 > T separate L from
# both the observed maximum length and the padded width.
REWARDS = [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
LENGTHS = [[2, 5, 3, 6], [1, 4, 6, 2], [3, 3, 2, 5]]
WIDTH = 7
MAX_LENGTH = 8
# exp(+-0.5) falls outside [0.8, 1.2]; the other log ratios stay inside.
DELTAS = [-0.5, -0.1, 0.0, 0.1, 0.5]


def synthetic_batch(pad_new=3.0, pad_old=-7.0):
    new, old, mask, log_ratios = [], [], [], []
    for p, group in enumerate(LENGTHS):
        new.append([]), old.append([]), mask.append([]), log_ratios.append([])
        for g, length in enumerate(group):
            old_row = [-0.5 - 0.1 * ((p + g + t) % 5) for t in range(WIDTH)]
            delta_row = [DELTAS[(p + 2 * g + t) % 5] for t in range(WIDTH)]
            new[p].append([
                o + d if t < length else pad_new
                for t, (o, d) in enumerate(zip(old_row, delta_row))
            ])
            old[p].append([o if t < length else pad_old for t, o in enumerate(old_row)])
            mask[p].append([float(t < length) for t in range(WIDTH)])
            log_ratios[p].append(delta_row)
    return (
        torch.tensor(new, dtype=torch.float64, requires_grad=True),
        torch.tensor(old, dtype=torch.float64),
        torch.tensor(REWARDS, dtype=torch.float64),
        torch.tensor(mask, dtype=torch.float64),
        log_ratios,
    )


def two_length_batch(first=2.0, second=0.5, width=128):
    """One group holding responses with T=10 and T=100."""
    surrogate = torch.zeros(1, 2, width, dtype=torch.float64)
    mask = torch.zeros_like(surrogate)
    surrogate[0, 0, :10], surrogate[0, 1, :100] = first, second
    mask[0, 0, :10], mask[0, 1, :100] = 1, 1
    return surrogate.requires_grad_(), mask


def test_advantages_match_hand_values():
    rewards = torch.tensor([[1., 0., 1., 0.], [1., 0., 0., 0.]], dtype=torch.float64)
    root3 = math.sqrt(3)
    expected_std = torch.tensor(
        [[1., -1., 1., -1.], [root3, -1 / root3, -1 / root3, -1 / root3]], dtype=torch.float64
    )
    expected_center = torch.tensor(
        [[.5, -.5, .5, -.5], [.75, -.25, -.25, -.25]], dtype=torch.float64
    )
    assert torch.allclose(group_advantages(rewards, "std"), expected_std)
    assert torch.allclose(group_advantages(rewards, "center"), expected_center)


@pytest.mark.parametrize("mode", ["std", "center"])
def test_advantages_match_reference_on_non_binary_groups(mode):
    groups = [[0.25, 1.5, -2.0], [3.0, 3.5, 7.0]]
    out = group_advantages(torch.tensor(groups, dtype=torch.float64), mode)
    assert out.shape == (2, 3) and torch.isfinite(out).all()
    expected = torch.tensor([ref_advantages(g, mode) for g in groups], dtype=torch.float64)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("mode", ["std", "center"])
@pytest.mark.parametrize("value", [0., 1.])
def test_homogeneous_binary_group_has_zero_advantage(mode, value):
    assert torch.equal(group_advantages(torch.full((2, 8), value), mode), torch.zeros(2, 8))


# float32 means of these groups round away from the value, leaving centered=-delta
# and std=|delta| > eps, which would otherwise normalize rounding noise to +-1.
@pytest.mark.parametrize("mode", ["std", "center"])
@pytest.mark.parametrize("value", [.1, .2, .3, .7, .9, 5.3])
@pytest.mark.parametrize("group_size", [3, 6, 7, 8])
def test_homogeneous_group_rounding_noise_is_not_amplified(mode, value, group_size):
    out = group_advantages(torch.full((1, group_size), value), mode)
    assert torch.equal(out, torch.zeros(1, group_size))


def test_small_real_reward_difference_is_still_standardized():
    rewards = torch.tensor([[0., 1e-3]], dtype=torch.float64)
    assert torch.allclose(group_advantages(rewards), torch.tensor([[-1., 1.]], dtype=torch.float64))


@pytest.mark.parametrize("mode", ["std", "center"])
def test_single_response_group_is_finite_zero(mode):
    assert torch.equal(group_advantages(torch.ones(2, 1), mode), torch.zeros(2, 1))


@pytest.mark.parametrize("mode", ["std", "center"])
def test_each_prompt_normalizes_independently(mode):
    rewards = torch.tensor(
        [[1., 0., 0.], [1., 1., 0.], [1., 1., 1.], [0., 0., 0.], [10., 20., 60.]],
        dtype=torch.float64,
    )
    joint = group_advantages(rewards, mode)
    assert joint.shape == rewards.shape
    for p in range(rewards.shape[0]):
        assert torch.equal(joint[p], group_advantages(rewards[p:p + 1], mode)[0])
    assert torch.allclose(joint.mean(dim=1), torch.zeros(5, dtype=torch.float64), atol=1e-12)
    if mode == "std":
        stds = joint.pow(2).mean(dim=1).sqrt()
        assert torch.allclose(stds, torch.tensor([1., 1., 0., 0., 1.], dtype=torch.float64))


def test_invalid_advantage_inputs_raise():
    with pytest.raises(ValueError, match="unsupported advantage mode"):
        group_advantages(torch.ones(1, 2), "rank")
    for shape in [(4,), (1, 2, 3)]:
        with pytest.raises(ValueError, match="rewards must have shape"):
            group_advantages(torch.ones(shape))


def test_response_aggregation_gives_each_response_total_weight_one_over_b():
    surrogate, mask = two_length_batch()
    loss = policy_loss(surrogate, mask, ObjectiveConfig(aggregation="response", max_completion_length=128))
    assert loss.item() == pytest.approx(-(2.0 + 0.5) / 2)
    grad = torch.autograd.grad(loss, surrogate)[0]
    assert torch.allclose(grad[0, 0, :10], torch.full((10,), -1 / 20, dtype=torch.float64))
    assert torch.allclose(grad[0, 1, :100], torch.full((100,), -1 / 200, dtype=torch.float64))
    assert grad[0, 0].sum().item() == pytest.approx(-1 / 2)
    assert grad[0, 1].sum().item() == pytest.approx(-1 / 2)


def test_fixed_aggregation_gives_each_token_one_over_b_l():
    surrogate, mask = two_length_batch()
    loss = policy_loss(surrogate, mask, ObjectiveConfig(aggregation="fixed", max_completion_length=128))
    assert loss.item() == pytest.approx(-(10 * 2.0 + 100 * 0.5) / (2 * 128))
    grad = torch.autograd.grad(loss, surrogate)[0]
    assert torch.allclose(grad[mask.bool()], torch.full((110,), -1 / 256, dtype=torch.float64))


@pytest.mark.parametrize("aggregation", ["response", "fixed"])
def test_padding_contributes_no_loss_or_gradient(aggregation):
    config = ObjectiveConfig(aggregation=aggregation, max_completion_length=128)
    surrogate, mask = two_length_batch()
    baseline = policy_loss(surrogate, mask, config)
    noisy = surrogate.detach().clone()
    noisy[mask == 0] = 1e6
    noisy.requires_grad_()
    loss = policy_loss(noisy, mask, config)
    assert loss.item() == pytest.approx(baseline.item())
    assert torch.equal(torch.autograd.grad(loss, noisy)[0][mask == 0], torch.zeros(146, dtype=torch.float64))


def test_fixed_length_ignores_observed_and_padded_lengths():
    config = ObjectiveConfig(aggregation="fixed", max_completion_length=128)
    expected = -70 / 256  # max(T_i)=100 would give -70/200; width only changes padding.
    for width in (100, 128, 300):
        surrogate, mask = two_length_batch(width=width)
        assert policy_loss(surrogate, mask, config).item() == pytest.approx(expected)
    doubled = dataclasses.replace(config, max_completion_length=256)
    assert policy_loss(surrogate, mask, doubled).item() == pytest.approx(expected / 2)


def test_boolean_mask_matches_float_mask():
    surrogate, mask = two_length_batch()
    for aggregation in ("response", "fixed"):
        config = ObjectiveConfig(aggregation=aggregation, max_completion_length=128)
        assert policy_loss(surrogate, mask.bool(), config).item() == policy_loss(surrogate, mask, config).item()


@pytest.mark.parametrize("aggregation", ["response", "token", "fixed"])
def test_empty_response_in_batch_is_rejected(aggregation):
    mask = torch.tensor([[[1., 1.], [0., 0.]]])
    with pytest.raises(ValueError, match="at least one valid token"):
        policy_loss(torch.ones(1, 2, 2), mask, ObjectiveConfig(aggregation=aggregation))
    with pytest.raises(ValueError, match="at least one valid token"):
        compute_loss(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.tensor([[1., 0.]]), mask,
                     ObjectiveConfig(aggregation=aggregation))


@pytest.mark.parametrize("mask_shape", [(1, 2), (1, 2, 4), (2, 1, 3)])
def test_mask_shape_mismatch_is_rejected(mask_shape):
    with pytest.raises(ValueError):
        policy_loss(torch.ones(1, 2, 3), torch.ones(mask_shape), ObjectiveConfig())
    with pytest.raises(ValueError):
        compute_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3), torch.tensor([[1., 0.]]),
                     torch.ones(mask_shape), ObjectiveConfig())


def test_invalid_aggregation_settings_are_rejected():
    surrogate, mask = torch.ones(1, 1, 2), torch.ones(1, 1, 2)
    with pytest.raises(ValueError, match="unsupported aggregation"):
        policy_loss(surrogate, mask, ObjectiveConfig(aggregation="sum"))
    for length in (0, -1):
        with pytest.raises(ValueError, match="max_completion_length"):
            policy_loss(surrogate, mask, ObjectiveConfig(aggregation="fixed", max_completion_length=length))


@pytest.mark.parametrize("name,advantage,aggregation", ALL_EXPERIMENTS)
def test_e0_e3_loss_and_gradient_match_reference(name, advantage, aggregation):
    new, old, rewards, mask, log_ratios = synthetic_batch()
    loss = compute_loss(new, old, rewards, mask, preset(name, MAX_LENGTH))
    expected, expected_grads = ref_loss_and_grads(
        REWARDS, log_ratios, LENGTHS, advantage, aggregation, MAX_LENGTH
    )
    assert loss.item() == pytest.approx(expected, rel=1e-12)
    grad = torch.autograd.grad(loss, new)[0]
    expected_grad = torch.zeros_like(grad)
    for index, value in expected_grads.items():
        expected_grad[index] = value
    assert torch.allclose(grad, expected_grad, atol=1e-12)
    assert torch.equal(grad[mask == 0], torch.zeros_like(grad[mask == 0]))


def test_e0_e3_are_numerically_distinct_on_identical_inputs():
    new, old, rewards, mask, _ = synthetic_batch()
    snapshot = [t.detach().clone() for t in (new, old, rewards, mask)]
    losses = [
        compute_loss(new, old, rewards, mask, preset(name, MAX_LENGTH)).item()
        for name, _, _ in ALL_EXPERIMENTS
    ]
    for t, before in zip((new, old, rewards, mask), snapshot):
        assert torch.equal(t, before)
    assert min(abs(a - b) for i, a in enumerate(losses) for b in losses[i + 1:]) > 1e-4


# Hand example: r=[1,0], T=[1,3], L=4, new=old so every ratio is 1, sigma=0.5.
# Each row is (loss, gradient on the correct token, gradient on each wrong token).
@pytest.mark.parametrize("name,loss,correct,wrong", [
    ("grpo_core", 0.0, -1 / 2, 1 / 6),
    ("length_only", 1 / 4, -1 / 8, 1 / 8),
    ("std_only", 0.0, -1 / 4, 1 / 12),
    ("dr_grpo_core", 1 / 8, -1 / 16, 1 / 16),
])
def test_hand_derived_signs_and_magnitudes(name, loss, correct, wrong):
    new = torch.zeros(1, 2, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[[1., 0., 0.], [1., 1., 1.]]], dtype=torch.float64)
    rewards = torch.tensor([[1., 0.]], dtype=torch.float64)
    value = compute_loss(new, torch.zeros_like(new), rewards, mask, preset(name, 4))
    assert value.item() == pytest.approx(loss, abs=1e-12)
    grad = torch.autograd.grad(value, new)[0]
    expected = torch.tensor([[[correct, 0., 0.], [wrong, wrong, wrong]]], dtype=torch.float64)
    assert torch.allclose(grad, expected, atol=1e-12)


def test_only_new_logp_receives_gradient():
    new = torch.tensor([[[-1.0, -2.0], [-1.5, -0.5]]], requires_grad=True)
    old = (new.detach() + 0.05).requires_grad_()
    rewards = torch.tensor([[1., 0.]], requires_grad=True)
    compute_loss(new, old, rewards, torch.ones(1, 2, 2), PRESETS["dr_grpo_core"]).backward()
    assert new.grad is not None and new.grad.abs().sum() > 0
    assert old.grad is None
    assert rewards.grad is None


def test_policy_objective_freezes_supplied_advantages():
    new = torch.zeros(1, 2, 1, requires_grad=True)
    adv = torch.tensor([[1., -1.]], requires_grad=True)
    policy_objective(new, torch.zeros_like(new), adv, ObjectiveConfig()).sum().backward()
    assert adv.grad is None


@pytest.mark.parametrize("name,advantage,aggregation", ALL_EXPERIMENTS)
@pytest.mark.parametrize("pad_new,pad_old", [
    (0.0, -math.inf),
    (math.inf, 0.0),
    (math.nan, 0.0),
    (-1.0, -100.0),  # Finite, but exp(99) overflows float32.
])
def test_arbitrary_padding_values_cannot_reach_loss_or_gradient(name, advantage, aggregation, pad_new, pad_old):
    new, old, rewards, mask, log_ratios = synthetic_batch(pad_new, pad_old)
    new, old, rewards, mask = (
        new.detach().float().requires_grad_(), old.float(), rewards.float(), mask.float()
    )
    loss = compute_loss(new, old, rewards, mask, preset(name, MAX_LENGTH))
    expected, _ = ref_loss_and_grads(REWARDS, log_ratios, LENGTHS, advantage, aggregation, MAX_LENGTH)
    assert loss.item() == pytest.approx(expected, rel=1e-5)
    grad = torch.autograd.grad(loss, new)[0]
    assert torch.isfinite(grad).all()
    assert torch.equal(grad[mask == 0], torch.zeros_like(grad[mask == 0]))


@pytest.mark.parametrize("name", ["length_only", "dr_grpo_core"])
def test_fixed_length_token_coefficients_do_not_depend_on_response_length(name):
    # Equal rewards pair up as (1,0)/(1,0): same |A|, all ratios 1, so nothing is clipped.
    lengths = torch.tensor([[1, 2, 4, 8]])
    mask = (torch.arange(8).expand(1, 4, 8) < lengths.unsqueeze(-1)).double()
    new = torch.zeros(1, 4, 8, dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[1., 0., 1., 0.]], dtype=torch.float64)
    loss = compute_loss(new, torch.zeros_like(new), rewards, mask, preset(name, 16))
    grad = torch.autograd.grad(loss, new)[0][mask.bool()].abs()
    assert grad.shape == (15,)
    assert torch.allclose(grad, grad[:1].expand(15))


@pytest.mark.parametrize("ratio,adv,clip_low,clip_high,value,slope", [
    (1.25, 1.0, 0.2, 0.2, 1.2, 0.0),  # Above upper bound, A>0: frozen.
    (0.75, -1.0, 0.2, 0.2, -0.8, 0.0),  # Below lower bound, A<0: frozen.
    (1.25, -1.0, 0.2, 0.2, -1.25, -1.25),  # Above upper bound, A<0: pessimistic term stays live.
    (0.75, 1.0, 0.2, 0.2, 0.75, 0.75),  # Below lower bound, A>0: pessimistic term stays live.
    (1.1, 2.0, 0.2, 0.2, 2.2, 2.2),  # Inside bounds both branches tie; slope must not halve.
    (0.9, -2.0, 0.2, 0.2, -1.8, -1.8),
    (1.25, 1.0, 0.1, 0.3, 1.25, 1.25),  # Each bound uses its own threshold.
    (0.85, -1.0, 0.1, 0.3, -0.9, 0.0),
])
def test_clipping_value_and_gradient(ratio, adv, clip_low, clip_high, value, slope):
    new = torch.full((1, 1, 1), math.log(ratio), dtype=torch.float64, requires_grad=True)
    config = ObjectiveConfig(clip_low=clip_low, clip_high=clip_high)
    out = policy_objective(new, torch.zeros_like(new), torch.tensor([[adv]], dtype=torch.float64), config)
    assert out.item() == pytest.approx(value)
    assert torch.autograd.grad(out.sum(), new)[0].item() == pytest.approx(slope)


@pytest.mark.parametrize("name,_advantage,_aggregation", ALL_EXPERIMENTS)
def test_homogeneous_group_has_zero_policy_gradient(name, _advantage, _aggregation):
    new, old, rewards, mask, _ = synthetic_batch()
    grad = torch.autograd.grad(compute_loss(new, old, rewards, mask, preset(name, MAX_LENGTH)), new)[0]
    assert torch.equal(grad[2].abs(), torch.zeros_like(grad[2]))
    assert grad[:2].abs().sum() > 0


@pytest.mark.parametrize("name,advantage,aggregation", ALL_EXPERIMENTS)
def test_preset_maps_to_experiment_definition(name, advantage, aggregation):
    assert (PRESETS[name].advantage, PRESETS[name].aggregation) == (advantage, aggregation)


def test_e0_e3_form_a_complete_two_by_two_factorial():
    cells = {(PRESETS[name].advantage, PRESETS[name].aggregation) for name, _, _ in ALL_EXPERIMENTS}
    assert cells == {("std", "response"), ("std", "fixed"), ("center", "response"), ("center", "fixed")}


def test_e0_e3_share_every_other_hyperparameter():
    def rest(name):
        fields = dataclasses.asdict(PRESETS[name])
        del fields["advantage"], fields["aggregation"]
        return fields

    baseline = rest("grpo_core")
    assert baseline == {"clip_low": 0.2, "clip_high": 0.2, "max_completion_length": 512, "eps": 1e-8}
    assert all(rest(name) == baseline for name, _, _ in ALL_EXPERIMENTS)
