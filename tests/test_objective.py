import pytest
import torch
from grpo_ablation.objective import ObjectiveConfig, group_advantages, policy_objective, policy_loss


def test_group_advantages_and_zero_variance():
    r = torch.tensor([[1., 0., 1., 0.], [1., 1., 1., 1.]])
    assert torch.allclose(group_advantages(r, "std")[0], torch.tensor([1., -1., 1., -1.]))
    assert torch.all(group_advantages(r, "std")[1] == 0)
    assert torch.allclose(group_advantages(r, "center")[0], torch.tensor([.5, -.5, .5, -.5]))


def test_aggregation_example():
    x = torch.ones(1, 2, 80)
    m = torch.zeros_like(x)
    m[0, 0, :20] = 1
    m[0, 1, :] = 1
    expected = {"response": -1., "token": -1., "fixed": -.5}
    for aggregation, val in expected.items():
        cfg = ObjectiveConfig(aggregation=aggregation, max_completion_length=100)
        assert policy_loss(x, m, cfg).item() == pytest.approx(val)


def test_clip_higher_positive_advantage():
    new = torch.tensor([[[torch.log(torch.tensor(1.25))]]], requires_grad=True)
    old = torch.zeros_like(new)
    pos = torch.ones(1, 1)
    grpo = policy_objective(new, old, pos, ObjectiveConfig(clip_high=.2))
    dapo = policy_objective(new, old, pos, ObjectiveConfig(clip_high=.28))
    assert grpo.item() == pytest.approx(1.2)
    assert dapo.item() == pytest.approx(1.25)
    assert torch.autograd.grad(grpo.sum(), new, retain_graph=True)[0].item() == pytest.approx(0.)
    assert torch.autograd.grad(dapo.sum(), new)[0].item() > 0


def test_negative_advantage_lower_clipping():
    new = torch.tensor([[[torch.log(torch.tensor(.75))]]], requires_grad=True)
    v = policy_objective(new, torch.zeros_like(new), -torch.ones(1, 1), ObjectiveConfig(clip_high=.28))
    assert v.item() == pytest.approx(-.8)
    assert torch.autograd.grad(v.sum(), new)[0].item() == pytest.approx(0.)


def test_padding_has_no_gradient():
    x = torch.ones(1, 1, 3, requires_grad=True)
    mask = torch.tensor([[[1., 1., 0.]]])
    policy_loss(x, mask, ObjectiveConfig()).backward()
    assert x.grad[0, 0, 2].item() == 0


def test_old_logp_is_detached():
    new = torch.zeros(1, 1, 1, requires_grad=True)
    old = torch.zeros(1, 1, 1, requires_grad=True)
    adv = torch.ones(1, 1)
    v = policy_objective(new, old, adv, ObjectiveConfig()).sum()
    v.backward()
    assert new.grad is not None and new.grad.abs().sum() > 0
    assert old.grad is None


def test_dapo_dr_scale_ratio():
    x = torch.tensor([[[1.] * 10 + [0.] * 90, [1.] * 100]], requires_grad=True)
    m = torch.zeros_like(x)
    m[0, 0, :10] = 1
    m[0, 1, :] = 1
    a = policy_loss(x, m, ObjectiveConfig(aggregation="token"))
    b = policy_loss(x, m, ObjectiveConfig(aggregation="fixed", max_completion_length=100))
    ga = torch.autograd.grad(a, x, retain_graph=True)[0]
    gb = torch.autograd.grad(b, x)[0]
    assert torch.allclose(ga, gb * (200 / 110), atol=1e-6)


def test_empty_response_fails():
    with pytest.raises(ValueError):
        policy_loss(torch.ones(1, 1, 2), torch.zeros(1, 1, 2), ObjectiveConfig())
