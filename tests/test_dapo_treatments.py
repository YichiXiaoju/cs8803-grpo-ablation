import pytest

from grpo_ablation.treatments import MULTI_UPDATE, SETTINGS, validate_treatment


def cfg(name, **extra):
    loss, scale, dynamic, required = SETTINGS[name]
    out = {"experiment": name, "loss_type": loss, "scale_rewards": scale, "dynamic_sampling": dynamic, **required}
    if name in MULTI_UPDATE:
        out["num_iterations"] = 4
    out.update(extra)
    return out


@pytest.mark.parametrize("name", list(SETTINGS))
def test_every_arm_validates(name):
    t = validate_treatment(cfg(name))
    assert t["num_iterations"] == (4 if name in MULTI_UPDATE else 1)


def test_original_arms_unchanged():
    assert validate_treatment({"experiment": "E5", "loss_type": "grpo", "scale_rewards": "group",
                               "dynamic_sampling": True}) == {
        "epsilon_high": None, "num_iterations": 1, "overlong_filtering": False, "overlong_penalty": False}


def test_e4_pair_differs_only_in_epsilon_high():
    a, b = cfg("E4C"), cfg("E4")
    assert {k for k in a if a[k] != b[k]} == {"experiment", "epsilon_high"}


def test_multi_update_arms_need_iterations():
    bad = cfg("E4"); bad["num_iterations"] = 1
    with pytest.raises(ValueError):
        validate_treatment(bad)
    missing = cfg("E4"); del missing["num_iterations"]
    with pytest.raises(ValueError):
        validate_treatment(missing)


def test_single_update_arms_reject_iterations():
    with pytest.raises(ValueError):
        validate_treatment(cfg("E7", num_iterations=2))


def test_wrong_treatment_rejected():
    with pytest.raises(ValueError):
        validate_treatment(cfg("E7", loss_type="grpo"))
    with pytest.raises(ValueError):
        validate_treatment(cfg("E4", epsilon_high=0.2))
    with pytest.raises(ValueError):
        validate_treatment(cfg("E7", overlong_filtering=True))
    with pytest.raises(ValueError):
        validate_treatment({"experiment": "E9"})
