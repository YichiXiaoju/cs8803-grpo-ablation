"""Treatment definitions for the 1B ablation arms (single source of truth).

Each arm fixes loss_type, scale_rewards and dynamic_sampling, plus optional DAPO
fields. Multi-update arms (E4C, E4, E6, E8) must set num_iterations >= 2: with one
update per rollout the importance ratio is exactly 1, so clipping can never fire.
"""

BASE_KEYS = ("experiment", "loss_type", "scale_rewards", "dynamic_sampling")
OPTIONAL_KEYS = ("epsilon_high", "num_iterations", "overlong_filtering", "overlong_penalty")
MULTI_UPDATE = ("E4C", "E4", "E6", "E8")

# arm -> (loss_type, scale_rewards, dynamic_sampling, required optional fields)
SETTINGS = {
    "E0": ("grpo", "group", False, {}),
    "E1": ("dr_grpo", "group", False, {}),
    "E2": ("grpo", "none", False, {}),
    "E3": ("dr_grpo", "none", False, {}),
    "E5": ("grpo", "group", True, {}),
    "E7": ("dapo", "group", False, {}),
    "E4C": ("grpo", "group", False, {"epsilon_high": 0.20}),  # control for E4: same update schedule
    "E4": ("grpo", "group", False, {"epsilon_high": 0.28}),
    "E6": ("grpo", "group", True, {"epsilon_high": 0.28}),
    "E8": ("dapo", "group", True, {
        "epsilon_high": 0.28, "overlong_filtering": True, "overlong_penalty": True,
    }),
}


def validate_treatment(overrides: dict) -> dict:
    """Check a treatment config against SETTINGS and return it with defaults filled in."""
    name = overrides.get("experiment")
    if name not in SETTINGS:
        raise ValueError(f"Unknown experiment {name!r}")
    keys = set(overrides)
    if not set(BASE_KEYS) <= keys or not keys <= set(BASE_KEYS) | set(OPTIONAL_KEYS):
        raise ValueError("An ablation config may change only the declared treatment fields.")
    loss, scale, dynamic, extra = SETTINGS[name]
    if (overrides["loss_type"], overrides["scale_rewards"], overrides["dynamic_sampling"]) != (loss, scale, dynamic):
        raise ValueError(f"Config does not implement {name}.")
    for key, value in extra.items():
        if overrides.get(key) != value:
            raise ValueError(f"{name} requires {key}={value!r}.")
    if keys - set(BASE_KEYS) - set(extra) - {"num_iterations"}:
        raise ValueError(f"{name} does not accept these fields: {sorted(keys - set(BASE_KEYS) - set(extra) - {'num_iterations'})}")
    iterations = overrides.get("num_iterations", 1)
    if name in MULTI_UPDATE and (not isinstance(iterations, int) or iterations < 2):
        raise ValueError(f"{name} requires num_iterations >= 2")
    if name not in MULTI_UPDATE and "num_iterations" in overrides:
        raise ValueError(f"{name} uses one update per rollout; remove num_iterations")
    return {
        "epsilon_high": overrides.get("epsilon_high"),
        "num_iterations": iterations,
        "overlong_filtering": bool(overrides.get("overlong_filtering", False)),
        "overlong_penalty": bool(overrides.get("overlong_penalty", False)),
    }
