"""Mechanism presets, not full reproductions of named papers."""
from .objective import ObjectiveConfig

PRESETS = {
    "grpo_core": ObjectiveConfig(advantage="std", aggregation="response"),
    "dr_grpo_core": ObjectiveConfig(advantage="center", aggregation="fixed"),
    "dapo_core": ObjectiveConfig(advantage="std", aggregation="token", clip_high=0.28),
    "length_only": ObjectiveConfig(advantage="std", aggregation="fixed"),
    "std_only": ObjectiveConfig(advantage="center", aggregation="response"),
    "token_no_std": ObjectiveConfig(advantage="center", aggregation="token"),
}
