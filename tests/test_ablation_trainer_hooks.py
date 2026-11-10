"""AblationTrainer rollout hook with TRL's generation stubbed out (needs trl, datasets, yaml)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("trl")
pytest.importorskip("datasets")
pytest.importorskip("yaml")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from trl import GRPOTrainer  # noqa: E402
from train_ice_ablation import AblationTrainer, AttemptLedger, BudgetReached  # noqa: E402
from grpo_ablation.overlong import OverlongShapedReward  # noqa: E402

EOS = 7
ROW = {"prompt": "p", "ground_truth": "1"}


def make(tmp_path, monkeypatch, outputs, shaped=None, dynamic=True, target=10**6):
    t = AblationTrainer.__new__(AblationTrainer)
    t.model = SimpleNamespace(training=True)
    t.accelerator = SimpleNamespace(num_processes=1)
    t.args = SimpleNamespace(seed=42, generation_batch_size=4)
    t.state = SimpleNamespace(global_step=3)
    t.train_dataset = [dict(prompt=f"q{i}", ground_truth=str(i)) for i in range(20)]
    t.eos_token_id = EOS
    t.ledger = AttemptLedger(tmp_path / "t.jsonl", target)
    t.dynamic_sampling, t.shaped = dynamic, shaped
    queue = iter(outputs)

    def fake(self, inputs):
        out = next(queue)
        if shaped is not None:
            shaped.last_correct = out.pop("correct")
        return out
    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", fake)
    return t


def out(lengths, adv, correct=None, masked=False):
    ids = torch.full((4, max(lengths)), EOS)
    for i, n in enumerate(lengths):
        ids[i, :n - 1] = 1                      # n tokens including the EOS
    o = {"completion_ids": ids, "completion_mask": torch.zeros_like(ids) if masked else torch.ones_like(ids),
         "advantages": torch.tensor(adv)}
    if correct is not None:
        o["correct"] = correct
    return o


def test_masked_completions_still_count(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, [out([10, 20, 30, 40], [0.0] * 4, masked=True), out([5, 5, 5, 5], [1, -1, 1, -1], masked=True)])
    t._generate_and_score_completions([ROW] * 4)
    assert t.ledger.tokens == 100 + 20 and t.ledger.rejected_tokens == 100 and t.ledger.attempts == 2


def test_overlong_arm_judges_group_by_correctness(tmp_path, monkeypatch):
    shaped = OverlongShapedReward(lambda c, ground_truth: [], 192, 40)
    # all-correct group: shaped rewards differ so advantages are nonzero, but there is no learning signal
    outputs = [out([100, 172, 100, 100], [0.7, -1.5, 0.4, 0.4], correct=[1.0] * 4),
               out([100] * 4, [1, -1, 1, -1], correct=[1.0, 0.0, 1.0, 0.0])]
    t = make(tmp_path, monkeypatch, outputs, shaped=shaped)
    res = t._generate_and_score_completions([ROW] * 4)
    assert t.ledger.attempts == 2 and t.ledger.rejected_tokens == 472
    assert res["advantages"].tolist() == [1, -1, 1, -1]


def test_plain_dynamic_sampling_uses_advantages(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, [out([9] * 4, [0.0] * 4), out([9] * 4, [1, -1, 1, -1])])
    t._generate_and_score_completions([ROW] * 4)
    assert t.ledger.attempts == 2


def test_disabled_dynamic_sampling_accepts_first_group(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, [out([9] * 4, [0.0] * 4)], dynamic=False)
    t._generate_and_score_completions([ROW] * 4)
    assert t.ledger.attempts == 1 and t.ledger.rejected_tokens == 0


def test_budget_stops_rejection_loop(tmp_path, monkeypatch):
    t = make(tmp_path, monkeypatch, [out([100] * 4, [0.0] * 4)], target=300)
    with pytest.raises(BudgetReached):
        t._generate_and_score_completions([ROW] * 4)
        t._generate_and_score_completions([ROW] * 4)
