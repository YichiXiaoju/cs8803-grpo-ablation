from decimal import Decimal

from grpo_ablation.rewards import (
    exact_number_reward,
    extract_final_number,
    gsm8k_ground_truth,
)


def test_ground_truth():
    assert gsm8k_ground_truth("workings... #### 1,234") == Decimal(1234)
    assert gsm8k_ground_truth("no delimiter") is None


def test_final_answer_only():
    gt = Decimal(42)
    assert exact_number_reward("We thought 42. Final answer: 41", gt) == 0
    assert exact_number_reward("Computation. Final answer: 42.0", gt) == 1
    assert exact_number_reward("Computation. \\boxed{42}", gt) == 1
    assert extract_final_number("I considered 42, but cannot finish") is None


def test_final_answer_rejects_partial_number():
    assert extract_final_number("Final answer: 7 1 / 4\n") is None
    assert exact_number_reward("Final answer: 7 1 / 4\n", Decimal(7)) == 0
