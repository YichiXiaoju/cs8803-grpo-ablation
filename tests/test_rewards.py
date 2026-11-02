from decimal import Decimal
from grpo_ablation.rewards import gsm8k_ground_truth, extract_final_number, exact_number_reward


def test_ground_truth():
    assert gsm8k_ground_truth("workings... #### 1,234") == Decimal("1234")
    assert gsm8k_ground_truth("no delimiter") is None


def test_final_answer_only():
    gt = Decimal("42")
    assert exact_number_reward("We thought 42. Final answer: 41", gt) == 0
    assert exact_number_reward("Computation. Final answer: 42.0", gt) == 1
    assert exact_number_reward("Computation. \\boxed{42}", gt) == 1
    assert extract_final_number("I considered 42, but cannot finish") is None
