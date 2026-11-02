"""GSM8K final-number verification. Strict by design; no process rewards."""
import re
from decimal import Decimal, InvalidOperation

_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_FINAL = re.compile(r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)", re.I)
_BOXED = re.compile(r"\\boxed\{\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*\}")


def _decimal(text: str):
    try:
        return Decimal(text.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def gsm8k_ground_truth(answer: str):
    """GSM8K references end with '#### number'."""
    if "####" not in answer:
        return None
    suffix = answer.rsplit("####", 1)[-1].strip()
    return _decimal(suffix) if _NUMBER.fullmatch(suffix) else None


def extract_final_number(response: str):
    """Require an explicit final-answer marker, avoiding numbers in reasoning.

    Allowed formats: 'Final answer: 42', 'Answer is 42', '\\boxed{42}'.
    If multiple markers exist, the LAST marker wins. This is intentionally
    conservative. Document parser miss rate before using for real training.
    """
    matches = [(m.start(), m.group(1)) for rx in (_FINAL, _BOXED) for m in rx.finditer(response)]
    return _decimal(max(matches)[1]) if matches else None


def exact_number_reward(response: str, ground_truth):
    pred = extract_final_number(response)
    return float(pred is not None and ground_truth is not None and pred == ground_truth)
