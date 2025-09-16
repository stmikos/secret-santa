import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("BOT_TOKEN", "123456:ABCDEF")
os.environ.setdefault("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")

from app import draw_pairs  # noqa: E402


def assert_no_two_cycles(pairs: List[Tuple[int, int]]) -> None:
    assignment = {giver: receiver for giver, receiver in pairs}
    for giver, receiver in pairs:
        assert giver != receiver, "Self pair detected"
        assert assignment.get(receiver) != giver, "Two-cycle detected"


def test_draw_pairs_basic_deterministic() -> None:
    pairs = draw_pairs([1, 2, 3], [])
    assert pairs == [(1, 2), (2, 3), (3, 1)]
    assert_no_two_cycles(pairs)


def test_draw_pairs_respects_forbidden_and_backtracks() -> None:
    forbidden = [(1, 2), (2, 3), (3, 4)]
    pairs = draw_pairs([1, 2, 3, 4], forbidden)
    assert pairs == [(1, 3), (2, 4), (3, 2), (4, 1)]
    for pair in pairs:
        assert pair not in forbidden
    assert_no_two_cycles(pairs)


def test_draw_pairs_avoids_two_cycles_for_larger_set() -> None:
    pairs = draw_pairs([10, 20, 30, 40], [])
    assert pairs == [(10, 20), (20, 30), (30, 40), (40, 10)]
    assert_no_two_cycles(pairs)


def test_draw_pairs_returns_none_when_impossible() -> None:
    forbidden = [
        (1, 2), (1, 3),
        (2, 1), (2, 3),
        (3, 1), (3, 2),
    ]
    assert draw_pairs([1, 2, 3], forbidden) is None
