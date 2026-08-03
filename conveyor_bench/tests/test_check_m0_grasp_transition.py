from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_m0_grasp_transition.py"
SPEC = importlib.util.spec_from_file_location("check_m0_grasp_transition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def test_transition_candidate_requires_future_descent_and_close() -> None:
    chunk = [[0.0] * 10 for _ in range(16)]
    for action in chunk:
        action[5] = 0.008
        action[9] = 1.0
    record = {"canonical_action10_chunk": chunk}
    assert not GATE._is_transition_candidate(record)

    chunk[3][5] = -0.008
    chunk[13][9] = -1.0
    assert GATE._is_transition_candidate(record)
    assert GATE._first_index([False, False, True]) == 2
    assert GATE._first_index([False, False]) is None


def test_spread_keeps_endpoints() -> None:
    rows = [{"index": index} for index in range(9)]
    assert [row["index"] for row in GATE._spread(rows, 3)] == [0, 4, 8]


def test_transition_gate_requires_timing_and_executable_events() -> None:
    assert GATE._transition_ok(
        5,
        8,
        15,
        14,
        index_tolerance=4,
        executed_prefix=12,
    )
    assert not GATE._transition_ok(
        3,
        11,
        13,
        14,
        index_tolerance=4,
        executed_prefix=12,
    )
    assert not GATE._transition_ok(
        1,
        2,
        11,
        13,
        index_tolerance=4,
        executed_prefix=12,
    )
