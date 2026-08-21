from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_waypoint_v2_videos.py"
SPEC = importlib.util.spec_from_file_location("extract_waypoint_v2_videos", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACT)


def _complete_rows(episode: str) -> list[dict]:
    routes = ("NAV_TO_SOURCE", "PICK", "NAV_TO_TARGET", "PLACE", "DONE")
    transitions = tuple(EXTRACT.BOUNDARY_EVENTS)
    return [
        {
            "source_episode_id": episode,
            "source_row_id": index,
            "route": route,
            "boundary_transition": transitions[index] if index < 4 else None,
        }
        for index, route in enumerate(routes)
    ]


def test_select_transition_prefers_nearest_before_row() -> None:
    rows = [
        {
            "transition_window": True,
            "boundary_transition": "PICK->NAV_TO_TARGET",
            "boundary_signed_time_s": value,
            "boundary_class": boundary_class,
            "source_episode_id": "episode",
            "source_row_id": index,
        }
        for index, (value, boundary_class) in enumerate(
            ((-0.2, "BEFORE"), (0.0, "AFTER"), (0.0, "BEFORE"))
        )
    ]
    assert EXTRACT._select_transition(rows, "PICK->NAV_TO_TARGET")["source_row_id"] == 2


def test_select_full_episode_requires_every_route_and_transition() -> None:
    complete = _complete_rows("episode_b")
    selected_id, _ = EXTRACT._select_full_episode(
        {
            "episode_b": complete,
            "episode_a": _complete_rows("episode_a"),
            "incomplete": complete[:-1],
        }
    )
    assert selected_id == "episode_a"
    with pytest.raises(RuntimeError, match="no complete"):
        EXTRACT._select_full_episode({"incomplete": complete[:-1]})
