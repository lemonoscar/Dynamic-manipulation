from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_dense_transition_videos.py"
SPEC = importlib.util.spec_from_file_location("extract_dense_transition_videos", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXTRACT)


def _rows(split: str, phases: tuple[str, ...]) -> list[dict[str, str]]:
    return [{"split": split, "phase_name": phase} for phase in phases]


def test_full_episode_selection_requires_all_four_phases() -> None:
    phases = tuple(phase.name for phase in EXTRACT.PHASE_ORDER)
    episodes = {
        "collection:episode_b": _rows("train", phases),
        "collection:episode_a": _rows("train", phases),
        "collection:episode_val": _rows("val", phases),
    }

    episode_id, rows = EXTRACT._select_full_episode(episodes, "train")

    assert episode_id == "collection:episode_a"
    assert [row["phase_name"] for row in rows] == list(phases)


def test_full_episode_selection_rejects_incomplete_split() -> None:
    phases = tuple(phase.name for phase in EXTRACT.PHASE_ORDER)

    with pytest.raises(RuntimeError, match="no complete four-phase episode"):
        EXTRACT._select_full_episode(
            {"collection:episode": _rows("test", phases[:-1])},
            "test",
        )
