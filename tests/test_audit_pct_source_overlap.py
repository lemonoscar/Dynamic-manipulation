from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_pct_source_overlap.py"
SPEC = importlib.util.spec_from_file_location("audit_pct_source_overlap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _episode(root: Path, name: str, frames: str) -> None:
    episode = root / name
    episode.mkdir(parents=True)
    (episode / "frames.jsonl").write_text(frames, encoding="utf-8")
    (episode / "samples.jsonl").write_text("{}\n", encoding="utf-8")


def test_source_overlap_audit_rejects_exact_cross_collection_trajectory(
    tmp_path: Path,
) -> None:
    first = tmp_path / "liangzhu_0815_n200"
    second = tmp_path / "liangzhu_0815_n400"
    _episode(first, "episode_000001", "same\n")
    _episode(second, "episode_000001", "same\n")
    output = tmp_path / "overlap.json"

    result = AUDIT.main(
        [
            "--source-root",
            str(first),
            "--source-root",
            str(second),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert result == 1
    assert report["fingerprinted_episodes"] == 2
    assert len(report["exact_duplicate_trajectory_groups"]) == 1
