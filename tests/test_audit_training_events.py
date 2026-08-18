from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_training_events.py"
SPEC = importlib.util.spec_from_file_location("audit_training_events", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _event(step: int, *, include_routing: bool = True) -> dict:
    event = {
        "event": "train_step",
        "step": step,
        "subtask_loss": 1.0,
        "action_loss": 1.0,
        "navigation_loss": 1.0,
        "manipulation_loss": 1.0,
        "gradient_norm": 1.0,
        "vlm_gradient_norm": 1.0,
        "navigation_gradient_norm": 1.0,
        "manipulation_gradient_norm": 1.0,
        "teacher_forcing_probability": 0.5,
        "learning_rates": [1.0e-5],
    }
    if include_routing:
        event["routing"] = {
            "observed_samples": 256,
            "action_routed_samples": 240,
            "action_routed_fraction": 240 / 256,
        }
    return event


def _write_events(path: Path, *, include_routing: bool = True) -> None:
    path.write_text(
        "".join(
            json.dumps(_event(step, include_routing=include_routing)) + "\n"
            for step in range(1, 21)
        ),
        encoding="utf-8",
    )


def test_health_audit_accepts_aggregated_routing_summary(tmp_path, capsys) -> None:
    events = tmp_path / "events.jsonl"
    _write_events(events)

    assert AUDIT.main(["--events", str(events)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "conveyor-vla-al0-training-health-audit-2"
    assert report["ok"] is True


def test_health_audit_rejects_missing_routing_summary(tmp_path, capsys) -> None:
    events = tmp_path / "events.jsonl"
    _write_events(events, include_routing=False)

    assert AUDIT.main(["--events", str(events)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any("no routing summary" in problem for problem in report["problems"])
