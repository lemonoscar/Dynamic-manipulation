import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "formal_closed_loop", Path(__file__).resolve().parents[1] / "scripts/run_formal_closed_loop.py")
closed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(closed)


def test_timeout_preserves_queries_and_physics_without_claiming_success(tmp_path):
    path = tmp_path / "joint_trajectory_trace.jsonl"
    events = [
        {"event": "model_query", "response": {"committed_route": "PICK", "manipulation": {
            "position_saturation_count": 1, "rate_saturation_count": 2, "gripper_saturation_count": 3}}},
        {"event": "physical_pick_verified"}, {"event": "physical_carry_verified"},
    ]
    path.write_text("".join(json.dumps(event) + "\n" for event in events) + '{"event":')
    result = closed.recover_partial_trace(tmp_path)
    assert result["query_count"] == 1
    assert result["final_route"] == "PICK"
    assert result["physics_evidence"] == {"pick_verified": True, "carry_verified": True}
    assert result["saturation"]["denominator"] == 70
    assert result["truncated_final_trace_line"] is True
    assert "success" not in result
    path.write_text('{"event":\n')
    with pytest.raises(json.JSONDecodeError):
        closed.recover_partial_trace(tmp_path)


def test_startup_failures_and_queried_timeouts_have_separate_denominators(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    manifest = {"environment": "migration", "tasks": [1, 2],
                "profiles": ["source_assisted"], "diffusion_seeds": [17]}
    args = SimpleNamespace(manifest=manifest_path, limit=0, max_queries=96)
    base = dict(profile="source_assisted", success=False, transfer_chain_success=False)
    report = closed.aggregate([
        dict(base, episode_id="a", failure_reason="runtime_launch_failed"),
        dict(base, episode_id="b", failure_reason="wall_clock_timeout", query_count=2, final_route="PICK",
             physics_evidence={"pick_verified": True}, saturation={
                 "position_events": 0, "rate_events": 0, "gripper_events": 2, "denominator": 70}),
    ], manifest, args)
    metrics = report["profiles"]["source_assisted"]
    assert report["status"] == "complete"
    assert metrics["attempts_with_model_queries"] == 1
    assert metrics["attempts_without_model_queries"] == 1
    assert metrics["contract_success"]["episodes"] == 2
    assert metrics["contract_success_on_queried_attempts"]["episodes"] == 1
    assert metrics["pick_success"]["episode_mean"] == .5
    assert metrics["saturation_gate"]["passed"] is False
