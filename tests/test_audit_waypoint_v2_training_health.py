from pathlib import Path

from scripts import audit_waypoint_v2_training_health as audit


def _event(step: int) -> dict:
    return {
        "event": "train_step",
        "step": step,
        "valid_optimizer_step": True,
        "loss": 100.0 / step,
        "answer_loss": 2.0,
        "route_loss": 1.0,
        "navigation_loss": 2.0,
        "manipulation_loss": 3.0,
        "gradient_norm": 10.0,
        "vlm_gradient_norm": 8.0,
        "navigation_gradient_norm": 4.0,
        "manipulation_gradient_norm": 5.0,
        "auxiliary_gradient_norm": 1.0,
        "optimizer_step_time_s": 20.0,
        "samples_per_second": 1.2,
        "gpu_hours_per_step": 4.0 * 20.0 / 3600.0,
        "peak_allocated_memory_mib": 40_000.0,
        "peak_reserved_memory_mib": 50_000.0,
        "learning_rates": [1.0e-6, 1.0e-5],
    }


def _resolved(output: Path) -> dict:
    return {
        "schema_version": audit.RUN_SCHEMA,
        "world_size": 4,
        "visible_gpu_uuids": "gpu0,gpu1,gpu2,gpu3",
        "max_steps": 10_000,
        "training_subset": False,
        "arguments": {
            "output_dir": str(output),
            "save_interval_steps": 500,
            "save_first_checkpoint_step": 500,
        },
        "resolved_policy_config": {
            "auxiliary": {
                "enable_boundary_progress": True,
                "enable_prefix": False,
                "enable_crl": False,
            }
        },
    }


def test_formal_health_accepts_twenty_live_finite_steps(tmp_path: Path) -> None:
    report = audit.audit_training_health(
        [_event(step) for step in range(1, 21)],
        _resolved(tmp_path),
        {"status": "running", "global_step": 20},
        events_path=tmp_path / "events.jsonl",
        minimum=20,
        gpu_memory_total_mib=97_871.0,
        formal=True,
    )
    assert report["ok"] is True
    assert report["observed_steps"] == list(range(1, 21))


def test_formal_health_rejects_stale_state_and_wrong_save_policy(
    tmp_path: Path,
) -> None:
    resolved = _resolved(tmp_path)
    resolved["arguments"]["save_interval_steps"] = 100
    report = audit.audit_training_health(
        [_event(step) for step in range(1, 21)],
        resolved,
        {"status": "running", "global_step": 0},
        events_path=tmp_path / "events.jsonl",
        minimum=20,
        gpu_memory_total_mib=97_871.0,
        formal=True,
    )
    assert report["ok"] is False
    assert any("interval" in problem for problem in report["problems"])
    assert any("lags" in problem for problem in report["problems"])
