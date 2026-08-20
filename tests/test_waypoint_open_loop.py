from scripts import evaluate_waypoint_open_loop as evaluation


def _passing_metrics():
    return {
        "route_accuracy": 1.0,
        "route_recover_rate": 0.0,
        "navigation_ade_mean_m": 0.01,
        "navigation_fde_mean_m": 0.02,
        "navigation_direction_accuracy": 1.0,
        "navigation_segment_violation_rate": 0.0,
        "navigation_normalization_oob_rate": 0.0,
        "arm_position_mean_m": 0.01,
        "arm_orientation_mean_rad": 0.02,
        "arm_gripper_accuracy": 1.0,
        "arm_workspace_violation_rate": 0.0,
        "arm_step_violation_rate": 0.0,
        "arm_normalization_oob_rate": 0.0,
    }


def test_overfit_quality_gate_requires_route_action_and_safety_metrics():
    metrics = _passing_metrics()
    assert all(evaluation._overfit_checks(metrics).values())

    metrics["route_accuracy"] = 0.94
    metrics["arm_step_violation_rate"] = 0.01
    checks = evaluation._overfit_checks(metrics)
    assert checks["route_accuracy"] is False
    assert checks["arm_step_safety"] is False
    assert all(
        value
        for name, value in checks.items()
        if name not in {"route_accuracy", "arm_step_safety"}
    )


def test_overfit_quality_gate_fails_when_a_required_domain_metric_is_absent():
    metrics = _passing_metrics()
    metrics["navigation_ade_mean_m"] = None
    assert evaluation._overfit_checks(metrics)["navigation_ade"] is False
