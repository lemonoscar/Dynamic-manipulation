
import numpy as np
import pytest

from conveyor_bench.conveyorvla.formal_checkpoint import validate_formal_checkpoint, write_json
from conveyor_bench.conveyorvla.formal_metrics import trajectory_metrics, summarize, cluster_mean


def row(route="NAV_TO_SOURCE", predicted="RECOVER", episode="ep0"):
    return dict(episode_id=episode, target_route=route, predicted_route=predicted,
                route_correct=route == predicted, raw_route_correct=route == predicted,
                route_valid=predicted != "RECOVER", predicted=None, oracle=None, baseline=None,
                action_failure="cross_domain" if route != predicted else None,
                recover_reason=None, transition_window=False, transition_id=None,
                gripper_transition=False, real_future_points=10)


def test_wrong_routes_stay_in_denominator_and_missing_actions_are_null():
    report = summarize([row(), row(predicted="NAV_TO_SOURCE", episode="ep1")])
    assert report["route_accuracy"]["sample_mean"] == .5
    assert report["confusion"]["NAV_TO_SOURCE"]["RECOVER"] == 1
    assert report["action_coverage"]["sample_mean"] == 0
    assert report["saturation_gate"]["predicted"]["passed"] is None


def test_yaw_wrap_and_real_future_hold_are_separate():
    p = [[0, 0, -np.pi + .01] for _ in range(10)]
    t = [[0, 0, np.pi - .01] for _ in range(10)]
    p[-1][0] = 10
    m = trajectory_metrics("NAV_TO_SOURCE", p, t, None, 9)
    assert m["yaw_mae_rad"] == pytest.approx(.02)
    assert m["real_future_error"] == 0
    assert m["hold_tail_error"] == 10
    assert m["xy_ade_m"] == 1


def test_clipping_event_rate_preserves_overlapping_events_and_unique_fraction():
    p = [[20.] * 6 + [2.]] * 10
    target = [[0.] * 7] * 10
    m = trajectory_metrics("PICK", p, target, [0.] * 13)
    assert m["position_saturation_rate"] == 1
    assert m["gripper_saturation_rate"] == 1
    assert m["saturation_rate"] > 1  # event rate, not unique clipped-channel probability
    assert m["unique_saturation_fraction"] == 1


def test_confidence_resamples_episodes_not_correlated_frames():
    m = cluster_mean([0.] * 99 + [1.], ["a"] * 99 + ["b"])
    assert m["sample_mean"] == .01
    assert m["episode_mean"] == .5
    assert m["episodes"] == 2
    assert cluster_mean([1], ["a"])["ci95"] is None
    assert cluster_mean([0., 0.], ["a", "b"])["ci95"] == [0., 0.]
    assert cluster_mean([False, False], ["a", "b"])["ci95"][1] > 0
    with pytest.raises(ValueError):
        cluster_mean([float("nan")], ["a"])


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_trajectory_rejected(bad):
    p = [[0., 0., 0.]] * 9 + [[bad, 0, 0]]
    with pytest.raises(ValueError, match="finiteness"):
        trajectory_metrics("NAV_TO_TARGET", p, [[0., 0., 0.]] * 10, None)


def test_atomic_json_never_serializes_nan(tmp_path):
    with pytest.raises(ValueError):
        write_json(tmp_path / "report.json", {"metric": float("nan")})
    assert not (tmp_path / "report.json").exists()


def test_binding_rejects_tampered_run_before_loading_weights(tmp_path):
    # Minimal gate fixture deliberately stops before config/data/model construction.
    ckpt = tmp_path / "checkpoints/step_000002"
    ckpt.mkdir(parents=True)
    manifest = {"global_step": 2, "max_steps": 2, "run_kind": "formal", "stage_a_steps": 1,
                "model_contract_id": "conveyorvla-joint-trajectory-policy-5hz-v1",
                "dataset_schema_version": "conveyorvla-joint-trajectory-5hz-v1",
                "dataset_manifest_sha256": "a", "normalization_sha256": "b",
                "normalizer_id": "n", "policy_config_sha256": "p"}
    write_json(ckpt / "joint_trajectory_checkpoint_manifest.json", manifest)
    write_json(tmp_path / "resolved_run.json", manifest)
    (tmp_path / "CHECKSUMS.sha256").write_text("0" * 64 + "  resolved_run.json\n")
    with pytest.raises(ValueError, match="saved checksum mismatch: resolved_run"):
        validate_formal_checkpoint(ckpt, tmp_path / "config.json")
    manifest["global_step"] = 1
    write_json(ckpt / "joint_trajectory_checkpoint_manifest.json", manifest)
    with pytest.raises(ValueError, match="directory/step"):
        validate_formal_checkpoint(ckpt, tmp_path / "config.json")
