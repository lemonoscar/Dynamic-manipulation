from types import SimpleNamespace
import numpy as np
import pytest

from conveyor_bench.conveyorvla.execution_consistency import (
    replay_schedule, navigation_decomposition, validate_dwa_inputs,
)


def test_frozen_saturation_gate_uses_samples_even_when_episode_mean_crosses_threshold():
    from conveyor_bench.conveyorvla.formal_metrics import saturation_gate
    assert saturation_gate({'sample_mean':.006, 'episode_mean':.004})['passed'] is False
    assert saturation_gate({'sample_mean':.005, 'episode_mean':.006})['passed'] is True
    assert saturation_gate({})['passed'] is None


def test_time_offset_advances_commands_but_keeps_same_state_duration_and_phase():
    phase = [({"timestamp": i*.2, "frame_index": i,
               "action": [0,0,0,i*.1,0,0,0,0,0,.04*(i != 1),.04*(i != 1)]}, {}, {}) for i in range(3)]
    now, ahead = replay_schedule(phase, 0), replay_schedule(phase, 1)
    assert [r["source_frame_index"] for r in now] == [0,1,2]
    assert [r["source_frame_index"] for r in ahead] == [1,2,2]
    assert now[0]["source_query_timestamp_s"] == ahead[0]["source_query_timestamp_s"] == 0
    assert sum(r["control_ticks"] for r in now) == sum(r["control_ticks"] for r in ahead) == 30
    assert now[0]["gripper_fraction"] == 1 and ahead[0]["gripper_fraction"] == 0
    assert ahead[-1]["end_hold"] and not now[-1]["end_hold"]


def test_navigation_poses_are_distinct_and_timeout_has_no_arrival_bound():
    r = navigation_decomposition(nominal=(0,0,0), requested=(.03,0,0), planned=(.13,0,0), measured=(.25,0,0), reached=True)
    assert r["errors"]["B_minus_A"]["xy_m"] == pytest.approx(.10)
    assert r["errors"]["C_minus_B"]["xy_m"] == pytest.approx(.12)
    assert r["errors"]["C_minus_A"]["xy_m"] == pytest.approx(.22)
    p = navigation_decomposition(nominal=(0,0,0), requested=(0,0,0), planned=(0,0,0))
    assert p["poses_xyyaw"]["C"] is None and p["errors"]["C_minus_B"] is None
    assert p["nominal_tolerance_bound_C_minus_A_m"] is None


def test_yaw_wrap_is_not_mixed_with_position_units():
    p = navigation_decomposition(nominal=(0,0,np.pi-.01), requested=(0,0,-np.pi+.01), planned=(0,0,0))
    assert p["errors"]["A_minus_G"] == pytest.approx({"xy_m":0,"yaw_rad":.02})


def test_all_free_nonempty_map_is_valid_but_missing_empty_and_invalid_maps_fail():
    path = [(0,0),(1,0)]
    def grid(a): return SimpleNamespace(occupancy=np.asarray(a),resolution=.1,origin=(0,0,0))
    assert validate_dwa_inputs(path,grid(np.zeros((4,4))))["valid_all_free_map"]
    for bad in [None,grid(np.zeros((0,4))),grid([[np.nan]])]:
        with pytest.raises(ValueError,match="dwa_invalid_map"):
            validate_dwa_inputs(path,bad)
    for bad in [[],[(0,0)],[(0,0),(float('nan'),1)]]:
        with pytest.raises(ValueError,match="dwa_invalid_path"):
            validate_dwa_inputs(bad,grid(np.zeros((4,4))))
    with pytest.raises(ValueError,match="dwa_degenerate_path"):
        validate_dwa_inputs([(0,0),(0,0)],grid(np.zeros((4,4))))


def test_dwa_rejects_collapsed_source_path_before_projection_and_preserves_reason():
    from conveyor_bench.conveyorvla.waypoint_planner_adapters import ArmVLADWAControllerAdapter, APPROVED_ARM_VLA_COMMIT
    def controller(*args, **kwargs):
        pytest.fail("invalid path must not reach reference projection")
    adapter=ArmVLADWAControllerAdapter(controller,None,reference_commit=APPROVED_ARM_VLA_COMMIT)
    grid=SimpleNamespace(occupancy=np.zeros((8,8)),resolution=.2,origin=(-2.,4.,0.))
    with pytest.raises(ValueError,match="dwa_degenerate_path"):
        adapter.command([(-1.515245,6.024866)]*2,(-1.45,5.89,1.72),(0,0,0),grid)
    assert adapter.last_trace["failure_status"] == "dwa_degenerate_path"


def test_no_feasible_control_is_distinct_from_an_all_free_map():
    from conveyor_bench.conveyorvla.waypoint_planner_adapters import ArmVLADWAControllerAdapter, APPROVED_ARM_VLA_COMMIT
    class NoCandidate:
        def __init__(self,*args,**kwargs): pass
        def compute_command(self,*args):
            return (0.,0.,0.),{"sampled_candidates":12,"feasible_candidates":0,"collision_rejections":12}
    grid=SimpleNamespace(occupancy=np.ones((8,8)),resolution=.2,origin=(0.,0.,0.))
    adapter=ArmVLADWAControllerAdapter(NoCandidate,None,reference_commit=APPROVED_ARM_VLA_COMMIT)
    with pytest.raises(ValueError,match="dwa_no_legal_control_candidates"):
        adapter.command([(0,0),(1,0)],(0,0,0),(0,0,0),grid)
    assert adapter.last_trace["debug"]["collision_rejections"] == 12


def test_nonfinite_path_retains_invalid_path_status_at_adapter_boundary():
    from conveyor_bench.conveyorvla.waypoint_planner_adapters import ArmVLADWAControllerAdapter, APPROVED_ARM_VLA_COMMIT
    adapter = ArmVLADWAControllerAdapter(None, None, reference_commit=APPROVED_ARM_VLA_COMMIT)
    with pytest.raises(ValueError, match="dwa_invalid_path"):
        adapter.command([(0,0),(np.nan,0)],(0,0,0),(0,0,0),None)
    assert adapter.last_trace["failure_status"] == "dwa_invalid_path"
