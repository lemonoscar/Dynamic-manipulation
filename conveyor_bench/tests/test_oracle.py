import pytest

from conveyor_bench.oracle import (
    DynamicPickOracle,
    OracleConfig,
    OracleObservation,
    OraclePhase,
)


def _obs(
    time: float,
    *,
    object_y: float,
    object_x: float = 0.70,
    gripper=(0.70, 0.0, 0.87),
    lifted: bool = False,
    crossed_exit: bool = False,
    contacts: bool = False,
) -> OracleObservation:
    return OracleObservation(
        sim_time=time,
        object_position=(object_x, object_y, 0.72 if not lifted else 0.84),
        object_velocity=(0.0, -0.10, 0.0),
        gripper_position=gripper,
        object_crossed_exit=crossed_exit,
        object_lifted=lifted,
        secure_grasp=lifted,
        left_contact=contacts,
        right_contact=contacts,
    )


def test_oracle_reaches_success_terminal():
    oracle = DynamicPickOracle(
        OracleConfig(
            transport_direction_xy=(0.0, -1.0),
            track_target_speed_mps=10.0,
            settle_duration=0.1,
            descend_settle_duration=0.0,
            descend_speed_mps=10.0,
            close_duration=0.1,
            contact_settle_duration=0.0,
            hold_duration=0.1,
        )
    )
    oracle.reset(sim_time=0.0, object_position=(0.70, 0.48, 0.72))

    oracle.step(_obs(0.2, object_y=0.30))
    oracle.step(_obs(0.3, object_y=0.20))
    assert oracle.phase is OraclePhase.TRACK

    oracle.step(_obs(0.4, object_y=0.10, gripper=(0.70, 0.10, 0.87)))
    assert oracle.phase is OraclePhase.DESCEND

    oracle.step(_obs(0.5, object_y=0.08, gripper=(0.70, 0.10, 0.725)))
    assert oracle.phase is OraclePhase.CLOSE

    oracle.step(
        _obs(
            0.7,
            object_y=0.06,
            gripper=(0.70, 0.10, 0.725),
            contacts=True,
        )
    )
    assert oracle.phase is OraclePhase.LIFT

    oracle.step(_obs(0.8, object_y=0.05, lifted=True))
    assert oracle.phase is OraclePhase.HOLD

    command = oracle.step(_obs(1.0, object_y=0.05, lifted=True))
    assert command.terminal
    assert command.success
    assert command.target_orientation == (-1.0, 0.0, 0.0, 0.0)
    assert oracle.phase is OraclePhase.COMPLETE


def test_oracle_fails_when_object_exits_before_lift():
    oracle = DynamicPickOracle()
    oracle.reset(sim_time=0.0, object_position=(0.70, 0.48, 0.72))

    command = oracle.step(_obs(0.1, object_y=-0.60, crossed_exit=True))

    assert command.terminal
    assert not command.success
    assert command.failure_reason == "belt_exit"
    assert oracle.phase is OraclePhase.FAILED


def test_oracle_uses_transport_prior_instead_of_rebound_velocity():
    oracle = DynamicPickOracle(
        OracleConfig(
            transport_direction_xy=(0.0, -1.0),
            track_target_speed_mps=10.0,
            settle_duration=0.0,
            expected_transport_speed=0.08,
            descend_settle_duration=0.0,
            descend_speed_mps=10.0,
        )
    )
    oracle.reset(sim_time=0.0, object_position=(0.70, 0.48, 0.71))
    oracle.phase = OraclePhase.TRACK

    oracle.step(
        OracleObservation(
            sim_time=0.5,
            object_position=(0.70, 0.12, 0.71),
            object_velocity=(0.0, 0.8, 0.0),
            gripper_position=(0.70, 0.1104, 0.86),
        )
    )
    assert oracle.phase is OraclePhase.DESCEND

    command = oracle.step(
        OracleObservation(
            sim_time=0.6,
            object_position=(0.70, 0.20, 0.71),
            object_velocity=(0.0, 0.8, 0.0),
            gripper_position=(0.70, 0.1024, 0.715),
            left_contact=True,
        )
    )

    assert oracle.phase is OraclePhase.CLOSE
    assert command.target_position[1] == pytest.approx(0.1024)


def test_oracle_requires_secure_grasp_during_hold():
    oracle = DynamicPickOracle(OracleConfig(hold_duration=0.1))
    oracle.reset(sim_time=0.0, object_position=(0.70, 0.0, 0.71))
    oracle.phase = OraclePhase.HOLD
    oracle._phase_started_at = 0.0

    command = oracle.step(
        OracleObservation(
            sim_time=0.2,
            object_position=(0.70, -0.05, 0.76),
            object_velocity=(0.0, 0.0, 0.0),
            gripper_position=(0.70, 0.0, 0.78),
            object_lifted=True,
            secure_grasp=False,
        )
    )

    assert command.terminal
    assert command.failure_reason == "dropped_during_hold"


def test_lateral_trigger_uses_forward_progress_not_world_x():
    oracle = DynamicPickOracle(
        OracleConfig(
            transport_direction_xy=(0.0, -1.0),
            track_target_speed_mps=10.0,
            settle_duration=0.0,
        )
    )
    oracle.reset(sim_time=0.0, object_position=(0.72, 0.48, 0.71))
    oracle.phase = OraclePhase.TRACK

    oracle.step(_obs(0.1, object_x=0.72, object_y=0.131))
    assert oracle.phase is OraclePhase.TRACK

    command = oracle.step(
        _obs(
            0.2,
            object_x=0.72,
            object_y=0.130,
            gripper=(0.72, 0.13, 0.86),
        )
    )
    assert oracle.phase is OraclePhase.DESCEND
    assert command.target_position[:2] == (0.72, 0.13)


def test_tracking_target_moves_gradually_from_the_intercept():
    oracle = DynamicPickOracle(
        OracleConfig(
            transport_direction_xy=(0.0, -1.0),
            track_target_speed_mps=0.30,
            settle_duration=0.0,
        )
    )
    oracle.reset(sim_time=0.0, object_position=(0.70, 0.48, 0.71))
    oracle.phase = OraclePhase.TRACK

    command = oracle.step(_obs(0.02, object_y=0.22))

    assert oracle.phase is OraclePhase.TRACK
    assert command.target_position[:2] == pytest.approx((0.70, 0.006))
