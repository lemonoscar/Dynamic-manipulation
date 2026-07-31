import pytest

from conveyor_bench.v2.coordinator import (
    CoordinatorStatus,
    CoordinatorTerminatedError,
    SequentialTargetCoordinator,
)


def test_success_advances_in_order_and_only_final_success_terminates() -> None:
    coordinator = SequentialTargetCoordinator(
        ("target-a", "target-b", "target-c"),
        episode_start_time_s=10.0,
        episode_timeout_s=30.0,
    )

    assert coordinator.status is CoordinatorStatus.ACTIVE
    assert coordinator.current_target_id == "target-a"
    assert coordinator.completed_target_ids == ()

    first = coordinator.mark_success("target-a", sim_time_s=12.0)
    assert first.target_id == "target-a"
    assert first.next_target_id == "target-b"
    assert first.status is CoordinatorStatus.ACTIVE
    assert not first.episode_terminal
    assert not first.episode_success
    assert first.remaining_time_s == pytest.approx(28.0)
    assert coordinator.current_target_id == "target-b"
    assert coordinator.completed_target_ids == ("target-a",)

    second = coordinator.mark_success("target-b", sim_time_s=18.0)
    assert second.next_target_id == "target-c"
    assert not second.episode_terminal

    final = coordinator.mark_success("target-c", sim_time_s=25.0)
    assert final.next_target_id is None
    assert final.status is CoordinatorStatus.SUCCEEDED
    assert final.episode_terminal
    assert final.episode_success
    assert final.remaining_time_s == pytest.approx(15.0)
    assert coordinator.status is CoordinatorStatus.SUCCEEDED
    assert coordinator.current_target_id is None
    assert coordinator.completed_target_ids == (
        "target-a",
        "target-b",
        "target-c",
    )


def test_failure_terminates_immediately_without_advancing() -> None:
    coordinator = SequentialTargetCoordinator(
        ("target-a", "target-b"),
        episode_start_time_s=0.0,
        episode_timeout_s=20.0,
    )

    transition = coordinator.mark_failure(
        "target-a",
        sim_time_s=3.0,
        reason="grasp_timeout",
    )

    assert transition.target_id == "target-a"
    assert transition.next_target_id is None
    assert transition.status is CoordinatorStatus.FAILED
    assert transition.episode_terminal
    assert not transition.episode_success
    assert transition.failure_reason == "grasp_timeout"
    assert transition.remaining_time_s == pytest.approx(17.0)
    assert coordinator.status is CoordinatorStatus.FAILED
    assert coordinator.current_target_id is None
    assert coordinator.completed_target_ids == ()
    assert coordinator.failure_reason == "grasp_timeout"


def test_remaining_time_and_transition_times_are_validated() -> None:
    coordinator = SequentialTargetCoordinator(
        ("target-a", "target-b"),
        episode_start_time_s=5.0,
        episode_timeout_s=10.0,
    )

    assert coordinator.remaining_time_s(5.0) == pytest.approx(10.0)
    assert coordinator.remaining_time_s(12.0) == pytest.approx(3.0)
    assert coordinator.remaining_time_s(15.0) == pytest.approx(0.0)
    assert coordinator.remaining_time_s(18.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="before episode start"):
        coordinator.remaining_time_s(4.99)
    with pytest.raises(ValueError, match="finite"):
        coordinator.remaining_time_s(float("nan"))

    coordinator.mark_success("target-a", sim_time_s=12.0)
    with pytest.raises(ValueError, match="cannot move backwards"):
        coordinator.mark_failure(
            "target-b",
            sim_time_s=11.0,
            reason="target_missed",
        )
    with pytest.raises(TimeoutError, match="episode time budget is exhausted"):
        coordinator.mark_success("target-b", sim_time_s=15.0)

    terminal = coordinator.mark_failure(
        "target-b",
        sim_time_s=15.0,
        reason="episode_timeout",
    )
    assert terminal.remaining_time_s == 0.0
    assert terminal.failure_reason == "episode_timeout"


@pytest.mark.parametrize(
    "target_ids",
    (
        (),
        ("only-one",),
        ("target-a", "target-a"),
        ("target-a", ""),
    ),
)
def test_requires_two_unique_nonempty_target_ids(target_ids) -> None:
    with pytest.raises(ValueError):
        SequentialTargetCoordinator(
            target_ids,
            episode_start_time_s=0.0,
            episode_timeout_s=10.0,
        )


@pytest.mark.parametrize(
    ("start_time", "timeout"),
    (
        (-1.0, 10.0),
        (float("inf"), 10.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (0.0, float("nan")),
    ),
)
def test_requires_a_finite_positive_episode_window(start_time, timeout) -> None:
    with pytest.raises(ValueError):
        SequentialTargetCoordinator(
            ("target-a", "target-b"),
            episode_start_time_s=start_time,
            episode_timeout_s=timeout,
        )


def test_stale_target_and_repeated_terminal_transitions_are_rejected() -> None:
    coordinator = SequentialTargetCoordinator(
        ("target-a", "target-b"),
        episode_start_time_s=0.0,
        episode_timeout_s=10.0,
    )
    coordinator.mark_success("target-a", sim_time_s=1.0)

    with pytest.raises(ValueError, match="does not match current target"):
        coordinator.mark_success("target-a", sim_time_s=1.1)
    with pytest.raises(ValueError, match="non-empty"):
        coordinator.mark_failure("target-b", sim_time_s=1.1, reason="")

    coordinator.mark_success("target-b", sim_time_s=2.0)
    with pytest.raises(CoordinatorTerminatedError, match="already terminated"):
        coordinator.mark_success("target-b", sim_time_s=2.1)
    with pytest.raises(CoordinatorTerminatedError, match="already terminated"):
        coordinator.mark_failure(
            "target-b",
            sim_time_s=2.1,
            reason="late_failure",
        )


def test_repeated_failure_transition_is_rejected() -> None:
    coordinator = SequentialTargetCoordinator(
        ("target-a", "target-b"),
        episode_start_time_s=0.0,
        episode_timeout_s=10.0,
    )
    coordinator.mark_failure(
        "target-a",
        sim_time_s=1.0,
        reason="wrong_object",
    )

    with pytest.raises(CoordinatorTerminatedError, match="already terminated"):
        coordinator.mark_failure(
            "target-a",
            sim_time_s=1.1,
            reason="wrong_object",
        )
