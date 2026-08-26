import inspect

import pytest

from conveyor_bench.conveyorvla.joint_trajectory import JointTrajectoryRoute
from conveyor_bench.conveyorvla.joint_trajectory_runtime import (
    DirectJointTrajectoryExecutor,
    JointTrajectoryInferenceSession,
    JointTrajectoryRuntimeRequest,
    JointSafetyLimits,
    RouteCommitStatus,
    RouteCommitter,
    TransferSuccessEvaluator,
    navigation_reference,
)
from conveyor_bench.conveyorvla.joint_trajectory_model import JointTrajectoryRouteDecision
from conveyor_bench.conveyorvla.joint_trajectory_training import (
    AccumulationMicroBatchSampler,
    StratifiedJointTrajectoryBatchSampler,
    TrainingStages,
    load_joint_trajectory_config,
    select_disposable_overfit_episodes,
    validate_global_batch,
)


def _probs(nav_source, pick, nav_target, place):
    return {
        "NAV_TO_SOURCE": nav_source,
        "PICK": pick,
        "NAV_TO_TARGET": nav_target,
        "PLACE": place,
    }


def test_route_commit_needs_two_fresh_wins_and_pending_holds():
    committer = RouteCommitter()
    first = committer.observe(_probs(0.6, 0.2, 0.1, 0.1), sequence_id=1)
    assert first.status is RouteCommitStatus.INITIAL_PENDING and not first.execute_action
    second = committer.observe(_probs(0.55, 0.25, 0.1, 0.1), sequence_id=2)
    assert second.committed_route is JointTrajectoryRoute.NAV_TO_SOURCE
    assert second.execute_action
    pending = committer.observe(_probs(0.3, 0.5, 0.1, 0.1), sequence_id=3)
    assert pending.status is RouteCommitStatus.SWITCH_PENDING and not pending.execute_action
    flicker = committer.observe(_probs(0.6, 0.2, 0.1, 0.1), sequence_id=4)
    assert flicker.status is RouteCommitStatus.UNCHANGED
    again = committer.observe(_probs(0.3, 0.5, 0.1, 0.1), sequence_id=5)
    assert not again.execute_action
    switched = committer.observe(_probs(0.25, 0.55, 0.1, 0.1), sequence_id=6)
    assert switched.committed_route is JointTrajectoryRoute.PICK
    with pytest.raises(ValueError, match="fresh"):
        committer.observe(_probs(0.25, 0.55, 0.1, 0.1), sequence_id=6)


def test_nav_passes_all_points_and_direct_joint_has_no_pose_planner():
    nav = navigation_reference([[0.1 * index, 0.0, 0.01 * index] for index in range(10)])
    assert len(nav.points_query_body) == 10
    assert nav.local_goal_query_body == nav.points_query_body[-1]
    limits = JointSafetyLimits(
        lower=(-1.0,) * 6,
        upper=(1.0,) * 6,
        max_rate_rad_s=(0.5,) * 6,
    )
    executor = DirectJointTrajectoryExecutor(limits)
    chunk = executor.prepare(
        [0.0] * 6,
        [[0.2 * (index + 1)] * 6 + [1.2] for index in range(10)],
    )
    assert len(chunk.commands) == 10
    assert all(command.base_velocity == (0.0, 0.0, 0.0) for command in chunk.commands)
    assert all(
        command.joint_position[0] == pytest.approx(0.02 * (command.index + 1))
        for command in chunk.commands
    )
    assert chunk.rate_saturation_count > 0 and chunk.gripper_saturation_count == 10
    source = inspect.getsource(DirectJointTrajectoryExecutor).lower()
    assert "plan_pose" not in source and "curobo" not in source
    assert not hasattr(executor, "planner") and not hasattr(executor, "ik_solver")


def test_success_is_release_inside_and_one_second_dwell_only():
    evaluator = TransferSuccessEvaluator()
    assert not evaluator.update(0.0, released=True, inside_target_valid_area=True).success
    assert not evaluator.update(0.8, released=True, inside_target_valid_area=True).success
    assert evaluator.update(1.0, released=True, inside_target_valid_area=True).success
    evaluator.reset()
    evaluator.update(2.0, released=True, inside_target_valid_area=True)
    evaluator.update(2.5, released=False, inside_target_valid_area=True)
    assert not evaluator.update(3.1, released=True, inside_target_valid_area=True).success


class _IdentityNormalizer:
    def normalize_mani_state(self, value):
        return tuple(value)

    def denormalize_action(self, _route, value):
        return value


class _SessionPolicy:
    def __init__(self):
        self.route_calls = 0
        self.action_calls = 0
        self.route = JointTrajectoryRoute.NAV_TO_SOURCE

    def predict_routes(self, _examples):
        self.route_calls += 1
        probabilities = {
            candidate.value: 0.7 if candidate is self.route else 0.1
            for candidate in JointTrajectoryRoute
        }
        return (
            JointTrajectoryRouteDecision(
                route=self.route,
                assistant_prefix=f"prefix-{self.route.value}",
                subtask_text="move",
                route_confidence=0.7,
                route_probs=probabilities,
                valid=True,
            ),
        )

    def predict_actions(self, _examples, _decisions):
        self.action_calls += 1
        width = 3 if self.route is JointTrajectoryRoute.NAV_TO_SOURCE else 7
        row = [0.01] * width
        if width == 7:
            row[-1] = 0.8
        return ((tuple(tuple(row) for _ in range(10))),)


def test_session_skips_pass2_on_first_pending_observation_then_executes():
    policy = _SessionPolicy()
    executor = DirectJointTrajectoryExecutor(
        JointSafetyLimits(lower=(-2.0,) * 6, upper=(2.0,) * 6, max_rate_rad_s=(2.0,) * 6)
    )
    session = JointTrajectoryInferenceSession(
        policy,
        _IdentityNormalizer(),
        executor,
        checkpoint_id="checkpoint",
        normalization_sha256="normalizer",
    )

    def request(sequence):
        return JointTrajectoryRuntimeRequest(
            request_id=f"request-{sequence}",
            episode_id="episode",
            sequence_id=sequence,
            instruction="move the Coke",
            head_images=(object(), object()),
            wrist_images=(object(), object()),
            joint_position=(0.0,) * 6,
            joint_velocity=(0.0,) * 6,
            gripper_open_fraction=1.0,
        )

    pending = session.step(request(1))
    assert not pending.pass2_executed and pending.hold is not None
    assert policy.action_calls == 0
    active = session.step(request(2))
    assert active.pass2_executed and active.navigation is not None
    assert active.hold is not None
    assert active.hold.joint_position == (0.0,) * 6
    assert active.hold.gripper_open_fraction == 1.0
    assert policy.action_calls == 1
    policy.route = JointTrajectoryRoute.PICK
    switch_pending = session.step(request(3))
    assert not switch_pending.pass2_executed and policy.action_calls == 1
    switched = session.step(request(4))
    assert switched.pass2_executed and switched.manipulation is not None
    assert switched.hold is None
    assert policy.action_calls == 2
    policy.route = JointTrajectoryRoute.NAV_TO_TARGET
    switch_back_pending = session.step(request(5))
    assert not switch_back_pending.pass2_executed
    assert switch_back_pending.hold is not None
    assert switch_back_pending.hold.joint_position == pytest.approx((0.01,) * 6)
    assert switch_back_pending.hold.gripper_open_fraction == pytest.approx(0.8)


def _sampler_metadata():
    routes = []
    episodes = []
    transitions = []
    signed = []
    buckets = []
    gripper = []
    for route in JointTrajectoryRoute:
        for bucket in ("early", "middle", "late"):
            for index in range(30):
                routes.append(route.value)
                episodes.append(f"ordinary-{route.value}-{bucket}-{index}")
                transitions.append(None)
                signed.append(None)
                buckets.append(bucket)
                gripper.append(
                    route in {JointTrajectoryRoute.PICK, JointTrajectoryRoute.PLACE}
                    and index < 15
                )
    pairs = [
        (JointTrajectoryRoute.NAV_TO_SOURCE, JointTrajectoryRoute.PICK),
        (JointTrajectoryRoute.PICK, JointTrajectoryRoute.NAV_TO_TARGET),
        (JointTrajectoryRoute.NAV_TO_TARGET, JointTrajectoryRoute.PLACE),
    ]
    for old, new in pairs:
        for episode in range(8):
            transition = f"{old.value}->{new.value}"
            transition_id = f"boundary-{transition}-{episode}"
            for route, time in ((old, -0.1), (new, 0.1)):
                routes.append(route.value)
                episodes.append(f"boundary-episode-{transition}-{episode}")
                transitions.append(transition_id)
                signed.append(time)
                buckets.append("late" if time < 0 else "early")
                gripper.append(False)
    return routes, episodes, transitions, signed, buckets, gripper


def test_sampler_builds_one_scientific_batch_then_micro_batches():
    metadata = _sampler_metadata()
    sampler = StratifiedJointTrajectoryBatchSampler(*metadata, seed=7, batches_per_epoch=1)
    batch = next(iter(sampler))
    assert len(batch) == 64
    interior = [index for index in batch if metadata[2][index] is None]
    boundary = [index for index in batch if metadata[2][index] is not None]
    assert len(interior) == 56 and len(boundary) == 8
    assert all(
        metadata[2][batch[index]] == metadata[2][batch[index + 1]]
        for index in range(0, 8, 2)
    )
    assert len({metadata[1][index] for index in batch}) >= 56
    for route in JointTrajectoryRoute:
        assert sum(metadata[0][index] == route.value for index in interior) == 14
    mani_interior = [
        index
        for index in interior
        if metadata[0][index] in {"PICK", "PLACE"}
    ]
    assert sum(metadata[5][index] for index in mani_interior) >= 7

    sampler = StratifiedJointTrajectoryBatchSampler(*metadata, seed=7, batches_per_epoch=1)
    micro = AccumulationMicroBatchSampler(
        sampler,
        world_size=4,
        micro_batch_per_rank=2,
        gradient_accumulation_steps=8,
    )
    chunks = list(micro)
    assert len(chunks) == 8 and all(len(chunk) == 8 for chunk in chunks)
    assert len({index for chunk in chunks for index in chunk}) == 64
    assert validate_global_batch(2, 2, 16) == 64
    with pytest.raises(Exception, match="boundary pairs"):
        validate_global_batch(4, 1, 16)


def test_disposable_overfit_selects_exactly_twelve_complete_episodes_and_reuses_only_them():
    routes = []
    episodes = []
    transitions = []
    signed = []
    buckets = []
    gripper = []
    pairs = list(zip(tuple(JointTrajectoryRoute), tuple(JointTrajectoryRoute)[1:]))
    for episode_index in range(14):
        episode = f"complete-{episode_index}"
        for route in JointTrajectoryRoute:
            routes.append(route.value)
            episodes.append(episode)
            transitions.append(None)
            signed.append(None)
            buckets.append("middle")
            gripper.append(route in {JointTrajectoryRoute.PICK, JointTrajectoryRoute.PLACE})
        for old, new in pairs:
            event = f"{episode}:{old.value}->{new.value}"
            for route, time in ((old, -0.1), (new, 0.1)):
                routes.append(route.value)
                episodes.append(episode)
                transitions.append(event)
                signed.append(time)
                buckets.append("late" if time < 0.0 else "early")
                gripper.append(False)
    selected = select_disposable_overfit_episodes(
        routes, episodes, transitions, signed, gripper, seed=3
    )
    assert len(selected) == len(set(selected)) == 12
    sampler = StratifiedJointTrajectoryBatchSampler(
        routes,
        episodes,
        transitions,
        signed,
        buckets,
        gripper,
        seed=3,
        batches_per_epoch=1,
        eligible_episode_ids=selected,
        allow_episode_reuse=True,
        minimum_distinct_episodes=4,
    )
    batch = next(iter(sampler))
    assert len(batch) == 64
    assert {episodes[index] for index in batch} <= set(selected)
    overfit_stages = TrainingStages.for_disposable_overfit(6400, max_steps=300)
    assert overfit_stages.stage_a_steps == 0
    assert overfit_stages.stage(0) == "B"


def test_config_and_stage_boundaries_are_frozen():
    config = load_joint_trajectory_config("configs/manipulation_navi_v1.json")
    assert config["loss"]["repeated_diffusion_steps"] == 1
    assert config["action_model"]["num_inference_timesteps"] == 10
    assert all(config["disabled"].values())
    stages = TrainingStages.from_rows(6400)
    assert stages.equivalent_epoch_steps == 100
    assert stages.stage_a_steps == 25
    assert stages.total_steps == 200
    assert stages.stage(24) == "A" and stages.stage(25) == "B"
