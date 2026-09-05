"""Sampler, stages, optimizer, and config gates for Joint-Trajectory v1."""

from __future__ import annotations

import json
import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import torch
from torch.utils.data import Sampler

from conveyor_bench.conveyorvla.config import M0MobileError
from conveyor_bench.conveyorvla.joint_trajectory import (
    ACTION_HORIZON,
    DATASET_PROFILE,
    DATASET_SCHEMA_VERSION,
    MANIPULATION_ACTION_DIM,
    MANIPULATION_STATE_DIM,
    MANIPULATION_STRIDE_S,
    MODEL_CONTRACT_ID,
    NAVIGATION_ACTION_DIM,
    NAVIGATION_STRIDE_S,
    POLICY_CONFIG_SCHEMA_VERSION,
    TRANSITION_TAU_S,
    TRAIN_GLOBAL_BATCH_SIZE,
    JointTrajectoryDomain,
    JointTrajectoryRoute,
    action_domain,
)
from conveyor_bench.conveyorvla.joint_trajectory_model import (
    ConveyorVLAJointTrajectoryPolicy,
)


GLOBAL_BATCH_SIZE = TRAIN_GLOBAL_BATCH_SIZE
NAV_INTERIOR_PER_BATCH = 28
MANI_INTERIOR_PER_BATCH = 28
BOUNDARY_ROWS_PER_BATCH = 8
MIN_DISTINCT_EPISODES = 56


@dataclass(frozen=True)
class TrainingStages:
    eligible_train_rows: int
    global_batch_size: int
    equivalent_epoch_steps: int
    stage_a_steps: int
    total_steps: int

    @classmethod
    def from_rows(
        cls,
        eligible_train_rows: int,
        *,
        global_batch_size: int = GLOBAL_BATCH_SIZE,
        stage_a_epochs: float = 0.25,
        total_epochs: float = 2.0,
    ) -> "TrainingStages":
        if eligible_train_rows < global_batch_size:
            raise ValueError("training dataset is smaller than one global batch")
        if global_batch_size != GLOBAL_BATCH_SIZE:
            raise ValueError(f"scientific global batch must be {GLOBAL_BATCH_SIZE}")
        if not 0.0 < stage_a_epochs < total_epochs:
            raise ValueError("stage epoch fractions are invalid")
        epoch_steps = math.ceil(eligible_train_rows / global_batch_size)
        return cls(
            eligible_train_rows=eligible_train_rows,
            global_batch_size=global_batch_size,
            equivalent_epoch_steps=epoch_steps,
            stage_a_steps=max(1, math.ceil(stage_a_epochs * epoch_steps)),
            total_steps=math.ceil(total_epochs * epoch_steps),
        )

    @classmethod
    def for_disposable_overfit(
        cls, eligible_train_rows: int, *, max_steps: int
    ) -> "TrainingStages":
        if eligible_train_rows <= 0 or max_steps <= 0:
            raise ValueError("disposable overfit rows and max_steps must be positive")
        return cls(
            eligible_train_rows=eligible_train_rows,
            global_batch_size=GLOBAL_BATCH_SIZE,
            equivalent_epoch_steps=math.ceil(eligible_train_rows / GLOBAL_BATCH_SIZE),
            stage_a_steps=0,
            total_steps=max_steps,
        )

    def stage(self, completed_optimizer_steps: int) -> str:
        if completed_optimizer_steps < 0:
            raise ValueError("completed optimizer steps cannot be negative")
        return "A" if completed_optimizer_steps < self.stage_a_steps else "B"


class StratifiedJointTrajectoryBatchSampler(Sampler[list[int]]):
    """Build global batches with explicit domain/route/event/episode structure.

    Accelerate must use ``split_batches=True`` so every yielded 64-index global
    batch is divided across ranks instead of independently resampled per rank.
    """

    def __init__(
        self,
        routes: Sequence[str],
        episode_ids: Sequence[str],
        transition_ids: Sequence[str | None],
        boundary_signed_times: Sequence[float | None],
        progress_buckets: Sequence[str | None],
        gripper_transitions: Sequence[bool],
        *,
        seed: int,
        global_batch_size: int = GLOBAL_BATCH_SIZE,
        batches_per_epoch: int | None = None,
        eligible_episode_ids: Sequence[str] | None = None,
        allow_episode_reuse: bool = False,
        minimum_distinct_episodes: int = MIN_DISTINCT_EPISODES,
    ) -> None:
        lengths = {
            len(routes),
            len(episode_ids),
            len(transition_ids),
            len(boundary_signed_times),
            len(progress_buckets),
            len(gripper_transitions),
        }
        if len(lengths) != 1 or not routes:
            raise ValueError("stratified sampler metadata is empty or misaligned")
        if global_batch_size != GLOBAL_BATCH_SIZE:
            raise ValueError(f"joint-trajectory global batch must be {GLOBAL_BATCH_SIZE}")
        self.routes = tuple(JointTrajectoryRoute(str(route)) for route in routes)
        self.episode_ids = tuple(str(value) for value in episode_ids)
        if any(not value for value in self.episode_ids):
            raise ValueError("sampler episode IDs must be non-empty")
        self.transition_ids = tuple(
            None if value is None else str(value) for value in transition_ids
        )
        self.boundary_signed_times = tuple(
            None if value is None else float(value) for value in boundary_signed_times
        )
        self.progress_buckets = tuple(progress_buckets)
        self.gripper_transitions = tuple(bool(value) for value in gripper_transitions)
        known_episodes = set(self.episode_ids)
        eligible = (
            known_episodes
            if eligible_episode_ids is None
            else {str(value) for value in eligible_episode_ids}
        )
        if not eligible or not eligible <= known_episodes:
            raise ValueError("sampler eligible episode IDs are empty or unknown")
        self.eligible_episode_ids = frozenset(eligible)
        self.allow_episode_reuse = bool(allow_episode_reuse)
        self.minimum_distinct_episodes = int(minimum_distinct_episodes)
        if self.minimum_distinct_episodes <= 0:
            raise ValueError("sampler minimum distinct episodes must be positive")
        self.seed = int(seed)
        self.global_batch_size = global_batch_size
        self._epoch = 0
        self._batches = (
            max(1, len(routes) // global_batch_size)
            if batches_per_epoch is None
            else int(batches_per_epoch)
        )
        if self._batches <= 0:
            raise ValueError("batches_per_epoch must be positive")
        self.interior_by_route_bucket: dict[
            JointTrajectoryRoute, dict[str | None, tuple[int, ...]]
        ] = {}
        for route in JointTrajectoryRoute:
            buckets: dict[str | None, list[int]] = defaultdict(list)
            for index, candidate in enumerate(self.routes):
                if (
                    candidate is route
                    and self.transition_ids[index] is None
                    and self.episode_ids[index] in self.eligible_episode_ids
                ):
                    buckets[self.progress_buckets[index]].append(index)
            self.interior_by_route_bucket[route] = {
                bucket: tuple(values) for bucket, values in buckets.items()
            }
            if not any(buckets.values()):
                raise M0MobileError(f"sampler has no interior rows for {route.value}")
        events: dict[tuple[str, str], tuple[list[int], list[int]]] = {}
        for index, transition_id in enumerate(self.transition_ids):
            if self.episode_ids[index] not in self.eligible_episode_ids:
                continue
            signed = self.boundary_signed_times[index]
            if transition_id is None:
                if signed is not None:
                    raise ValueError("interior sampler row has boundary signed time")
                continue
            if signed is None or not math.isfinite(signed):
                raise ValueError("boundary sampler row needs finite signed time")
            key = (self.episode_ids[index], transition_id)
            before, after = events.setdefault(key, ([], []))
            (before if signed < 0.0 else after).append(index)
        self.events = tuple(
            (key, tuple(before), tuple(after))
            for key, (before, after) in sorted(events.items())
            if before and after
        )
        if len(self.events) < BOUNDARY_ROWS_PER_BATCH // 2:
            raise M0MobileError("sampler needs at least four complete boundary events")
        event_transition_names = {
            self._event_transition(before, after)
            for _key, before, after in self.events
        }
        if event_transition_names != set(TRANSITION_TAU_S):
            missing = sorted(set(TRANSITION_TAU_S) - event_transition_names)
            raise M0MobileError(f"sampler is missing boundary transitions: {missing}")
        self.last_exposure: Mapping[str, int] = {}

    def __len__(self) -> int:
        return self._batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch * 1_000_003)
        exposure: Counter[str] = Counter()
        event_cursor = self._epoch
        for batch_index in range(self._batches):
            used_episodes: set[str] = set()
            batch: list[int] = []
            event_rows = self._boundary_rows(rng, event_cursor, used_episodes)
            event_cursor += BOUNDARY_ROWS_PER_BATCH // 2
            batch.extend(event_rows)
            for route in JointTrajectoryRoute:
                gripper_quota = 0
                if route is JointTrajectoryRoute.PICK:
                    gripper_quota = 4
                elif route is JointTrajectoryRoute.PLACE:
                    gripper_quota = 3
                selected = self._interior_rows(
                    route,
                    14,
                    gripper_quota,
                    rng,
                    used_episodes,
                    bucket_offset=batch_index,
                )
                batch.extend(selected)
            if len(batch) != GLOBAL_BATCH_SIZE:
                raise AssertionError("sampler constructed the wrong global batch size")
            distinct = len({self.episode_ids[index] for index in batch})
            if distinct < self.minimum_distinct_episodes:
                raise M0MobileError(
                    f"sampler global batch has only {distinct} distinct episodes"
                )
            # Keep each before/after pair adjacent at the front of the
            # accumulation window.  With an even per-rank micro-batch, the
            # distributed batch sharder cannot split a pair across ranks.
            ordinary = batch[BOUNDARY_ROWS_PER_BATCH:]
            rng.shuffle(ordinary)
            batch = batch[:BOUNDARY_ROWS_PER_BATCH] + ordinary
            for index in batch:
                route = self.routes[index]
                exposure[f"route:{route.value}"] += 1
                exposure[f"domain:{action_domain(route).value}"] += 1
                exposure[f"progress:{self.progress_buckets[index]}"] += 1
                exposure[f"boundary:{self.transition_ids[index] is not None}"] += 1
                exposure[f"gripper_transition:{self.gripper_transitions[index]}"] += 1
            yield batch
        self.last_exposure = dict(sorted(exposure.items()))
        self._epoch += 1

    def _boundary_rows(
        self, rng: random.Random, cursor: int, used_episodes: set[str]
    ) -> list[int]:
        by_transition: dict[str, list[int]] = defaultdict(list)
        for event_index, (_key, before, after) in enumerate(self.events):
            by_transition[self._event_transition(before, after)].append(event_index)
        required = list(TRANSITION_TAU_S)
        chosen_events = []
        for offset, transition in enumerate(required):
            candidates = by_transition[transition]
            ordered = candidates[(cursor + offset) % len(candidates) :] + candidates[: (cursor + offset) % len(candidates)]
            candidate = next(
                (
                    value
                    for value in ordered
                    if self.events[value][0][0]
                    not in {self.events[selected][0][0] for selected in chosen_events}
                ),
                None,
            )
            if candidate is None:
                raise M0MobileError("sampler cannot assign boundary transitions to unique episodes")
            chosen_events.append(candidate)
        remaining = [
            index
            for index in range(len(self.events))
            if index not in chosen_events
        ]
        rng.shuffle(remaining)
        extra = next(
            (
                value
                for value in remaining
                if self.events[value][0][0]
                not in {self.events[selected][0][0] for selected in chosen_events}
            ),
            None,
        )
        if extra is not None:
            chosen_events.append(extra)
        if len(chosen_events) != 4:
            raise M0MobileError("sampler cannot select four distinct boundary events")
        result = []
        for event_index in chosen_events:
            (episode, _transition_id), before, after = self.events[event_index]
            if episode in used_episodes:
                raise M0MobileError("sampler boundary events repeat an episode")
            used_episodes.add(episode)
            result.append(rng.choice(before))
            result.append(rng.choice(after))
        return result

    def _interior_rows(
        self,
        route: JointTrajectoryRoute,
        count: int,
        gripper_quota: int,
        rng: random.Random,
        used_episodes: set[str],
        *,
        bucket_offset: int,
    ) -> list[int]:
        bucket_targets = ["early"] * 5 + ["middle"] * 5 + ["late"] * 4
        bucket_targets = bucket_targets[bucket_offset % count :] + bucket_targets[: bucket_offset % count]
        result: list[int] = []
        for position, bucket in enumerate(bucket_targets):
            require_gripper = position < gripper_quota
            index = self._choose_interior(
                route,
                bucket,
                require_gripper,
                rng,
                used_episodes,
            )
            if not self.allow_episode_reuse:
                used_episodes.add(self.episode_ids[index])
            result.append(index)
        if sum(self.gripper_transitions[index] for index in result) < gripper_quota:
            raise M0MobileError(f"sampler cannot meet {route.value} gripper-transition quota")
        return result

    def _choose_interior(
        self,
        route: JointTrajectoryRoute,
        bucket: str,
        require_gripper: bool,
        rng: random.Random,
        used_episodes: set[str],
    ) -> int:
        buckets = self.interior_by_route_bucket[route]
        ordered_pools = [buckets.get(bucket, ()), *buckets.values()]
        for pool in ordered_pools:
            candidates = [
                index
                for index in pool
                if (
                    self.allow_episode_reuse
                    or self.episode_ids[index] not in used_episodes
                )
                and (not require_gripper or self.gripper_transitions[index])
            ]
            if candidates:
                return rng.choice(candidates)
        qualifier = " gripper-transition" if require_gripper else ""
        raise M0MobileError(
            f"sampler lacks a unique-episode {route.value}/{bucket}{qualifier} row"
        )

    def _event_transition(
        self, before: Sequence[int], after: Sequence[int]
    ) -> str:
        before_route = self.routes[before[0]]
        after_route = self.routes[after[0]]
        return f"{before_route.value}->{after_route.value}"


def select_disposable_overfit_episodes(
    routes: Sequence[str],
    episode_ids: Sequence[str],
    transition_ids: Sequence[str | None],
    boundary_signed_times: Sequence[float | None],
    gripper_transitions: Sequence[bool],
    *,
    seed: int,
    count: int = 12,
) -> tuple[str, ...]:
    lengths = {
        len(routes),
        len(episode_ids),
        len(transition_ids),
        len(boundary_signed_times),
        len(gripper_transitions),
    }
    if len(lengths) != 1 or not routes or count <= 0:
        raise ValueError("overfit episode metadata is empty or misaligned")
    route_sets: dict[str, set[JointTrajectoryRoute]] = defaultdict(set)
    gripper_routes: dict[str, set[JointTrajectoryRoute]] = defaultdict(set)
    events: dict[tuple[str, str], tuple[list[JointTrajectoryRoute], list[JointTrajectoryRoute]]] = {}
    for route_value, episode_value, transition_id, signed, gripper in zip(
        routes,
        episode_ids,
        transition_ids,
        boundary_signed_times,
        gripper_transitions,
        strict=True,
    ):
        route = JointTrajectoryRoute(str(route_value))
        episode = str(episode_value)
        route_sets[episode].add(route)
        if bool(gripper):
            gripper_routes[episode].add(route)
        if transition_id is None:
            continue
        if signed is None or not math.isfinite(float(signed)):
            raise ValueError("overfit boundary metadata needs finite signed time")
        before, after = events.setdefault((episode, str(transition_id)), ([], []))
        (before if float(signed) < 0.0 else after).append(route)
    transition_sets: dict[str, set[str]] = defaultdict(set)
    for (episode, _event), (before, after) in events.items():
        if before and after:
            transition_sets[episode].add(f"{before[0].value}->{after[0].value}")
    required_gripper = {JointTrajectoryRoute.PICK, JointTrajectoryRoute.PLACE}
    candidates = [
        episode
        for episode in sorted(route_sets)
        if route_sets[episode] == set(JointTrajectoryRoute)
        and transition_sets[episode] == set(TRANSITION_TAU_S)
        and required_gripper <= gripper_routes[episode]
    ]
    random.Random(int(seed)).shuffle(candidates)
    if len(candidates) < count:
        raise M0MobileError(
            f"disposable overfit needs {count} complete episodes, found {len(candidates)}"
        )
    return tuple(candidates[:count])


class AccumulationMicroBatchSampler(Sampler[list[int]]):
    """Split each scientific global batch into accumulation micro-batches."""

    def __init__(
        self,
        global_sampler: StratifiedJointTrajectoryBatchSampler,
        *,
        world_size: int,
        micro_batch_per_rank: int,
        gradient_accumulation_steps: int,
    ) -> None:
        validate_global_batch(
            world_size, micro_batch_per_rank, gradient_accumulation_steps
        )
        self.global_sampler = global_sampler
        self.micro_global_batch = world_size * micro_batch_per_rank
        # Accelerate's split-batch adapter reads this public BatchSampler
        # contract before dividing each global micro-batch across ranks.
        self.batch_size = self.micro_global_batch
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def __len__(self) -> int:
        return len(self.global_sampler) * self.gradient_accumulation_steps

    def __iter__(self) -> Iterator[list[int]]:
        for global_batch in self.global_sampler:
            if len(global_batch) != GLOBAL_BATCH_SIZE:
                raise AssertionError("upstream sampler yielded a non-global batch")
            for start in range(0, GLOBAL_BATCH_SIZE, self.micro_global_batch):
                yield global_batch[start : start + self.micro_global_batch]


def validate_global_batch(
    world_size: int, micro_batch_per_rank: int, gradient_accumulation_steps: int
) -> int:
    values = (world_size, micro_batch_per_rank, gradient_accumulation_steps)
    if any(isinstance(value, bool) or value <= 0 for value in values):
        raise ValueError("distributed batch factors must be positive integers")
    if micro_batch_per_rank < 2 or micro_batch_per_rank % 2:
        raise M0MobileError(
            "per-rank micro-batch must be an even value >=2 so boundary pairs stay local"
        )
    effective = math.prod(values)
    if effective != GLOBAL_BATCH_SIZE:
        raise M0MobileError(
            f"effective global batch is {effective}, expected {GLOBAL_BATCH_SIZE}"
        )
    return effective


def configure_deepspeed_micro_batch(
    deepspeed_config: MutableMapping[str, Any], micro_batch_per_rank: int
) -> None:
    """Bind a custom split-batch sampler to DeepSpeed's per-device batch."""

    if micro_batch_per_rank <= 0:
        raise ValueError("micro batch per rank must be positive")
    key = "train_micro_batch_size_per_gpu"
    configured = deepspeed_config.get(key)
    if configured not in (None, "auto", micro_batch_per_rank):
        raise M0MobileError(
            "DeepSpeed train micro batch conflicts with --micro-batch-per-rank"
        )
    deepspeed_config[key] = micro_batch_per_rank


def set_training_stage(
    model: ConveyorVLAJointTrajectoryPolicy,
    completed_optimizer_steps: int,
    stages: TrainingStages,
) -> str:
    stage = stages.stage(completed_optimizer_steps)
    if stage == "A":
        model.enable_action_warmup()
    else:
        model.enable_full_finetuning()
    return stage


def build_optimizer(
    model: ConveyorVLAJointTrajectoryPolicy,
    config: Mapping[str, Any],
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    optimization = _mapping(config.get("optimization"), "optimization")
    groups: dict[str, list[torch.nn.Parameter]] = {
        "action_experts": list(model.navigation_expert.parameters())
        + list(model.manipulation_expert.parameters()),
        "qwen_core": [],
        "qwen_vision": [],
        "route_embeddings_lm_head": [],
        "auxiliary_heads": list(model.auxiliary_heads.parameters()),
    }
    lm_suffixes = ("embed_tokens.weight", "lm_head.weight")
    for name, parameter in model.qwen.named_parameters():
        if name.endswith(lm_suffixes):
            group = "route_embeddings_lm_head"
        elif _is_vision_parameter(name):
            group = "qwen_vision"
        else:
            group = "qwen_core"
        groups[group].append(parameter)
    if any(not values for values in groups.values()):
        empty = [name for name, values in groups.items() if not values]
        raise M0MobileError(f"joint-trajectory optimizer groups are empty: {empty}")
    rates = {
        "action_experts": float(optimization["action_learning_rate"]),
        "qwen_core": float(optimization["qwen_learning_rate"]),
        "qwen_vision": float(optimization["vision_learning_rate"]),
        "route_embeddings_lm_head": float(optimization["route_lm_learning_rate"]),
        "auxiliary_heads": float(optimization["auxiliary_learning_rate"]),
    }
    flat = [parameter for values in groups.values() for parameter in values]
    if len({id(parameter) for parameter in flat}) != len(flat):
        raise M0MobileError("joint-trajectory optimizer groups overlap")
    expected = {id(parameter) for parameter in model.parameters()}
    if {id(parameter) for parameter in flat} != expected:
        raise M0MobileError("joint-trajectory optimizer does not cover every parameter")
    optimizer = torch.optim.AdamW(
        [
            {"name": name, "params": parameters, "lr": rates[name]}
            for name, parameters in groups.items()
        ],
        betas=tuple(float(value) for value in optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    report = [
        {
            "name": name,
            "learning_rate": rates[name],
            "parameter_tensors": len(parameters),
            "parameters": sum(parameter.numel() for parameter in parameters),
        }
        for name, parameters in groups.items()
    ]
    return optimizer, report


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    stages: TrainingStages,
    config: Mapping[str, Any],
) -> torch.optim.lr_scheduler.LambdaLR:
    optimization = _mapping(config.get("optimization"), "optimization")
    action_warmup = int(optimization["action_warmup_steps"])
    qwen_warmup = int(optimization["qwen_warmup_steps"])
    floor = float(optimization["cosine_min_ratio"])
    if action_warmup < 0 or qwen_warmup < 0 or not 0.0 <= floor <= 1.0:
        raise ValueError("joint-trajectory scheduler configuration is invalid")

    def action(step: int) -> float:
        if action_warmup and step < action_warmup:
            return max(1, step + 1) / action_warmup
        return _cosine_ratio(step, action_warmup, stages.total_steps, floor)

    def stage_b(step: int) -> float:
        if step < stages.stage_a_steps:
            return 0.0
        local = step - stages.stage_a_steps
        if qwen_warmup and local < qwen_warmup:
            return max(1, local + 1) / qwen_warmup
        return _cosine_ratio(
            local,
            qwen_warmup,
            stages.total_steps - stages.stage_a_steps,
            floor,
        )

    lambdas = []
    for group in optimizer.param_groups:
        lambdas.append(action if group["name"] == "action_experts" else stage_b)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def load_joint_trajectory_config(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise M0MobileError(f"joint-trajectory config does not exist: {config_path}")
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise M0MobileError("joint-trajectory config must be a JSON object")
    validate_joint_trajectory_config(value)
    return value


def load_consolidated_checkpoint(path: str | Path) -> Mapping[str, torch.Tensor]:
    """Load one consolidated torch/safetensors file or a safetensors index."""

    source = Path(path).expanduser().resolve()
    if source.is_file():
        return _load_tensor_file(source)
    if not source.is_dir():
        raise M0MobileError(f"warm-start checkpoint does not exist: {source}")
    for name in ("model.safetensors", "pytorch_model.bin", "model.pt"):
        candidate = source / name
        if candidate.is_file():
            return _load_tensor_file(candidate)
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = _mapping(index.get("weight_map"), "safetensors weight_map")
        shards = sorted(set(str(value) for value in weight_map.values()))
        state: dict[str, torch.Tensor] = {}
        for shard in shards:
            loaded = _load_tensor_file(source / shard)
            overlap = set(state).intersection(loaded)
            if overlap:
                raise M0MobileError(f"warm-start shards duplicate keys: {sorted(overlap)[:3]}")
            state.update(loaded)
        if set(state) != set(weight_map):
            raise M0MobileError("warm-start safetensors index and shards do not align")
        return state
    if (source / "zero_to_fp32.py").is_file():
        raise M0MobileError(
            "ZeRO checkpoint must first be exported to consolidated safetensors; "
            "training never resumes the old optimizer state"
        )
    raise M0MobileError("warm-start directory has no consolidated model weights")


def consolidated_checkpoint_identity(path: str | Path) -> Mapping[str, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    if source.is_file():
        paths = (source,)
        root = source.parent
    elif source.is_dir():
        candidates = [
            source / name
            for name in (
                "model.safetensors",
                "pytorch_model.bin",
                "model.pt",
                "model.safetensors.index.json",
            )
            if (source / name).is_file()
        ]
        if (source / "model.safetensors.index.json").is_file():
            index = json.loads(
                (source / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            weight_map = _mapping(index.get("weight_map"), "safetensors weight_map")
            candidates.extend(source / str(name) for name in sorted(set(weight_map.values())))
        paths = tuple(dict.fromkeys(candidates))
        root = source
    else:
        raise M0MobileError(f"warm-start checkpoint does not exist: {source}")
    if not paths or any(not candidate.is_file() for candidate in paths):
        raise M0MobileError("warm-start identity cannot resolve all weight files")
    return {
        candidate.relative_to(root).as_posix(): {
            "size": candidate.stat().st_size,
            "sha256": _file_sha256(candidate),
        }
        for candidate in paths
    }


def _load_tensor_file(path: Path) -> Mapping[str, torch.Tensor]:
    if not path.is_file():
        raise M0MobileError(f"warm-start shard is missing: {path}")
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise M0MobileError("safetensors is required for warm-start loading") from error
        value = load_file(str(path), device="cpu")
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, Mapping) and isinstance(value.get("state_dict"), Mapping):
            value = value["state_dict"]
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor)
        for key, tensor in value.items()
    ):
        raise M0MobileError(f"warm-start file is not a tensor state dict: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_joint_trajectory_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != POLICY_CONFIG_SCHEMA_VERSION:
        raise M0MobileError("joint-trajectory policy config schema is incompatible")
    if config.get("model_contract_id") != MODEL_CONTRACT_ID:
        raise M0MobileError("joint-trajectory model contract ID is incompatible")
    if config.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise M0MobileError("joint-trajectory dataset schema ID is incompatible")
    if config.get("dataset_profile") != DATASET_PROFILE:
        raise M0MobileError("joint-trajectory dataset profile is incompatible")
    dataset = _mapping(config.get("dataset"), "dataset")
    if dataset != {
        "source": "modelscope",
        "dataset_id": "OscarXu/liangzhuNeW_500",
        "revision": "6806fadf2e8e125ca871f576676b60e7db1605dc",
    }:
        raise M0MobileError(
            "joint-trajectory source dataset must be the pinned OscarXu/liangzhuNeW_500 snapshot"
        )
    vlm = _mapping(config.get("vlm"), "vlm")
    _require_config_fields(
        vlm,
        {
            "architecture": "Qwen3VLForConditionalGeneration",
            "relative_path": "Qwen3-VL-4B-Instruct",
            "hidden_size": 2560,
            "checkpoint_vocab_size": 153984,
            "action_feature_layer": "last_hidden_state",
            "full_finetuning_stage_b": True,
            "dtype": "bfloat16",
        },
        "vlm",
    )
    router = _mapping(config.get("router"), "router")
    _require_config_fields(
        router,
        {
            "active_routes": [route.value for route in JointTrajectoryRoute],
            "max_subtask_tokens": 24,
            "done_token_active": False,
        },
        "router",
    )
    action = _mapping(config.get("action_model"), "action_model")
    expected_action = {
        "architecture": "ABotM0LastHiddenDualDiT",
        "action_horizon": ACTION_HORIZON,
        "navigation_action_dim": NAVIGATION_ACTION_DIM,
        "manipulation_action_dim": MANIPULATION_ACTION_DIM,
        "manipulation_state_dim": MANIPULATION_STATE_DIM,
        "navigation_stride_s": NAVIGATION_STRIDE_S,
        "manipulation_stride_s": MANIPULATION_STRIDE_S,
        "cross_attention_dim": 2560,
        "input_embedding_dim": 768,
        "hidden_size": 1024,
        "num_layers": 16,
        "num_attention_heads": 12,
        "attention_head_dim": 64,
        "dropout": 0.2,
        "max_seq_len": 1024,
        "num_target_vision_tokens": 32,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "interleave_self_attention": True,
        "expert_parameters_shared": False,
        "future_tokens": True,
        "block_order": [
            "qwen_cross_attention_on_even_blocks",
            "self_attention_on_odd_blocks",
            "ffn_each_block",
        ],
    }
    _require_config_fields(action, expected_action, "action_model")
    auxiliary = _mapping(config.get("auxiliary"), "auxiliary")
    _require_config_fields(
        auxiliary,
        {
            "progress_hidden_size": 256,
            "progress_target": "unavailable_in_sampled_5hz_source_masked",
            "elapsed_time_fallback": False,
            "row_index_fallback": False,
        },
        "auxiliary",
    )
    loss = _mapping(config.get("loss"), "loss")
    expected_loss = {
        "lambda_answer": 1.0,
        "lambda_route": 1.0,
        "lambda_navigation": 1.0,
        "lambda_manipulation": 1.0,
        "repeated_diffusion_steps": 1,
        "manipulation_joint_weight": 0.75,
        "manipulation_gripper_weight": 0.25,
        "lambda_boundary": 0.2,
        "lambda_progress": 0.0,
        "boundary_rank_margin": 0.2,
    }
    _require_config_fields(loss, expected_loss, "loss")
    initialization = _mapping(config.get("initialization"), "initialization")
    expected_initialization = {
        "mode": "abot_m0_pretrain_strict_domain_transfer",
        "source_model_id": "amap_cvlab/ABot-M0-Pretrain",
        "relative_path": "ABot-M0-Pretrain/checkpoints/ABot_M0_Pretrain.pt",
        "checkpoint_sha256": "94478682b5c9eecf6f02179ba67ae47ea41257ca059bea6dd20e161716f5e16b",
        "qwen": "strict_load_then_reinitialize_waypoint_token_rows",
        "action": "strict_trunk_load_reinitialize_domain_boundaries",
        "progress_head": "reinitialize",
        "optimizer_scheduler_rng": "reinitialize",
        "normalizer": "fit_sampled_5hz_train_only",
    }
    _require_config_fields(initialization, expected_initialization, "initialization")
    disabled = _mapping(config.get("disabled"), "disabled")
    required_disabled = {
        "done",
        "prefix",
        "crl",
        "on_policy_correction",
        "self_conditioned_auxiliary",
        "image_augmentation",
        "ik",
        "curobo",
    }
    if set(disabled) != required_disabled or any(disabled[name] is not True for name in required_disabled):
        raise M0MobileError("joint-trajectory disabled feature set changed")
    sampling = _mapping(config.get("sampling"), "sampling")
    if sampling.get("global_batch_size") != GLOBAL_BATCH_SIZE:
        raise M0MobileError(f"sampling.global_batch_size must be {GLOBAL_BATCH_SIZE}")
    if [sampling.get(key) for key in ("nav_interior", "mani_interior", "boundary_rows")] != [
        NAV_INTERIOR_PER_BATCH,
        MANI_INTERIOR_PER_BATCH,
        BOUNDARY_ROWS_PER_BATCH,
    ]:
        raise M0MobileError("joint-trajectory batch mixture changed")
    _require_config_fields(
        sampling,
        {
            "ordinary_rows_per_episode_max": 1,
            "minimum_distinct_episodes": MIN_DISTINCT_EPISODES,
            "mani_gripper_transition_fraction_min": 0.25,
        },
        "sampling",
    )
    training = _mapping(config.get("training"), "training")
    _require_config_fields(
        training,
        {
            "stage_a_equivalent_epochs": 0.25,
            "stage_b_equivalent_epochs": 1.75,
            "total_equivalent_epochs": 2.0,
            "save_interval_steps": 250,
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "precision": "bf16",
        },
        "training",
    )
    if not math.isclose(
        float(training["stage_a_equivalent_epochs"])
        + float(training["stage_b_equivalent_epochs"]),
        float(training["total_equivalent_epochs"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise M0MobileError("joint-trajectory training stage lengths do not sum to total")
    optimization = _mapping(config.get("optimization"), "optimization")
    _require_config_fields(
        optimization,
        {
            "optimizer": "AdamW",
            "action_learning_rate": 2.0e-5,
            "qwen_learning_rate": 2.0e-6,
            "vision_learning_rate": 5.0e-7,
            "route_lm_learning_rate": 1.0e-5,
            "auxiliary_learning_rate": 1.0e-5,
            "betas": [0.9, 0.95],
            "epsilon": 1.0e-8,
            "weight_decay": 1.0e-8,
            "max_gradient_norm": 1.0,
            "action_warmup_steps": 200,
            "qwen_warmup_steps": 100,
            "decay": "cosine",
            "cosine_min_ratio": 0.1,
        },
        "optimization",
    )
    route = _mapping(config.get("route"), "route")
    _require_config_fields(
        route,
        {
            "interior_target": "hard_ce",
            "transition_target": "old_new_soft_ce",
            "transition_tau_s": dict(TRANSITION_TAU_S),
            "confirmation_observations": 2,
            "confidence_threshold": None,
        },
        "route",
    )
    runtime = _mapping(config.get("runtime"), "runtime")
    _require_config_fields(
        runtime,
        {
            "navigation": "full_10_point_reference_to_pct_dwa",
            "manipulation": "direct_joint_sequential_10_point",
            "manipulation_command_period_s": MANIPULATION_STRIDE_S,
            "manipulation_horizon_s": ACTION_HORIZON * MANIPULATION_STRIDE_S,
            "manipulation_base_velocity": [0, 0, 0],
            "pending_behavior": "base_zero_and_hold_last_joint_target",
            "joint_position_saturation": True,
            "joint_rate_saturation": True,
            "validation_saturation_rate_max": 0.005,
            "success": "released_and_inside_target_for_1.0s_orientation_free",
        },
        "runtime",
    )


def _require_config_fields(
    value: Mapping[str, Any], expected: Mapping[str, Any], section: str
) -> None:
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise M0MobileError(f"{section}.{key} must be {expected_value!r}")


def _cosine_ratio(step: int, warmup: int, total: int, floor: float) -> float:
    if total <= warmup:
        return floor
    progress = max(0.0, min(1.0, (step - warmup) / (total - warmup)))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _is_vision_parameter(name: str) -> bool:
    components = name.split(".")
    return "visual" in components or "vision_model" in components or "vision_tower" in components


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be a mapping")
    return value


__all__ = [
    "BOUNDARY_ROWS_PER_BATCH",
    "GLOBAL_BATCH_SIZE",
    "MANI_INTERIOR_PER_BATCH",
    "MIN_DISTINCT_EPISODES",
    "NAV_INTERIOR_PER_BATCH",
    "StratifiedJointTrajectoryBatchSampler",
    "AccumulationMicroBatchSampler",
    "TrainingStages",
    "build_optimizer",
    "build_scheduler",
    "configure_deepspeed_micro_batch",
    "load_joint_trajectory_config",
    "load_consolidated_checkpoint",
    "consolidated_checkpoint_identity",
    "set_training_stage",
    "select_disposable_overfit_episodes",
    "validate_global_batch",
    "validate_joint_trajectory_config",
]
