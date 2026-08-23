import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from conveyor_bench.conveyorvla.waypoint import ACTION_HORIZON, WaypointRoute
from conveyor_bench.conveyorvla.waypoint_data import FORBIDDEN_MODEL_KEYS
from conveyor_bench.conveyorvla.waypoint_v2 import (
    DATASET_SCHEMA_VERSION_V2_COMMAND_GRIPPER,
)
from conveyor_bench.conveyorvla.waypoint_v2_data import (
    MODEL_BATCH_KEYS_V2,
    NORMALIZATION_SCHEMA_VERSION_V2,
    ConveyorVLAWaypointV2Dataset,
    select_waypoint_v2_episodes_per_split,
    upgrade_waypoint_records,
)


def test_model_input_contract_does_not_expand_after_materialization():
    assert "transition_id" in MODEL_BATCH_KEYS_V2
    assert "boundary_transition" not in MODEL_BATCH_KEYS_V2


def _record(route: WaypointRoute, position: int, prefix_k: int) -> dict:
    width = 3 if route in {WaypointRoute.NAV_TO_SOURCE, WaypointRoute.NAV_TO_TARGET} else 7
    action = [
        [float(100 * position + 10 * step + dimension) for dimension in range(width)]
        for step in range(ACTION_HORIZON)
    ]
    is_done = route is WaypointRoute.DONE
    return {
        "schema_version": "conveyorvla-waypoint-dense-transition-v1",
        "source_dataset_id": "liangzhu_0815_n200",
        "source_episode_id": "liangzhu_0815_n200:episode_000001",
        "source_row_id": position,
        "split": "train",
        "timestamp": position * 0.2,
        "route": route.value,
        "route_token": None if is_done else f"<{route.value}>",
        "action_domain": (
            "NONE"
            if is_done
            else "NAVIGATION"
            if width == 3
            else "MANIPULATION"
        ),
        "nav_waypoints_body": None if is_done or width != 3 else action,
        "arm_targets_base": None if is_done or width != 7 else action,
        "action_valid_mask": [not is_done and index < prefix_k for index in range(ACTION_HORIZON)],
        "label_provenance": {
            "target_source_rows": [position + index + 1 for index in range(ACTION_HORIZON)]
        },
        "roundtrip_error": {
            "navigation_max_m_or_rad": 0.0,
            "arm_max_m_or_rad": 0.0,
        },
    }


def _raw_samples(count: int) -> dict[int, dict]:
    explicit_commands = {0: "close", 3: "open", 5: "close", 9: "open"}
    return {
        index: {
            "base_pose": [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
            "tcp_pose": [0.3 + index * 0.01, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            "gripper_position": 0.03,
            "subtask_signals": {
                "gripper_command": explicit_commands.get(index, "hold")
            },
        }
        for index in range(count)
    }


def _four_stage_records() -> list[dict]:
    routes_and_prefixes = (
        [(WaypointRoute.NAV_TO_SOURCE, 0)] * 3
        + [
            (WaypointRoute.PICK, 2),
            (WaypointRoute.PICK, 1),
            (WaypointRoute.PICK, 0),
        ]
        + [(WaypointRoute.NAV_TO_TARGET, 0)] * 3
        + [
            (WaypointRoute.PLACE, 2),
            (WaypointRoute.PLACE, 1),
            (WaypointRoute.PLACE, 0),
        ]
        + [(WaypointRoute.DONE, 0)]
    )
    return [
        _record(route, position, prefix_k)
        for position, (route, prefix_k) in enumerate(routes_and_prefixes)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_terminal_hold_preserves_original_k_and_uses_nearest_boundary() -> None:
    source = _four_stage_records()
    upgraded = upgrade_waypoint_records(source, _raw_samples(len(source)))

    last_nav = upgraded[2]
    first_pick = upgraded[3]
    assert last_nav["boundary_class"] == "BEFORE"
    assert first_pick["boundary_class"] == "AFTER"
    assert last_nav["transition_id"] == first_pick["transition_id"]
    assert last_nav["boundary_transition"] == "NAV_TO_SOURCE->PICK"
    assert first_pick["next_route"] == WaypointRoute.NAV_TO_TARGET.value
    assert last_nav["boundary_uncertainty_s"] == pytest.approx(0.2)

    zero_nav = upgraded[0]
    assert zero_nav["original_valid_prefix_k"] == 0
    assert zero_nav["prefix_target_k"] == 1
    assert zero_nav["suffix_reason"] == "boundary"
    assert all(zero_nav["action_valid_mask"])
    assert all(zero_nav["terminal_hold_mask"])
    assert zero_nav["padded_action"] == [[0.0, 0.0, 0.0]] * ACTION_HORIZON

    two_pick = upgraded[3]
    assert two_pick["original_valid_prefix_k"] == 2
    assert [row[6] for row in two_pick["padded_action"][:2]] == [1.0, 0.0]
    assert all(
        two_pick["padded_action"][index][:6]
        == source[3]["arm_targets_base"][index][:6]
        for index in range(2)
    )
    assert two_pick["padded_action"][2:] == [two_pick["padded_action"][1]] * 18
    assert two_pick["terminal_hold_mask"] == [False, False] + [True] * 18

    zero_pick = upgraded[5]
    assert zero_pick["original_valid_prefix_k"] == 0
    assert zero_pick["terminal_hold_target"] == pytest.approx(
        [0.35, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]
    )
    assert all(row == zero_pick["terminal_hold_target"] for row in zero_pick["padded_action"])

    assert upgraded[0]["phase_progress"] == pytest.approx(0.0)
    assert upgraded[2]["phase_progress"] == pytest.approx(1.0)
    done = upgraded[-1]
    assert done["suffix_reason"] == "done-no-action"
    assert done["padded_action"] is None
    assert not any(done["action_valid_mask"])


def test_command_gripper_label_uses_expert_command_not_measured_opening() -> None:
    source = [_record(WaypointRoute.PICK, 0, 1)]
    raw = _raw_samples(2)
    raw[0]["subtask_signals"]["gripper_command"] = "open"
    raw[1]["subtask_signals"]["gripper_command"] = "close"
    assert raw[1]["gripper_position"] / 0.04 > 0.5

    upgraded = upgrade_waypoint_records(source, raw)[0]

    assert upgraded["schema_version"] == DATASET_SCHEMA_VERSION_V2_COMMAND_GRIPPER
    assert upgraded["arm_targets_base"][0][6] == 0.0
    assert upgraded["label_provenance"]["gripper_action_source"] == (
        "future_expert_gripper_command"
    )
    assert upgraded["label_provenance"]["measured_joint_opening_used_as_target"] is False


def test_source_and_episode_tails_are_never_terminal_hold() -> None:
    records = [
        _record(WaypointRoute.NAV_TO_SOURCE, 0, 3),
        _record(WaypointRoute.NAV_TO_SOURCE, 100, ACTION_HORIZON),
        _record(WaypointRoute.NAV_TO_SOURCE, 200, 3),
    ]
    records[0]["timestamp"] = 0.0
    records[1]["timestamp"] = 20.0
    records[2]["timestamp"] = 40.0
    upgraded = upgrade_waypoint_records(records, _raw_samples(201))
    assert [row["suffix_reason"] for row in upgraded] == [
        "source-tail",
        "none",
        "episode-tail",
    ]
    for source, row in zip(records, upgraded, strict=True):
        assert row["action_valid_mask"] == source["action_valid_mask"]
        assert not any(row["terminal_hold_mask"])
        assert row["terminal_hold_applied"] is False


def test_v2_loader_exposes_supervision_without_robot_state(tmp_path: Path) -> None:
    record = dict(upgrade_waypoint_records(_four_stage_records(), _raw_samples(13))[3])
    images = []
    for index in range(4):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (4, 4), (index, 0, 0)).save(path)
        images.append(str(path))
    record.update(
        {
            "global_instruction": "Pick up the Coke can and place it in the other box.",
            "assistant_solution": "ACTION: PICK\nSUBTASK: grasp the can",
            "head_images": images[:2],
            "wrist_images": images[2:],
        }
    )
    root = tmp_path / "waypoint-v2"
    root.mkdir()
    train = root / "train.jsonl"
    train.write_text(json.dumps(record) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        (root / f"{split}.jsonl").write_text("", encoding="utf-8")
    normalization = {
        "schema_version": NORMALIZATION_SCHEMA_VERSION_V2,
        "dataset_schema_version": DATASET_SCHEMA_VERSION_V2_COMMAND_GRIPPER,
        "navigation": {
            "q01": [[-1.0] * 3 for _ in range(ACTION_HORIZON)],
            "q99": [[1.0] * 3 for _ in range(ACTION_HORIZON)],
        },
        "manipulation": {
            "q01": [[-1.0] * 6 for _ in range(ACTION_HORIZON)],
            "q99": [[1.0] * 6 for _ in range(ACTION_HORIZON)],
        },
    }
    normalization_path = root / "normalization.json"
    normalization_path.write_text(json.dumps(normalization) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION_V2_COMMAND_GRIPPER,
        "records": {
            split: {
                "relative_path": f"{split}.jsonl",
                "sha256": _sha256(root / f"{split}.jsonl"),
            }
            for split in ("train", "val", "test")
        },
        "normalization_relative_path": "normalization.json",
        "normalization_sha256": _sha256(normalization_path),
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    example = ConveyorVLAWaypointV2Dataset(root, split="train")[0]
    assert set(example) == MODEL_BATCH_KEYS_V2
    assert not FORBIDDEN_MODEL_KEYS.intersection(example)
    assert example["route"] == WaypointRoute.PICK.value
    assert example["original_valid_prefix_k"] == 2
    assert example["suffix_reason"] == "boundary"
    assert len(example["action"]) == ACTION_HORIZON
    assert all(len(clip) == 2 for clip in example["video"])


def test_smoke_episode_selection_is_balanced_and_deterministic(tmp_path: Path) -> None:
    roots = []
    expected = {"train": [], "val": [], "test": []}
    from conveyor_bench.conveyorvla.waypoint_data import _episode_split, _source_episode_id

    for index in range(1000):
        root = tmp_path / "liangzhu_0815_n200" / f"episode_{index:06d}"
        split = _episode_split(_source_episode_id(root.resolve()))
        roots.append(root)
        if len(expected[split]) < 2:
            expected[split].append(root.resolve())
        if all(len(values) == 2 for values in expected.values()):
            break
    selected = select_waypoint_v2_episodes_per_split(roots, 2)
    assert selected == tuple(
        root.resolve()
        for root in roots
        if root.resolve() in {value for values in expected.values() for value in values}
    )
