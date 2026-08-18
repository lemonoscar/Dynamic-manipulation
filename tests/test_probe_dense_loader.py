from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_dense_loader.py"
SPEC = importlib.util.spec_from_file_location("probe_dense_loader", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _manifest() -> dict:
    return {
        "train_navigation_action_statistics": {
            "action_indices_in_action10": [0, 2],
            "recommended_physical_scale": [0.47, 0.52],
        },
        "train_manipulation_action_statistics": {
            "action_indices_in_action10": [3, 4, 5, 6, 7, 8, 9],
            "recommended_physical_scale": [0.2, 0.1, 0.25, 0.1, 1.4, 0.1, None],
        },
    }


def test_action_scale_check_accepts_all_train_recommendations() -> None:
    config = {
        "normalization": {
            "action": {
                "scale": [0.47, 1.0, 0.52, 0.3, 0.3, 0.25, 0.5, 1.4, 0.5, 1.0]
            }
        }
    }
    report = PROBE._check_action_scales(config, _manifest())
    assert report["ok"] is True
    assert report["problems"] == []


def test_action_scale_check_rejects_under_scaled_channel() -> None:
    config = {
        "normalization": {
            "action": {
                "scale": [0.3, 1.0, 0.52, 0.3, 0.3, 0.25, 0.5, 1.4, 0.5, 1.0]
            }
        }
    }
    report = PROBE._check_action_scales(config, _manifest())
    assert report["ok"] is False
    assert report["problems"] == ["action[0]=0.3 < recommended=0.47"]
