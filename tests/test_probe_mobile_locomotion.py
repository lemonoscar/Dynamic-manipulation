import importlib.util
from math import cos, sin
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_mobile_locomotion.py"
)
SPEC = importlib.util.spec_from_file_location(
    "probe_mobile_locomotion",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def _sample(
    *,
    phase: str,
    x: float,
    vx: float,
    z: float = 0.30,
    roll: float = 0.0,
    pitch: float = 0.0,
):
    return {
        "phase": phase,
        "root_position_m": [x, 0.0, z],
        "base_linear_velocity_mps": [vx, 0.0, 0.0],
        "base_angular_velocity_radps": [0.0, 0.0, 0.0],
        "roll_rad": roll,
        "pitch_rad": pitch,
    }


def test_roll_pitch_uses_scalar_first_quaternion():
    half_angle = 0.25
    quaternion = (cos(half_angle), sin(half_angle), 0.0, 0.0)

    roll, pitch = probe.roll_pitch_from_quaternion(quaternion)

    assert roll == pytest.approx(0.5)
    assert pitch == pytest.approx(0.0)


def test_evaluate_samples_accepts_tracking_and_stopping():
    samples = [_sample(phase="initial", x=0.0, vx=0.0)]
    samples.extend(
        _sample(phase="settle", x=index * 0.0001, vx=0.0)
        for index in range(1, 6)
    )
    samples.extend(
        _sample(phase="track", x=0.004 * index, vx=0.2)
        for index in range(1, 21)
    )
    samples.extend(
        _sample(phase="stop", x=0.08, vx=0.0)
        for _ in range(10)
    )

    metrics = probe.evaluate_samples(samples, (0.2, 0.0, 0.0))

    assert metrics["passed"] is True
    assert metrics["failures"] == []
    assert metrics["fall_detected"] is False
    assert metrics["steady_tracking"]["vx_gain"] == pytest.approx(1.0)
    assert metrics["steady_tracking"]["rmse_vx_vy_wz"] == [0.0, 0.0, 0.0]


def test_evaluate_samples_reports_tracking_and_fall_failures():
    samples = [_sample(phase="initial", x=0.0, vx=0.0)]
    samples.extend(
        _sample(
            phase="track",
            x=0.0,
            vx=0.01,
            z=0.15 if index == 4 else 0.30,
            pitch=0.8 if index == 4 else 0.0,
        )
        for index in range(10)
    )

    metrics = probe.evaluate_samples(samples, (0.2, 0.0, 0.0))

    assert metrics["passed"] is False
    assert metrics["fall_detected"] is True
    assert {"fall_detected", "posture_tilt", "vx_gain", "vx_rmse"} <= set(
        metrics["failures"]
    )


def test_evaluate_samples_rejects_nonfinite_state():
    samples = [
        _sample(phase="initial", x=0.0, vx=0.0),
        _sample(phase="track", x=np.nan, vx=0.2),
    ]

    metrics = probe.evaluate_samples(samples, (0.2, 0.0, 0.0))

    assert metrics == {
        "passed": False,
        "failures": ["nonfinite_state"],
        "fall_detected": True,
        "sample_count": 2,
    }
