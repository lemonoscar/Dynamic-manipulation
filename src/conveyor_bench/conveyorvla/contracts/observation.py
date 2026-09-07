"""Explicit policy allowlist. Evaluator payloads never enter planner inputs."""
from dataclasses import dataclass
import math
from .action import finite_array


@dataclass(frozen=True)
class CurrentObservation:
    mission_id: str
    observation_id: str
    time_s: float
    q: tuple
    dq: tuple
    gripper: float
    images: tuple = ()
    image_times_s: tuple = ()
    base_xyyaw: tuple = (0., 0., 0.)

    def __post_init__(self):
        if not self.mission_id or not self.observation_id or not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError('invalid observation identity/time')
        finite_array(self.q, (6,), 'q'); finite_array(self.dq, (6,), 'dq')
        finite_array(self.base_xyyaw, (3,), 'base_xyyaw')
        # Measured calibrated state may overshoot sensor endpoints; commands have a separate gate.
        if not math.isfinite(self.gripper):
            raise ValueError('invalid measured gripper')
        if len(self.images) != len(self.image_times_s) or any(
                not math.isfinite(t) or t > self.time_s or t < 0 for t in self.image_times_s):
            raise ValueError('invalid/future image timestamps')

    @classmethod
    def from_record(cls, record):
        # Deliberately ignore evaluator_truth, teacher_phase and all undeclared keys.
        return cls(**{key: record[key] for key in cls.__dataclass_fields__ if key in record})

    @property
    def mani_state(self):
        return (*self.q, *self.dq, self.gripper)
