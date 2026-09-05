#!/usr/bin/env python3
"""Serve a consolidated Joint-Trajectory checkpoint on loopback HTTP."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers.modeling_utils import load_sharded_checkpoint  # noqa: E402

from scripts import train_joint_trajectory as training  # noqa: E402
from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.formal_checkpoint import (  # noqa: E402
    validate_formal_checkpoint, load_formal_policy, public_identity, source_identity,
)
from conveyor_bench.conveyorvla.joint_trajectory_data import (  # noqa: E402
    ConveyorVLAJointTrajectoryDataset,
)
from conveyor_bench.conveyorvla.joint_trajectory_runtime import (  # noqa: E402
    DirectJointTrajectoryExecutor,
    JointSafetyLimits,
    JointTrajectoryInferenceSession,
    JointTrajectoryRuntimeRequest,
)


ARM_LOWER = (-2.618, 0.0, 0.0, -1.5708, -1.5708, -1.5708)
ARM_UPPER = (3.14, 3.14, 3.14, 1.5708, 1.5708, 1.5708)
ARM_MAX_RATE_RAD_S = (3.0,) * 6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--weights", type=Path, help="legacy overfit bundle only")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/manipulation_navi_v1.json")
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    return parser


class JointTrajectoryService:
    def __init__(
        self,
        session: JointTrajectoryInferenceSession,
        *,
        seed: int,
        identity: Mapping[str, Any],
    ) -> None:
        self.session = session
        self.seed = int(seed)
        self.identity = dict(identity)

    def health(self) -> Mapping[str, Any]:
        return {
            "ok": True,
            "protocol_version": "conveyorvla-joint-trajectory-runtime/v1",
            **self.identity,
        }

    def infer(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {
            "protocol_version",
            "request_id",
            "episode_id",
            "sequence_id",
            "instruction",
            "head_images",
            "wrist_images",
            "joint_position",
            "joint_velocity",
            "gripper_open_fraction",
            "diffusion_seed",
        }
        extra = set(payload).difference(allowed)
        if extra:
            raise ValueError(f"runtime request contains forbidden fields: {sorted(extra)}")
        if payload.get("protocol_version") != "conveyorvla-joint-trajectory-runtime/v1":
            raise ValueError("joint-trajectory runtime protocol differs")
        request = JointTrajectoryRuntimeRequest(
            request_id=str(payload["request_id"]),
            episode_id=str(payload["episode_id"]),
            sequence_id=int(payload["sequence_id"]),
            instruction=str(payload["instruction"]),
            head_images=_decode_pair(payload["head_images"], "head_images"),
            wrist_images=_decode_pair(payload["wrist_images"], "wrist_images"),
            joint_position=_vector(payload["joint_position"], 6, "joint_position"),
            joint_velocity=_vector(payload["joint_velocity"], 6, "joint_velocity"),
            gripper_open_fraction=float(payload["gripper_open_fraction"]),
        )
        seed = payload.get("diffusion_seed", self.seed)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("diffusion_seed must be a uint32")
        torch.manual_seed(seed + request.sequence_id * 1009)
        torch.cuda.manual_seed_all(seed + request.sequence_id * 1009)
        step = self.session.step(request)
        return _runtime_step_mapping(step)


class _Handler(BaseHTTPRequestHandler):
    server_version = "ConveyorVLA-Joint-Trajectory/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(404, {"ok": False, "error": "not_found"})
            return
        self._send(200, self.server.service.health())  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send(404, {"ok": False, "error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16 * 1024 * 1024:
                raise ValueError("runtime request size is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, Mapping):
                raise ValueError("runtime request must be a JSON object")
            result = self.server.service.infer(payload)  # type: ignore[attr-defined]
        except Exception as error:
            self._send(
                400,
                {"ok": False, "error": f"{type(error).__name__}:{error}"},
            )
            return
        self._send(200, {"ok": True, "response": result})

    def log_message(self, format: str, *args: Any) -> None:
        print(
            json.dumps(
                {
                    "event": "http",
                    "client": self.client_address[0],
                    "message": format % args,
                }
            ),
            flush=True,
        )

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def load_service(args: argparse.Namespace) -> tuple[JointTrajectoryService, Mapping[str, Any]]:
    if not torch.cuda.is_available():
        raise M0MobileError("Joint-Trajectory service requires CUDA")
    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_manifest = _json(checkpoint / "joint_trajectory_checkpoint_manifest.json")
    resolved = _json(checkpoint.parents[1] / "resolved_run.json")
    if checkpoint_manifest.get("run_kind") == "formal":
        if args.weights is not None:
            raise M0MobileError("formal service loads only its bound model.safetensors")
        binding = validate_formal_checkpoint(checkpoint, args.config)
        policy = load_formal_policy(binding, args.model_root, args.attention_implementation)
        from conveyor_bench.conveyorvla.joint_trajectory_data import JointTrajectoryNormalizer
        normalizer = JointTrajectoryNormalizer.from_path(Path(binding["dataset_root"]) / "normalization.json")
        session = JointTrajectoryInferenceSession(
            policy, normalizer,
            DirectJointTrajectoryExecutor(JointSafetyLimits(ARM_LOWER, ARM_UPPER, ARM_MAX_RATE_RAD_S)),
            checkpoint_id=binding["checkpoint_id"], normalization_sha256=binding["normalization_sha256"],
        )
        identity = {**public_identity(binding), "source_sha256": source_identity(PROJECT_ROOT)["sha256"],
                    "strict_load": True, "device": str(next(policy.parameters()).device),
                    "dtype": str(next(policy.parameters()).dtype), "action_stride_s": .2,
                    "action_horizon": 10, "inference_seed": args.seed}
        return JointTrajectoryService(session, seed=args.seed, identity=identity), identity
    if checkpoint_manifest.get("global_step") != training.common._checkpoint_step(checkpoint):
        raise M0MobileError("checkpoint step binding differs")
    if checkpoint_manifest.get("run_kind") != "disposable_32_episode_overfit":
        raise M0MobileError("closed-loop service requires the 32-episode overfit checkpoint")
    if args.weights is None:
        raise M0MobileError("legacy overfit service requires --weights")
    for key in (
        "model_contract_id",
        "dataset_manifest_sha256",
        "normalization_sha256",
        "normalizer_id",
        "policy_config_sha256",
    ):
        if checkpoint_manifest.get(key) != resolved.get(key):
            raise M0MobileError(f"checkpoint/resolved-run binding differs: {key}")
    dataset = ConveyorVLAJointTrajectoryDataset(
        Path(str(resolved["dataset_root"])), split="train"
    )
    config = _mapping(resolved.get("resolved_policy_config"), "policy config")
    policy, token_ids = training._build_model(
        config,
        args.model_root.expanduser().resolve(),
        args.attention_implementation,
    )
    if dict(token_ids) != _mapping(resolved.get("special_token_ids"), "special token IDs"):
        raise M0MobileError("current processor token IDs differ from the checkpoint")
    incompatible = load_sharded_checkpoint(
        policy,
        str(args.weights.expanduser().resolve()),
        strict=True,
        prefer_safe=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise M0MobileError("consolidated checkpoint did not load strictly")
    policy.to(device=torch.device("cuda:0"), dtype=torch.bfloat16).eval()
    checkpoint_id = f"step_{int(checkpoint_manifest['global_step']):06d}"
    session = JointTrajectoryInferenceSession(
        policy,
        dataset.normalizer,
        DirectJointTrajectoryExecutor(
            JointSafetyLimits(ARM_LOWER, ARM_UPPER, ARM_MAX_RATE_RAD_S)
        ),
        checkpoint_id=checkpoint_id,
        normalization_sha256=str(checkpoint_manifest["normalization_sha256"]),
    )
    identity = {
        "checkpoint_id": checkpoint_id,
        "global_step": int(checkpoint_manifest["global_step"]),
        "model_contract_id": checkpoint_manifest["model_contract_id"],
        "normalization_sha256": checkpoint_manifest["normalization_sha256"],
        "dataset_manifest_sha256": checkpoint_manifest["dataset_manifest_sha256"],
        "device": str(next(policy.parameters()).device),
        "dtype": str(next(policy.parameters()).dtype),
    }
    return JointTrajectoryService(session, seed=args.seed, identity=identity), identity


def _runtime_step_mapping(step: Any) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "request_id": step.request_id,
        "sequence_id": step.sequence_id,
        "predicted_route": None if step.predicted_route is None else step.predicted_route.value,
        "committed_route": None if step.committed_route is None else step.committed_route.value,
        "commit_status": None if step.commit_status is None else step.commit_status.value,
        "route_probs": dict(step.route_probs),
        "subtask": step.subtask,
        "action_domain": None if step.action_domain is None else step.action_domain.value,
        "pass2_executed": step.pass2_executed,
        "checkpoint_id": step.checkpoint_id,
        "normalization_sha256": step.normalization_sha256,
        "elapsed_ms": step.elapsed_ms,
        "recover_reason": step.recover_reason,
        "navigation": None,
        "manipulation": None,
        "hold": None,
    }
    if step.navigation is not None:
        result["navigation"] = {
            "points_query_body": [list(row) for row in step.navigation.points_query_body],
            "local_goal_query_body": list(step.navigation.local_goal_query_body),
            "stride_s": step.navigation.stride_s,
        }
    if step.manipulation is not None:
        result["manipulation"] = {
            "commands": [
                {
                    "index": command.index,
                    "joint_position": list(command.joint_position),
                    "gripper_open_fraction": command.gripper_open_fraction,
                    "base_velocity": list(command.base_velocity),
                    "duration_s": command.duration_s,
                }
                for command in step.manipulation.commands
            ],
            "position_saturation_count": step.manipulation.position_saturation_count,
            "rate_saturation_count": step.manipulation.rate_saturation_count,
            "gripper_saturation_count": step.manipulation.gripper_saturation_count,
            "saturation_rate": step.manipulation.saturation_rate,
        }
    if step.hold is not None:
        result["hold"] = {
            "index": step.hold.index,
            "joint_position": list(step.hold.joint_position),
            "gripper_open_fraction": step.hold.gripper_open_fraction,
            "base_velocity": list(step.hold.base_velocity),
            "duration_s": step.hold.duration_s,
        }
    return result


def _decode_pair(value: Any, name: str) -> tuple[Image.Image, Image.Image]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two JPEG strings")
    images = []
    for encoded in value:
        raw = base64.b64decode(str(encoded), validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            images.append(image.convert("RGB"))
    return tuple(images)  # type: ignore[return-value]


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(float(item) for item in value)


def _json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise M0MobileError(f"binding file is missing: {path}")
    return _mapping(json.loads(path.read_text()), str(path))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        service, identity = load_service(args)
        server = HTTPServer(("127.0.0.1", args.port), _Handler)
        server.service = service  # type: ignore[attr-defined]
    except Exception as error:
        print(
            json.dumps({"event": "startup_failed", "error": f"{type(error).__name__}:{error}"}),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "event": "ready",
                "bind": f"127.0.0.1:{server.server_port}",
                "model": identity,
                "timestamp": time.time(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
