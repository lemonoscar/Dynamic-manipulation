#!/usr/bin/env python3
"""Serve a consolidated state-free Waypoint v1/v2 policy on localhost."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import subprocess
import sys
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts import export_waypoint_inference as exporter  # noqa: E402
from scripts import train_waypoint as training  # noqa: E402

from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402
from conveyor_bench.conveyorvla.waypoint import MODEL_CONTRACT_ID  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_data import WaypointNormalizer  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_model import waypoint_token_ids  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_protocol import (  # noqa: E402
    WaypointProtocolError,
    WaypointRequest,
)
from conveyor_bench.conveyorvla.waypoint_runtime import (  # noqa: E402
    WaypointInferenceSession,
)
from conveyor_bench.conveyorvla.waypoint_v2 import MODEL_CONTRACT_ID_V2  # noqa: E402
from conveyor_bench.conveyorvla.waypoint_v2_data import (  # noqa: E402
    WaypointV2Normalizer,
)


MAX_REQUEST_BYTES = 16 * 1024 * 1024
REQUEST_KEYS = {
    "protocol_version",
    "request_id",
    "episode_id",
    "sequence_id",
    "instruction",
    "images",
    "camera_calibration_id",
}


class WaypointPolicyService:
    def __init__(self, session: WaypointInferenceSession, torch: Any, report: Mapping[str, Any], seed: int) -> None:
        self.session = session
        self.torch = torch
        self.report = dict(report)
        self.seed = int(seed)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "protocol_version": "conveyorvla-waypoint-runtime/v1",
            "model_inputs": ["instruction", "head[t-0.20,t]", "wrist[t-0.20,t]"],
            "robot_state_fields": 0,
            **self.report,
        }

    def infer(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = _decode_request(payload)
        seed = (self.seed + request.sequence_id) % (2**63 - 1)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)
        result = self.session.infer(request)
        return {
            "response": result.response.to_mapping(),
            "trace": asdict(result.trace),
            "diffusion_seed": seed,
        }


class WaypointRequestHandler(BaseHTTPRequestHandler):
    server_version = "ConveyorVLA-Waypoint/1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, self.server.service.health())  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            payload = self._read_payload()
            result = self.server.service.infer(payload)  # type: ignore[attr-defined]
        except (WaypointProtocolError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error, ValueError) as error:
            self._send(400, {"error": str(error)})
            return
        except Exception as error:
            print(
                json.dumps(
                    {"event": "waypoint_inference_failed", "error": type(error).__name__},
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
            self._send(500, {"error": "inference failed"})
            return
        self._send(200, result)

    def _read_payload(self) -> Mapping[str, Any]:
        if self.headers.get_content_type() != "application/json":
            raise ValueError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        length = int(raw_length)
        if not 1 <= length <= MAX_REQUEST_BYTES:
            raise ValueError("request body has an invalid size")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("request body ended before Content-Length")
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise ValueError("request body must be an object")
        return value

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
    )
    return parser


def load_service(args: argparse.Namespace) -> tuple[WaypointPolicyService, dict[str, Any]]:
    import torch
    from transformers import AutoProcessor
    from transformers.modeling_utils import load_sharded_checkpoint

    root = args.export_dir.expanduser().resolve()
    manifest = _validate_export(root)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise M0MobileError("waypoint online inference requires a CUDA device")
    torch.cuda.set_device(device)
    config = training._load_config(root / "policy_config.json")
    attention = args.attention_implementation or str(
        manifest["attention_implementation"]
    )
    policy, ids = training._build_model(
        config,
        Path(str(manifest["model_root"])),
        attention,
    )
    processor = AutoProcessor.from_pretrained(
        root / "processor",
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"
    policy.qwen.processor = processor
    resolved_ids = waypoint_token_ids(policy.qwen)
    ids = {
        "pred_action": resolved_ids.pred_action,
        "pred_done": resolved_ids.pred_done,
        "routes": list(resolved_ids.route_ids),
        "subtask_start": resolved_ids.subtask_start,
        "subtask_end": resolved_ids.subtask_end,
    }
    if ids != manifest["special_token_ids"]:
        raise M0MobileError("inference processor token IDs do not match the export")
    load_sharded_checkpoint(
        policy,
        root / str(_mapping(manifest["weights"], "weights")["relative_path"]),
        strict=True,
        prefer_safe=True,
    )
    policy.requires_grad_(False)
    policy.eval()
    policy.to(device)
    is_v2 = manifest["model_contract_id"] == MODEL_CONTRACT_ID_V2
    if training._is_v2_config(config) != is_v2:
        raise M0MobileError("inference export config/model contract mismatch")
    normalizer = (
        WaypointV2Normalizer.from_path(root / "normalization.json")
        if is_v2
        else WaypointNormalizer.from_path(root / "normalization.json")
    )
    camera = _mapping(manifest["camera_contract"], "camera contract")
    checkpoint_id = (
        f"step_{int(manifest['global_step']):06d}@"
        f"{_mapping(manifest['source_git'], 'source Git')['commit'][:12]}"
    )
    session = WaypointInferenceSession(
        policy,
        normalizer,
        checkpoint_id=checkpoint_id,
        normalization_sha256=str(manifest["normalization_sha256"]),
        camera_calibration_id=str(camera["camera_calibration_id"]),
    )
    report = {
        "checkpoint_id": checkpoint_id,
        "model_contract_id": manifest["model_contract_id"],
        "global_step": int(manifest["global_step"]),
        "device": str(device),
        "normalization_sha256": manifest["normalization_sha256"],
        "source_git": manifest["source_git"],
        "trusted_prefix_runtime": is_v2,
    }
    return WaypointPolicyService(session, torch, report, args.seed), report


def create_server(port: int, service: WaypointPolicyService) -> HTTPServer:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be within [0,65535]")
    server = HTTPServer(("127.0.0.1", port), WaypointRequestHandler)
    server.service = service  # type: ignore[attr-defined]
    return server


def _decode_request(payload: Mapping[str, Any]) -> WaypointRequest:
    if set(payload) != REQUEST_KEYS:
        raise WaypointProtocolError(
            "waypoint HTTP request keys are not exact: "
            + ", ".join(sorted(set(payload).symmetric_difference(REQUEST_KEYS)))
        )
    wire = WaypointRequest.from_mapping(payload)
    return WaypointRequest(
        request_id=wire.request_id,
        episode_id=wire.episode_id,
        sequence_id=wire.sequence_id,
        instruction=wire.instruction,
        head_images=tuple(_decode_jpeg(value) for value in wire.head_images),
        wrist_images=tuple(_decode_jpeg(value) for value in wire.wrist_images),
        camera_calibration_id=wire.camera_calibration_id,
        protocol_version=wire.protocol_version,
    )


def _decode_jpeg(value: Any) -> Any:
    from PIL import Image, UnidentifiedImageError

    if not isinstance(value, str) or not value:
        raise WaypointProtocolError("waypoint images must be base64 JPEG strings")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "JPEG":
                raise WaypointProtocolError("waypoint images must use JPEG encoding")
            return image.convert("RGB").copy()
    except UnidentifiedImageError as error:
        raise WaypointProtocolError("waypoint image is not a valid JPEG") from error


def _validate_export(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "inference_manifest.json")
    contract = manifest.get("model_contract_id")
    expected_schema = {
        MODEL_CONTRACT_ID: exporter.EXPORT_SCHEMA,
        MODEL_CONTRACT_ID_V2: exporter.EXPORT_SCHEMA_V2,
    }.get(contract)
    if manifest.get("schema_version") != expected_schema or manifest.get("status") != "complete":
        raise M0MobileError("waypoint inference export is incomplete or incompatible")
    if contract not in {MODEL_CONTRACT_ID, MODEL_CONTRACT_ID_V2}:
        raise M0MobileError("waypoint inference model contract is incompatible")
    if training.common._sha256(root / "policy_config.json") != manifest["policy_config_sha256"]:
        raise M0MobileError("waypoint inference policy config binding is corrupt")
    if training.common._sha256(root / "normalization.json") != manifest["normalization_sha256"]:
        raise M0MobileError("waypoint inference normalization binding is corrupt")
    _verify_files(root / "processor", _mapping(manifest["processor_files"], "processor files"))
    weights = _mapping(manifest["weights"], "weights")
    _verify_files(root / str(weights["relative_path"]), _mapping(weights["files"], "weight files"))
    if training.common._sha256(root / "source_checkpoint_manifest.json") != manifest["source_checkpoint_manifest_sha256"]:
        raise M0MobileError("source checkpoint manifest binding is corrupt")
    if training.common._sha256(root / "source_resolved_run.json") != manifest["source_resolved_run_sha256"]:
        raise M0MobileError("source resolved-run binding is corrupt")
    qwen = _mapping(manifest["qwen_base"], "Qwen base")
    qwen_root = Path(str(qwen["model_dir"]))
    _verify_files(qwen_root, _mapping(qwen["files"], "Qwen files"))
    current_git = training._source_git_identity(PROJECT_ROOT)
    if current_git["dirty_state_artifact"]["is_dirty"]:
        raise M0MobileError("current waypoint inference checkout is dirty")
    source_commit = str(_mapping(manifest["source_git"], "source Git")["commit"])
    if source_commit != current_git["commit"]:
        ancestor = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "merge-base", "--is-ancestor", source_commit, current_git["commit"]],
            check=False,
        )
        if ancestor.returncode:
            raise M0MobileError("inference export source is not an ancestor of this checkout")
    return manifest


def _verify_files(root: Path, identities: Mapping[str, Any]) -> None:
    if not identities:
        raise M0MobileError(f"artifact binding is empty: {root}")
    for relative, raw_identity in identities.items():
        identity = _mapping(raw_identity, f"file identity {relative}")
        path = root / str(relative)
        if not path.is_file() or path.stat().st_size != int(identity["size"]):
            raise M0MobileError(f"artifact size binding failed: {path}")
        if training.common._sha256(path) != identity["sha256"]:
            raise M0MobileError(f"artifact hash binding failed: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0MobileError(f"cannot read inference manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise M0MobileError("waypoint inference manifest must be an object")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise M0MobileError(f"{name} must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        service, report = load_service(args)
        server = create_server(args.port, service)
    except (M0MobileError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"event": "startup_failed", "error": str(error)}), file=sys.stderr, flush=True)
        return 2
    print(
        json.dumps(
            {"event": "ready", "bind": f"127.0.0.1:{server.server_port}", "model": report},
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
