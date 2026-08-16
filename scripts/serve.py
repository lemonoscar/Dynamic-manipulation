#!/usr/bin/env python3
"""Serve ConveyorVLA AL0 behind an SSH-forwarded HTTP endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    MODEL_FAMILY,
    MODEL_NAME,
    MODEL_VARIANT,
    M0MobileError,
    M0MobileNormalizer,
    load_m0_mobile_config,
    resolve_model_root,
)
from conveyor_bench.conveyorvla.online import (  # noqa: E402
    ACTION_HORIZON as ONLINE_ACTION_HORIZON,
    MAX_REQUEST_BYTES,
    M0InferRequest,
    M0OnlineError,
    decode_rgb_jpeg,
    health_payload,
    load_state_statistics,
    make_infer_response,
    parse_infer_request,
)
from conveyor_bench.conveyorvla.temporal import (  # noqa: E402
    DEFAULT_TEMPORAL_CONFIG_PATH,
    build_temporal_policy_config,
    load_temporal_config,
)


class M0PolicyService:
    def __init__(
        self,
        policy: Any,
        normalizer: M0MobileNormalizer,
        action_mask: tuple[bool, ...],
        image_size: tuple[int, int],
        torch: Any,
        model_identity: Mapping[str, Any],
        temporal_history: bool = False,
        temporal_history_span_s: float | None = None,
    ) -> None:
        self.policy = policy
        self.normalizer = normalizer
        self.action_mask = action_mask
        self.image_size = image_size
        self.torch = torch
        self.model_identity = dict(model_identity)
        self.temporal_history = temporal_history
        self.temporal_history_span_s = temporal_history_span_s
        if temporal_history and temporal_history_span_s != 0.20:
            raise M0OnlineError(
                "temporal cache requires the Liangzhu 0.20 s observation interval"
            )
        self.temporal_cache_contract = (
            {
                "history_offsets_model_ticks": [-5, 0],
                "history_span_s": 0.20,
                "cache": "previous_contiguous_5hz_request",
            }
            if temporal_history
            else None
        )
        self._previous_images: list[Any] | None = None
        self._previous_sequence: int | None = None

    def health(self) -> Mapping[str, Any]:
        return health_payload(self.model_identity)

    def infer(self, request: M0InferRequest) -> Mapping[str, Any]:
        started = time.perf_counter()
        images = [decode_rgb_jpeg(payload) for payload in request.jpeg_images]
        if any(image.size != self.image_size for image in images):
            raise M0OnlineError(
                f"policy images must be exactly {self.image_size} pixels"
            )
        self.torch.manual_seed(request.seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(request.seed)
            self.torch.cuda.synchronize()
        example = {
            "lang": request.instruction,
            "state": (self.normalizer.normalize_state(request.state28),),
            "action_mask": self.action_mask,
        }
        if self.temporal_history:
            if request.sequence_id == 0:
                previous = images
            elif self._previous_sequence != request.sequence_id - 1:
                raise M0OnlineError(
                    "temporal requests must start at sequence 0 and remain contiguous"
                )
            else:
                assert self._previous_images is not None
                previous = self._previous_images
            example["video"] = (
                (previous[0], images[0]),
                (previous[1], images[1]),
            )
        else:
            example["image"] = images
        normalized = self.policy.predict_normalized_actions([example])
        if self.temporal_history:
            normalized = normalized[:, :ONLINE_ACTION_HORIZON]
        if self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        response = make_infer_response(
            request,
            normalized.detach().float().cpu().tolist()[0],
            latency_ms,
        )
        if self.temporal_history:
            self._previous_images = images
            self._previous_sequence = request.sequence_id
        return response


class M0RequestHandler(BaseHTTPRequestHandler):
    server_version = "ConveyorVLA-AL0/1"

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
            if self.headers.get_content_type() != "application/json":
                raise M0OnlineError("Content-Type must be application/json")
            length_text = self.headers.get("Content-Length")
            if length_text is None:
                raise M0OnlineError("Content-Length is required")
            try:
                length = int(length_text)
            except ValueError as error:
                raise M0OnlineError("Content-Length must be an integer") from error
            if not 1 <= length <= MAX_REQUEST_BYTES:
                raise M0OnlineError("request body has an invalid size")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise M0OnlineError("request body ended before Content-Length")
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise M0OnlineError(f"request body is not valid JSON: {error}") from error
            request = parse_infer_request(payload)
            response = self.server.service.infer(request)  # type: ignore[attr-defined]
        except (M0OnlineError, ValueError) as error:
            self._send(400, {"error": str(error)})
            return
        except Exception as error:  # fail closed without exposing a traceback remotely
            print(json.dumps({"event": "inference_failed", "error": str(error)}), file=sys.stderr, flush=True)
            self._send(500, {"error": "inference failed"})
            return
        self._send(200, response)

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(port: int, service: Any) -> HTTPServer:
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise M0OnlineError("port must be an integer within [0, 65535]")
    server = HTTPServer(("127.0.0.1", port), M0RequestHandler)
    server.service = service  # type: ignore[attr-defined]
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-checkpoint", required=True, type=Path)
    parser.add_argument("--state-statistics", required=True, type=Path)
    parser.add_argument(
        "--training-report",
        type=Path,
        help="Defaults to training_report.json beside the action checkpoint.",
    )
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--temporal-config",
        type=Path,
        default=DEFAULT_TEMPORAL_CONFIG_PATH,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--attention-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    return parser


def _checkpoint_path(config: Mapping[str, Any], root: Path) -> Path:
    path = (root / config["checkpoint_transfer"]["relative_path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise M0OnlineError("official checkpoint path escapes model root") from error
    if not path.is_file():
        raise M0OnlineError(f"official checkpoint does not exist: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_training_artifacts(
    action_path: Path,
    statistics_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise M0OnlineError(
            f"cannot read training report {report_path}: {error}"
        ) from error
    if not isinstance(report, dict) or not report.get("ok"):
        raise M0OnlineError("training report must describe a successful run")
    identity = report.get("model_identity")
    if identity is not None and identity != {
        "family": MODEL_FAMILY,
        "variant": MODEL_VARIANT,
        "name": MODEL_NAME,
    }:
        raise M0OnlineError(f"training report does not identify {MODEL_NAME}")
    action_sha256 = _sha256(action_path)
    statistics_sha256 = _sha256(statistics_path)
    if report.get("action_model_sha256") != action_sha256:
        raise M0OnlineError("action checkpoint SHA-256 disagrees with training report")
    if report.get("state_statistics_sha256") != statistics_sha256:
        raise M0OnlineError("state statistics SHA-256 disagrees with training report")
    config_relative = report.get("config_relative_path")
    config_sha256 = report.get("config_sha256")
    config_path = None
    if config_relative is not None:
        if not isinstance(config_relative, str) or not config_relative:
            raise M0OnlineError("training report config path is invalid")
        config_path = (report_path.parent / config_relative).resolve()
        try:
            config_path.relative_to(report_path.parent.resolve())
        except ValueError as error:
            raise M0OnlineError("training config escapes its artifact directory") from error
        if not config_path.is_file():
            raise M0OnlineError("training config does not exist")
        if config_sha256 is not None and _sha256(config_path) != config_sha256:
            raise M0OnlineError("training config SHA-256 disagrees with training report")
    elif config_sha256 is not None:
        raise M0OnlineError("training report config path is missing")
    return {
        "action_model_sha256": action_sha256,
        "state_statistics_sha256": statistics_sha256,
        "training_report_sha256": _sha256(report_path),
        "training_steps": report.get("max_steps"),
        "dataset_records": report.get("dataset_records"),
        "data_format": report.get("data_format"),
        "sources": report.get("sources"),
        "config_path": None if config_path is None else str(config_path),
        "config_sha256": config_sha256,
    }


def load_service(args: argparse.Namespace) -> tuple[M0PolicyService, Mapping[str, Any]]:
    import torch
    from safetensors.torch import load_file

    from conveyor_bench.conveyorvla.dit import M0DiTActionHead
    from conveyor_bench.conveyorvla.policy import (
        ConveyorVLAAL0Policy,
        ConveyorVLAAL0TemporalPolicy,
        Qwen3VLInterface,
        m0_dit_config,
        transfer_robocasa_policy_weights,
    )

    statistics_path = args.state_statistics.expanduser().resolve()
    statistics = load_state_statistics(statistics_path)
    action_path = args.action_checkpoint.expanduser().resolve()
    if not action_path.is_file():
        raise M0OnlineError(f"action checkpoint does not exist: {action_path}")
    training_report = (
        args.training_report.expanduser().resolve()
        if args.training_report is not None
        else action_path.with_name("training_report.json")
    )
    artifact_report = _verify_training_artifacts(
        action_path, statistics_path, training_report
    )
    temporal_history = artifact_report["data_format"] == "lerobot_v3"
    config = load_m0_mobile_config(args.config)
    if temporal_history:
        saved_config = artifact_report.get("config_path")
        config = (
            load_m0_mobile_config(saved_config)
            if saved_config is not None
            else build_temporal_policy_config(
                config,
                load_temporal_config(args.temporal_config),
            )
        )
        sources = artifact_report.get("sources")
        if (
            not isinstance(sources, list)
            or len(sources) != 1
            or not isinstance(sources[0], Mapping)
            or sources[0].get("history_offsets_model_ticks") != [-5, 0]
            or sources[0].get("history_span_s") != 0.2
            or config["data"].get("history_offsets_model_ticks") != [-5, 0]
            or config["data"].get("history_span_s") != 0.2
        ):
            raise M0OnlineError(
                "temporal deployment requires the PCT [-5, 0] / 0.20 s history contract"
            )
    normalizer = M0MobileNormalizer.from_config(config, statistics)
    root = resolve_model_root(config, args.model_root)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise M0OnlineError(f"{MODEL_NAME} online inference requires a CUDA device")
    torch.cuda.set_device(device)
    qwen = Qwen3VLInterface.from_local(
        root / config["vlm"]["relative_path"],
        checkpoint_vocab_size=config["vlm"]["checkpoint_vocab_size"],
        dtype=torch.bfloat16,
        attention_implementation=args.attention_implementation,
    )
    policy_class = (
        ConveyorVLAAL0TemporalPolicy
        if temporal_history
        else ConveyorVLAAL0Policy
    )
    policy_kwargs = {
        "repeated_diffusion_steps": config["training"][
            "repeated_diffusion_steps"
        ],
    }
    if temporal_history:
        policy_kwargs["temporal_history_span_s"] = config["data"][
            "history_span_s"
        ]
    policy = policy_class(
        qwen,
        M0DiTActionHead(m0_dit_config(config)),
        **policy_kwargs,
    )
    transfer = transfer_robocasa_policy_weights(policy, _checkpoint_path(config, root))
    policy.action_model.load_state_dict(load_file(action_path, device="cpu"), strict=True)
    policy.freeze_qwen()
    policy.qwen_vl_interface.to(device)
    policy.action_model.to(device)
    policy.eval()
    report = {
        "model_identity": {
            "family": MODEL_FAMILY,
            "variant": MODEL_VARIANT,
            "name": MODEL_NAME,
        },
        "loaded_qwen_tensors": transfer.loaded_qwen_tensors,
        "loaded_official_action_tensors": transfer.loaded_action_tensors,
        "reinitialized_tensors": len(transfer.reinitialized_keys),
        "fine_tuned_action_tensors": len(policy.action_model.state_dict()),
        "device": str(device),
        "temporal_history": temporal_history,
        "temporal_cache_contract": (
            {
                "history_offsets_model_ticks": [-5, 0],
                "history_span_s": 0.20,
                "cache": "previous_contiguous_5hz_request",
            }
            if temporal_history
            else None
        ),
        **artifact_report,
    }
    service = M0PolicyService(
        policy,
        normalizer,
        tuple(config["data"]["action_dimension_mask"]),
        tuple(config["data"]["image_size"]),
        torch,
        {
            key: artifact_report[key]
            for key in (
                "action_model_sha256",
                "state_statistics_sha256",
                "training_report_sha256",
                "training_steps",
                "dataset_records",
            )
        },
        temporal_history=temporal_history,
        temporal_history_span_s=(
            float(config["data"]["history_span_s"])
            if temporal_history
            else None
        ),
    )
    return service, report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        service, report = load_service(args)
        server = create_server(args.port, service)
    except (M0MobileError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"event": "startup_failed", "error": str(error)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "event": "ready",
                "bind": f"127.0.0.1:{server.server_port}",
                "model": report,
            },
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
