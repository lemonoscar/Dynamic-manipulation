#!/usr/bin/env python3
"""Audit local M0-Mobile artifacts, checkpoint shapes, and one exported sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.m0_mobile import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    M0MobileError,
    audit_model_artifacts,
    iter_m0_mobile_samples,
    load_m0_mobile_config,
    resolve_model_root,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--check-checkpoint-shapes", action="store_true")
    parser.add_argument("--check-qwen-processor", action="store_true")
    parser.add_argument("--sample-jsonl", type=Path)
    parser.add_argument("--episode-root", type=Path)
    return parser


def _artifact_path(config: dict, root: Path, artifact_id: str) -> Path:
    for artifact in config["artifacts"]:
        if artifact["id"] == artifact_id:
            files = artifact["files"]
            if len(files) != 1:
                raise M0MobileError(
                    f"artifact {artifact_id!r} must contain exactly one file"
                )
            return root / files[0]["path"]
    raise M0MobileError(f"unknown artifact_id: {artifact_id}")


def _checkpoint_shapes(config: dict, root: Path) -> dict:
    try:
        import torch
    except ImportError as error:
        raise M0MobileError("PyTorch is required for checkpoint shape checks") from error
    transfer = config["checkpoint_transfer"]
    checkpoint = _artifact_path(config, root, transfer["artifact_id"])
    state = torch.load(
        checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    expected = transfer["source_shapes"]
    observed: dict[str, list[int]] = {}
    for key, shape in expected.items():
        if key not in state:
            raise M0MobileError(f"checkpoint is missing required tensor: {key}")
        actual = list(state[key].shape)
        if actual != shape:
            raise M0MobileError(
                f"checkpoint shape mismatch for {key}: expected {shape}, got {actual}"
            )
        observed[key] = actual
    prefixes = sorted({key.split(".", 1)[0] for key in state})
    if prefixes != ["action_model", "qwen_vl_interface"]:
        raise M0MobileError(f"unexpected checkpoint prefixes: {prefixes}")
    from conveyor_bench.m0_dit import M0DiTActionHead
    from conveyor_bench.m0_policy import m0_dit_config

    with torch.device("meta"):
        target_action = M0DiTActionHead(m0_dit_config(config)).state_dict()
    source_action = {
        key.removeprefix("action_model."): value
        for key, value in state.items()
        if key.startswith("action_model.")
    }
    if set(source_action) != set(target_action):
        unexpected = sorted(set(source_action) - set(target_action))
        missing = sorted(set(target_action) - set(source_action))
        raise M0MobileError(
            f"action checkpoint key mismatch: unexpected={unexpected}, missing={missing}"
        )
    shape_mismatches = sorted(
        key
        for key, target_value in target_action.items()
        if source_action[key].shape != target_value.shape
    )
    expected_mismatches = sorted(transfer["shape_mismatch_action_keys"])
    if shape_mismatches != expected_mismatches:
        raise M0MobileError(
            "action checkpoint shape mismatch set differs from migration contract: "
            f"expected={expected_mismatches}, got={shape_mismatches}"
        )
    try:
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise M0MobileError(
            "Transformers is required for Qwen checkpoint structure checks"
        ) from error
    qwen_path = root / config["vlm"]["relative_path"]
    qwen_config = AutoConfig.from_pretrained(qwen_path, local_files_only=True)
    with torch.device("meta"):
        target_qwen_model = Qwen3VLForConditionalGeneration(qwen_config)
        target_qwen_model.resize_token_embeddings(
            config["vlm"]["checkpoint_vocab_size"],
            mean_resizing=False,
        )
    target_qwen = target_qwen_model.state_dict()
    source_qwen = {
        key.removeprefix("qwen_vl_interface.model."): value
        for key, value in state.items()
        if key.startswith("qwen_vl_interface.model.")
    }
    if set(source_qwen) != set(target_qwen):
        unexpected = sorted(set(source_qwen) - set(target_qwen))
        missing = sorted(set(target_qwen) - set(source_qwen))
        raise M0MobileError(
            f"Qwen checkpoint key mismatch: unexpected={unexpected}, missing={missing}"
        )
    qwen_shape_mismatches = sorted(
        key
        for key, target_value in target_qwen.items()
        if source_qwen[key].shape != target_value.shape
    )
    if qwen_shape_mismatches:
        raise M0MobileError(
            f"Qwen checkpoint shape mismatches: {qwen_shape_mismatches}"
        )
    return {
        "tensor_count": len(state),
        "prefixes": prefixes,
        "required_shapes": observed,
        "action_tensor_count": len(source_action),
        "action_key_set_exact": True,
        "action_shape_mismatches": shape_mismatches,
        "qwen_tensor_count": len(source_qwen),
        "qwen_key_set_exact": True,
        "qwen_shape_mismatches": qwen_shape_mismatches,
    }


def _qwen_processor(config: dict, root: Path) -> dict:
    try:
        from transformers import AutoConfig, AutoProcessor
    except ImportError as error:
        raise M0MobileError(
            "Transformers is required for Qwen processor checks"
        ) from error
    path = root / config["vlm"]["relative_path"]
    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    hidden_size = model_config.text_config.hidden_size
    vocab_size = model_config.text_config.vocab_size
    if hidden_size != config["vlm"]["hidden_size"]:
        raise M0MobileError("Qwen hidden size disagrees with M0-Mobile config")
    if vocab_size != config["vlm"]["base_vocab_size"]:
        raise M0MobileError("Qwen vocabulary size disagrees with M0-Mobile config")
    tokenizer_size = len(processor.tokenizer)
    if tokenizer_size != config["vlm"]["processor_tokenizer_size"]:
        raise M0MobileError("Qwen tokenizer size disagrees with M0-Mobile config")
    try:
        from PIL import Image
    except ImportError as error:
        raise M0MobileError("Pillow is required for Qwen image checks") from error
    images = [Image.new("RGB", (224, 224), color) for color in ("red", "blue")]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image} for image in images
            ]
            + [
                {
                    "type": "text",
                    "text": "Pick the moving part and place it in the sorting tray.",
                }
            ],
        }
    ]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    )
    required_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not required_keys <= set(encoded):
        raise M0MobileError("Qwen processor omitted a required multimodal tensor")
    if encoded["image_grid_thw"].shape != (2, 3):
        raise M0MobileError("Qwen processor did not preserve the two camera images")
    return {
        "model_type": model_config.model_type,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "tokenizer_size": tokenizer_size,
        "two_image_input_shapes": {
            key: list(encoded[key].shape) for key in sorted(required_keys)
        },
    }


def _sample(config: dict, path: Path, episode_root: Path | None) -> dict:
    if episode_root is None:
        if path.parent.name != "exports":
            raise M0MobileError(
                "pass --episode-root when sample JSONL is not inside episode/exports"
            )
        episode_root = path.parent.parent
    sample = next(iter_m0_mobile_samples(path, episode_root, config), None)
    if sample is None:
        raise M0MobileError(f"sample JSONL is empty: {path}")
    return {
        "sample_id": sample.sample_id,
        "camera_files": [str(item) for item in sample.image_paths],
        "state_shape": [1, len(sample.state)],
        "action_shape": [len(sample.actions), len(sample.actions[0])],
        "action_mask": list(sample.action_mask),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_m0_mobile_config(args.config)
        root = resolve_model_root(config, args.model_root)
        artifacts = audit_model_artifacts(
            config,
            root,
            verify_hashes=args.verify_hashes,
        )
        report = {
            "ok": True,
            "config": str(args.config.resolve()),
            "model_root": str(root),
            "artifact_files": len(artifacts),
            "artifact_bytes": sum(item.size for item in artifacts),
            "hashes_verified": args.verify_hashes,
        }
        if args.check_checkpoint_shapes:
            report["checkpoint"] = _checkpoint_shapes(config, root)
        if args.check_qwen_processor:
            report["qwen"] = _qwen_processor(config, root)
        if args.sample_jsonl is not None:
            report["sample"] = _sample(
                config,
                args.sample_jsonl,
                args.episode_root,
            )
    except (M0MobileError, OSError, RuntimeError, StopIteration) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
