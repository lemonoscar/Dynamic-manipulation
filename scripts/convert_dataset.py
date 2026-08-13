#!/usr/bin/env python3
"""Convert successful ConveyorVLA AL0 temporal episodes to LeRobot v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conveyor_bench.conveyorvla.lerobot_v3 import (  # noqa: E402
    DEFAULT_LEROBOT_V3_CONFIG_PATH,
    materialize_lerobot_v3,
)
from conveyor_bench.conveyorvla.config import M0MobileError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode-root",
        action="append",
        type=Path,
        default=[],
        help="Successful episode containing the temporal export; repeatable.",
    )
    parser.add_argument(
        "--episode-list",
        action="append",
        type=Path,
        default=[],
        help="Text file with one successful episode root per line; repeatable.",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_LEROBOT_V3_CONFIG_PATH)
    parser.add_argument("--repo-id", help="Local LeRobot repository identity override.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Convert only the first N episodes for a smoke test.",
    )
    return parser


def _episode_roots(args: argparse.Namespace) -> tuple[Path, ...]:
    roots = [path.expanduser().resolve() for path in args.episode_root]
    for raw_list in args.episode_list:
        list_path = raw_list.expanduser().resolve()
        try:
            lines = list_path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise M0MobileError(f"cannot read episode list {list_path}: {error}") from error
        for line in lines:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value).expanduser()
            roots.append((list_path.parent / path).resolve() if not path.is_absolute() else path.resolve())
    roots = list(dict.fromkeys(roots))
    if not roots:
        raise M0MobileError("pass --episode-root and/or --episode-list")
    if args.max_episodes is not None:
        if args.max_episodes <= 0:
            raise M0MobileError("--max-episodes must be positive")
        roots = roots[: args.max_episodes]
    return tuple(roots)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = materialize_lerobot_v3(
            _episode_roots(args),
            args.output_root,
            config_path=args.config,
            repo_id=args.repo_id,
        )
    except (M0MobileError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, **report}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
