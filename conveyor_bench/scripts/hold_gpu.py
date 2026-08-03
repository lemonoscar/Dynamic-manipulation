#!/usr/bin/env python3
"""Reserve one visible CUDA device for an explicitly authorized placeholder."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="starVLA-placeholder")
    parser.add_argument("--memory-mib", type=int, default=90000)
    parser.add_argument("--heartbeat-seconds", type=int, default=60)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.memory_mib <= 0 or args.heartbeat_seconds <= 0:
        raise SystemExit("memory and heartbeat values must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("the placeholder must see exactly one CUDA device")
    total_mib = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    if args.memory_mib >= total_mib:
        raise SystemExit(
            f"requested {args.memory_mib} MiB but visible GPU has {total_mib} MiB"
        )
    reservation = torch.empty(
        args.memory_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda:0",
    )
    reservation.zero_()
    stop = threading.Event()

    def request_stop(signum, _frame) -> None:
        print(json.dumps({"event": "signal", "signal": signum}), flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    identity = {
        "event": "ready",
        "label": args.label,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "physical_gpu": visible,
        "visible_device_name": torch.cuda.get_device_name(0),
        "reserved_mib": args.memory_mib,
    }
    print(json.dumps(identity, sort_keys=True), flush=True)
    while not stop.wait(args.heartbeat_seconds):
        identity["event"] = "heartbeat"
        identity["unix_time"] = int(time.time())
        identity["sentinel"] = int(reservation[0].item())
        print(json.dumps(identity, sort_keys=True), flush=True)
    del reservation
    torch.cuda.empty_cache()
    print(json.dumps({"event": "stopped", "pid": os.getpid()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
