from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets"
V1_LOCK_PATH = ASSET_ROOT / "asset_lock.json"
V2_LOCK_PATH = ASSET_ROOT / "asset_lock_v2.json"
V2_ONLY_ASSETS = {
    "receptacles/remote_delivery_v2.json",
    "workcells/remote_delivery_v2/ASSET_MANIFEST.json",
}


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_lock(path: Path) -> dict[str, object]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_v2_lock_is_exact_v1_superset_plus_remote_manifests() -> None:
    v1 = _load_lock(V1_LOCK_PATH)
    v2 = _load_lock(V2_LOCK_PATH)

    assert v2["schema_version"] == "conveyor-bench-asset-lock-v1"
    assert v2["generated_for"] == "conveyor-bench-v2"
    v1_files = v1["files"]
    v2_files = v2["files"]
    assert isinstance(v1_files, dict)
    assert isinstance(v2_files, dict)
    assert set(v2_files) == set(v1_files) | V2_ONLY_ASSETS
    assert {name: v2_files[name] for name in v1_files} == v1_files


def test_v2_lock_paths_are_local_regular_files_with_exact_hashes() -> None:
    lock = _load_lock(V2_LOCK_PATH)
    files = lock["files"]
    assert isinstance(files, dict)
    resolved_root = ASSET_ROOT.resolve()

    for relative_name, expected_digest in files.items():
        assert isinstance(relative_name, str)
        assert isinstance(expected_digest, str)
        relative = PurePosixPath(relative_name)
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        path = (ASSET_ROOT / relative_name).resolve()
        assert path.is_relative_to(resolved_root)
        assert path.is_file()
        assert len(expected_digest) == 64
        assert _sha256(path) == expected_digest


def test_v2_lock_and_remote_manifests_contain_no_runtime_url() -> None:
    payloads = [V2_LOCK_PATH.read_text(encoding="utf-8")]
    payloads.extend(
        (ASSET_ROOT / name).read_text(encoding="utf-8")
        for name in sorted(V2_ONLY_ASSETS)
    )
    payload = "\n".join(payloads).lower()
    for marker in (
        "http://",
        "https://",
        "omniverse://",
        "omniverse-content",
        "s3://",
        "s3-us-",
    ):
        assert marker not in payload
