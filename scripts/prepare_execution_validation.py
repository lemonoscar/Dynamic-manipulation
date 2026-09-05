#!/usr/bin/env python3
"""Extract only validation source evidence for no-model execution diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
import copy
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.run_formal_closed_loop import bind_scene
from conveyor_bench.conveyorvla.formal_checkpoint import read_json, write_json, sha256, source_identity
from conveyor_bench.conveyorvla.execution_consistency import sampled_phase


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    dataset, source, output = (getattr(args,n).resolve() for n in ("dataset_root","source_root","output_dir"))
    if output.is_relative_to(ROOT):
        raise ValueError("validation evidence must remain outside the worktree")
    manifest = read_json(dataset / "manifest.json")
    if sha256(dataset / "val.jsonl") != manifest["records"]["val"]["sha256"]:
        raise ValueError("validation records hash differs")
    if sha256(source / "manifest.ndjson") != manifest["source_snapshot_manifest_sha256"]:
        raise ValueError("source snapshot identity differs")
    episodes = sorted({json.loads(x)["episode_id"] for x in (dataset / "val.jsonl").open()})
    output.mkdir(parents=True, exist_ok=False)
    scene = bind_scene(output)
    entries = {int(x["index"]):x for x in (json.loads(line) for line in (source/"manifest.ndjson").open())}
    groups = defaultdict(list)
    for episode in episodes:
        entry = entries[int(episode.rsplit("-",1)[1])]
        groups[entry["archive"]].append((episode,entry))
    hashes, audit = {}, []
    wanted_names = ("task.json","summary.json","samples.jsonl","frames.jsonl","events.jsonl")
    for archive, group in groups.items():
        path = (source/archive).resolve()
        if not path.is_relative_to(source):
            raise ValueError("unsafe source archive path")
        print(json.dumps({"event":"hashing_source_archive","archive":archive}),flush=True)
        digest = sha256(path)
        if digest != manifest["source_archive_sha256"].get(archive,manifest["source_archive_sha256"].get(path.name)):
            raise ValueError("archive checksum differs")
        wanted = {entry["member_path"]+"/"+name:(episode,name) for episode,entry in group for name in wanted_names}
        with tarfile.open(path,"r:") as tar:
            for member in tar:
                key = member.name.removeprefix("./")
                if key not in wanted:
                    continue
                episode,name = wanted.pop(key)
                if not member.isfile() or member.size > 64*1024*1024:
                    raise ValueError("invalid source evidence member")
                dest = output/"source"/episode/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(tar.extractfile(member).read())
                hashes[str(dest.relative_to(output))] = sha256(dest)
                if not wanted:
                    break
        if wanted:
            raise ValueError("source evidence members missing")
        for episode,entry in group:
            directory = output/"source"/episode
            samples = [json.loads(x) for x in (directory/"samples.jsonl").open()]
            frames = [json.loads(x) for x in (directory/"frames.jsonl").open()]
            phase = sampled_phase(samples,frames)
            if read_json(directory/"summary.json").get("success") is not True:
                raise ValueError("source episode not successful")
            task = copy.deepcopy(read_json(directory/"task.json"))
            for key in ("annotation_config","annotation_config_report","scene_asset_binding_runtime"):
                task.pop(key,None)
            task["scene_usd"] = str(scene)
            write_json(directory/"migration_task.json",task)
            hashes[str((directory/"migration_task.json").relative_to(output))] = sha256(directory/"migration_task.json")
            steps = [f["observation"]["step_index"] for f in frames]
            audit.append({"episode_id":episode,"seed":entry["seed"],"sample_count":len(samples),
                          "frame_count":len(frames),"frame_step_differences":dict(Counter(b-a for a,b in zip(steps,steps[1:]))),
                          "pick_samples":len(phase),"first_pick_timestamp_s":phase[0][0]["timestamp"],
                          "source_archive_sha256":digest})
        print(json.dumps({"event":"archive_prepared","episodes":len(group)}),flush=True)
    write_json(output/"manifest.json",{"schema":"execution-consistency-validation-v1","split":"val",
               "episode_ids":episodes,"first_pilot_episodes":episodes[:3],"source_identity":source_identity(ROOT),
               "dataset_manifest_sha256":sha256(dataset/"manifest.json"),"files":hashes,"audit":audit,
               "source_environment_replay":"unavailable_until_source_Sim6_runtime_is_supplied",
               "replay_contract":"isolated PICK, absolute sampled command held 10 ticks; offsets 0/1; both formal assistance profiles"})


if __name__ == "__main__":
    main()
