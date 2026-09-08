#!/usr/bin/env python3
"""Bind complete independent cache shards without copying their tensor files."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.staged_data import digest,jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--shards',type=int,required=True)
    a=p.parse_args();output=a.cache_root/'manifest.json'
    if output.exists():raise ValueError('merged manifest already exists')
    entries={};identity=None
    for index in range(a.shards):
        directory=a.cache_root/f'shard-{index}'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['shard_count']!=a.shards or manifest['shard_index']!=index:
            raise ValueError('wrong shard identity')
        current={k:manifest[k] for k in ('schema','release_sha256','encoder_model_id','input_modalities','splits')}
        if identity is not None and current!=identity:raise ValueError('cache shard contract mismatch')
        identity=current
        for key,entry in manifest['entries'].items():
            if key in entries:raise ValueError('duplicate cached query')
            path=(directory/entry['file']).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file():raise ValueError('missing or escaping cache')
            entries[key]={**entry,'file':str(path.relative_to(a.cache_root.resolve()))}
    if identity is None or identity['release_sha256']!=digest(a.release/'manifest.json'):
        raise ValueError('cache/release identity mismatch')
    expected={r['episode_uuid']+':'+r['observation_id'] for r in jsonl(a.release/'actions.jsonl') if r['split'] in identity['splits']}
    if set(entries)!=expected:raise ValueError('incomplete or foreign query cache')
    with output.open('x') as stream:json.dump({**identity,'entries':entries,'merged_shards':a.shards},stream,indent=2)
    print(json.dumps({'cached_queries':len(entries),'release_sha256':identity['release_sha256']}))

if __name__=='__main__':main()
