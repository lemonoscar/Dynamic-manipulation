#!/usr/bin/env python3
"""Validate raw-control-v2; derive immutable task/action views and quarantine report."""
import argparse,json,sys
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.staged_data import validate_family_splits,digest
from conveyor_bench.conveyorvla.collection_importer import load_episode
from conveyor_bench.conveyorvla.contracts.action import TIME_PROFILES


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',nargs='+',type=Path,required=True)
    p.add_argument('--split-manifest',type=Path,required=True,help='episode UUID -> train/validation/test JSON')
    p.add_argument('--profile',choices=TIME_PROFILES,default='causal_command_5hz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit-normalizer',action='store_true')
    p.add_argument('--development-smoke',action='store_true')
    p.add_argument('--max-rows-per-route',type=int,default=0)
    p.add_argument('--modalities',choices=('rgb','rgbd'),default='rgbd')
    p.add_argument('--query-stride-ticks',type=int,default=1,help='query cadence on the native 50Hz timeline; labels retain the selected profile')
    a=p.parse_args(argv)
    if a.output.exists():raise ValueError('output exists; never overwrite an existing release')
    if a.query_stride_ticks<1:raise ValueError('positive query stride required')
    episodes=[SimpleNamespace(manifest=json.loads((path/'manifest.json').read_text())) for path in a.episodes]
    splits=json.loads(a.split_manifest.read_text());validate_family_splits(episodes,splits)
    actions=[];tasks=[];quarantine=[];audits=[];source_files={}
    for path in a.episodes:
        ep=load_episode(path,modalities=a.modalities)
        audits.append(getattr(ep,'import_audit',{}))
        source_files.update({str(f.resolve()):digest(f) for f in ep.root.glob('*.json*')})
        for oid,(obs,raw) in ep.observations.items():
            if not raw.get('action_query',False):continue
            if a.query_stride_ticks>1:
                if 'control_tick' not in raw:raise ValueError('query stride requires explicit native control tick')
                if raw['control_tick']%a.query_stride_ticks:continue
            try:
                row=ep.action_view(oid,profile=a.profile)
                row['split']=splits[ep.manifest['episode_uuid']]
                row['episode_root']=str(ep.root.resolve())
                actions.append(row)
            except ValueError as error:
                quarantine.append({'episode_uuid':ep.manifest['episode_uuid'],'observation_id':oid,'reason':str(error)})
        for row in ep.task_view():
            row['split']=splits[ep.manifest['episode_uuid']]
            row['episode_root']=str(ep.root.resolve());tasks.append(row)
    eligible_action_rows=len(actions)
    if a.max_rows_per_route:
        if not a.development_smoke or a.max_rows_per_route<2:
            raise ValueError('bounded row selection requires development-smoke and at least two rows/route')
        selected=[]
        for route in ('NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE'):
            pool=[r for r in actions if r['route']==route]
            count=min(len(pool),a.max_rows_per_route)
            selected.extend(pool[round(i*(len(pool)-1)/max(count-1,1))] for i in range(count))
        actions=selected
    normalizer=None
    if a.fit_normalizer:
        from conveyor_bench.conveyorvla.staged_training import StagedNormalizer
        normalizer=StagedNormalizer.fit(r for r in actions if r['split']=='train')
    a.output.mkdir(parents=True)
    for name,rows in (('actions.jsonl',actions),('tasks.jsonl',tasks),('quarantine.jsonl',quarantine)):
        with (a.output/name).open('x') as f:
            for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    if normalizer:(a.output/'normalization.json').write_text(json.dumps(normalizer.payload,indent=2)+'\n')
    manifest={'schema':'staged-release-v2','time_profile':a.profile,'input_modalities':a.modalities,'query_stride_ticks':a.query_stride_ticks,'action_rows':len(actions),
        'task_rows':len(tasks),'eligible_action_rows_before_selection':eligible_action_rows,
        'purpose':'development_smoke_only' if a.development_smoke else 'derived_release',
        'not_formal_training_release':a.development_smoke,'max_rows_per_route':a.max_rows_per_route,'quarantined_chunks':len(quarantine),'synthetic':any(e.manifest.get('synthetic',False) for e in episodes),
        'import_audits':audits,
        'source_files':source_files,
        'split_manifest_sha256':digest(a.split_manifest),'files':{p.name:digest(p) for p in a.output.iterdir()},
        'normalizer_id':None if normalizer is None else normalizer.payload['normalizer_id']}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in {'source_files','files'}}))
    return 0 if actions else 2

if __name__=='__main__':raise SystemExit(main())
