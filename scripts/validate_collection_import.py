#!/usr/bin/env python3
"""Real collection import, RGB/PNG decode and cross-machine golden view check."""
import argparse,collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from conveyor_bench.conveyorvla.collection_importer import CollectionEpisode,IMPORTER_VERSION
from conveyor_bench.conveyorvla.staged_data import digest
from conveyor_bench.conveyorvla.contracts.action import decode_mani,encode_mani


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--golden',type=Path)
    a=p.parse_args()
    if a.output.exists():raise ValueError('new validation output required')
    ep=CollectionEpisode(a.episode);rows=[];quarantine=[]
    for oid,(_,raw) in ep.observations.items():
        if not raw['action_query']:continue
        try:rows.append(ep.action_view(oid,profile='causal_command_5hz'))
        except ValueError as e:quarantine.append({'observation_id':oid,'reason':str(e)})
    assets=set();depth_samples=0
    for _,raw in ep.observations.values():
        assets.update(raw['images'])
        for d in raw.get('depth',[]):
            assets.add(d['path']);depth_samples+=1
    hashes={}
    for name in sorted(assets):
        path=(a.episode/name).resolve()
        if not path.is_relative_to(a.episode.resolve()):raise ValueError('asset escape')
        with Image.open(path) as im:
            value=np.asarray(im)
            if name.startswith('depth/') and (value.dtype!=np.uint16 or value.ndim!=2):
                raise ValueError('depth encoding mismatch')
            im.load()
        hashes[name]=digest(path)
    golden=[];bindings=[]
    for route in ('NAV_TO_SOURCE','PICK','NAV_TO_TARGET','PLACE'):
        candidates=[r for r in rows if r['route']==route]
        if not candidates:continue
        row=candidates[len(candidates)//2]
        if route in ('PICK','PLACE'):
            physical=decode_mani(row['actions'],row['mani_state'][:6])
            np.testing.assert_allclose(encode_mani(physical,row['mani_state'][:6]),row['actions'],atol=1e-15)
            cid=row['source_command_ids'][0];c=next(c for c in ep.controls if c['resolved']['command_id']==cid)
            assert physical[0,6]==c['source_channels']['gripper']['target'][0]/.04
        golden.append(row)
        control=next(c for c in ep.controls if c['observation_ref']==row['observation_id'])
        raw=ep.observations[row['observation_id']][1]
        depths=[]
        for d in row['depth']:
            with Image.open(a.episode/d['path']) as im:depth=np.asarray(im)
            pixels=[(0,0),(depth.shape[0]//2,depth.shape[1]//2),(depth.shape[0]-1,depth.shape[1]-1)]
            depths.append({'camera':d['camera'],'path':d['path'],'sha256':digest(a.episode/d['path']),
                'invalid_pixels':int((depth==0).sum()),'pixel_samples':[{'v':v,'u':u,'mm':int(depth[v,u]),
                    'm':float(depth[v,u])*.001,'valid':bool(depth[v,u]>0)} for v,u in pixels],
                'intrinsics':d['intrinsics'],'camera_to_base':d['camera_to_base'],
                'world_base':d['world_base'],'world_camera':d['world_camera'],'pixel_offset':d['pixel_offset']})
        bindings.append({'observation_id':row['observation_id'],'source_command_id':control['resolved']['command_id'],
            'source_channels':control['source_channels'],'normalized_parent_command_ids':control['resolved']['parent_command_ids'],
            'named_q6':list(ep.observations[row['observation_id']][0].q),
            'named_dq6':list(ep.observations[row['observation_id']][0].dq),
            'measured_finger_m':raw['measured_finger_m'],'measured_gripper_unclipped_fraction':raw['measured_gripper_unclipped_fraction'],
            'images':row['images'],'image_times_s':row['image_times_s'],'depth':depths})
    golden_payload={'schema':'collection-import-golden-v1','importer':IMPORTER_VERSION,
        'episode_uuid':ep.manifest['episode_uuid'],'source_manifest_sha256':ep.manifest['source_manifest_sha256'],
        'views':golden,'raw_bindings':bindings}
    if a.golden and json.loads(a.golden.read_text())!=golden_payload:raise ValueError('cross-machine golden mismatch')
    a.output.mkdir(parents=True)
    (a.output/'golden.json').write_text(json.dumps(golden_payload,sort_keys=True,indent=2)+'\n')
    report={'schema':'collection-import-validation-v1','episode':str(a.episode.resolve()),
        'source_commit':ep.manifest['source_commit'],'import_audit':ep.import_audit,
        'action_rows':len(rows),'route_counts':dict(collections.Counter(r['route'] for r in rows)),
        'quarantine_counts':dict(collections.Counter(r['reason'] for r in quarantine)),
        'decoded_assets':len(assets),'depth_records':depth_samples,'planner_rows':0,
        'golden_sha256':digest(a.output/'golden.json'),'asset_hashes_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        'cross_machine_golden_compared':bool(a.golden),'physical_capability_evidence':False}
    for name,data in [('report.json',report),('quarantine.json',quarantine),('asset_hashes.json',hashes)]:
        (a.output/name).write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='import_audit'}))

if __name__=='__main__':main()
