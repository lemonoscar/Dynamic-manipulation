#!/usr/bin/env python3
"""One real RGB request over an authorized loopback tunnel; same-client-clock RTT."""
import argparse,base64,json,sys,time,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.collection_importer import CollectionEpisode


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True)
    p.add_argument('--query-tick',type=int,default=220);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',help='optional http://127.0.0.1:PORT tunnel; omitted prepares request only')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    ep=CollectionEpisode(a.episode);uuid=ep.manifest['episode_uuid'];oid=f'{uuid}:r1:t{a.query_tick}:pre'
    row=ep.action_view(oid,profile='legacy_future_5hz');task=json.loads((a.episode/'task.json').read_text(encoding='utf-8'))
    images=[base64.b64encode((a.episode/name).read_bytes()).decode() for name in row['images']]
    request={'protocol_version':'conveyorvla-conditioned-pick-rtc-diagnostic/v1','request_id':f'cross-machine-probe-{a.query_tick}',
        'episode_id':uuid,'sequence_id':0,'instruction':task['base_instruction'],
        'head_images':images[:2],'wrist_images':images[2:],'joint_position':row['mani_state'][:6],
        'joint_velocity':row['mani_state'][6:12],'gripper_open_fraction':row['mani_state'][12],
        'observation_id':oid,'query_time_s':row['query_time_s'],'image_times_s':row['image_times_s'],
        'active_task_id':row['active_task_id'],'active_task_epoch':row['active_task_epoch'],
        'time_profile':'legacy_future_5hz','rtc_context':None}
    (a.output/'request.json').write_text(json.dumps(request)+'\n')
    if not a.endpoint:return 0
    url=urllib.parse.urlparse(a.endpoint)
    if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost','::1'):raise ValueError('authorized loopback tunnel required')
    data=json.dumps(request).encode();started=time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(a.endpoint.rstrip('/')+'/infer',data=data,headers={'Content-Type':'application/json'}),timeout=120) as r:response=json.load(r)
    elapsed=time.perf_counter()-started
    (a.output/'response.json').write_text(json.dumps(response,indent=2)+'\n')
    report={'client_roundtrip_s':elapsed,'clock':'single_client_perf_counter',
            'service_elapsed_s':response['response']['service_elapsed_s'],'physical_execution':False,
            'source_manifest_sha256':ep.manifest['source_manifest_sha256'],'observation_id':oid}
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
