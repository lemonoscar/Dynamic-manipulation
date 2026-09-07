#!/usr/bin/env python3
"""Separate distinct source points, overlapping references and chunk isolation counts."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conveyor_bench.conveyorvla.formal_checkpoint import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    byep=defaultdict(list)
    for line in (a.audit_dir/'points.jsonl').open():
        r=json.loads(line);byep[r['episode_id']].append(r)
    idx={(ep,x['source_frame']):i for ep,rs in byep.items() for i,x in enumerate(rs)};unique={ch:set() for ch in ('arm','gripper')};classes={ch:Counter() for ch in unique};reasons={ch:Counter() for ch in unique};kept=Counter();n=0;isolated_channels=Counter();chunks=0
    for line in (a.audit_dir/'chunks.jsonl').open():
        c=json.loads(line);ep=c['episode_id'];i=idx[ep,c['query_frame']];chunks+=1
        for x in byep[ep][i+1:i+1+c['real_future_points']]:
            for ch in unique:unique[ch].add((ep,x['source_frame']))
        for ch,v in c['affected_source_frames'].items():isolated_channels[ch]+=bool(v)
        if not c['isolate']:
            n+=1
            for k in ('position_events','rate_events','gripper_events'):kept[k]+=c[k]
    for ch,points in unique.items():
        for ep,fr in points:
            v=byep[ep][idx[ep,fr]]['channels'][ch];classes[ch][v['class']]+=1;reasons[ch][v['reason']]+=1
    write_json(a.output,{'schema':'train-source-reference-summary-v1','audit_report_sha256':sha256(a.audit_dir/'report.json'),'points_sha256':sha256(a.audit_dir/'points.jsonl'),'chunks_sha256':sha256(a.audit_dir/'chunks.jsonl'),'mani_chunks':chunks,'distinct_source_points_referenced_by_mani_horizons':{ch:len(v) for ch,v in unique.items()},'referenced_distinct_source_point_classes':classes,'referenced_distinct_source_reasons':reasons,'isolated_chunks_by_channel_nonexclusive':isolated_channels,'conservative_retained_chunks':n,'retained_subset_saturation':{'counts':kept,'denominator':n*70,'rate':sum(kept.values())/(n*70) if n else None,'threshold':.005},'interpretation':'Unknown means insufficient provenance, not confirmed invalid. The conservative retained subset is a biased audit selection, not a released dataset; zero clipping does not show task coverage or successful replay. No new normalizer or training.'})
if __name__=='__main__':main()
