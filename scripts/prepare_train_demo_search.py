from pathlib import Path
import json,collections,hashlib,subprocess,copy
from PIL import Image
b=Path('/diff/wallx_workspace/dzb').resolve();repo=b/'ConveyorVLA-demo-search-20260909';release=b/'integration_runs/rgb_full_episode_20260908_v1/release';old=b/'integration_runs/train_seed_fit_20260908_timer60_v1';out=b/'integration_runs/train_demo_search_20260909_v1';head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert repo.resolve().is_relative_to(b) and out.parent.resolve().is_relative_to(b)
assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==head
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
actions=[json.loads(x) for x in (release/'actions.jsonl').read_text().splitlines()];trans=[json.loads(x) for x in (release/'transitions.jsonl').read_text().splitlines()];norm=json.loads((release/'normalization.json').read_text());groups=collections.defaultdict(list)
for a in actions:
 if a['split']=='train':groups[a['task_family_id']].append(a)
candidates=json.loads((repo/'docs/demo_search_20260909/candidate_audit.json').read_text());selected=[(e['family'],groups[e['family']]) for e in candidates['selected_candidates']];excluded=candidates['rejected']
assert len(selected)==24 and len(set(x[0] for x in selected))==24
manifest={'schema':'train-demo-search-timer60-v2','source_head':head,'runtime_code_origin':head,'prior_runtime_origin':'920ccc5b74e097ddf83dfc9ee0c1f039098684ef','checkpoint_step':1700,'checkpoint_sha256':'474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0','normalizer_id':norm['normalizer_id'],'selection_rule':candidates['ranking'],'candidate_audit_sha256':sha(repo/'docs/demo_search_20260909/candidate_audit.json'),'excluded':excluded,'normal_termination_per_attempt':'60 simulation seconds; loading excluded; no wall-clock or diagnostic-gate exit','max_attempts':24,'budget_started_at':'2026-09-09T00:08:57+08:00','deadline_at':'2026-09-09T06:08:57+08:00','stop_new_attempts_on_qualified_candidate':True,'retries':0,'resources':{'inference_gpu':2,'simulation_gpu':3,'parallel_simulation_processes':1,'model_service_reused':True,'expected_wall_minutes':280},'training':False,'depth':False,'rtc':False,'release_sha256':sha(release/'manifest.json'),'runs':[]}
prepared=[]
for family,rows in selected:
 roots={x['episode_root'] for x in rows};assert len(roots)==1;src=Path(roots.pop()).resolve();assert src.is_relative_to(b)
 candidate=next(e for e in candidates['selected_candidates'] if e['family']==family);assert src==Path(candidate['episode_root']).resolve() and sha(src/'task.json')==candidate['task_sha256'] and sha(src/'manifest.json')==candidate['manifest_sha256']
 seed=int(family.split('_')[-1]);m=json.loads((src/'manifest.json').read_text());assert m['task_family_id']==family and family in norm['families']
 pool=rows+[x for x in trans if x['task_family_id']==family];assert all(x['split']=='train' for x in pool)
 image_hashes={}
 for row in pool:
  for name in row.get('input',row)['images']:
   p=(src/name).resolve();assert p.is_relative_to(src)
   if name not in image_hashes:
    with Image.open(p) as im:im.convert('RGB').load()
    image_hashes[name]=sha(p)
 controls=[json.loads(x) for x in (src/'control_effective_50hz.jsonl').read_text().splitlines()];at=[x for x in controls if x['control_tick']==80];pa=[x for x in controls if x['control_tick']==79];assert len(at)==len(pa)==1
 cur,parent=at[0],pa[0];assert parent['verified'] and parent['reset_generation']==cur['reset_generation'];assert all(parent['channels'][x]['valid'] for x in ('arm','gripper'))
 obs=[json.loads(x) for x in (src/'observations.jsonl').read_text().splitlines()];obs=[x for x in obs if x['observation_id']==cur['observation_ref']];samples=[json.loads(x) for x in (src/'samples.jsonl').read_text().splitlines()];sample=[x for x in samples if x.get('effective_command',{}).get('command_id')==cur['command_id']];assert len(obs)==len(sample)==1
 assert sample[0]['camera_capture_step']==sample[0]['simulation_step']==80
 task=json.loads((src/'task.json').read_text());removed={k:task.pop(k) for k in ('annotation_config','annotation_config_report','scene_asset_binding_runtime') if k in task};oldscene=task['scene_usd'];task['scene_usd']=str(b/'integration_runs/train_seed_fit_20260908_v1/physical_protocol/scene_bound.usda')
 name='seed_'+str(seed);d=out/name;cmd=json.loads((old/'physical_timer60/command.json').read_text());argv=[x.replace('/diff/wallx_workspace/dzb/ConveyorVLA-train-seed-fit-20260908',str(repo)) for x in cmd['argv']];cmd['argv']=argv;cmd['cwd']=cmd['cwd'].replace('/diff/wallx_workspace/dzb/ConveyorVLA-train-seed-fit-20260908',str(repo))
 instructions={x.get('original_instruction',x.get('input',{}).get('original_instruction')) for x in pool};assert len(instructions)==1 and None not in instructions;model_instruction=instructions.pop()
 argv[argv.index('--'):argv.index('--')]=['--model-instruction',model_instruction]
 for key,value in {'--source-episode':str(src),'--task-json':str(d/'task.json'),'--output-dir':str(d/'runtime'),'--seed':str(seed),'--query-tick':'80'}.items():argv[argv.index(key)+1]=value
 assert '--timer-only' in argv
 cmd['source_head']=head
 entry={'name':name,'seed':seed,'family':family,'source_episode':str(src),'source_uuid':m['episode_uuid'],'action_rows':len(rows),'transition_rows':len(pool)-len(rows),'routes':dict(collections.Counter(x['route'] for x in rows)),'source_query_tick':80,'source_time_s':1.6,'source_parent_command_id':parent['command_id'],'original_instruction':model_instruction,'source_task_instruction':task.get('instruction'),'source_task_sha256':sha(src/'task.json'),'source_files':{n:sha(src/n) for n in ('manifest.json','task.json','control_effective_50hz.jsonl','observations.jsonl','samples.jsonl')},'image_count':len(image_hashes),'image_hashes':image_hashes,'task_migration':{'removed_metadata_keys':list(removed),'old_scene':oldscene,'scene':task['scene_usd'],'scene_sha256':sha(task['scene_usd']),'other_task_fields_unchanged':True},'source_phases':dict(collections.Counter(x['pipeline_state'] for x in samples))}
 manifest['runs'].append(entry);prepared.append((d,task,cmd))
manifest['protocol_version']='full-episode-rgb-diagnostic-v2'
manifest['launcher_sha256']=sha(repo/'scripts/launch_train_demo_search.py')
manifest['audit_sha256']=sha(repo/'scripts/audit_demo_candidate.py')
manifest['contract_sha256']=sha(repo/'docs/demo_search_20260909/SEARCH_CONTRACT.md')
out.mkdir(exist_ok=False)
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));digest=sha(out/'manifest.json')
for d,task,cmd in prepared:
 d.mkdir();(d/'task.json').write_text(json.dumps(task,ensure_ascii=False,indent=2));cmd['manifest_sha256']=digest;cmd['task_sha256']=sha(d/'task.json');(d/'command.json').write_text(json.dumps(cmd,indent=2))
print(json.dumps({'output':str(out),'source_head':head,'selected':[{'seed':x['seed'],'routes':x['routes'],'image_count':x['image_count']} for x in manifest['runs']],'excluded':excluded},ensure_ascii=False))
