"""Finite, serial step1700 demo search; physics always closes on its 60s clock."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback
import urllib.request
from audit_demo_candidate import audit

BASE=Path('/diff/wallx_workspace/dzb').resolve()
REPO=BASE/'ConveyorVLA-demo-search-20260909'
ROOT=BASE/'integration_runs/train_demo_search_20260909_v1'
PRIOR=BASE/'integration_runs/train_seed_fit_20260908_multiseed60_v1'

def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def gpu_pids(g):
    return subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def env_for(root,gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    for key,child in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','XDG_CONFIG_HOME':'config','XDG_DATA_HOME':'data','HF_HOME':'hf','TORCH_HOME':'torch','TRITON_CACHE_DIR':'triton','MPLCONFIGDIR':'mpl','WANDB_DIR':'wandb','CUDA_CACHE_PATH':'cuda_cache'}.items():
        p=root/child;p.mkdir(exist_ok=True);e[key]=str(p)
    return e


def main():
    assert REPO.resolve().is_relative_to(BASE) and ROOT.resolve().is_relative_to(BASE)
    m=json.loads((ROOT/'manifest.json').read_text());head=m['source_head']
    assert len(m['runs'])==m['max_attempts']==24 and m['retries']==0
    assert sha(Path(__file__))==m['launcher_sha256'] and sha(REPO/'scripts/audit_demo_candidate.py')==m['audit_sha256']
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==head
    assert not subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip()
    deadline=datetime.datetime.fromisoformat(m['deadline_at']).timestamp()
    for e in m['runs']:
        d=ROOT/e['name'];c=json.loads((d/'command.json').read_text())
        assert not (d/'attempt.json').exists()
        assert c['manifest_sha256']==sha(ROOT/'manifest.json') and c['task_sha256']==sha(d/'task.json')
        assert '--timer-only' in c['argv'] and c['source_head']==head
        for name,digest in e['source_files'].items():assert sha(Path(e['source_episode'])/name)==digest
    state={'status':'waiting_for_prior_batch','pid':os.getpid(),'started_at':now(),'source_head':head,'closed':0,'attempted':0,'max_attempts':24,'deadline_at':m['deadline_at']}
    write(ROOT/'search_status.json',state)
    while not (PRIOR/'service_stop.json').exists():
        if time.time()>=deadline or (ROOT/'STOP_AFTER_CURRENT').exists():
            state.update(status='complete',reason='budget_or_stop_before_first_attempt',ended_at=now());write(ROOT/'search_complete.json',state);write(ROOT/'search_status.json',state);return
        time.sleep(10)
    if not (PRIOR/'batch_complete.json').exists():raise RuntimeError('prior batch failed; inspect retained artifacts before claiming clean handoff')
    for g in (2,3):assert not gpu_pids(g),f'GPU{g} unavailable; no foreign process will be altered'
    probe=socket.socket();assert probe.connect_ex(('127.0.0.1',18170))!=0;probe.close()
    if time.time()>=deadline or (ROOT/'STOP_AFTER_CURRENT').exists():
        state.update(status='complete',reason='budget_or_stop_before_model_load',ended_at=now());write(ROOT/'search_complete.json',state);write(ROOT/'search_status.json',state);return
    cmd=[str(REPO/'artifacts/.conda-envs/conveyorvla-al0-lerobot044/bin/python'),str(REPO/'scripts/serve_full_episode_diagnostic.py'),'--checkpoint',str(BASE/'integration_runs/rgb_full_episode_20260908_v1/trained/best.pt'),'--legacy-checkpoint',str(BASE/'training_runs/conveyorvla-abot-m0-liangzhunew500-5hz-formal-20260905-r1/checkpoints/step_002414'),'--expected-sha256',m['checkpoint_sha256'],'--identity-output',str(ROOT/'service_identity.json')]
    with (ROOT/'service.log').open('x') as log:
        service=subprocess.Popen(cmd,cwd=REPO,env=env_for(ROOT,2),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(ROOT/'service_launch.json',{'pid':service.pid,'pgid':service.pid,'command':cmd,'gpu':2,'gpu_uuid':'GPU-0b4f9aa0-7c5b-0aa1-3652-51b7ec9fc8b7','started_at':now(),'source_head':head})
        try:
            started=time.monotonic()
            while True:
                if service.poll() is not None:raise RuntimeError('model initialization failed')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18170/health',timeout=2) as resp:health=json.load(resp)
                    break
                except (OSError,ValueError):
                    if time.monotonic()-started>300:raise RuntimeError('model preflight unavailable; no simulation clock started')
                    time.sleep(1)
            assert health['strict_load'] and health['global_step']==1700 and health['source_head']==head and health['protocol_version']=='full-episode-rgb-diagnostic-v2'
            write(ROOT/'service_health.json',health)
            reason='attempt_budget_exhausted'
            for e in m['runs']:
                if time.time()>=deadline:reason='wall_budget_exhausted';break
                if (ROOT/'STOP_AFTER_CURRENT').exists():reason='stop_after_current_requested';break
                assert service.poll() is None,'model service exited'
                d=ROOT/e['name'];c=json.loads((d/'command.json').read_text())
                assert c['task_sha256']==sha(d/'task.json') and c['manifest_sha256']==sha(ROOT/'manifest.json')
                assert not gpu_pids(3),'GPU3 is in use; never alter another process'
                env=env_for(d,3);assets=REPO/'artifacts/assets/conveyorvla-v3/liangzhu'
                env.update(LIANGZHU_VISUAL_USDZ=str(assets/'usdz/liangzhu.usdz'),LIANGZHU_COLLISION_USD=str(assets/'usd/liangzhu_collision.usda'))
                with (d/'process.log').open('x') as physical_log:
                    if time.time()>=deadline:reason='wall_budget_exhausted';break
                    if (ROOT/'STOP_AFTER_CURRENT').exists():reason='stop_after_current_requested';break
                    p=subprocess.Popen(c['argv'],cwd=c['cwd'],env=env,stdout=physical_log,stderr=subprocess.STDOUT,start_new_session=True)
                    result={'seed':e['seed'],'pid':p.pid,'pgid':p.pid,'started_at':now(),'status':'running','gpu':3,'gpu_uuid':'GPU-75c5e401-ba99-d869-5237-77bde2370458','simulation_clock_limit_s':60,'wall_clock_termination':False}
                    write(d/'attempt.json',result);state.update(status='running',seed=e['seed'],attempted=state['attempted']+1,physical_pid=p.pid,updated_at=now());write(ROOT/'search_status.json',state)
                    started=time.monotonic();code=p.wait()
                    result.update(status='exited',returncode=code,wall_s=time.monotonic()-started,ended_at=now());write(d/'attempt.json',result)
                try:verdict=audit(d)
                except Exception:verdict={'candidate_pending_video_review':False,'audit_error':traceback.format_exc(),'run':str(d),'seed':e['seed']}
                write(d/'demo_audit.json',verdict)
                state.update(closed=state['closed']+1,last_audit=str(d/'demo_audit.json'),updated_at=now());write(ROOT/'search_status.json',state)
                print(json.dumps({'seed':e['seed'],'closed':state['closed'],'candidate':verdict['candidate_pending_video_review'],'wall_s':result['wall_s']}),flush=True)
                if verdict['candidate_pending_video_review']:
                    reason='candidate_pending_video_review';write(ROOT/'candidate_found.json',verdict);break
            state.update(status='complete',reason=reason,ended_at=now());write(ROOT/'search_complete.json',state);write(ROOT/'search_status.json',state)
        finally:
            if service.poll() is None:
                service.terminate()
                try:service.wait(timeout=15)
                except subprocess.TimeoutExpired:service.kill();service.wait()
            write(ROOT/'service_stop.json',{'pid':service.pid,'ended_at':now(),'returncode':service.returncode,'reason':'only this search model service closed'})


if __name__=='__main__':
    try:main()
    except BaseException:
        write(ROOT/'search_failure.json',{'at':now(),'traceback':traceback.format_exc()});raise
