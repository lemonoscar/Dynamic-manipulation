"""Read-only analysis of completed recorded-observation fit; CPU plotting only."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'results'
read = lambda name: json.loads((DATA / name).read_text())
rows = [json.loads(line) for line in (DATA/'train_predictions.jsonl').read_text().splitlines()]
transitions = [json.loads(line) for line in (DATA/'train_transitions.jsonl').read_text().splitlines()]
report, selection = read('train_report.json'), read('selection.json')
for name, key in [('train_predictions.jsonl','predictions_sha256'), ('train_transitions.jsonl','transitions_sha256')]:
    assert hashlib.sha256((DATA/name).read_bytes()).hexdigest() == report[key]
assert report['status'] == 'complete' and len(rows) == selection['action_rows'] == 96
assert len(transitions) == selection['transition_rows'] == 6
assert {r['diffusion_seed'] for r in rows} == {17}
assert {r['family'] for r in rows} == {'liangzhu_seed_16100000'}
assert report['protocol']['identity']['step'] == 1700
assert report['protocol']['identity']['checkpoint_sha256'] == '474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0'

routes = ['NAV_TO_SOURCE', 'PICK', 'NAV_TO_TARGET', 'PLACE']
windows = ['first', 'executed_prefix', 'full_valid']
result = dict(schema='train-family-fit-independent-analysis-v1', family='liangzhu_seed_16100000',
              model_step=1700, action_queries=len(rows), transition_queries=len(transitions),
              artifact_hashes_verified=True, recorded_observations=True, autonomous_physics=False,
              weighting='query-equal across valid-prefix means; point-micro separately identified', routes={})

for route in routes:
    pool = [r for r in rows if r['route'] == route]
    mani = route in {'PICK','PLACE'}
    entry = dict(queries=len(pool), windows={}, strata={}, gripper={})
    for window in windows:
        values, ade, grip_correct, pred_grip, true_grip = [], [], [], [], []
        per_query_grip = []
        for row in pool:
            mask = row['mask']; assert mask == sorted(mask, reverse=True) and any(mask)
            n = min(1 if window=='first' else 2 if window=='executed_prefix' else 10, sum(mask))
            p, t = np.asarray(row['prediction'])[:n], np.asarray(row['target'])[:n]
            d = p-t
            values.append(float(np.abs(d[:,:6] if mani else d[:,:2]).mean()))
            if not mani:
                ade.append(float(np.linalg.norm(d[:,:2],axis=1).mean()))
            else:
                correct = (p[:,6]>=.5)==(t[:,6]>=.5)
                per_query_grip.append(float(correct.mean())); grip_correct.extend(correct.tolist())
                pred_grip.extend(p[:,6].tolist()); true_grip.extend(t[:,6].tolist())
        entry['windows'][window] = dict(mae_query_mean=float(np.mean(values)),
            unit='rad' if mani else 'm', dimensions='arm_joint1..6' if mani else 'XY coordinate errors',
            xy_euclidean_ade_m=None if mani else float(np.mean(ade)))
        old = report['metrics']['routes'][route]['strata']['all'][window]
        assert np.isclose(np.mean(values) if mani else np.mean(ade), old['joint_mae_rad' if mani else 'xy_ade_m']['sample_mean'])
        if mani:
            p,t=np.asarray(pred_grip),np.asarray(true_grip); wrong=(p>=.5)!=(t>=.5)
            info=dict(query_macro_accuracy=float(np.mean(per_query_grip)), point_micro_accuracy=float(np.mean(grip_correct)),
                valid_points=len(p), wrong_points=int(wrong.sum()),
                expected_closed_predicted_open=int(((t<.5)&(p>=.5)).sum()),
                expected_open_predicted_closed=int(((t>=.5)&(p<.5)).sum()),
                continuous_gripper_mae=float(abs(p-t).mean()),
                wrong_prediction_quantiles=None if not wrong.any() else np.quantile(p[wrong],[0,.25,.5,.75,1]).tolist(),
                wrong_target_quantiles=None if not wrong.any() else np.quantile(t[wrong],[0,.25,.5,.75,1]).tolist(),
                wrong_predictions_within_005_of_threshold=int((wrong&(abs(p-.5)<=.05)).sum()),
                wrong_predictions_within_010_of_threshold=int((wrong&(abs(p-.5)<=.10)).sum()),
                wrong_median_joint7_target_error_mm=None if not wrong.any() else float(np.median(abs(p[wrong]-t[wrong]))*.04*1000))
            entry['gripper'][window]=info
            assert np.isclose(info['query_macro_accuracy'],old['gripper_accuracy']['sample_mean'])
    for name in ['boundary_1s','interior','partial']:
        raw=report['metrics']['routes'][route]['strata'][name]
        entry['strata'][name]={'queries':raw['rows'],**{w:raw[w]['joint_mae_rad' if mani else 'xy_ade_m']['sample_mean'] for w in windows}}
    result['routes'][route]=entry

mani=[r for r in rows if r['route'] in {'PICK','PLACE'}]
counts=Counter({key:sum(r['metrics'][key] for r in mani) for key in ['position_events','rate_events','gripper_events']})
rate=sum(counts.values())/(70*len(mani))
assert np.isclose(rate,report['metrics']['saturation_gate']['statistics']['sample_mean'])
result['saturation']=dict(counts=counts,denominator=70*len(mani),rate=rate,limit=.005,passed=rate<=.005)
result['transitions']=[]
for r in transitions:
    tick=int(r['observation_id'].rsplit(':t',1)[1].split(':',1)[0])
    result['transitions'].append(dict(tick=tick,time_s=tick*.02,truth=r['label'],prediction=r['prediction'],correct=r['correct']))
result['transition_correct']=sum(r['correct'] for r in transitions)
(ROOT/'fit_metrics_independent.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')

from matplotlib import font_manager
font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_path)
font_name=font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({'font.family':font_name,'axes.unicode_minus':False,'font.size':10})
fig, axes=plt.subplots(4,2,figsize=(15,14),layout='constrained')
fig.suptitle('训练种子 16100000 · 第 1700 步模型的已录观测拟合\n96 个独立动作查询；扩散 seed 17；不代表自由物理闭环轨迹',fontsize=17)
colors={'target':'#233b57','prediction':'#d65b33'}
for col,route in enumerate(['PICK','PLACE']):
    pool=[r for r in rows if r['route']==route]
    for point in [0,1]:
        ax=axes[point,col]; selected=[r for r in pool if r['mask'][point]]
        x=[r['query_time_s'] for r in selected]
        ax.plot(x,[r['target'][point][6] for r in selected],'-o',markersize=4,color=colors['target'],label='真实生效目标')
        ax.plot(x,[r['prediction'][point][6] for r in selected],'-s',markersize=4,color=colors['prediction'],label='模型原值（未裁剪）')
        ax.axhline(.5,color='#666666',linestyle='--',linewidth=1,label='0.5 分类阈值')
        ax.set(title=f'{route} · '+('首点 t' if point==0 else '第二点 t+0.2 s'),xlabel='已录观测查询时间 t（秒）',ylabel='夹爪 open_fraction')
        ax.set_ylim(-.08,1.12);ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
for i,route in enumerate(routes):
    ax=axes[2+i//2,i%2];pool=[r for r in rows if r['route']==route]
    key='joint_mae_rad' if route in {'PICK','PLACE'} else 'xy_ade_m'
    for window,label,color in [('first','首点','#3880a1'),('executed_prefix','前两点 / 0.4 s','#d48c32'),('full_valid','全部有效前缀','#7456a3')]:
        ax.plot([r['query_time_s'] for r in pool],[r['metrics'][window][key] for r in pool],'-o',markersize=3,color=color,label=label)
    ax.set(title=f'{route} · 各独立查询的动作误差',xlabel='已录观测查询时间 t（秒）',ylabel='关节 MAE（rad）' if key=='joint_mae_rad' else 'XY 欧氏误差 / ADE（m）')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.text(.5,-.005,'每个标记对应一次真实已录观测下的独立预测；连线仅帮助读数。无效后缀未评分，夹爪曲线保留原始预测值。',ha='center',fontsize=10)
fig.savefig(ROOT/'training_seed16100000_fit.png',dpi=180,bbox_inches='tight')
plt.close(fig)
print(json.dumps({'verified':True,'rows':len(rows),'transitions':len(transitions),'saturation':rate,'chart':str(ROOT/'training_seed16100000_fit.png')}))
