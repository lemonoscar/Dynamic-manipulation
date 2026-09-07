"""Train-only normalization, immutable cached-condition identity and staged batches."""
import hashlib
import json
import numpy as np
import torch
from .contracts.action import TIME_PROFILES, finite_array


def payload_id(payload):
    data={k:v for k,v in payload.items() if k!='normalizer_id'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


class StagedNormalizer:
    def __init__(self,payload):
        if payload.get('schema')!='staged-normalizer-v2' or payload.get('normalizer_id')!=payload_id(payload):
            raise ValueError('staged normalizer identity mismatch')
        if payload.get('fit_split')!='train':
            raise ValueError('normalizer must be fitted only on train')
        self.payload=payload

    @classmethod
    def fit(cls,records):
        rows=list(records)
        if not rows or any(r.get('split')!='train' or not r.get('training_eligible') or r.get('synthetic') for r in rows):
            raise ValueError('fit requires nonsynthetic, verified train actions only')
        mani=[r for r in rows if r['route'] in {'PICK','PLACE'}]
        nav=[r for r in rows if r['route'] in {'NAV_TO_SOURCE','NAV_TO_TARGET'}]
        if not mani or not nav:
            raise ValueError('common normalization pool needs NAV and Mani')
        def bounds(values):
            a=np.asarray(values,float)
            if not np.isfinite(a).all():raise ValueError('nonfinite normalization pool')
            lo,hi=np.quantile(a,[.01,.99],axis=0)
            hi=np.maximum(hi,lo+1e-6)
            return {'low':lo.tolist(),'high':hi.tolist()}
        payload={'schema':'staged-normalizer-v2','fit_split':'train','fit_row_count':len(rows),
            'families':sorted({r['task_family_id'] for r in rows}),
            'time_profiles':sorted({r['time_profile'] for r in rows}),
            'nav':bounds([p for r in nav for p in r['actions']]),
            'mani':bounds([p[:6] for r in mani for p in r['actions']]),
            'state_q':bounds([r['mani_state'][:6] for r in mani]),
            'state_dq':bounds([r['mani_state'][6:12] for r in mani]),
            'gripper':'open_fraction_[0,1]_to_[-1,1]'}
        payload['normalizer_id']=payload_id(payload)
        return cls(payload)

    def _scale(self,value,key,inverse=False):
        b=self.payload[key];low=np.asarray(b['low']);span=np.asarray(b['high'])-low
        if isinstance(value,torch.Tensor):
            low=torch.as_tensor(low,device=value.device,dtype=value.dtype)
            span=torch.as_tensor(span,device=value.device,dtype=value.dtype)
        return (value+1)*span/2+low if inverse else 2*(value-low)/span-1

    def normalize_action(self,route,value):
        a=np.asarray(value,float)
        if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('invalid action')
        if route in {'NAV_TO_SOURCE','NAV_TO_TARGET'}:
            if a.shape[1]!=3:raise ValueError('NAV shape')
            return self._scale(a,'nav')
        if a.shape[1]!=7:raise ValueError('Mani shape')
        return np.concatenate((self._scale(a[:,:6],'mani'),2*a[:,6:]-1),-1)

    def denormalize_action(self,route,value):
        a=value if isinstance(value,torch.Tensor) else np.asarray(value,float)
        if route in {'NAV_TO_SOURCE','NAV_TO_TARGET'}:return self._scale(a,'nav',True)
        parts=(self._scale(a[...,:6],'mani',True),(a[...,6:]+1)/2)
        return torch.cat(parts,-1) if isinstance(a,torch.Tensor) else np.concatenate(parts,-1)

    def normalize_mani_state(self,value):
        a=finite_array(value,(13,),'mani_state')
        return np.concatenate((self._scale(a[:6],'state_q'),self._scale(a[6:12],'state_dq'),[2*a[12]-1]))


def validate_condition_cache(cache,record,model_id):
    expected={'observation_id':record['observation_id'],'active_task_id':record['active_task_id'],
        'active_task_epoch':record['active_task_epoch'],'encoder_model_id':model_id,
        'query_time_s':record['query_time_s']}
    if any(cache.get(k)!=v for k,v in expected.items()):
        raise ValueError('stale/foreign cached task or live visual condition')
    for name in ('task_tokens','live_tokens'):
        value=cache[name]
        if not isinstance(value,torch.Tensor) or value.ndim!=3 or value.shape[0]!=1 or not torch.isfinite(value).all():
            raise ValueError('invalid cached token tensor')
    if cache.get('calibration_id')!=record.get('calibration_id'):
        raise ValueError('cached calibration mismatch')
    return {k:cache[k] for k in ('task_tokens','live_tokens','task_mask','live_mask','points','depth_valid') if k in cache}
