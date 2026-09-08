import json,sys
import pytest
from scripts.merge_staged_conditions import main
from conveyor_bench.conveyorvla.staged_data import digest


def test_merge_requires_complete_bound_query_set(tmp_path,monkeypatch):
    release=tmp_path/'release';release.mkdir()
    (release/'manifest.json').write_text('{}')
    (release/'actions.jsonl').write_text(json.dumps({'episode_uuid':'e','observation_id':'o','split':'train'})+'\n')
    root=tmp_path/'cache';shard=root/'shard-0';shard.mkdir(parents=True)
    payload={'schema':'staged-condition-cache-v2','release_sha256':digest(release/'manifest.json'),'encoder_model_id':'frozen',
        'input_modalities':'rgb','splits':['train','validation'],'shard_count':1,'shard_index':0,'entries':{}}
    manifest=shard/'manifest.json';manifest.write_text(json.dumps(payload))
    monkeypatch.setattr(sys,'argv',['merge','--release',str(release),'--cache-root',str(root),'--shards','1'])
    with pytest.raises(ValueError,match='incomplete'):main()
    assert not (root/'manifest.json').exists()
    (shard/'one.pt').write_bytes(b'cached-condition')
    payload['entries']['e:o']={'file':'one.pt','sha256':digest(shard/'one.pt')}
    manifest.write_text(json.dumps(payload));main()
    result=json.loads((root/'manifest.json').read_text())
    assert result['entries']['e:o']['file']=='shard-0/one.pt'
