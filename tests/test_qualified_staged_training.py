"""Qualified trainer gates and exact optimizer/RNG continuation on tiny CPU heads."""
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qualified_trainer', ROOT / 'scripts/train_staged_vla.py')
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def fixture(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    tiny = trainer.M0DiTConfig(vlm_hidden_dim=8, input_embedding_dim=8, hidden_size=12,
        num_attention_heads=2, attention_head_dim=4, num_layers=4, dropout=0.,
        max_seq_len=64, num_target_vision_tokens=2)
    monkeypatch.setattr(trainer, 'M0DiTConfig', lambda **kwargs: tiny)
    config = trainer.StagedConfig()
    training = dict(steps=3, effective_batch=2, learning_rate=2e-5, seed=17,
        checkpoint_every=1, validation_every=3, validation_rows=4, max_wall_seconds=60)
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps({'candidate': asdict(config), 'training': training}))
    release, cache_dir = tmp_path / 'release', tmp_path / 'cache'
    release.mkdir(); cache_dir.mkdir()
    rows = []
    for split in ('train', 'validation'):
        for index, route in enumerate(trainer.ROUTES):
            mani = route in {'PICK', 'PLACE'}
            rows.append(dict(split=split, task_family_id=split + '-family', episode_uuid=split,
                observation_id=str(index), route=route, query_time_s=float(index),
                time_profile=config.time_profile, active_task_id=route, active_task_epoch=index,
                training_eligible=True, actions=[[.1 * (index + 1)] * (7 if mani else 3)] * 10,
                mani_state=[.1 * (index + 1)] * 13))
    normalizer = trainer.StagedNormalizer.fit(r for r in rows if r['split'] == 'train')
    (release / 'normalization.json').write_text(json.dumps(normalizer.payload))
    (release / 'actions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = dict(schema='staged-release-v2', purpose='derived_release',
        time_profile=config.time_profile, synthetic=False, normalizer_id=normalizer.payload['normalizer_id'],
        files={p.name: trainer.digest(p) for p in release.iterdir()})
    (release / 'manifest.json').write_text(json.dumps(manifest))
    torch.manual_seed(10)
    entries = {}
    for index, row in enumerate(rows):
        condition = {k: row[k] for k in ('observation_id', 'active_task_id', 'active_task_epoch', 'query_time_s')}
        condition.update(encoder_model_id='frozen-encoder', task_tokens=torch.randn(1, 2, 8),
            live_tokens=torch.randn(1, 4, 8), task_mask=torch.ones(1, 2, dtype=torch.bool),
            live_mask=torch.ones(1, 4, dtype=torch.bool))
        path = cache_dir / f'{index}.pt'
        torch.save(condition, path)
        entries[row['episode_uuid'] + ':' + row['observation_id']] = dict(file=path.name, sha256=trainer.digest(path))
    (cache_dir / 'manifest.json').write_text(json.dumps(dict(release_sha256=trainer.digest(release / 'manifest.json'),
        encoder_model_id='frozen-encoder', entries=entries)))
    initialization = tmp_path / 'init.pt'
    torch.save(dict(candidate=asdict(config), model=trainer.StagedExperts(config, tiny).state_dict(),
        encoder_model_id='frozen-encoder', normalizer=normalizer.payload), initialization)
    args = ['--config', str(cfg), '--release', str(release), '--condition-cache', str(cache_dir),
        '--encoder-model-id', 'frozen-encoder']
    return args, initialization, manifest, rows, normalizer


def test_release_gates_and_validation_selection(tmp_path, monkeypatch):
    _, _, manifest, rows, normalizer = fixture(tmp_path, monkeypatch)
    train, val = trainer.validate_release_rows(manifest, rows, normalizer)
    assert len(train) == len(val) == 4
    assert {r['route'] for r in trainer.select_validation(val, 4)} == set(trainer.ROUTES)
    with pytest.raises(ValueError, match='stratum'):
        trainer.select_validation(val, 3)
    with pytest.raises(ValueError, match='smoke'):
        trainer.validate_release_rows({**manifest, 'not_formal_training_release': True}, rows, normalizer)
    leaked = [dict(r, task_family_id='train-family') if r['split'] == 'validation' else r for r in rows]
    with pytest.raises(ValueError, match='family crosses'):
        trainer.validate_release_rows(manifest, leaked, normalizer)
    with pytest.raises(ValueError, match='non-validation'):
        trainer.select_validation(train, 4)


def test_preflight_strict_load_and_resume_exact(tmp_path, monkeypatch):
    args, initialization, _, _, _ = fixture(tmp_path, monkeypatch)
    assert trainer.main(args + ['--initialization', str(initialization)]) == 0
    full = tmp_path / 'full'
    assert trainer.main(args + ['--initialization', str(initialization), '--execute', '--output', str(full)]) == 0
    partial = tmp_path / 'partial'
    original_step = torch.optim.AdamW.step
    calls = []
    def interrupt(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError('simulated process failure before optimizer update')
        return original_step(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', interrupt)
    with pytest.raises(RuntimeError, match='simulated process failure'):
        trainer.main(args + ['--initialization', str(initialization), '--execute', '--output', str(partial)])
    state = torch.load(partial / 'last.pt', weights_only=True)
    assert state['training_steps'] == 2
    assert state['optimizer']['state']
    altered = dict(state['training_binding'], encoder_model_id='foreign')
    with pytest.raises(ValueError, match='identity mismatch'):
        trainer.check_resume(state, altered)
    monkeypatch.setattr(torch.optim.AdamW, 'step', original_step)
    resumed = tmp_path / 'resumed'
    assert trainer.main(args + ['--resume', str(partial / 'last.pt'), '--execute', '--output', str(resumed)]) == 0
    expected = torch.load(full / 'last.pt', weights_only=True)
    actual = torch.load(resumed / 'last.pt', weights_only=True)
    assert actual['training_steps'] == expected['training_steps'] == 3
    assert actual['elapsed_wall_s'] >= state['elapsed_wall_s']
    for key, value in expected['model'].items():
        assert torch.equal(value, actual['model'][key]), key
    for key, value in expected['rng'].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual['rng'][key])
    assert not list(tmp_path.rglob('*.tmp'))
    assert (full / 'best.pt').exists()
    assert (resumed / 'model.pt').exists()


def test_rgb_rejects_depth_cache_and_bad_initialization(tmp_path, monkeypatch):
    args, initialization, _, _, _ = fixture(tmp_path, monkeypatch)
    bad = torch.load(initialization, weights_only=True)
    bad['encoder_model_id'] = 'different'
    torch.save(bad, initialization)
    with pytest.raises(ValueError, match='encoder/normalizer mismatch'):
        trainer.main(args + ['--initialization', str(initialization)])
    cache_dir = tmp_path / 'cache'
    path = cache_dir / '0.pt'
    cache = torch.load(path, weights_only=True)
    cache['points'] = torch.zeros(1, 1, 3)
    torch.save(cache, path)
    manifest = json.loads((cache_dir / 'manifest.json').read_text())
    manifest['entries']['train:0']['sha256'] = trainer.digest(path)
    (cache_dir / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='without depth'):
        trainer.main(args + ['--initialization', str(initialization)])
