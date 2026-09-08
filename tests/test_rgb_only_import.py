"""RGB selection must not require or validate the unused depth branch."""
import json
import pytest
from conveyor_bench.conveyorvla.collection_importer import load_episode
from test_staged_data_experts import fixture, mutate


def test_rgb_selection_ignores_malformed_depth(tmp_path):
    root = fixture(tmp_path / 'ep')
    mutate(root / 'observations.jsonl', lambda rows: [r.update(depth={'definition': 'invalid-unused-depth'}) for r in rows])
    rgb = load_episode(root, modalities='rgb')
    assert rgb.action_view('o10')['depth'] is None
    with pytest.raises(ValueError, match='depth'):
        load_episode(root, modalities='rgbd').action_view('o10')


def test_unknown_modalities_rejected(tmp_path):
    with pytest.raises(ValueError, match='modalities'):
        load_episode(fixture(tmp_path / 'ep'), modalities='rgbx')
