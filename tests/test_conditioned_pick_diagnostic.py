from types import SimpleNamespace
import importlib.util
from pathlib import Path
import pytest


def load_script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_diagnostic_service_rejects_truth_and_autonomous_route_injection():
    pytest.importorskip('torch')
    module=load_script('serve_conditioned_pick')
    service=module.ConditionedPickService(None)
    for key in ('object_pose','robot_root_pose','target_pose','route','assistant_prefix'):
        with pytest.raises(ValueError,match='invalid diagnostic request'):
            service.infer({'protocol_version':module.PROTOCOL,key:[]})


def test_conditioned_history_hold_uses_pick_locks_without_changing_base_method_signature(monkeypatch):
    module=load_script('run_conditioned_pick')
    cls=module.pipeline_type(SimpleNamespace(execute_points=2))
    pipeline=cls.__new__(cls)
    pipeline.physics=SimpleNamespace(armed=True)
    pipeline.simulation=SimpleNamespace(read=lambda:None)
    pipeline._query_count=0
    monkeypatch.setattr(module,'measured_named_joint_state',lambda s:SimpleNamespace(joint_position=(0.,)*6,gripper_open_fraction=.75))
    pipeline.action_adapter=SimpleNamespace(hold=lambda command,**kw:(command,kw))
    command,kwargs=pipeline._measured_hold('camera_history')
    assert kwargs['route'].value=='PICK'
    assert command.gripper_open_fraction==.75
    pipeline.physics=None
    monkeypatch.setattr(module.runner.JointTrajectoryRolloutPipeline,'_measured_hold',lambda self,source:source)
    assert pipeline._measured_hold('prepare')=='prepare'
