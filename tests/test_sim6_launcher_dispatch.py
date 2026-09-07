import ast,json
from pathlib import Path
from types import SimpleNamespace


def test_source_gpu_subclass_dispatch_survives_device_binding():
    root=Path(__file__).resolve().parents[1]
    module=ast.parse((root/'scripts/run_joint_trajectory_rollout.py').read_text())
    wrapper=next(n for n in ast.walk(module) if isinstance(n,ast.ClassDef) and n.name=='_DeviceBoundAppLauncher')
    class Launcher:
        def __init__(self,launcher_args):
            self._resolve_device_settings(launcher_args)
            self.result=launcher_args
        def _resolve_device_settings(self,launcher_args):
            self.device_id=int(launcher_args['device'].split(':')[1])
            launcher_args['physics_gpu']=self.device_id
            launcher_args['active_gpu']=self.device_id
    scope={'approved_app_launcher':Launcher,'args':SimpleNamespace(isaac_device='cuda:0'),'Mapping':dict,'Any':object,'json':json}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual-device-wrapper>','exec'),scope)
    class SourceBridge(scope['_DeviceBoundAppLauncher']):
        def _resolve_device_settings(self,launcher_args):
            super()._resolve_device_settings(launcher_args)
            launcher_args['active_gpu']=4
    result=SourceBridge({'device':'cuda:3'})
    assert isinstance(result,SourceBridge)
    assert result.result['physics_gpu']==0 and result.result['active_gpu']==4
    assert result.result['enable_cameras'] is True
