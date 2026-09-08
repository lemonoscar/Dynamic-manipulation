"""CPU checks of the actual new object proof and measured-state tolerance."""
import ast,math
from pathlib import Path
from types import SimpleNamespace as N
import unittest
source=ast.parse((Path(__file__).with_name('run_full_episode_diagnostic.py')).read_text())
live=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='live_object_tensor')
pipeline=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='pipeline_type')
cls=next(x for x in pipeline.body if isinstance(x,ast.ClassDef))
safety=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='safety')
ns={'math':math,'first_array_row':lambda v:v[0],'options':N(measured_speed_limit=30,measured_position_tolerance=.02),'LIMITS':N(lower=(0,)*6,upper=(3.14,)*6)}
exec(compile(ast.Module(body=[live,safety],type_ignores=[]),'actual_live_proof','exec'),ns)
class Checks(unittest.TestCase):
 def test_live_tensor_and_no_usd_fallback(self):
  view=N(is_physics_handle_valid=lambda:True,_physics_view=N(get_transforms=lambda:[[1,2,3,0,0,0,1]],get_velocities=lambda:[[0]*6]))
  sim=N(_object=N(_rigid_prim_view=view));self.assertEqual(ns['live_object_tensor'](sim)['pose_wxyz'],[1,2,3,1,0,0,0])
  view.is_physics_handle_valid=lambda:False
  with self.assertRaisesRegex(RuntimeError,'fallback_forbidden'):ns['live_object_tensor'](sim)
 def test_relaxed_measurement_still_bounds_commands(self):
  target=(0,)*6+(.5,);obs=N(q=(-1.66e-5,0,0,0,0,0),dq=(3.4,0,0,0,0,0))
  self.assertTrue(ns['safety'](None,target,obs)[0])
  obs.q=(-.03,0,0,0,0,0);self.assertFalse(ns['safety'](None,target,obs)[0]);obs.q=(0,)*6
  obs.dq=(30.1,0,0,0,0,0);self.assertFalse(ns['safety'](None,target,obs)[0]);obs.dq=(0,)*6
  self.assertFalse(ns['safety'](None,(-.001,0,0,0,0,0,.5),obs)[0])
  obs.q=(float('nan'),0,0,0,0,0);self.assertFalse(ns['safety'](None,target,obs)[0])
if __name__=='__main__':unittest.main()
