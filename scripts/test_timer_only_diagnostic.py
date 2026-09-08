"""Independent AST extraction tests of actual timer runner; CPU only."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

PATH=Path(__file__).with_name('run_full_episode_diagnostic.py')
tree=ast.parse(PATH.read_text())
pipeline=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pipeline_type')
cls=next(n for n in pipeline.body if isinstance(n,ast.ClassDef))
# Pure API doubles replace imports, not control logic.
for method in cls.body:
    if isinstance(method,ast.FunctionDef):
        method.body=[n for n in method.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
class Expired(Exception):pass
class Sim:
    def __init__(self):self.n=0
    def read(self):return NS(timestamp=self.n*.02,step_index=self.n)
    def apply(self,a):pass
    def step(self,**k):self.n+=1
ns={'math':math,'SimulationClockExpired':Expired,
    'old':NS(JointTrajectoryRolloutPipeline=object,waypoint_runner=NS(_jsonable=lambda x:x)),
    'replace':lambda obj,**kw:NS(**{**vars(obj),**kw}),
    'NavigationReference':lambda *a:a,
    'measured_body_velocity':lambda s:(0.,0.,0.),
    'LIMITS':NS(lower=(-1.,)*6,upper=(1.,)*6)}
exec(compile(ast.fix_missing_locations(ast.Module(body=[pipeline],type_ignores=[])),str(PATH),'exec'),ns)
options=NS(timer_only=True,measured_position_tolerance=.02,measured_speed_limit=30.,diagnostic_pct_snap_max=.5)
C=ns['pipeline_type'](options)
def instance():
    c=C();c.raw_sim=Sim();c.simulation=c.raw_sim;c.clock_deadline_s=60.;c.config=NS(render=False)
    c.events=[];c._record=lambda name,data:c.events.append((name,data));c._capture_tick=lambda *a:None
    return c

class TestTimer(unittest.TestCase):
    def test_final_mechanical_clip(self):
        c=instance();c.route='PICK';c._query_count=1
        ns['DirectJointCommand']=lambda i,q,g:NS(joint_position=q,gripper_open_fraction=g)
        def build(command,**kw):c.trusted=command;return command
        c.action_adapter=NS(manipulation=build);c._read_trusted=lambda:None
        applied=c.apply((5.,-5.,5.,-5.,5.,-5.,2.),None)
        self.assertEqual(applied,(1.,-1.,1.,-1.,1.,-1.,1.))
        self.assertEqual(c.raw_sim.n,1)
    def test_timer_catch_precedes_recovery(self):
        runner=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_episode')
        loops=[n for n in ast.walk(runner) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='query']
        inner=loops[0].body[0]
        self.assertIsInstance(inner,ast.Try)
        self.assertEqual(inner.handlers[0].type.id,'SimulationClockExpired')
        self.assertIsInstance(inner.handlers[0].body[0],ast.Raise)
        self.assertEqual(inner.handlers[1].type.id,'Exception')

    def test_exact_3000_steps(self):
        c=instance()
        for _ in range(3000):c._physical_step(None,route=None,command_index=None)
        self.assertEqual(c.raw_sim.n,3000)
        with self.assertRaises(Expired):c._physical_step(None,route=None,command_index=None)
        self.assertEqual(c.raw_sim.n,3000)
    def test_evaluator_exception_after_step_not_repeated(self):
        c=instance()
        def step(**kw):c.raw_sim.n+=1;raise ValueError('evaluator broke after tick')
        c.simulation=NS(apply=lambda a:None,step=step)
        c._physical_step(None,route=None,command_index=None)
        self.assertEqual(c.raw_sim.n,1)
        self.assertEqual(c.events[0][1]['reason'],'post_step_evaluator_error')
        self.assertFalse(c.scoring_valid)
    def test_thresholds_advisory_no_latch(self):
        c=instance();obs=NS(q=(10.,)*6,dq=(100.,)*6,time_s=0.)
        self.assertTrue(c.safety((5.,)*6+(2.,),obs)[0])
        self.assertEqual(c.events[0][1]['reason'],'nonterminating_joint_diagnostic')
        self.assertFalse(hasattr(c,'safety_stop_reason'))
    def test_zero_nav_control_keeps_20_ticks(self):
        c=instance();c.route='NAV_TO_SOURCE';c._query_count=1
        c.trusted=NS(joint_position=(0.,)*6,gripper_open_fraction=0.)
        c._obs=lambda state:NS(q=(0.,)*6,dq=(0.,)*6,time_s=state.timestamp)
        c._read_trusted=lambda:None;c._local_map=lambda route:{}
        path=NS(trace={},pct_plan=NS(snap_distance_m=.2))
        control=NS(base_velocity=(0.,0.,0.),reason='dwa_zero_control_before_reach',requires_requery=True,reached_local_goal=False,trace={})
        nav=NS(config=NS(),begin=lambda *a,**k:path,command=lambda *a,**k:control,
            dwa_controller=NS(last_trace={'command':[0.,0.,0.],'debug':{}}),reset=lambda:None)
        c.system=NS(navigation_executor=nav);c.action_adapter=NS(navigation=lambda *a,**k:None)
        state=NS(robot_root_pose=(0.,0.,0.,1.,0.,0.,0.),timestamp=0.)
        original=c.raw_sim.read
        c.raw_sim.read=lambda:NS(**vars(original()),robot_root_pose=state.robot_root_pose)
        c._navigation_window({'reference_query_body':[[0.,0.,0.]]*10},state)
        self.assertEqual(c.raw_sim.n,20)
        self.assertEqual(sum(e[0]=='navigation_control_rejected' for e in c.events),20)
        self.assertEqual(sum(e[0]=='navigation_control' for e in c.events),20)

if __name__=='__main__':unittest.main()
