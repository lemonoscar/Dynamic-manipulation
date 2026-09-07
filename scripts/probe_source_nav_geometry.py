#!/usr/bin/env python3
"""Live scene preflight for a source continuous NAV endpoint; never bypass certificates."""
import argparse,json,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import run_conditioned_pick as base
from scripts.replay_sampled_joint_targets import initialize_source_state
from conveyor_bench.conveyorvla.joint_trajectory_data import _align_sampled_5hz_rows
from conveyor_bench.conveyorvla.waypoint import yaw_from_quaternion
from conveyor_bench.conveyorvla.formal_checkpoint import write_json,sha256


def factory(options,probe):
    parent=base.pipeline_type(options)
    class Nav(parent):
        def _execute_policy(self,options,summary,health,payload,state):
            from pxr import Usd,UsdGeom,UsdPhysics
            import omni.usd
            source=options.source_episode
            seq=_align_sampled_5hz_rows(list(map(json.loads,(source/'samples.jsonl').open())),list(map(json.loads,(source/'frames.jsonl').open())))
            obs=next(o for s,f,o in seq if s['frame_index']==probe.source_frame)
            sim=self.physics.simulation;summary['nav_initialization']=initialize_source_state(sim,obs)
            sim._runtime.sim.forward();sim._render_without_physics(valid_state_step=int(sim.read().step_index),reason='nav_source_sync',force=True)
            current=sim.read();xyzyaw=(*current.robot_root_pose[:3],yaw_from_quaternion(current.robot_root_pose[3:]))
            rows=[json.loads(l) for l in probe.source_probes.open()]
            row=next(r for r in rows if r['episode_id']==options.source_episode.name and r['sample_id'].endswith(f'{probe.source_frame:06d}') and r['kind']=='source' and r['route']=='NAV_TO_SOURCE')
            poses=row['poses_xyyaw'];goal=poses['A'];requested=(goal[0],goal[1],current.robot_root_pose[2],goal[2])
            planner=self.system.navigation_executor.pct_planner;plan=planner.plan(xyzyaw,requested)
            stage=omni.usd.get_context().get_stage();cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy,UsdGeom.Tokens.guide],useExtentsHint=False,ignoreVisibility=True)
            robot_root=next(str(p.GetPath()).split('/Robot/')[0]+'/Robot' for p in stage.Traverse() if '/Robot/' in str(p.GetPath()))
            colliders=[]
            for prim in stage.Traverse():
                if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
                enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                if enabled is False:continue
                bbox=cache.ComputeWorldBound(prim).ComputeAlignedRange()
                colliders.append({'path':str(prim.GetPath()),'type':prim.GetTypeName(),'minimum':list(bbox.GetMin()),'maximum':list(bbox.GetMax()),'robot':str(prim.GetPath()).startswith(robot_root+'/'),'bounds_valid':bool(np.all(np.asarray(bbox.GetMin())<=np.asarray(bbox.GetMax()))),'mesh_approximation':str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()) if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None})
            # Current bounds are observations, not a certificate of all gait/turn envelopes.
            robot=[c for c in colliders if c['robot']];radius=max((math.hypot(x-current.robot_root_pose[0],y-current.robot_root_pose[1]) for c in robot for x in (c['minimum'][0],c['maximum'][0]) for y in (c['minimum'][1],c['maximum'][1])),default=None)
            result={'schema':'source-nav-live-geometry-preflight-v1','source_episode':options.source_episode.name,'source_frame':probe.source_frame,
                'G':poses['G'],'A':goal,'B':list(plan.snapped_goal_world),'candidate_continuous_target':requested,
                'source_start_pose':xyzyaw,'coarse_path':plan.path_world,'coarse_snap_m':plan.snap_distance_m,
                'colliders':colliders,'observed_robot_xy_radius_m':radius,
                'geometry_coverage':{'full_gait_turn_envelope_certified':False,'continuous_support_coverage_certified':False,'live_collision_approximations_bound':True},
                'robot_motion_executed':False,'actual_arrival_C':None,'initial_pose_only':list(current.robot_root_pose),
                'arrival_verified':False,'operation_handoff_verified':None,
                'status':'preflight_incomplete','reason':'observed_collision_bounds_do_not_certify_full_gait_turn_or_support',
                'position_tolerance_m':.12,'yaw_tolerance_rad':.14,'timeout_s':10,
                'source_probe_sha256':sha256(probe.source_probes)}
            write_json(self.episode_dir/'nav_preflight_raw.json',result)
            if probe.execute_certified:
                from conveyor_bench.conveyorvla.nav_geometry_audit import rectangle_geometry_certificate
                from scripts.source_nav_motion import live_full_reach, execute_certified_motion, live_navigation_posture_bound
                import trimesh
                result['live_full_reach']=live_full_reach(stage,cache,robot_root,colliders)
                terrain=next(p for p in stage.Traverse() if str(p.GetPath())=='/World/nav_collision/terrain/collision_mesh')
                mesh=UsdGeom.Mesh(terrain);vertices=np.asarray(mesh.GetPointsAttr().Get(),float);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get());indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                if not np.all(counts==3):raise ValueError('terrain not triangular')
                matrix=np.asarray(UsdGeom.Xformable(terrain).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                vertices=(np.c_[vertices,np.ones(len(vertices))]@matrix)[:,:3]
                tm=trimesh.Trimesh(vertices=vertices,faces=indices.reshape(-1,3),process=False)
                hits,_,floor_faces=tm.ray.intersects_location(np.asarray([[xyzyaw[0],xyzyaw[1],xyzyaw[2]]]),np.asarray([[0.,0.,-1.]]),multiple_hits=True)
                if len(hits)==0:raise ValueError('source ground ray misses terrain')
                floor_z=float(hits[:,2].max());path=[xyzyaw[:2],*plan.path_world,goal[:2]]
                environment=[c for c in colliders if not c['robot'] and c['path']!=str(terrain.GetPath())]
                radius=result['live_full_reach']['radius_m'];tolerance=None
                if probe.nav_posture_bound:
                    from conveyor_bench.conveyorvla.nav_geometry_audit import connected_sweep_certificate
                    result['nav_posture_bound']=live_navigation_posture_bound(result['live_full_reach'],stage,cache,sim._adapter.robot,current)
                    radius=result['nav_posture_bound']['radius_m'];tolerance=result['nav_posture_bound']['arm_angular_error_bound_rad']
                    result['geometry_certificate']=connected_sweep_certificate(tm,path,radius,floor_faces[hits[:,2].argmax()],environment)
                else:
                    result['geometry_certificate']=rectangle_geometry_certificate(tm,path,radius,floor_z,environment)
                result['terrain_vertex_face_sha256']=__import__('hashlib').sha256(vertices.tobytes()+indices.tobytes()).hexdigest()
                write_json(self.episode_dir/'nav_preflight_before_motion.json',result)
                if result['geometry_certificate']['valid']:
                    result.update(execute_certified_motion(self,path,goal,radius,arm_tracking_tolerance=tolerance))
                else:
                    result.update(status='geometry_rejected',reason=result['geometry_certificate']['reason'])
            write_json(self.episode_dir/'nav_preflight.json',result)
            summary.update(execution_mode='source-nav-live-preflight',nav_result=result,policy_score=False,full_task_success=None)
    return Nav


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--nav-posture-bound',action='store_true');p.add_argument('--execute-certified',action='store_true');p.add_argument('--source-frame',type=int,required=True);p.add_argument('--source-probes',type=Path,required=True);probe,rest=p.parse_known_args();original=base.pipeline_type
    def patched(options):
        base.pipeline_type=original;return factory(options,probe)
    base.pipeline_type=patched;sys.argv=[sys.argv[0],*rest];return base.main()
if __name__=='__main__':raise SystemExit(main())
