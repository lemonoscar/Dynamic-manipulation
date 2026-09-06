"""GPU tensor contact measurements for the two existing robot contact reporters.

Optional object reporting enumerates external collider paths. Missing support
coverage remains unknown even when both fingers make contact.
"""
from __future__ import annotations
import numpy as np


def support_coverage_result(external_loaded, finger_loaded, reverse_finger_loaded,
                            *, external_witness, articulated_witness):
    """Absence needs calibrated rigid AND articulated contact coverage."""
    reciprocal=list(finger_loaded)==list(reverse_finger_loaded)
    external_witness=bool(external_witness or external_loaded>0)
    articulated_witness=bool(articulated_witness or (all(finger_loaded) and all(reverse_finger_loaded)))
    complete=external_witness and articulated_witness and reciprocal
    support=True if external_loaded>0 else False if complete else None
    return support,external_witness,articulated_witness,reciprocal


class IsaacFingerContactTensorProbe:
    def __init__(self, simulation, record, *, object_support_probe=False):
        import omni.physics.tensors as tensors
        self.object_path=simulation._metadata['object_reader_report']['rigid_body_prim_path']
        if object_support_probe:
            import omni.usd
            from pxr import PhysxSchema, Usd, UsdPhysics
            stage=omni.usd.get_context().get_stage()
            prim=stage.GetPrimAtPath(self.object_path)
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.)
            # Report flags are parsed while building PhysX actors. Recreate all
            # readers BEFORE allocating contact views or restoring trial state.
            reset=simulation._hard_reset_stage_reuse_physics(simulation._episode_spec)
            record('object_contact_report_initialization',reset)
        robot=simulation._adapter.robot
        paths=robot.root_physx_view.link_paths[0]
        self.sim_view=tensors.create_simulation_view('torch')
        self.sim_view.set_subspace_roots('/')
        self.views={}
        self.record=record
        self.physics_dt=float(simulation._runtime.physics_dt)
        self.object_view=None
        self.support_coverage_witness=False
        self.articulated_coverage_witness=False
        for name in ('arm_link7','arm_link8'):
            path=paths[list(robot.body_names).index(name)]
            view=self.sim_view.create_rigid_contact_view(path,filter_patterns=[self.object_path],max_contact_data_count=256)
            if view.sensor_count != 1 or view.filter_count != 1:
                raise ValueError(f'finger contact tensor binding mismatch: {name}')
            self.views[name]=view
        if object_support_probe:
            finger_paths=[paths[list(robot.body_names).index(n)] for n in ('arm_link7','arm_link8')]
            actors=set()
            for collider in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
                if not collider.HasAPI(UsdPhysics.CollisionAPI):continue
                if UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:continue
                actor=collider
                while actor.IsValid() and not actor.IsPseudoRoot() and not actor.HasAPI(UsdPhysics.RigidBodyAPI):
                    actor=actor.GetParent()
                actor_path=str((actor if actor.IsValid() and not actor.IsPseudoRoot() else collider).GetPath())
                if actor_path != self.object_path:actors.add(actor_path)
            self.external_filter_paths=sorted(actors-set(finger_paths))
            if not self.external_filter_paths:raise ValueError('no external collider coverage for contact calibration')
            self.object_filters=[*finger_paths,*self.external_filter_paths]
            self.object_view=self.sim_view.create_rigid_contact_view(
                self.object_path,filter_patterns=self.object_filters,max_contact_data_count=4096)
            if self.object_view.sensor_count!=1 or self.object_view.filter_count!=len(self.object_filters):
                raise ValueError('object support contact filter binding mismatch')
            self.record('object_support_tensor_probe_installed',
                        {'filters':self.object_view.filter_paths,'requested_filters':self.object_filters,
                         'sensor_paths':self.object_view.sensor_paths,
                         'coverage_requires_nonfinger_contact_witness':True})
        self.record('finger_contact_tensor_probe_installed',
                    {'object':self.object_path, 'finger_sensor_paths':{k:v.sensor_paths for k,v in self.views.items()},
                     'external_support_coverage':False,'physics_dt_s':self.physics_dt})

    def read(self):
        records={};directions=[]
        for name,view in self.views.items():
            force,point,normal,separation,count,start=view.get_contact_data(dt=self.physics_dt)
            n=int(count[0,0]);first=int(start[0,0])
            if n >= 256 or first+n > 256:
                raise RuntimeError('finger contact tensor buffer may be truncated')
            normals=normal[first:first+n].detach().cpu().numpy()
            forces=force[first:first+n].detach().cpu().numpy()
            if not np.isfinite(normals).all() or not np.isfinite(forces).all():
                raise RuntimeError('nonfinite finger contact tensor')
            # Require loaded contact, not only a speculative near-contact point.
            loaded=forces.reshape(-1)>1.e-5
            vector=np.mean(normals[loaded],axis=0) if loaded.any() else np.zeros(3)
            norm=np.linalg.norm(vector);directions.append(vector/norm if norm>1.e-8 else None)
            records[name]={'contact_count':n, 'normal_forces_N':forces.reshape(-1).tolist(),
                           'normals':normals.tolist(),
                           'points_world':point[first:first+n].detach().cpu().tolist(),
                           'separation_m':separation[first:first+n].detach().cpu().reshape(-1).tolist()}
        bilateral=all(v is not None for v in directions)
        support=None;support_evidence=None
        if self.object_view is not None:
            force,_,_,_,count,start=self.object_view.get_contact_data(dt=self.physics_dt)
            loaded=[]
            for index in range(len(self.object_filters)):
                n,first=int(count[0,index]),int(start[0,index])
                if n>=4096 or first+n>4096:raise RuntimeError('object contact buffer may be truncated')
                values=force[first:first+n].detach().cpu().numpy()
                if not np.isfinite(values).all():raise RuntimeError('nonfinite object contact data')
                loaded.append(int((values>1.e-5).sum()))
            residual=sum(loaded[2:])
            support,self.support_coverage_witness,self.articulated_coverage_witness,reciprocal = support_coverage_result(
                residual,[v is not None for v in directions],[n>0 for n in loaded[:2]],
                external_witness=self.support_coverage_witness,articulated_witness=self.articulated_coverage_witness)
            support_evidence={'loaded_contacts_by_filter':dict(zip(self.object_filters,loaded)),
                              'nonfinger_loaded_contacts':residual,
                              'nonfinger_contact_coverage_witness':self.support_coverage_witness,
                              'articulated_contact_coverage_witness':self.articulated_coverage_witness,
                              'finger_object_reciprocity_consistent':reciprocal}
        result={'schema':'physx-finger-contact-tensors-v1','bilateral_finger_contact':bilateral,
                'opposing_finger_normals':bool(bilateral and float(np.dot(*directions))<=-.5),
                'external_support':support,'external_support_coverage':support is not None,'fingers':records,
                'object_support_evidence':support_evidence,
                'provenance':'GPU rigid contact tensor; finger-body colliders; support unknown unless witnessed aggregate coverage'}
        self.record('grasp_contact_measurement',result)
        return result

    def close(self):
        self.views.clear()
        self.object_view=None
        self.sim_view=None
