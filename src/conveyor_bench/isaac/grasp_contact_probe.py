"""Opt-in Isaac 5.1 PhysX contact recording; no forces or constraints applied.

Finger-body contact is recorded explicitly, not represented as segmented pad
contact. Opposing contact normals distinguish squeezing from two-sided support.
"""
from __future__ import annotations
import numpy as np


class IsaacGraspContactProbe:
    def __init__(self, simulation, record):
        import carb
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, PhysicsSchemaTools
        from omni.physx.bindings._physx import ContactEventType
        self.decode = lambda value: str(PhysicsSchemaTools.intToSdfPath(value))
        self.lost_type = ContactEventType.CONTACT_LOST
        self.record = record
        self.pairs = {}
        self.events = 0
        self.total_headers = self.reads = 0
        self.actor_examples = set()
        self.error = None
        self.settings = carb.settings.get_settings()
        self.previous_disable_processing = self.settings.get('/physics/disableContactProcessing')
        # IsaacLab SimulationContext disables report processing by default;
        # registering a callback alone otherwise produces silent unknowns.
        self.settings.set_bool('/physics/disableContactProcessing', False)
        robot=simulation._adapter.robot
        self.fingers={str(robot.root_physx_view.link_paths[0][list(robot.body_names).index(n)]):n
                      for n in ('arm_link7','arm_link8')}
        self.object_path=str(simulation._metadata['object_reader_report']['rigid_body_prim_path'])
        stage=omni.usd.get_context().get_stage()
        prim=stage.GetPrimAtPath(self.object_path)
        if not prim.IsValid():raise ValueError('contact probe object prim missing')
        self.api=PhysxSchema.PhysxContactReportAPI.Apply(prim)
        self.api.CreateThresholdAttr().Set(0.)
        self.subscription=get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact)
        self.record('grasp_contact_probe_installed',{'object':self.object_path,'finger_bodies':self.fingers,
                    'contact_semantics':'finger-body colliders, not segmented fingerpads','threshold':0.,
                    'previous_disable_contact_processing':self.previous_disable_processing,
                    'disable_contact_processing':False})

    def _on_contact(self, headers, data):
        try:
            for h in headers:
                self.total_headers += 1
                actors=[self.decode(h.actor0),self.decode(h.actor1)]
                if len(self.actor_examples) < 32:
                    self.actor_examples.update(actors)
                if self.object_path not in actors:continue
                colliders=[self.decode(h.collider0),self.decode(h.collider1)]
                key=tuple(colliders);self.events+=1
                if h.type==self.lost_type:
                    self.pairs.pop(key,None);continue
                other=1-actors.index(self.object_path)
                normals=[]
                for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                    # Keep normals consistently directed relative to the object.
                    normals.append((np.asarray(data[i].normal,dtype=float)*(1 if other==1 else -1)).tolist())
                self.pairs[key]={'other_actor':actors[other],'other_collider':colliders[other],'normals':normals}
        except Exception as error:
            self.error=f'{type(error).__name__}:{error}'

    def read(self):
        if self.error:raise RuntimeError(f'contact probe failed: {self.error}')
        self.reads += 1
        if not self.events:
            if self.reads % 50 == 1:
                self.record('grasp_contact_coverage_unknown',
                            {'total_headers':self.total_headers, 'object_path':self.object_path,
                             'actor_examples':sorted(self.actor_examples)})
            return None  # Recording coverage not demonstrated yet.
        normals={name:[] for name in self.fingers.values()};support=False
        for value in self.pairs.values():
            name=self.fingers.get(value['other_actor'])
            if name is None:support=True
            else:normals[name].extend(value['normals'])
        vectors=[]
        for name in ('arm_link7','arm_link8'):
            v=np.mean(normals[name],axis=0) if normals[name] else np.zeros(3)
            length=np.linalg.norm(v);vectors.append(v/length if length>1e-8 else None)
        bilateral=all(v is not None for v in vectors)
        opposing=bool(bilateral and float(np.dot(*vectors))<=-.5)
        result={'schema':'physx-grasp-contact-v1','bilateral_finger_contact':bilateral,
                'opposing_finger_normals':opposing,'external_support':support,
                'contact_events':self.events,'pairs':list(self.pairs.values()),
                'provenance':'PhysX contact reports; finger-body colliders'}
        self.record('grasp_contact_measurement',result)
        return result

    def close(self):
        self.subscription=None
        if self.previous_disable_processing is not None:
            self.settings.set_bool('/physics/disableContactProcessing', bool(self.previous_disable_processing))
