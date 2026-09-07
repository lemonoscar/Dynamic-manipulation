"""Conservative source-scene rectangle coverage from triangle geometry, not map interpolation."""
import math,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def area(poly):
    return 0. if len(poly)<3 else abs(float(np.sum(poly[:,0]*np.roll(poly[:,1],-1)-poly[:,1]*np.roll(poly[:,0],-1))))/2

def clip(poly,a,b,inside=True):
    if len(poly)==0:return poly
    sign=1 if inside else -1
    d=lambda p:sign*((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]))
    out=[];prev=poly[-1];dp=d(prev)
    for cur in poly:
        dc=d(cur)
        if (dc>=0)!=(dp>=0):out.append(prev+(cur-prev)*(dp/(dp-dc)))
        if dc>=0:out.append(cur)
        prev=cur;dp=dc
    return np.asarray(out).reshape(-1,2)

def uncovered_rectangle(lo,hi,triangles):
    remainder=[np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])]
    for tri in triangles:
        tri=np.asarray(tri)[:,:2]
        if area(tri)<1e-12:continue
        a,b=tri[1]-tri[0],tri[2]-tri[0]
        if a[0]*b[1]-a[1]*b[0]<0:tri=tri[::-1]
        next_parts=[]
        for poly in remainder:
            pending=poly
            for a,b in zip(tri,np.roll(tri,-1,axis=0)):
                outside=clip(pending,a,b,False)
                if area(outside)>1e-12:next_parts.append(outside)
                pending=clip(pending,a,b,True)
                if area(pending)<=1e-12:break
        remainder=next_parts
        if not remainder:break
        if len(remainder)>10000:raise ValueError('support union complexity exceeds diagnostic bound')
    return sum(area(p) for p in remainder)

def urdf_full_reach(path):
    """Triangle inequality over kinematic chain and collision primitives; all rotations allowed."""
    import trimesh
    path=Path(path);root=ET.parse(path).getroot();joints={j.find('child').get('link'):j for j in root.findall('joint')};memo={};hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest()}
    def vec(el,key,default):return np.asarray([float(v) for v in el.get(key,default).split()]) if el is not None else np.asarray([float(v) for v in default.split()])
    def reach(link):
        if link not in joints:return 0.
        if link not in memo:
            j=joints[link];travel=0.
            if j.get('type')=='prismatic':travel=max(abs(float(j.find('limit').get(k))) for k in ('lower','upper'))
            memo[link]=reach(j.find('parent').get('link'))+float(np.linalg.norm(vec(j.find('origin'),'xyz','0 0 0')))+travel
        return memo[link]
    bounds=[]
    for link in root.findall('link'):
        for collision in link.findall('collision'):
            g=list(collision.find('geometry'))[0]
            if g.tag=='box':radius=np.linalg.norm(vec(g,'size','0 0 0'))/2
            elif g.tag=='sphere':radius=float(g.get('radius'))
            elif g.tag=='cylinder':radius=math.hypot(float(g.get('radius')),float(g.get('length'))/2)
            elif g.tag=='mesh':
                p=(path.parent/g.get('filename')).resolve()
                if not p.is_file():p=(path.parent.parent/g.get('filename')).resolve()
                mesh=trimesh.load(p,process=False);radius=np.linalg.norm(np.asarray(mesh.vertices)*vec(g,'scale','1 1 1'),axis=1).max();hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            else:raise ValueError('unsupported collision primitive')
            bounds.append({'link':link.get('name'),'reach_m':reach(link.get('name'))+np.linalg.norm(vec(collision.find('origin'),'xyz','0 0 0'))+float(radius)})
    return {'radius_m':float(max(b['reach_m'] for b in bounds)),'bounds':bounds,'asset_hashes':hashes,'method':'all_joint_rotations_triangle_inequality_collision_bounds','limits':'URDF must match runtime collision geometry and root before acceptance'}

def rectangle_geometry_certificate(mesh,points,radius,floor_z,obstacles):
    points=np.asarray(points);lo=points.min(axis=0)-radius;hi=points.max(axis=0)+radius;tri=np.asarray(mesh.triangles)
    near=np.all(tri[:,:,:2].max(axis=1)>=lo,axis=1)&np.all(tri[:,:,:2].min(axis=1)<=hi,axis=1)
    t=tri[near];norm=np.asarray(mesh.face_normals)[near];support=(np.min(t[:,:,2],axis=1)>=floor_z-.03)&(np.max(t[:,:,2],axis=1)<=floor_z+.03)&(np.abs(norm[:,2])>=math.cos(.1))
    overlap_height=(t[:,:,2].max(axis=1)>floor_z+.03)&(t[:,:,2].min(axis=1)<floor_z+2*radius+.5)
    obstructing=int(np.sum(overlap_height&~support));others=[]
    for c in obstacles:
        mn=np.array(c['minimum']);mx=np.array(c['maximum'])
        if np.any(mn>mx):return {'valid':False,'reason':'invalid_collider_bounds','collider':c['path']}
        if np.all(mx[:2]>=lo) and np.all(mn[:2]<=hi) and mx[2]>floor_z+.03 and mn[2]<floor_z+2*radius+.5:others.append(c['path'])
    uncovered=uncovered_rectangle(lo,hi,t[support])
    return {'valid':obstructing==0 and not others and uncovered<=1e-10,'reason':'checked_full_rectangle' if obstructing==0 and not others and uncovered<=1e-10 else 'obstacle_or_support_not_certified','rectangle_lo':lo.tolist(),'rectangle_hi':hi.tolist(),'radius_m':radius,'floor_z':floor_z,'support_triangles':int(support.sum()),'obstacle_triangles':obstructing,'other_obstacles':others,'uncovered_support_area_m2':uncovered,'support_height_band_m':.03,'max_support_slope_rad':.1}

def subtract_support(poly,triangles):
    remainder=[np.asarray(poly,float)]
    for tri in triangles:
        tri=np.asarray(tri)[:,:2]
        if area(tri)<1e-12:continue
        a,b=tri[1]-tri[0],tri[2]-tri[0]
        if a[0]*b[1]-a[1]*b[0]<0:tri=tri[::-1]
        parts=[]
        for original in remainder:
            pending=original
            for a,b in zip(tri,np.roll(tri,-1,axis=0)):
                outside=clip(pending,a,b,False)
                if area(outside)>1e-12:parts.append(outside)
                pending=clip(pending,a,b,True)
                if area(pending)<=1e-12:break
        remainder=parts
        if not remainder:break
        if len(remainder)>10000:raise ValueError('support union complexity bound')
    return sum(area(p) for p in remainder)

def intersect_convex(poly,other):
    for a,b in zip(other,np.roll(other,-1,axis=0)):
        poly=clip(poly,a,b,True)
        if len(poly)<3:break
    return area(poly)

def connected_sweep_certificate(mesh,points,radius,floor_face,obstacles):
    """Circumscribed sweep polygon; slope-limited connected floor faces, no filled holes."""
    from scipy.spatial import ConvexHull
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    points=np.asarray(points)[:,:2];angles=np.arange(32)*2*math.pi/32
    ring=np.c_[np.cos(angles),np.sin(angles)]*(radius/math.cos(math.pi/32))
    cloud=(points[:,None,:]+ring[None,:,:]).reshape(-1,2);poly=cloud[ConvexHull(cloud).vertices]
    normals=np.asarray(mesh.face_normals);allowed=np.abs(normals[:,2])>=math.cos(.1)
    if not allowed[floor_face]:return {'valid':False,'reason':'source_support_slope_exceeds_limit'}
    adjacency=np.asarray(mesh.face_adjacency);edges=adjacency[np.all(allowed[adjacency],axis=1)]
    matrix=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(normals),len(normals))).tocsr();_,labels=connected_components(matrix,directed=False)
    component=allowed&(labels==labels[floor_face]);tri=np.asarray(mesh.triangles);lo=poly.min(axis=0);hi=poly.max(axis=0)
    near=np.all(tri[:,:,:2].max(axis=1)>=lo,axis=1)&np.all(tri[:,:,:2].min(axis=1)<=hi,axis=1)
    support=tri[near&component];uncovered=subtract_support(poly,support);floor_min=float(support[:,:,2].min()) if len(support) else None;floor_max=float(support[:,:,2].max()) if len(support) else None
    obstacle_ids=[]
    for i in np.flatnonzero(near&~component):
        t=tri[i]
        if floor_min is not None and (t[:,2].max()<floor_min-.03 or t[:,2].min()>floor_max+2*radius+.5):continue
        xy=t[:,:2]
        # Degenerate projected walls still require rejection when their bounding rectangle touches the sweep.
        mn=xy.min(axis=0)-1e-7;mx=xy.max(axis=0)+1e-7;box=np.array([[mn[0],mn[1]],[mx[0],mn[1]],[mx[0],mx[1]],[mn[0],mx[1]]])
        if intersect_convex(poly,box)>1e-12:obstacle_ids.append(int(i))
    other=[]
    for c in obstacles:
        mn=np.asarray(c['minimum']);mx=np.asarray(c['maximum'])
        if np.any(mn>mx):return {'valid':False,'reason':'invalid_collider_bounds','collider':c['path']}
        if floor_min is not None and (mx[2]<floor_min-.03 or mn[2]>floor_max+2*radius+.5):continue
        box=np.array([[mn[0],mn[1]],[mx[0],mn[1]],[mx[0],mx[1]],[mn[0],mx[1]]])
        if intersect_convex(poly,box)>1e-12:other.append(c['path'])
    valid=uncovered<=1e-10 and not obstacle_ids and not other
    return {'schema':'connected-floor-circumscribed-sweep-v2','valid':valid,'reason':'certified_connected_support_sweep' if valid else 'obstacle_or_connected_support_not_certified','radius_m':radius,'circumscribed_polygon_xy':poly.tolist(),'support_triangles':len(support),'source_floor_face':int(floor_face),'source_connected_component_faces':int(component.sum()),'floor_height_range_m':[floor_min,floor_max],'max_support_slope_rad':.1,'uncovered_support_area_m2':uncovered,'obstacle_triangle_ids':obstacle_ids,'other_obstacles':other,'map_interpolation_used':False}
