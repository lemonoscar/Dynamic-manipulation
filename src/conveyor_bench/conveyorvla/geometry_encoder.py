"""Calibrated metric depth candidate. Invalid pixels never become valid zero depth."""
import torch
from torch import nn


def project_depth(depth, valid, intrinsics, camera_to_base, *, definition, unit_scale=1., pixel_offset=0.):
    if definition not in {'z_depth', 'ray_range'} or unit_scale <= 0:
        raise ValueError('explicit z_depth/ray_range and positive unit scale required')
    if depth.ndim != 3 or valid.shape != depth.shape or valid.dtype != torch.bool:
        raise ValueError('depth/valid must be B x H x W with boolean mask')
    batch, height, width = depth.shape
    if intrinsics.shape != (batch,3,3) or camera_to_base.shape != (batch,4,4):
        raise ValueError('invalid K/T shapes')
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(camera_to_base).all():
        raise ValueError('nonfinite calibration')
    rotation = camera_to_base[:,:3,:3]
    eye = torch.eye(3,device=depth.device,dtype=depth.dtype).expand(batch,-1,-1)
    if not torch.allclose(rotation.transpose(1,2)@rotation, eye, atol=1e-4) or not torch.allclose(torch.det(rotation),torch.ones(batch,device=depth.device),atol=1e-4):
        raise ValueError('extrinsics must be a proper rigid transform')
    if not torch.allclose(camera_to_base[:,3],torch.tensor([0,0,0,1],device=depth.device,dtype=depth.dtype).expand(batch,-1)):
        raise ValueError('invalid homogeneous transform')
    if (intrinsics[:,0,0] <= 0).any() or (intrinsics[:,1,1] <= 0).any():
        raise ValueError('nonpositive focal length')
    y,x=torch.meshgrid(torch.arange(height,device=depth.device),torch.arange(width,device=depth.device),indexing='ij')
    pixels=torch.stack((x+pixel_offset,y+pixel_offset,torch.ones_like(x)),dim=-1).to(depth).reshape(1,-1,3).expand(batch,-1,-1)
    rays=pixels @ torch.linalg.inv(intrinsics).transpose(1,2)
    if definition=='ray_range':
        rays=rays/torch.linalg.vector_norm(rays,dim=-1,keepdim=True)
    valid=valid & torch.isfinite(depth) & (depth>0)
    metric=torch.where(valid,depth*unit_scale,0).flatten(1)
    points=(rays*metric[...,None])@rotation.transpose(1,2)+camera_to_base[:,None,:3,3]
    points=torch.where(valid.flatten(1)[...,None],points,0.)
    return points,valid.flatten(1)


class GeometryEncoder(nn.Module):
    def __init__(self, hidden_dim, token_budget=16):
        super().__init__()
        self.token_budget=token_budget
        self.projection=nn.Sequential(nn.Linear(3,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))

    def forward(self, points, valid):
        if points.ndim!=3 or points.shape[-1]!=3 or valid.shape!=points.shape[:2]:
            raise ValueError('point/mask shape mismatch')
        # Fixed raster bins, masked average; missing bins retain false masks.
        tokens=[];masks=[]
        for p,m in zip(torch.tensor_split(points,self.token_budget,dim=1),torch.tensor_split(valid,self.token_budget,dim=1)):
            safe=torch.where(m[...,None],p,0.)
            mean=safe.sum(1)/m.sum(1,keepdim=True).clamp_min(1)
            active=m.any(1)
            tokens.append(torch.where(active[:,None],self.projection(mean),0.))
            masks.append(active)
        return torch.stack(tokens,1),torch.stack(masks,1)
