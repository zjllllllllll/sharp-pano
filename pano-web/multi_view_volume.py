"""Union of accepted RGB-D free volumes; never AND projected 2-D masks."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'room-outline-experiment'))
from occlusion import RGBDVolume,erp_rays

VERSION='accepted-free-volume-union-v1'


class PlacedVolume:
    def __init__(self,volume,position):
        self.volume=volume
        self.position=np.asarray(position,np.float32)

    def boundary(self,points,directions):
        v=self.volume
        hit=v.intersect(v.boundary,np.asarray(points-self.position,np.float32),directions,
                        v.wall_triangles,v.wall_start,v.validation_lookup)
        return np.where(hit['primID']>=0,hit['tfar'],np.inf)

    def contains(self,points):
        local=np.asarray(points-self.position,np.float32)
        radius=np.linalg.norm(local,axis=1)
        rays=local/np.maximum(radius[:,None],1e-12)
        rays[radius<1e-7]=[0,0,1]
        origin=np.broadcast_to(self.position,points.shape)
        boundary=self.boundary(origin,rays)
        return (radius<1e-7)|(np.isfinite(boundary)&(radius<boundary-1e-5))

    def surface(self,points,directions):
        if not self.volume.has_surface:return np.full(len(points),np.inf)
        hit=self.volume.intersect(self.volume.surface,np.asarray(points-self.position,np.float32),directions)
        return np.where(hit['primID']>=0,hit['tfar'],np.inf)


def covered_segments(volumes,origin,directions,targets,epsilon=1e-4,max_crossings=128):
    """Extend continuous coverage at exact boundary events, not fixed ray steps.

    A gap in one volume can be bridged by another. A gap in their UNION cannot.
    Unresolved/numerically degenerate paths remain unknown conservatively.
    """
    origin=np.asarray(origin,np.float32)
    targets=np.asarray(targets)
    distance=np.zeros(len(directions),np.float64)
    known=np.zeros(len(directions),bool)
    active=np.flatnonzero(np.isfinite(targets)&(targets>0))
    for _ in range(max_crossings):
        if not len(active):break
        points=(origin+distance[active,None]*directions[active]).astype(np.float32)
        reach=distance[active].copy()
        for volume in volumes:
            inside=volume.contains(points)
            if not inside.any():continue
            exits=volume.boundary(points[inside],directions[active[inside]])
            extension=np.where(np.isfinite(exits),exits,0)
            reach[inside]=np.maximum(reach[inside],distance[active[inside]]+extension)
        arrived=reach>=targets[active]-epsilon
        known[active[arrived]]=True
        advancing=(reach>distance[active]+epsilon)&~arrived
        distance[active[advancing]]=reach[advancing]+epsilon
        active=active[advancing]
    return known


class AccumulatedVolume:
    def __init__(self,volumes):
        if not volumes:raise ValueError('At least one accepted observation is required')
        self.volumes=volumes

    def mask(self,position,width=1024,height=512,supersampling=2):
        directions=erp_rays(width*supersampling,height*supersampling).reshape(-1,3)
        output=[]
        first=self.volumes[0]
        for start in range(0,len(directions),65536):
            rays=directions[start:start+65536]
            _,reason,ts,_,_=first.volume.query(np.asarray(position)-first.position,rays)
            unknown=reason!=0
            ids=np.flatnonzero(unknown)
            if len(ids):
                origins=np.broadcast_to(np.asarray(position,np.float32),(len(ids),3))
                targets=ts[ids].copy()
                for volume in self.volumes[1:]:targets=np.minimum(targets,volume.surface(origins,rays[ids]))
                unknown[ids]=~covered_segments(self.volumes,position,rays[ids],targets)
            output.append(unknown)
        s=supersampling
        return np.concatenate(output).reshape(height,s,width,s).any(axis=(1,3))
