"""RGB-D free-space boundary queries for novel-view conservative repair masks.

The radial depth field defines a star-shaped observed free volume.  A BVH of
its boundary finds the *first exit*, without fixed ray-marching steps that
could skip narrow occluders.  This boundary is only an acceleration proxy,
never a reconstructed object or geometry to be fed back to SHARP.

Every valid depth pixel retains its angular footprint, including silhouettes.
Discontinuous neighbours have separate surface patches. Radial side walls
close ONLY the unknown-volume boundary; they are never observed surfaces.
Geometry is a discrete approximation, not ground-truth occupancy.
"""
from pathlib import Path
from dataclasses import dataclass, asdict
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
from embreex.rtcore_scene import EmbreeScene
from embreex.mesh_construction import TriangleMesh

VERSION = 'unknown-volume-v2-footprints'
REASONS = {1: 'no_reliable_surface', 2: 'unknown_before_surface',
           4: 'invalid_or_filtered_path', 8: 'camera_not_confirmed_free'}
BACKGROUND = np.array([0, 255, 0], np.uint8)


@dataclass(frozen=True)
class Config:
    proxy_max_width: int = 2048
    depth_jump_abs_m: float = .05
    depth_jump_rel: float = .03
    surface_epsilon_m: float = .02
    surface_epsilon_rel: float = .005
    comparison_epsilon_m: float = .001
    supersampling: int = 2
    invalid_near_m: float = .001


def erp_rays(width, height):
    lon = ((np.arange(width) + .5) / width - .5) * (2*np.pi)
    lat = ((np.arange(height) + .5) / height - .5) * np.pi
    x = np.cos(lat)[:, None] * np.sin(lon)[None, :]
    y = np.broadcast_to(np.sin(lat)[:, None], (height, width))
    z = np.cos(lat)[:, None] * np.cos(lon)[None, :]
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def pixel_patches(depth, valid, config):
    """Cover every pixel's solid angle without bridging a depth discontinuity.

    Continuous corners are shared averages. At a discontinuity each pixel
    keeps its own depth instead. Connections between differing corner radii
    lie on angular cell edges and belong ONLY to the unknown boundary.
    """
    h, w = depth.shape
    north=np.concatenate((depth[:1],depth),axis=0)
    south=np.concatenate((depth,depth[-1:]),axis=0)
    samples=np.stack((np.roll(north,1,axis=1),north,np.roll(south,1,axis=1),south))
    low=samples.min(axis=0); high=samples.max(axis=0)
    shared=samples.mean(axis=0)
    vn=np.concatenate((valid[:1],valid),axis=0)
    vs=np.concatenate((valid,valid[-1:]),axis=0)
    continuous=(vn & vs & np.roll(vn,1,axis=1) & np.roll(vs,1,axis=1) &
                ((high-low)<=np.maximum(config.depth_jump_abs_m,config.depth_jump_rel*low)))
    # A pole is one direction, not w different observations. Use a shared
    # pole only if the whole polar row is continuous and valid.
    for row,source in ((0,0),(-1,-1)):
        continuous[row]=valid[source].all() and np.ptp(depth[source])<=max(
            config.depth_jump_abs_m,config.depth_jump_rel*depth[source].min())
        shared[row]=depth[source].mean()
    lon=(np.arange(w)/w-.5)*(2*np.pi)
    lat=(np.arange(h+1)/h-.5)*np.pi
    directions=np.stack((np.cos(lat)[:,None]*np.sin(lon)[None,:],
        np.broadcast_to(np.sin(lat)[:,None],(h+1,w)),
        np.cos(lat)[:,None]*np.cos(lon)[None,:]),axis=-1).astype(np.float32)
    directions[0]=[0,-1,0]; directions[-1]=[0,1,0]
    # TL, TR, BL, BR: identical direction bits on both sides of every edge.
    corners=np.stack((directions[:-1],np.roll(directions[:-1],-1,axis=1),
                      directions[1:],np.roll(directions[1:],-1,axis=1)),axis=2)
    values=np.stack((shared[:-1],np.roll(shared[:-1],-1,axis=1),
                     shared[1:],np.roll(shared[1:],-1,axis=1)),axis=2)
    smooth=np.stack((continuous[:-1],np.roll(continuous[:-1],-1,axis=1),
                     continuous[1:],np.roll(continuous[1:],-1,axis=1)),axis=2)
    radii=np.where(smooth,values,depth[...,None]).astype(np.float32)
    ids=np.arange(h*w*4,dtype=np.int32).reshape(h,w,4)
    faces=np.stack((ids[:,:,[0,1,2]],ids[:,:,[1,3,2]]),axis=2)
    keep=np.ones((h,w,2),bool); keep[0,:,0]=False; keep[-1,:,1]=False
    faces=np.ascontiguousarray(faces[keep])
    observed=np.broadcast_to(valid[...,None],(h,w,2))[keep].copy()
    # Radial walls close step discontinuities without widening their angles.
    walls=[];uncertain=[]
    pairs=((ids[:,:,[1,3]],np.roll(ids[:,:,[0,2]],-1,axis=1),
            radii[:,:,[1,3]],np.roll(radii[:,:,[0,2]],-1,axis=1),
            ~valid | ~np.roll(valid,-1,axis=1)),
           (ids[:-1,:,[2,3]],ids[1:,:,[0,1]],
            radii[:-1,:,[2,3]],radii[1:,:,[0,1]],~valid[:-1] | ~valid[1:]))
    for a,b,ra,rb,invalid in pairs:
        different=np.any(ra!=rb,axis=-1)
        a=a[different];b=b[different]
        walls.append(np.stack((np.stack((a[:,0],b[:,0],a[:,1]),axis=-1),
                               np.stack((b[:,0],b[:,1],a[:,1]),axis=-1)),axis=1).reshape(-1,3))
        uncertain.append(np.repeat(invalid[different],2))
    return (corners.reshape(-1,3),radii.reshape(-1),faces,observed,
            np.ascontiguousarray(np.concatenate(walls)),np.concatenate(uncertain),
            float((~smooth).any(axis=-1).mean()))


class RGBDVolume:
    def __init__(self, depth, rgb, valid=None, config=None):
        started = time.perf_counter()
        self.config = config or Config()
        cfg = self.config
        depth = np.asarray(depth, np.float32)
        if depth.ndim != 2 or depth.shape[1] != depth.shape[0]*2:
            raise ValueError('Expected 2:1 radial ERP depth')
        self.rgb = np.asarray(rgb, np.uint8)
        if self.rgb.shape != (*depth.shape, 3):
            raise ValueError('RGB and radial depth must be pixel-aligned')
        valid = np.isfinite(depth) & (depth > 0) if valid is None else (
            np.asarray(valid, bool) & np.isfinite(depth) & (depth > 0))
        # Block reduction preserves even a one-source-pixel foreground in the
        # unknown boundary. Never average a foreground/background discontinuity.
        factor = max(1, int(np.ceil(depth.shape[1]/cfg.proxy_max_width)))
        if depth.shape[0] % factor or depth.shape[1] % factor:
            factor = 1
        h, w = depth.shape[0]//factor, depth.shape[1]//factor
        blocks = np.where(valid, depth, np.inf).reshape(h,factor,w,factor)
        low = blocks.min(axis=(1,3))
        high = np.where(valid, depth, -np.inf).reshape(h,factor,w,factor).max(axis=(1,3))
        good = valid.reshape(h,factor,w,factor).all(axis=(1,3))
        average = np.where(valid,depth,0).reshape(h,factor,w,factor).mean(axis=(1,3))
        jump = (high-low) > np.maximum(cfg.depth_jump_abs_m,cfg.depth_jump_rel*low)
        d = np.where(good & ~jump, average, low)
        d = np.where(np.isfinite(d), d, cfg.invalid_near_m).astype(np.float32)
        d[~good]=cfg.invalid_near_m
        dirs,radii,faces,observed,walls,wall_uncertain,split_fraction=pixel_patches(d,good,cfg)
        self.surface = EmbreeScene(robust=True)
        self.has_surface = bool(observed.any())
        if self.has_surface:
            TriangleMesh(self.surface,np.ascontiguousarray(dirs*radii[:,None]),
                         np.ascontiguousarray(faces[observed]))
        boundary_range=radii+cfg.surface_epsilon_m+cfg.surface_epsilon_rel*radii
        boundary_range[np.repeat(~good.ravel(),4)]=cfg.invalid_near_m
        self.boundary = EmbreeScene(robust=True)
        boundary_vertices=np.ascontiguousarray(dirs*boundary_range[:,None])
        TriangleMesh(self.boundary,boundary_vertices,
                     np.ascontiguousarray(np.concatenate((faces,walls))))
        self.wall_start=len(faces)
        self.wall_triangles=boundary_vertices[walls].copy()
        self.boundary_uncertain=np.concatenate((~observed,wall_uncertain))
        self.validation_lookup=None
        if not good.all():
            # Near-camera invalid caps and their long radial walls need a
            # relative, not centimetre-scale, intersection residual check.
            checked=np.concatenate((np.flatnonzero(~observed),
                                    np.arange(len(faces),len(faces)+len(walls))))
            self.validation_lookup=np.full(len(faces)+len(walls),-1,np.int32)
            self.validation_lookup[checked]=np.arange(len(checked),dtype=np.int32)
            self.wall_triangles=np.concatenate((boundary_vertices[faces[~observed]],self.wall_triangles))
        self.stats = dict(source_size_wh=[depth.shape[1],depth.shape[0]], proxy_size_wh=[w,h],
                          source_valid_fraction=float(valid.mean()),
                          surface_triangles=int(observed.sum()),boundary_triangles=len(faces)+len(walls),
                          boundary_only_wall_triangles=len(walls),split_pixel_fraction=split_fraction,
                          surface_representation='complete angular pixel footprints; discontinuity walls unknown-only',
                          config=asdict(cfg), version=VERSION)
        # Commit both BVHs during construction so timing is not charged to C001.
        zero=np.zeros((1,3),np.float32); forward=np.array([[0,0,1]],np.float32)
        self.boundary.run(zero,forward)
        if self.has_surface:self.surface.run(zero,forward)
        self.stats['build_seconds']=time.perf_counter()-started

    @classmethod
    def from_job(cls, job, config=None):
        job=Path(job)
        with Image.open(job/'input.png') as image: rgb=np.asarray(image.convert('RGB'))
        # This offline batch has raw RGB-D, no cleaning sidecar. A supplied
        # depth-valid.npy must be ERP-aligned; removed samples remain unknown.
        valid_path=job/'depth-valid.npy'
        valid=np.load(valid_path,allow_pickle=False) if valid_path.exists() else None
        result=cls(np.load(job/'depth.npy',allow_pickle=False),rgb,valid,config)
        result.stats['validity_source']='depth-valid.npy' if valid_path.exists() else 'finite positive raw depth; no cleanup mask in this batch'
        return result

    def camera_state(self, position):
        pos=np.asarray(position,np.float32)
        r=float(np.linalg.norm(pos))
        if r < 1e-7:return 'known_free'
        hit=self.intersect(self.boundary,np.zeros((1,3),np.float32),(pos/r).reshape(1,3),
                           self.wall_triangles,self.wall_start,self.validation_lookup)
        idx=int(hit['primID'][0]); distance=float(hit['tfar'][0])
        if idx<0:return 'unconfirmed'
        if r < distance-self.config.comparison_epsilon_m:return 'known_free'
        return 'unconfirmed' if self.boundary_uncertain[idx] else 'unknown'

    @staticmethod
    def intersect(scene, origins, directions, wall_triangles=None, wall_start=0, validation_lookup=None):
        hit=scene.run(origins,directions,output=True)
        current=directions.copy()
        def rejected():
            bad=hit['primID']<0
            if wall_triangles is not None:
                if validation_lookup is None:
                    idx=np.flatnonzero(hit['primID']>=wall_start)
                    triangle_ids=hit['primID'][idx]-wall_start
                else:
                    idx=np.flatnonzero((hit['primID']>=0)&(validation_lookup[np.maximum(hit['primID'],0)]>=0))
                    triangle_ids=validation_lookup[hit['primID'][idx]]
                if len(idx):
                    tri=wall_triangles[triangle_ids].astype(np.float64)
                    u=hit['u'][idx,None].astype(np.float64);v=hit['v'][idx,None].astype(np.float64)
                    point=tri[:,0]*(1-u-v)+tri[:,1]*u+tri[:,2]*v
                    ray_point=origins[idx].astype(np.float64)+current[idx].astype(np.float64)*hit['tfar'][idx,None]
                    tolerance=1e-5+2e-5*np.linalg.norm(point,axis=1)
                    if validation_lookup is not None:
                        tolerance=1e-9+1e-6*np.linalg.norm(point,axis=1)
                    bad[idx]=np.linalg.norm(point-ray_point,axis=1)>tolerance
            return bad
        # Exact vertex/edge rays can miss under float32 arithmetic even with
        # robust mode. Near-coplanar radial walls can also produce impossible
        # hits (ray point is nowhere near the barycentric triangle point).
        # Validate in float64 and retry only rejected rays, up to ~0.01 pixel.
        # This is identical at every camera, never an origin-view exemption.
        scales=(1.,-1.,10.,-10.,100.,-100.) if wall_triangles is not None else (1.,-1.)
        for sign in scales:
            missing=np.flatnonzero(rejected())
            if not len(missing):break
            rays=directions[missing].copy()
            rays += sign*np.array([1.7e-7,2.3e-7,1.1e-7],np.float32)
            rays /= np.linalg.norm(rays,axis=1,keepdims=True)
            retry=scene.run(origins[missing],rays,output=True)
            for key in hit:hit[key][missing]=retry[key]
            current[missing]=rays
        if validation_lookup is not None:
            # Long walls ending at a 1mm invalid cap can repeatedly return a
            # false near-origin intersection even after angular retries. Walk
            # past ONLY rejected hits (10 micrometres, below the 1mm comparison
            # tolerance), then query the next hit along the ORIGINAL ray.
            # Never skip a physically consistent hit or declare a miss free.
            offsets=np.zeros(len(directions),np.float64)
            for _ in range(256):
                bad=np.flatnonzero(rejected() & (hit['primID']>=0))
                if not len(bad):break
                offsets[bad]=np.maximum(offsets[bad],hit['tfar'][bad])+1e-5
                moved=(origins[bad].astype(np.float64)+directions[bad]*offsets[bad,None]).astype(np.float32)
                retry=scene.run(moved,directions[bad],output=True)
                retry['tfar']+=offsets[bad].astype(np.float32)
                for key in hit:hit[key][bad]=retry[key]
                current[bad]=directions[bad]
        hit['primID'][rejected()]=-1  # unresolved queries remain uncertain
        return hit

    def query(self, position, directions):
        """Two exact BVH first-hit queries; never test only the final endpoint."""
        directions=np.ascontiguousarray(directions,dtype=np.float32).reshape(-1,3)
        origins=np.broadcast_to(np.asarray(position,np.float32),directions.shape).copy()
        n=len(directions)
        if self.has_surface:
            surface=self.intersect(self.surface,origins,directions)
            ts=np.where(surface['primID']>=0,surface['tfar'],np.inf)
        else:ts=np.full(n,np.inf,np.float32)
        boundary=self.intersect(self.boundary,origins,directions,self.wall_triangles,self.wall_start,self.validation_lookup)
        ids=boundary['primID']; found=ids>=0
        tb=np.where(found,boundary['tfar'],np.inf)
        reasons=np.zeros(n,np.uint8)
        reasons[~np.isfinite(ts)] |= 1
        before=found & (tb+self.config.comparison_epsilon_m < ts)
        unsure=found & self.boundary_uncertain[np.maximum(ids,0)]
        reasons[before & ~unsure] |= 2
        reasons[before & unsure] |= 4
        reasons[~found] |= 4
        state=self.camera_state(position)
        if state != 'known_free':reasons |= 8
        rgb=np.tile(BACKGROUND,(n,1))
        seen=np.isfinite(ts)
        if seen.any():
            points=origins[seen]+directions[seen]*ts[seen,None]
            # Color is observed RGB, not SHARP's second layer or synthesized data.
            h,w=self.rgb.shape[:2]
            u=(np.arctan2(points[:,0],points[:,2])/(2*np.pi)+.5)*w
            v=(np.arctan2(points[:,1],np.hypot(points[:,0],points[:,2]))/np.pi+.5)*h
            rgb[seen]=self.rgb[np.clip(v.astype(np.int32),0,h-1),u.astype(np.int32)%w]
        return rgb,reasons,ts,tb,state

    def render(self, position, width=1024, height=512):
        started=time.perf_counter(); s=self.config.supersampling
        directions=erp_rays(width*s,height*s).reshape(-1,3)
        # Limit temporary query buffers independently of source cloud size.
        colors=[];reasons=[];surface_depth=[];boundary_depth=[]
        for start in range(0,len(directions),262144):
            rgb,reason,ts,tb,state=self.query(position,directions[start:start+262144])
            colors.append(rgb);reasons.append(reason);surface_depth.append(ts);boundary_depth.append(tb)
        raw=np.concatenate(colors).reshape(height,s,width,s,3).mean(axis=(1,3)).round().astype(np.uint8)
        reason=np.bitwise_or.reduce(np.bitwise_or.reduce(
            np.concatenate(reasons).reshape(height,s,width,s),axis=3),axis=1)
        # Pixel footprint is conservative: ANY masked sub-ray masks the pixel.
        mask=reason!=0; masked=raw.copy();masked[mask]=BACKGROUND
        palette=np.zeros((height,width,3),np.uint8)
        palette[reason&1!=0]=[245,186,64]
        palette[reason&2!=0]=[255,77,99]
        palette[reason&4!=0]=[174,104,245]
        palette[reason&8!=0]=[76,160,255]
        overlay=raw.copy();overlay[mask]=np.rint(.4*raw[mask]+.6*palette[mask]).astype(np.uint8)
        def reduce_depth(items):
            return np.concatenate(items).reshape(height,s,width,s).min(axis=(1,3))
        stats=dict(camera_state=state,mask_fraction=float(mask.mean()),
                   coverage=float((~mask).mean()),surface_coverage=float((reason&1==0).mean()),
                   reasons={name:float((reason&bit!=0).mean()) for bit,name in REASONS.items()},
                   render_seconds=time.perf_counter()-started)
        return dict(raw=raw,masked=masked,mask=mask.astype(np.uint8)*255,overlay=overlay,
                    reason=reason,reason_color=palette,coverage=(~mask).astype(np.uint8)*255,
                    unknown=(reason&2!=0).astype(np.uint8)*255,
                    uncertain=(reason&12!=0).astype(np.uint8)*255,
                    holes=(reason&1!=0).astype(np.uint8)*255,
                    surface_depth=reduce_depth(surface_depth),boundary_depth=reduce_depth(boundary_depth),stats=stats)
