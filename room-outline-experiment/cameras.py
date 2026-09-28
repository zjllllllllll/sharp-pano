"""Exact inward-offset lattice, diagnostic figure and RGB-D mask previews."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
from shapely.geometry import Polygon, Point, mapping
from shapely import make_valid

SPACING = .5
INSET = .5
WIDTH, HEIGHT = 1024, 512
NUMBERING = 'origin-zero-square-clockwise-v1'
SAMPLING_VERSION = 'adaptive-eight-neighbors-v1'
OFFSET_STEP = .05
OFFSET_STOP = .1
MIN_NEIGHBORS = 4
from candidate_selection import MASK_THRESHOLD, apply_selection, write_candidates


class SamplingError(ValueError):
    def __init__(self, info):
        self.info = info
        super().__init__(info['sampling_error'])


def square_order(camera):
    """Top is +Z in draw_camera_map; begin at (0,+r), clockwise in X/Z."""
    x,z=camera['grid'];r=max(abs(x),abs(z))
    if r==0:return (0,0)
    if z==r and x>=0:t=x
    elif x==r:t=r+(r-z)
    elif z==-r:t=3*r+(r-x)
    elif x==-r:t=5*r+(z+r)
    else:t=7*r+(x+r)
    return r,t


def number_cameras(info):
    """Renumber without moving samples; always include the original reference."""
    from copy import deepcopy
    from shapely.geometry import shape
    info=deepcopy(info)
    cameras=info['cameras']
    for camera in cameras:
        camera.setdefault('previous_id',camera['id'])
        camera.setdefault('reference_only',False)
    if not any(c['grid']==[0,0] for c in cameras):
        cameras.append(dict(grid=[0,0],position_world_m=[0.,0.,0.],
            boundary_distance_m=float(Point(0,0).distance(shape(info['room_geometry']).boundary)),
            coincides_with_original=True,reference_only=True,previous_id=None))
    cameras.sort(key=square_order)
    for i,camera in enumerate(cameras):
        camera['id']=f'C{i:03d}'
        camera['square_ring'],camera['ring_clockwise_index']=square_order(camera)
    info.update(version=2,numbering=NUMBERING,cameras=cameras,count=len(cameras),
        sampled_camera_count=sum(not c['reference_only'] for c in cameras),
        grid_anchor='Original camera C000; square rings outward; +Z top, clockwise',
        numbering_note='Reference C000 is retained even outside the inset; other sample positions unchanged.')
    return info


def polygons(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == 'Polygon':
        return [geometry]
    return [p for g in getattr(geometry, 'geoms', []) for p in polygons(g)]


def _sample_fixed(vertices, spacing=SPACING, inset=INSET):
    raw = Polygon(vertices)
    if raw.is_empty or raw.area <= 0:
        raise ValueError('Empty room outline')
    repaired = not raw.is_valid
    shape = make_valid(raw) if repaired else raw
    parts = polygons(shape)
    if not parts:
        raise ValueError('Outline repair produced no polygon')
    # Preserve all valid room pieces, but never reinterpret repair as evidence.
    from shapely.ops import unary_union
    room = unary_union(parts)
    inner = room.buffer(-inset, quad_segs=32, join_style='round')
    cameras = []
    if not inner.is_empty:
        xmin, zmin, xmax, zmax = inner.bounds
        for iz in range(int(np.ceil(zmin/spacing)), int(np.floor(zmax/spacing))+1):
            for ix in range(int(np.ceil(xmin/spacing)), int(np.floor(xmax/spacing))+1):
                x, z = ix*spacing, iz*spacing
                point = Point(x,z)
                distance = point.distance(room.boundary)
                if room.contains(point) and inner.covers(point) and distance+1e-9 >= inset:
                    cameras.append(dict(id=f'C{len(cameras)+1:03d}', grid=[ix,iz],
                                        position_world_m=[x,0.,z], boundary_distance_m=float(distance),
                                        coincides_with_original=(ix==0 and iz==0)))
    return number_cameras(dict(version=1, spacing_m=spacing, inward_offset_m=inset,
                height_world_y_m=0., origin_world_m=[0.,0.,0.],
                grid_anchor='Original camera, X/Z lattice; row-major Z then X',
                outline_repaired=repaired, room_geometry=mapping(room), inset_geometry=mapping(inner),
                cameras=cameras, count=len(cameras),
                empty_reason=('offset_erased_room' if inner.is_empty else 'no_grid_node_inside_offset') if not cameras else None,
                safety_note='Only user-requested contour offset and grid are enforced; no furniture/unknown-space collision certification.'))


def sample_cameras(vertices, spacing=SPACING, inset=INSET, adaptive=True):
    """Relax geometry only; fixed Mask/component selection happens after render."""
    if not np.isfinite(spacing) or spacing <= 0 or not np.isfinite(inset) or inset < 0:
        raise ValueError('Spacing must be positive and offset must be non-negative')
    attempts=[]
    current=float(inset)
    while True:
        info=_sample_fixed(vertices, spacing, current)
        neighbors=sum(max(abs(v) for v in c['grid'])==1 for c in info['cameras'])
        attempts.append(dict(offset_m=current,neighbor_count=neighbors,camera_count=info['count']))
        if not adaptive or neighbors>=MIN_NEIGHBORS or current<OFFSET_STOP:
            break
        current=max(0.,round(current-OFFSET_STEP,8))
    info.update(sampling_version=SAMPLING_VERSION,requested_offset_m=float(inset),
        adaptive_offset=bool(adaptive),offset_step_m=OFFSET_STEP,offset_stop_below_m=OFFSET_STOP,
        minimum_neighbor_count=MIN_NEIGHBORS,neighbor_count=neighbors,offset_attempts=attempts,
        offset_stop_reason='enough_neighbors' if neighbors>=MIN_NEIGHBORS else ('below_min_offset' if adaptive else 'fixed'),
        sampling_error=None,sampling_warning=None,
        mask_filter=dict(default_threshold=MASK_THRESHOLD,metric='ERP masked pixels / all pixels',
            comparison='strictly_greater',origin_exempt=True))
    if adaptive and info['count']==1:
        info['sampling_error']=f'采样失败：offset 已降至 {current:g} m，仍只有原相机 C000，没有其他候选点。'
        raise SamplingError(info)
    if adaptive and neighbors<MIN_NEIGHBORS:
        info['sampling_warning']=f'offset 已降至 {current:g} m，周围八格仍仅有 {neighbors}/8 个候选点；保留现有候选，未达到 4 个目标。'
    return info


def draw_camera_map(job, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import shape
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    with np.load(job/'bev-evidence.npz') as data:
        counts = data['count'].astype(float)
        origin, grid = data['origin'], float(data['grid'])
    density = np.log1p(counts)
    density = np.clip(density / max(float(np.quantile(density[density>0],.97)),1e-6),0,1)
    room, inner = shape(info['room_geometry']), shape(info['inset_geometry'])
    # Keep adjacent 0.5 m lattice labels legible in large-room source images.
    # Font sizes stay fixed while the physical canvas grows with room extent.
    xmin,zmin,xmax,zmax=room.bounds
    figure_size=max(10.,min(32.,max(xmax-xmin,zmax-zmin)*.95+1.6))
    fig,ax=plt.subplots(figsize=(figure_size,figure_size),dpi=180,facecolor='#101716')
    ax.set_facecolor('#101716')
    h,w=counts.shape
    ax.imshow(density,origin='lower',extent=[origin[0],origin[0]+w*grid,origin[1],origin[1]+h*grid],cmap='gray',vmin=0,vmax=1)
    outline=json.loads((job/'outline.json').read_text(encoding='utf-8'))
    for row in outline['segments']:
        a,b=row['a'],row['b']
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#b6f76c' if row['kind']=='supported' else '#ffb55a',
                lw=1.8,ls='-' if row['kind']=='supported' else '--')
    for part in polygons(inner):
        xy=np.asarray(part.exterior.coords)
        ax.fill(xy[:,0],xy[:,1],color='#55bfff',alpha=.16)
        ax.plot(xy[:,0],xy[:,1],color='#55bfff',lw=1.7)
        for hole in part.interiors:
            xy=np.asarray(hole.coords)
            ax.plot(xy[:,0],xy[:,1],color='#55bfff',lw=1.7)
    xmin,zmin,xmax,zmax=room.bounds
    xmin,zmin,xmax,zmax=min(xmin,0),min(zmin,0),max(xmax,0),max(zmax,0)
    margin=max(.6,max(xmax-xmin,zmax-zmin)*.08)
    ax.set_xlim(xmin-margin,xmax+margin);ax.set_ylim(zmin-margin,zmax+margin)
    ax.set_aspect('equal');ax.grid(alpha=.16)
    ax.set_xlabel('X / m',color='#c1d4c9');ax.set_ylabel('Z / m',color='#c1d4c9')
    ax.tick_params(colors='#aac1b4')
    for spine in ax.spines.values():spine.set_color('#4a6356')
    pool=info.get('all_cameras',info['cameras'])
    ax.set_title(f'{job.name} · 候选 {len(info["cameras"])} / 几何点 {len(pool)}（含 C000）',color='#e3eee6',fontsize=16,pad=16)
    error=info.get('sampling_error') or info.get('selection_error')
    if error:
        ax.text(.5,.8,error,transform=ax.transAxes,
                ha='center',va='center',color='#ffe1ab',fontsize=16,
                bbox=dict(facecolor='#142018',alpha=.9,pad=12))
    fig.text(.5,.035,f'绿色：支撑边界 / 橙色：推断边界  ·  蓝色：内缩 {info["inward_offset_m"]:g} m  ·  {info["spacing_m"]:g} m 点阵 / 原相机同高',
             ha='center',color='#c2d5c9',fontsize=10)
    fig.text(.5,.016,'固定 Mask ≤15% → 原点四邻接分量；绿点保留 / 红叉 Mask 超限 / 橙叉不连通 / 蓝星原点。非碰撞认证。',ha='center',color='#91a69a',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));fig.canvas.draw()
    width,height=fig.canvas.get_width_height()
    info['camera_map_projection']=dict(image_size=[width,height],xlim=list(ax.get_xlim()),zlim=list(ax.get_ylim()),
        axes_px=[float(ax.bbox.x0),float(height-ax.bbox.y1),float(ax.bbox.width),float(ax.bbox.height)])
    fig.savefig(job/'camera-map-base.png')
    for row in pool:
        x,_,z=row['position_world_m']
        state=row.get('selection_state','pending')
        removed=state in ('excluded','disconnected')
        color={'origin':'#55bfff','retained':'#b9f66f','excluded':'#ff7777','disconnected':'#ffb55a','pending':'#a4a9aa'}[state]
        ax.scatter([x],[z],s=30,c=color,marker='x' if removed else 'o',zorder=5)
        ax.annotate(row['id'],(x,z),xytext=(4,5),textcoords='offset points',fontsize=7.5,
                    color=color,bbox=dict(facecolor='#17201a',alpha=.78,pad=.6,edgecolor='none'),zorder=6)
    ax.scatter([0],[0],s=70,marker='*',c='#55bfff',zorder=7)
    ax.annotate('原相机 C000',(0,0),xytext=(5,-14),textcoords='offset points',color='#55bfff',fontsize=9,zorder=8)
    fig.savefig(job/'cameras.png');plt.close(fig)


def render_points(cloud, position, width=WIDTH, height=HEIGHT):
    """Legacy point-splat baseline for regression tests; NOT the active renderer."""
    count=len(cloud)
    pixels=np.empty(count,np.int32)
    ranges=np.empty(count,np.float32)
    position=np.asarray(position,np.float32)
    for start in range(0,count,500000):
        end=min(start+500000,count)
        xyz=np.asarray(cloud[start:end,:3])-position
        radius=np.linalg.norm(xyz,axis=1)
        horizontal=np.hypot(xyz[:,0],xyz[:,2])
        u=(np.arctan2(xyz[:,0],xyz[:,2])/(2*np.pi)+.5)*width
        v=(np.arctan2(xyz[:,1],horizontal)/np.pi+.5)*height
        px=np.floor(u).astype(np.int32)%width
        py=np.clip(np.floor(v).astype(np.int32),0,height-1)
        pixels[start:end]=py*width+px
        ranges[start:end]=np.where(np.isfinite(radius)&(radius>.015),radius,np.inf)
    depth=np.full(width*height,np.inf,np.float32)
    np.minimum.at(depth,pixels,ranges)
    front=(ranges==depth[pixels]) & np.isfinite(ranges)
    rgb=np.zeros((height*width,3),np.uint8)
    rgb[pixels[front]]=np.rint(np.clip(cloud[front,3:6],0,1)*255).astype(np.uint8)
    rgb,depth=rgb.reshape(height,width,3),depth.reshape(height,width)
    # A one-pixel point footprint, resolved by nearest depth. This is splatting,
    # not semantic filling; gaps beyond that footprint remain explicitly black.
    out_rgb,out_depth=rgb.copy(),depth.copy()
    for dy,dx in [(0,-1),(0,1),(-1,0),(1,0)]:
        d=np.roll(depth,(dy,dx),(0,1))
        c=np.roll(rgb,(dy,dx),(0,1))
        if dy<0:d[dy:]=np.inf
        if dy>0:d[:dy]=np.inf
        take=d<out_depth
        out_depth[take]=d[take];out_rgb[take]=c[take]
    valid=np.isfinite(out_depth)
    out_rgb[~valid]=[0,255,0]
    return out_rgb,valid


def process(job, render=True, inset=INSET, reuse_from=None):
    job=Path(job)
    outline=json.loads((job/'outline.json').read_text(encoding='utf-8'))
    try:
        info=sample_cameras(outline['polygon_m'],inset=inset)
    except SamplingError as error:
        # Persist the failure for the lab; C000 is diagnostic, not success.
        info=error.info
    draw_camera_map(job,info)
    from render_unknown import render_job, RENDER_SOURCE
    from occlusion import VERSION
    info['render_source']=RENDER_SOURCE
    info['render_version']=VERSION
    info['render_resolution']=[WIDTH,HEIGHT]
    info['panorama_projection']='ERP: u=(atan2(X,Z)/2pi+.5), v=(atan2(Y,hypot(X,Z))/pi+.5); OpenCV world'
    info['render_complete']=False
    # Old point-splat previews cannot serve as a cache for the unknown-volume
    # renderer. Keep the argument for old resample callers, but recompute RGB-D.
    def save():
        temp=job/'cameras.json.partial'
        temp.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        # Do not overwrite the canonical step-3 six-face cameras.json.
        temp.replace(job/'sample-cameras.json')
    save()
    if render:
        info=render_job(job,width=WIDTH,height=HEIGHT)
    info=apply_selection(info)
    draw_camera_map(job,info)
    save()
    write_candidates(job,info)
    print(f'CAMERAS {job.name}: {info["count"]} render_complete={info["render_complete"]}',flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--no-render',action='store_true')
    a=p.parse_args();process(a.job,not a.no_render)
