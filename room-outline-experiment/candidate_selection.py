"""Fixed camera selection, after adaptive geometry and RGB-D Mask rendering.

Never change poses, numbering, renders or the adaptive-offset stop condition.
Keep the full geometric pool for diagnostics; `cameras` is the final component.
"""
from copy import deepcopy
from math import isfinite

VERSION = 'mask15-origin-four-connected-v1'
MASK_THRESHOLD = .15
SPACING = .5
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def mask_fraction(camera):
    value = camera.get('mask_stats', {}).get('mask_fraction')
    return value if type(value) in (int, float) and isfinite(value) and 0 <= value <= 1 else None


def select(cameras):
    """Four-connected flood fill AFTER thresholding. Unknown masks cannot bridge."""
    by_grid = {}
    ids = set()
    states = {}
    for camera in cameras:
        grid = tuple(camera['grid'])
        if len(grid) != 2 or any(type(v) is not int for v in grid):
            raise ValueError('Camera grid must contain two integers')
        if grid in by_grid or camera['id'] in ids:
            raise ValueError('Duplicate camera grid or ID')
        if any(abs(a-b) > 1e-8 for a,b in zip(camera['position_world_m'], (grid[0]*SPACING, 0., grid[1]*SPACING))):
            raise ValueError('Camera must be on the fixed 0.5 m grid at source height')
        if bool(camera.get('coincides_with_original')) != (grid == (0, 0)):
            raise ValueError('Original camera flag does not match grid origin')
        by_grid[grid] = camera
        ids.add(camera['id'])
        fraction = mask_fraction(camera)
        states[camera['id']] = ('origin' if grid == (0, 0) else
            'pending' if fraction is None else 'excluded' if fraction > MASK_THRESHOLD else 'disconnected')
    if (0, 0) not in by_grid or by_grid[(0, 0)]['id'] != 'C000':
        raise ValueError('C000 at grid origin is required')
    reached = {(0, 0)}
    queue = [(0, 0)]
    while queue:
        x,z = queue.pop()
        for dx,dz in DIRECTIONS:
            grid = (x+dx,z+dz)
            camera = by_grid.get(grid)
            if grid not in reached and camera and states[camera['id']] == 'disconnected':
                reached.add(grid)
                states[camera['id']] = 'retained'
                queue.append(grid)
    groups = {key: [] for key in ('origin','retained','excluded','disconnected','pending')}
    for camera in cameras:
        groups[states[camera['id']]].append(camera)
    kept = [c for c in cameras if states[c['id']] in ('origin','retained')]
    return dict(states=states, groups=groups, kept=kept,
                only_origin=len(kept)==1 and not groups['pending'])


def apply_selection(info):
    info = deepcopy(info)
    if info['spacing_m'] != SPACING:
        raise ValueError('Fixed selection requires a 0.5 m grid')
    pool = info.get('all_cameras', info['cameras'])
    result = select(pool)
    for camera in pool:
        camera['selection_state'] = result['states'][camera['id']]
    info.update(all_cameras=pool, cameras=result['kept'], count=len(result['kept']),
        geometric_camera_count=len(pool),
        geometric_sampled_camera_count=sum(not c.get('reference_only',False) for c in pool),
        sampled_camera_count=sum(not c.get('coincides_with_original') for c in result['kept']),
        selection_version=VERSION,
        default_retained_camera_ids=[c['id'] for c in result['kept']],
        selection_error=('候选采样失败：固定 Mask ≤15% 筛选及 0.5 m 四邻接连通筛选后，仅剩原相机 C000，没有可用的新相机。'
            if result['only_origin'] else None),
        selection_complete=not result['groups']['pending'],
        mask_filter=dict(default_threshold=MASK_THRESHOLD, threshold=MASK_THRESHOLD, fixed=True,
            metric='ERP masked pixels / all pixels', comparison='strictly_greater', origin_exempt=True),
        selection=dict(version=VERSION, mask_threshold=MASK_THRESHOLD, spacing_m=SPACING,
            connectivity=4, anchor='C000', origin_exempt=True,
            order=['adaptive_offset','mask_at_most_15_percent','origin_component'],
            counts={k:len(v) for k,v in result['groups'].items()},
            excluded_camera_ids=[c['id'] for c in result['groups']['excluded']],
            disconnected_camera_ids=[c['id'] for c in result['groups']['disconnected']],
            pending_camera_ids=[c['id'] for c in result['groups']['pending']]))
    return info


def write_candidates(job,info):
    """Small canonical downstream export, shared by fresh and migrated scenes."""
    from batch import write_json
    write_json(job/'candidate-cameras.json',dict(scene=job.name,**{key:info[key] for key in
        ['selection_version','selection','spacing_m','inward_offset_m','offset_attempts','sampling_error','selection_error','count','cameras']}))
