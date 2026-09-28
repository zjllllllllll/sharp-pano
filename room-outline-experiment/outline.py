"""Geometry-only bird-eye wall envelope. Never modifies source point clouds.

Vertical occupancy, rather than raw pixel density, supplies wall evidence.
An angular path through that evidence gives an explicitly star-shaped visible
envelope, NOT a claim of a complete architectural floorplan. Short line segments
are fitted in bird-eye coordinates; unsupported closure remains marked inferred.
"""
from pathlib import Path
import argparse
import json
import math
import time

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

VERSION = 1
GRID = 0.05
BINS = 24
ANGLE_COUNT = 360
SIMPLIFY_M = 0.13


def rasterize(xyz, grid=GRID):
    xyz = np.asarray(xyz)
    sample = xyz[::max(1, len(xyz) // 400000)]
    sample = sample[np.isfinite(sample).all(axis=1)]
    height = -sample[:, 1]
    hlow, hhigh = np.quantile(height, [.025, .975])
    if hhigh - hlow < .50:
        raise ValueError('Insufficient vertical extent for wall evidence')
    low = np.minimum(np.quantile(sample[:, [0, 2]], .001, axis=0), 0) - .3
    high = np.maximum(np.quantile(sample[:, [0, 2]], .999, axis=0), 0) + .3
    # Coarsen only for pathological range extents, keeping memory bounded.
    grid = max(grid, float(np.max(high-low)) / 1800)
    shape = np.ceil((high-low)[::-1] / grid).astype(int) + 1
    count = np.zeros(shape, np.uint32)
    occupancy = np.zeros((BINS, *shape), np.uint8)
    finite_count = inside_count = selected_count = 0
    height_margin = max(.12, .055 * (hhigh-hlow))
    h0, h1 = hlow + height_margin, hhigh - height_margin
    if h1 <= h0:
        raise ValueError('Degenerate height band')
    for start in range(0, len(xyz), 500000):
        part = np.asarray(xyz[start:start+500000])
        good = np.isfinite(part).all(axis=1)
        finite_count += int(good.sum())
        part = part[good]
        ij = np.floor((part[:, [0, 2]] - low) / grid).astype(np.int64)
        good = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
        part, ij = part[good], ij[good]
        inside_count += len(part)
        np.add.at(count, (ij[:, 1], ij[:, 0]), 1)
        h = -part[:, 1]
        sel = (h >= h0) & (h <= h1)
        selected_count += int(sel.sum())
        b = np.minimum(((h[sel]-h0)/(h1-h0)*BINS).astype(int), BINS-1)
        occupancy[b, ij[sel, 1], ij[sel, 0]] = 1
    # Pool locally within each height bin, never infer support from min/max
    # height alone (floor+ceiling would otherwise impersonate full-height walls).
    radius = max(1, round(.075 / grid))
    occupied = ndi.maximum_filter(occupancy, size=(1, 2*radius+1, 2*radius+1)) > 0
    coverage = occupied.sum(axis=0).astype(np.float32) / BINS
    longest = np.zeros(shape, np.uint8)
    run = np.zeros(shape, np.uint8)
    for layer in occupied:
        run = np.where(layer, run+1, 0).astype(np.uint8)
        longest = np.maximum(longest, run)
    evidence = np.sqrt(coverage * (longest.astype(np.float32)/BINS))
    evidence = ndi.gaussian_filter(evidence, .65)
    return dict(count=count, coverage=coverage, evidence=evidence,
                origin=low.astype(float), grid=grid, height_band=[float(h0), float(h1)],
                diagnostics=dict(input_points=len(xyz), finite_points=finite_count,
                                 raster_points=inside_count, height_band_points=selected_count,
                                 raster_clipped_points=finite_count-inside_count,
                                 height_p025_p975_m=[float(hlow), float(hhigh)],
                                 grid_m=grid, grid_shape=shape.tolist()))


def sample_grid(field, points, origin, grid):
    q = (np.asarray(points)-origin) / grid - .5
    return ndi.map_coordinates(field, [q[..., 1], q[..., 0]], order=1,
                               mode='constant', cval=0)


def detect_lines(raster):
    e, origin, grid = raster['evidence'], raster['origin'], raster['grid']
    image = np.rint(np.clip((e-.14)/.65, 0, 1)*255).astype(np.uint8)
    detections = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(image)[0]
    lines = []
    if detections is None:
        return lines
    for coords in detections[:, 0]:
        a, b = (coords.reshape(2, 2)+.5)*grid + origin
        length = np.linalg.norm(b-a)
        if length < .40:
            continue
        pts = np.linspace(a, b, max(5, int(length/.05)))
        support = float(np.mean(sample_grid(e, pts, origin, grid)))
        if support < .21:
            continue
        lines.append(dict(a=a.tolist(), b=b.tolist(), length_m=float(length), support=support))
    return sorted(lines, key=lambda row: -row['length_m']*row['support'])


def radial_envelope(raster):
    evidence = raster['evidence']
    origin, grid = raster['origin'], raster['grid']
    corners = np.array([origin, origin + np.array(evidence.shape[::-1])*grid])
    rmax = float(np.max(np.linalg.norm(corners, axis=1)))
    radii = np.arange(.30, rmax+.025, .025)
    angles = np.arange(ANGLE_COUNT) * (2*np.pi/ANGLE_COUNT)
    dirs = np.column_stack([np.sin(angles), np.cos(angles)])
    positions = dirs[:, None, :] * radii[None, :, None]
    profiles = sample_grid(evidence, positions, origin, grid)
    # Angular pooling makes 1-pixel projected gaps less sensitive, while retaining
    # radial discontinuities. No RGB, semantic labels or ceiling footprint used.
    profiles = ndi.maximum_filter(profiles, size=(3, 1), mode=('wrap', 'nearest'))
    profiles = ndi.gaussian_filter1d(profiles, 1.0, axis=1)
    candidates, scores = [], []
    supported = profiles.max(axis=1) > .22
    if supported.mean() < .25:
        raise ValueError(f'Insufficient wall evidence in angular sectors: {supported.mean():.1%}')
    for profile in profiles:
        peaks, _ = find_peaks(profile, distance=6, prominence=.025)
        peaks = peaks[profile[peaks] > .12]
        if not len(peaks):
            peaks = np.array([int(profile.argmax())])
        peaks = peaks[np.argsort(profile[peaks])[-7:]]
        peaks = peaks[np.argsort(radii[peaks])]
        candidates.append(radii[peaks])
        scores.append(profile[peaks])
    # Empty sectors inherit candidate ranges from both neighboring supported
    # sectors. They are still zero-evidence inferred segments, never observed.
    for i in np.flatnonzero(~supported):
        left = next((int((i-k) % ANGLE_COUNT) for k in range(1, ANGLE_COUNT)
                     if supported[(i-k) % ANGLE_COUNT]), None)
        right = next((int((i+k) % ANGLE_COUNT) for k in range(1, ANGLE_COUNT)
                      if supported[(i+k) % ANGLE_COUNT]), None)
        options = np.unique(np.concatenate([candidates[left], candidates[right]]))
        if len(options) > 7:
            options = options[np.linspace(0, len(options)-1, 7).astype(int)]
        candidates[i], scores[i] = options, np.zeros(len(options))
    anchor = int(np.argmax([s.max() for s in scores]))
    order = (np.arange(ANGLE_COUNT)+anchor) % ANGLE_COUNT
    best = None
    for initial in range(len(candidates[anchor])):
        costs = np.full(len(candidates[anchor]), np.inf)
        costs[initial] = 0
        back = []
        for index in range(1, ANGLE_COUNT):
            previous, current = order[index-1], order[index]
            delta = np.abs(candidates[previous][:, None]-candidates[current][None, :])
            transition = np.minimum(.8*delta + .7*delta**2, 3.0)
            combined = costs[:, None]+transition
            parent = combined.argmin(axis=0)
            costs = combined[parent, np.arange(len(parent))] - 4*scores[current]
            # A very mild outward preference breaks ties; never use the furthest
            # point alone, which would expand through windows/outlier clusters.
            costs -= .03*np.log(np.maximum(candidates[current], .1))
            back.append(parent)
        delta = np.abs(candidates[order[-1]]-candidates[anchor][initial])
        total = costs + np.minimum(.8*delta+.7*delta**2, 3.0) - 4*scores[anchor][initial]
        last = int(total.argmin())
        if best is None or total[last] < best[0]:
            path = [last]
            for parent in back[::-1]:
                path.append(int(parent[path[-1]]))
            path = path[::-1]
            selected = np.empty(ANGLE_COUNT)
            quality = np.empty(ANGLE_COUNT)
            for i, node in enumerate(path):
                selected[order[i]] = candidates[order[i]][node]
                quality[order[i]] = scores[order[i]][node]
            best = (float(total[last]), selected, quality)
    _, selected, quality = best
    # Mild angular median removes isolated one-sector spikes, not large gaps.
    selected = ndi.median_filter(selected, size=3, mode='wrap')
    raw = dirs * selected[:, None]
    poly = cv2.approxPolyDP(raw.astype(np.float32).reshape(-1, 1, 2), SIMPLIFY_M, True)[:, 0].astype(float)
    return poly, raw, dict(angular_supported_fraction=float(supported.mean()),
                          angular_selected_evidence_mean=float(quality.mean()),
                          angular_samples=ANGLE_COUNT)


def fit_polygon(poly, raster):
    """Fit each supported boundary span to bird-eye evidence, with bounded moves."""
    origin, grid, evidence = raster['origin'], raster['grid'], raster['evidence']
    yy, xx = np.nonzero(evidence >= .25)
    points = np.column_stack([xx+.5, yy+.5])*grid+origin
    weights = evidence[yy, xx]**3
    fitted = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        vector = b-a
        length = np.linalg.norm(vector)
        if length < 1e-6:
            fitted.append((a, vector))
            continue
        direction = vector / length
        normal = np.array([-direction[1], direction[0]])
        rel = points-a
        along, across = rel@direction, rel@normal
        sel = (along >= .03) & (along <= length-.03) & (np.abs(across) <= .18)
        if sel.sum() < 6 or length < .25:
            fitted.append((a, direction))
            continue
        p, w = points[sel], weights[sel]
        center = np.average(p, axis=0, weights=w)
        delta = p-center
        covariance = (delta*w[:, None]).T@delta / w.sum()
        _, vec = np.linalg.eigh(covariance)
        axis = vec[:, -1]
        if abs(axis@direction) >= math.cos(math.radians(12)):
            fitted.append((center, axis))
        else:
            fitted.append((a, direction))
    out = []
    for i, old in enumerate(poly):
        a, u = fitted[i-1]
        b, v = fitted[i]
        mat = np.column_stack([u, -v])
        if abs(np.linalg.det(mat)) < .10:
            out.append(old)
            continue
        intersection = a+np.linalg.solve(mat, b-a)[0]*u
        out.append(intersection if np.linalg.norm(intersection-old) <= .25 else old)
    out = np.asarray(out)
    # The origin must remain in the envelope. Don't use convex hull fallback.
    if cv2.pointPolygonTest(out.astype(np.float32), (0., 0.), False) < 0:
        return poly
    return out


def describe_segments(poly, raster):
    e, origin, grid = raster['evidence'], raster['origin'], raster['grid']
    near = ndi.maximum_filter(e, size=5)
    segments = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        length = float(np.linalg.norm(b-a))
        points = np.linspace(a, b, max(5, int(length/.04)))
        values = sample_grid(near, points, origin, grid)
        fraction = float(np.mean(values >= .28))
        mean = float(np.mean(values))
        observed = fraction >= .65 and mean >= .32
        segments.append(dict(a=a.tolist(), b=b.tolist(), length_m=length,
                             supported_fraction=fraction, mean_evidence=mean,
                             kind='supported' if observed else 'inferred'))
    return segments


def fit(xyz):
    raster = rasterize(xyz)
    lines = detect_lines(raster)
    poly, raw, diagnostics = radial_envelope(raster)
    poly = fit_polygon(poly, raster)
    if len(poly) < 3 or not np.isfinite(poly).all():
        raise ValueError('Invalid contour')
    segments = describe_segments(poly, raster)
    perimeter = sum(s['length_m'] for s in segments)
    inferred = sum(s['length_m'] for s in segments if s['kind'] == 'inferred')
    result = dict(version=VERSION, method='BEV vertical occupancy + cyclic radial evidence path + bounded line fitting',
                  source='Native SHARP raw stage-3 two-layer initializer; no semantic labels or Gaussian densification',
                  coordinates='polygon [X,Z] meters; source cloud OpenCV X-right,Y-down,Z-forward',
                  polygon_m=poly.tolist(), segments=segments, detected_lines=lines,
                  diagnostics={**raster['diagnostics'], **diagnostics,
                               'area_m2':abs(float(cv2.contourArea(poly.astype(np.float32)))),
                               'perimeter_m':perimeter, 'inferred_perimeter_fraction':inferred/max(perimeter, 1e-8),
                               'vertices':len(poly), 'detected_lines':len(lines)},
                  limitations=['Single-center visible wall envelope, not complete room reconstruction',
                               'Tall furniture, windows and monocular depth errors can bias the boundary',
                               'Dashed spans are geometric closure hypotheses, not measured walls',
                               'Not a collision-free camera placement map'])
    return result, raster


def plot_result(job, result, raster, output=None, name=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps
    plt.rcParams.update({'font.family':'Microsoft YaHei', 'axes.unicode_minus':False,
                         'figure.facecolor':'#101716', 'axes.facecolor':'#101716',
                         'text.color':'#e5ece9', 'axes.labelcolor':'#b3c5bd',
                         'xtick.color':'#8eaaa0', 'ytick.color':'#8eaaa0',
                         'axes.edgecolor':'#42594e', 'font.size':10})
    output = Path(output or job)
    output.mkdir(parents=True, exist_ok=True)
    name = name or job.name
    origin, grid = raster['origin'], raster['grid']
    h, w = raster['count'].shape
    extent = [origin[0], origin[0]+w*grid, origin[1], origin[1]+h*grid]
    density = np.log1p(raster['count'].astype(float))
    density /= max(float(np.quantile(density[density > 0], .97)), 1e-6)
    density = np.clip(density, 0, 1)
    poly = np.asarray(result['polygon_m'])
    pmin, pmax = poly.min(axis=0), poly.max(axis=0)
    margin = max(.6, float(np.max(pmax-pmin))*.08)
    def base(ax, evidence=False, full=False):
        ax.imshow(raster['evidence'] if evidence else density, origin='lower', extent=extent,
                  cmap='magma' if evidence else 'gray', vmin=0, vmax=1, interpolation='nearest')
        ax.scatter([0], [0], c='#68d5ff', s=35, marker='o', zorder=10, edgecolors='#101716')
        ax.annotate('', xy=(0, .45), xytext=(0, 0), arrowprops=dict(arrowstyle='->',color='#68d5ff',lw=1.5))
        ax.set_xlabel('X / m'); ax.set_ylabel('Z / m')
        ax.set_aspect('equal')
        ax.grid(alpha=.12)
        if not full:
            ax.set_xlim(pmin[0]-margin, pmax[0]+margin)
            ax.set_ylim(pmin[1]-margin, pmax[1]+margin)
    def outline(ax):
        for line in result['segments']:
            a, b = line['a'], line['b']
            supported = line['kind'] == 'supported'
            ax.plot([a[0],b[0]], [a[1],b[1]], color='#b6f76c' if supported else '#ffb55a',
                    lw=2, ls='-' if supported else (0, (4, 3)))
    for kind in ['bev', 'overlay', 'evidence', 'bev-full']:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=160)
        base(ax, kind=='evidence', kind=='bev-full')
        if kind not in ('bev', 'bev-full'): outline(ax)
        ax.set_title(f'{name} · '+{'bev':'点云俯视投影（房间区域）', 'overlay':'点云俯视 + 拟合轮廓', 'evidence':'多高度支撑 + 拟合轮廓', 'bev-full':'完整点云投影范围（含远处离群点）'}[kind],pad=14)
        fig.text(.5, .025, '蓝点：原相机  |  绿实线：点云支撑边  |  橙虚线：推测连接边', ha='center', fontsize=9)
        fig.tight_layout(rect=(0,.045,1,1))
        fig.savefig(output/f'{kind}.png')
        plt.close(fig)
    fig = plt.figure(figsize=(16, 9.5), dpi=150)
    layout = fig.add_gridspec(2, 3, height_ratios=[.8, 3.0], hspace=.25, wspace=.22)
    ax = fig.add_subplot(layout[0, :])
    with Image.open(job/'input.png') as image:
        image.thumbnail((1800,900))
        ax.imshow(image)
    ax.axis('off')
    ax.set_title(f'{name}  |  原版 DepthMaster → SHARP 反投影 → 仅几何俯视轮廓',loc='left',fontsize=15)
    titles=['全部点云俯视（密度显示）', '墙状结构的多高度支撑', '俯视点云 + 轮廓线']
    for i in range(3):
        ax = fig.add_subplot(layout[1,i]); base(ax, i==1)
        if i == 1:
            for line in result['detected_lines']:
                a,b=line['a'],line['b']
                ax.plot([a[0],b[0]],[a[1],b[1]],color='#69d8ff',lw=.6,alpha=.6)
        if i == 2: outline(ax)
        ax.set_title(titles[i],fontsize=12)
    d=result['diagnostics']
    text = (f"{d['input_points']:,} 点 · 网格 {d['grid_m']*100:.1f} cm · "
            f"轮廓 {d['vertices']} 边 / {d['area_m2']:.1f} m² · 推测边占周长 {d['inferred_perimeter_fraction']:.0%}")
    fig.text(.5,.045,text,ha='center',fontsize=11)
    fig.text(.5,.018,'绿实线 = 几何支撑；橙虚线 = 推测闭合；蓝点 = 原相机。仅实验轮廓，不是建筑真值或相机可通行区。',ha='center',fontsize=10,color='#aec1b6')
    fig.savefig(output/'overview.png',bbox_inches='tight')
    plt.close(fig)


def process(job, output=None, name=None):
    job = Path(job)
    output = Path(output or job)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cloud = np.memmap(job/'cloud-full.bin', dtype='<f4', mode='r').reshape(-1,6)
    result, raster = fit(cloud[:, :3])
    result['source_cloud_file'] = str(job/'cloud-full.bin')
    result['fit_seconds'] = time.perf_counter()-start
    np.savez_compressed(output/'bev-evidence.npz', count=raster['count'], evidence=raster['evidence'],
                        coverage=raster['coverage'], origin=raster['origin'], grid=raster['grid'])
    plot_result(job, result, raster, output, name)
    result['total_seconds'] = time.perf_counter()-start
    (output/'outline.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(name=name or job.name, **result['diagnostics'], seconds=result['total_seconds'])),flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('job',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--name')
    a=p.parse_args()
    process(a.job,a.output,a.name)
