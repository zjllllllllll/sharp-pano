"""Audit five shallow-depth scenes, calibrated inputs, PLYs and actual renders."""
from pathlib import Path
import argparse
import gc
import io
import json
import math
import subprocess
import unittest
import numpy as np
from PIL import Image
from plyfile import PlyData
from gaussian_ops import digest, FIELDS

ROOT = Path(__file__).resolve().parents[1]
SCENES = ['0.1', '1.1', '2.1', '3.1', '4.1']
FACES = ['front', 'right', 'back', 'left', 'up', 'down']
FORWARDS = [(0,0,1), (1,0,0), (0,0,-1), (-1,0,0), (0,-1,0), (0,1,0)]
CHECKPOINT_SHA = '94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d'


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def independent_sample(panorama, xyz):
    """Float64 ERP sampler: asin latitude rather than pipeline atan2 latitude."""
    xyz = np.asarray(xyz, dtype=np.float64)
    unit = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    h, w = panorama.shape[:2]
    u = (np.arctan2(unit[...,0], unit[...,2]) / (2*np.pi) + .5)*w - .5
    v = (np.arcsin(np.clip(unit[...,1], -1, 1)) / np.pi + .5)*h - .5
    v = np.clip(v, 0, h-1)
    x, y = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    dx, dy = u-x, v-y
    if panorama.ndim == 3:
        dx, dy = dx[...,None], dy[...,None]
    a, b = panorama[y, x%w], panorama[y, (x+1)%w]
    c, d = panorama[np.minimum(y+1,h-1), x%w], panorama[np.minimum(y+1,h-1), (x+1)%w]
    return (a*(1-dx)+b*dx)*(1-dy) + (c*(1-dx)+d*dx)*dy


def audit_inputs(stem, output):
    manifest = read_json(output/'inputs'/stem/'cameras.json')
    source = manifest['sources']
    check(digest(source['rgb']) == source['rgb_sha256'], 'RGB source changed')
    check(digest(source['radial_depth']) == source['depth_sha256'], 'Depth source changed')
    rgb = np.asarray(Image.open(source['rgb']).convert('RGB'))
    radial = np.load(source['radial_depth'])
    check(rgb.shape == (2048,4096,3) and radial.shape == (2048,4096), 'Source dimensions')
    check(np.all(np.isfinite(radial) & (radial>0)), 'Source depth invalid')
    check([c['face'] for c in manifest['cameras']] == FACES, 'Six face order')
    result = []
    for i, cam in enumerate(manifest['cameras']):
        n = 1536
        c2w, w2c = np.asarray(cam['c2w']), np.asarray(cam['w2c'])
        r = c2w[:3,:3]
        np.testing.assert_allclose(c2w@w2c, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(r[:,2], FORWARDS[i], atol=1e-12)
        check(abs(np.linalg.det(r)-1)<1e-12 and np.all(c2w[:3,3]==0), 'Pose handedness/origin')
        check(not cam['focal_estimated'] and not cam['pose_estimated'], 'Estimated camera used')
        focal = n/(2*math.tan(math.radians(50)))
        k = np.asarray(cam['K_pixel_edge'])
        np.testing.assert_allclose(k, [[focal,0,n/2],[0,focal,n/2],[0,0,1]], atol=1e-10)
        check(cam['size_wh']==[n,n] and cam['hfov_deg']==100 and cam['vfov_deg']==100, 'Face FOV/size')
        check(read_json(output/'inputs'/stem/f"{cam['face']}.camera.json")==cam, 'Sidecar mismatch')
        for name, path in cam['files'].items():
            check(digest(path)==cam['hashes'][name], 'Face hash mismatch')
        face_rgb = np.asarray(Image.open(cam['files']['rgb']).convert('RGB'))
        face_z, face_range = np.load(cam['files']['camera_z']), np.load(cam['files']['range'])
        check(face_rgb.shape==(n,n,3) and face_z.shape==face_range.shape==(n,n), 'Face dimensions')
        check(np.all(np.isfinite(face_z)&(face_z>0)), 'Camera depth invalid')
        # 2,401 locations include boundaries, corners, center vicinity and overlaps.
        axis = np.unique(np.linspace(0,n-1,49).round().astype(int))
        xx, yy = np.meshgrid(axis,axis)
        pixels = np.stack([xx+.5,yy+.5,np.ones_like(xx)],axis=-1)
        rays = pixels@np.linalg.inv(k).T
        rays /= np.linalg.norm(rays,axis=-1,keepdims=True)
        world = rays@r.T
        expected_range = independent_sample(radial,world)
        expected_rgb = independent_sample(rgb,world)
        range_error = np.abs(face_range[yy,xx]-expected_range)
        z_error = np.abs(face_z[yy,xx]-expected_range*rays[...,2])
        rgb_error = np.abs(face_rgb[yy,xx].astype(float)-expected_rgb)
        check(range_error.max()<5e-4 and z_error.max()<5e-4, 'ERP-to-face depth projection')
        check(rgb_error.max()<2, 'ERP-to-face RGB projection')
        result.append({'face':cam['face'],'checked_pixels':int(xx.size),
                       'range_error_max_m':float(range_error.max()),'z_error_max_m':float(z_error.max()),
                       'rgb_error_max_8bit':float(rgb_error.max()),'focal_px':focal})
    return manifest, radial, result


def audit_ply(path):
    ply = PlyData.read(path,mmap='r')
    v = ply['vertex'].data
    check(tuple(v.dtype.names)==tuple(FIELDS) and len(v)>0, 'PLY schema/count')
    max_q_error = 0.
    scale_min, scale_max = float('inf'), 0.
    for start in range(0,len(v),200000):
        part = v[start:start+200000]
        for key in FIELDS:
            check(np.isfinite(part[key]).all(), f'Nonfinite PLY {key}')
        qnorm = np.sqrt(sum(part[f'rot_{j}'].astype(float)**2 for j in range(4)))
        max_q_error = max(max_q_error,float(np.max(np.abs(qnorm-1))))
        scales = np.exp(np.column_stack([part[f'scale_{j}'] for j in range(3)]).astype(float))
        check(np.isfinite(scales).all() and (scales>0).all(), 'Invalid decoded covariance scales')
        scale_min, scale_max = min(scale_min,float(scales.min())),max(scale_max,float(scales.max()))
    check(max_q_error<2e-5, 'Quaternion normalization')
    return v, {'gaussians':len(v),'bytes':path.stat().st_size,'sha256':digest(path),
               'max_quaternion_norm_error':max_q_error,'scale_min_m':scale_min,'scale_max_m':scale_max}


def audit_scene(stem, output):
    manifest, radial, input_checks = audit_inputs(stem,output)
    result = {'inputs':input_checks,'routes':{}}
    for mode in ('shallow',):
        path = output/mode/f'{stem}.ply'
        merged = read_json(output/'reports'/stem/f'{mode}.merged.json')
        vertices, data = audit_ply(path)
        check(data['sha256']==merged['ply']['sha256'], 'Merged hash mismatch')
        offset = 0
        face_medians, face_p99 = [], []
        for face in FACES:
            fp = output/'work'/stem/f'{face}.{mode}.ply'
            face_v, fdata = audit_ply(fp)
            report = read_json(output/'reports'/stem/f'{face}.{mode}.json')
            check(digest(fp)==report['ply']['sha256'], 'Face PLY hash mismatch')
            check(np.array_equal(vertices[offset:offset+len(face_v)],face_v), 'Merge changed face attributes/order')
            check(len(face_v)==merged['face_counts'][face], 'Face count mismatch')
            offset += len(face_v)
            check(report['model']['api']=='RGBGaussianPredictor(image, disparity_factor, depth=target_camera_z)', 'Native depth API')
            check(report['model']['layers']==2, 'Both predicted layers required')
            face_medians.append(report['depth_diagnostics']['radial_relative_error']['median'])
            face_p99.append(report['depth_diagnostics']['radial_relative_error']['p99'])
        check(offset==len(vertices), 'Exact concatenation count')
        data['six_faces_bit_exact_concatenation'] = True
        data['layer0_face_relative_error_medians'] = dict(zip(FACES,face_medians))
        data['layer0_face_relative_error_p99'] = dict(zip(FACES,face_p99))
        result['routes'][mode] = data
        del vertices
        gc.collect()
    render_path = output/'validation'/stem/'renders.json'
    render = read_json(render_path)
    check(len(render['views'])==10, 'Need six source, two seam and two novel views')
    for mode in ('shallow',):
        check(render['ply_sha256'][mode]==result['routes'][mode]['sha256'], 'Render from stale PLY')
    metrics = {mode: {'source_psnr_db':[],'source_depth_relative_median':[],
                     'novel_coverage':[]} for mode in ('shallow',)}
    hashes = {}
    for name, view in render['views'].items():
        for mode in ('shallow',):
            image_path = render_path.parent/f'{name}.{mode}.png'
            depth_path = render_path.parent/f'{name}.{mode}.depth.npy'
            image = np.asarray(Image.open(image_path))
            depth = np.load(depth_path)
            check(image.shape==(512,512,3) and depth.shape==(512,512), 'Rendered dimensions')
            check(np.isfinite(depth).all() and depth.max()>0 and image.std()>1, 'Invalid/blank render')
            hashes[image_path.name],hashes[depth_path.name] = digest(image_path),digest(depth_path)
            if view['has_reference']:
                reference = np.asarray(Image.open(render_path.parent/f'{name}.input.png'))
                mse = np.mean(((image.astype(np.float32)-reference)/255)**2)
                psnr = float(-10*np.log10(max(float(mse),1e-12)))
                check(abs(psnr-view[mode]['rgb_psnr_db'])<1e-5, 'Rendered RGB metrics differ')
                metrics[mode]['source_psnr_db'].append(psnr)
                metrics[mode]['source_depth_relative_median'].append(view[mode]['radial_or_z_relative_error']['median'])
            else:
                metrics[mode]['novel_coverage'].append(view[mode]['coverage_alpha_gt_half'])
    result['renders'] = {'view_count_per_route':10,'metrics':metrics,'artifact_sha256':hashes}
    print('INDEPENDENT_SCENE_AUDIT_OK',stem,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'sharp-sixface-results')
    args=p.parse_args(); output=args.results
    model=read_json(output/'reports'/'model.json')
    check(model['precision']=='fp32' and model['strict_keys']==1038 and model['alignment_keys']==156, 'Original model identity')
    check(model['sha256']==CHECKPOINT_SHA==digest(model['checkpoint']), 'Checkpoint identity changed')
    check(not list(output.rglob('*.partial.ply')), 'Incomplete PLY remains')
    for mode in ('shallow',):
        check(sorted(x.stem for x in (output/mode).glob('*.ply'))==SCENES, 'Must deliver five shallow PLYs')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_geometry','test_gaussian_ops'])
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    check(tests.wasSuccessful() and tests.testsRun==8, 'Geometry/covariance/PLY tests failed')
    revision=subprocess.run(['git','-C',str(ROOT/'ml-sharp'),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    diff=subprocess.run(['git','-C',str(ROOT/'ml-sharp'),'diff','--name-only','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    check(not diff, 'Original SHARP tracked code changed')
    audit={'status':'passed','scope':'five panoramas x native depth input x calibrated six faces',
           'model':model,'original_sharp_git_revision':revision,'original_sharp_tracked_diff':diff,
           'unit_tests':log.getvalue(),
           'implementation_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},'scenes':{}}
    for stem in SCENES:
        audit['scenes'][stem]=audit_scene(stem,output)
    audit['limitations']=['Depth consistency is with supplied DepthMaster estimates, not ground truth.',
                         'Native external-depth input retains both learned Gaussian layers.',
                         'Optical-thickness feather is not multi-view optimization; no hidden geometry is observed.']
    (output/'reports'/'delivery_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('FULL_DELIVERY_AUDIT_PASSED: 5 final PLYs, 30 calibrated views, 50 actual renders',flush=True)


if __name__=='__main__': main()
