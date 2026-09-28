"""Independent audit of stage-9 inputs, row provenance, geometry and PLY output."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,distance_transform_edt
from scipy.spatial.transform import Rotation
from plyfile import PlyData
from novel_pipeline import read,digest
from novel_geometry import FACE_NAMES,camera,view_rays,sample_erp,angular_weights


def verify_point_filter(out,signature):
    """Check all identity masks and independently sample normals and global KNN."""
    from scipy.spatial import cKDTree
    from point_normals import estimate_angles,PARAMETERS as normal_parameters
    expected=dict(normal_angle_degrees=82.,neighbors=64,distance_meters=.15,iterations=1)
    assert signature['settings']['point_filter']==expected
    report=read(out/'point-filter.json');assert report['parameters']==expected
    assert report['normal_parameters']==normal_parameters==signature['settings']['normal_estimation']
    reference=[];queries=[];expected_scores=[];normal_samples=0
    rng=np.random.default_rng(42)
    active_faces={row['face'] for row in report['faces']}
    for i,name in enumerate(FACE_NAMES):
        if name not in active_faces:continue
        face=out/name;cam=read(face/'camera.json');pose=np.asarray(cam['c2w'])
        ndc=np.load(face/'initializer-ndc.npy');world=np.load(face/'initializer-world.npy')
        assert ndc.shape==world.shape==(2,768,768,3)
        assert np.isfinite(ndc).all() and np.all(ndc[...,2]>0)
        axis=2*(np.arange(768,dtype=np.float32)+.5)/768-1
        np.testing.assert_allclose(ndc[...,0]/ndc[...,2],np.broadcast_to(axis[None,None,:],ndc.shape[:-1]),atol=3e-7)
        np.testing.assert_allclose(ndc[...,1]/ndc[...,2],np.broadcast_to(axis[None,:,None],ndc.shape[:-1]),atol=3e-7)
        scale=np.array([512/(2*cam['intrinsic_K_opencv'][0][0])]*2+[1.])
        expected_world=((ndc*scale).astype(np.float32).astype(np.float64)@pose[:3,:3].T+pose[:3,3]).astype(np.float32)
        np.testing.assert_array_equal(world,expected_world)
        xyz=world.reshape(-1,3);angles=np.load(face/'point-angles.npy');scores=np.load(face/'point-distance.npy')
        keep=np.load(face/'point-keep.npy');domain=np.load(face/'point-domain.npy')
        expected_domain=angular_weights(xyz.astype(np.float64)-pose[:3,3],i,fov=100)>1e-4
        normal_grid=world
        if signature.get('bundle_sha256'):
            from scipy.ndimage import zoom,minimum_filter
            support=zoom((np.asarray(Image.open(face/'support.png'))>0).astype(np.float32),768/512,order=1,mode='nearest')>.999999
            supported_normals=minimum_filter(support.astype(np.uint8),size=5,mode='constant',cval=0)>0
            expected_domain &= np.tile(supported_normals.ravel(),2)
            normal_grid=np.where(support[None,...,None],world,np.nan)
        np.testing.assert_array_equal(domain,expected_domain)
        assert all(v.shape==(len(xyz),) for v in (angles,scores,keep,domain))
        assert keep.dtype==domain.dtype==np.bool_ and np.isfinite(angles).all()
        normal=(angles<0)|(angles>90)|(angles<=82.)
        active=domain&normal
        assert np.all(scores[active]>=0) and np.isfinite(scores[active]).all()
        assert np.all(scores[~active]==-1)
        np.testing.assert_array_equal(keep,active&(scores<=np.float32(.15)))
        for layer in range(2):
            for y,x in rng.integers(2,766,size=(24,2)):
                angle=estimate_angles(normal_grid[layer,y-2:y+3,x-2:x+3],pose[:3,3],**normal_parameters)[2,2]
                np.testing.assert_allclose(angles.reshape(2,768,768)[layer,y,x],angle,atol=1e-5)
                normal_samples+=1
        active_ids=np.flatnonzero(active);ref=xyz[active_ids];offset=sum(len(v) for v in reference)
        choice=rng.choice(len(ref),min(128,len(ref)),replace=False)
        queries.extend((offset+choice).tolist());expected_scores.extend(scores[active_ids[choice]].tolist());reference.append(ref)
    xyz=np.concatenate(reference).astype(np.float64);del reference
    tree=cKDTree(xyz);query=np.asarray(queries)
    distances,ids=tree.query(xyz[query],k=min(65,len(xyz)),workers=4)
    distances=np.where(ids==query[:,None],np.inf,distances)
    means=np.sort(distances,axis=1)[:,:min(64,len(xyz)-1)].mean(axis=1).astype(np.float32)
    np.testing.assert_allclose(means,expected_scores,rtol=1e-6,atol=1e-7)
    return dict(passed=True,parameters=expected,normal_samples=normal_samples,knn_samples=len(query),
        reference_points=len(xyz),patch_before=report['patch_before'],patch_removed=report['patch_removed'],patch_kept=report['patch_kept'])


def covariance(vertices):
    q=np.column_stack([vertices[f'rot_{i}'] for i in (1,2,3,0)]).astype(np.float64)
    rotation=Rotation.from_quat(q).as_matrix()
    variance=np.exp(2*np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    return (rotation*variance[:,None,:])@rotation.swapaxes(1,2)


def expected_target_grid(depth):
    # Match the reference's two resizes, explicitly in array pixel coordinates:
    # input -> 1536 align_corners=True -> 768 align_corners=False.
    y,x=np.meshgrid(np.linspace(0,depth.shape[0]-1,1536),np.linspace(0,depth.shape[1]-1,1536),indexing='ij')
    enlarged=map_coordinates(depth,[y,x],order=1,mode='nearest',prefilter=False)
    y,x=np.meshgrid(np.arange(768)*2+.5,np.arange(768)*2+.5,indexing='ij')
    return map_coordinates(enlarged,[y,x],order=1,mode='nearest',prefilter=False)


def verify(job,result=None,*,state=None):
    state=state or read(job/'job.json');result=result or state['stages']['9'];rgb_state=state['stages']['7'];depth_state=state['stages']['8']
    folder=job/'novel'/result['revision'];out=folder/result['path'];signature=read(out/'inputs.json')
    diagnostic_only=signature['settings'].get('sharp_quality_gate') is False
    from project_paths import REPAIR as reference, MODELS
    for name,sha in signature['reference'].items():assert digest(reference/name)==sha,name
    for name,sha in signature['adapter'].items():assert digest(Path(__file__).with_name(name))==sha,name
    assert digest(MODELS/'sharp_2572gikvuh.pt')==signature['checkpoint_sha256']
    assert result['revision']==rgb_state['revision']==depth_state['revision']
    assert result['camera_id']==rgb_state['camera_id']==depth_state['camera_id']
    assert result['depth_revision']==depth_state['depth_revision']
    assert depth_state['panorama_source']=='lingbot_raw_faces'
    assert result['source']==depth_state['source']==rgb_state['capture']['source']
    assert result['position_world_m']==depth_state['position_world_m']==signature['position_world_m']
    paths=dict(base=job/result['source']['file'],rgb=folder/rgb_state['image'],depth=folder/depth_state['arrays']['completed'],
        mask=folder/rgb_state['capture']['path']/'mask.png')
    bundle=depth_state.get('accepted_bundle');valid_faces=None;accepted=set(FACE_NAMES)
    if bundle:
        depth_folder=folder/depth_state['path']
        assert read(depth_folder/'accepted-bundle.json')==bundle
        assert digest(depth_folder/'accepted-bundle.json')==signature['bundle_sha256']==result['accepted_bundle_sha256']
        paths.update(rgb=folder/bundle['rgb'],mask=folder/bundle['effective_mask'],support=folder/bundle['support'])
        accepted={row['face'] for row in bundle['faces'] if row['accepted']}
        valid_faces=[np.asarray(Image.open(depth_folder/f'{name}-accepted-valid.png'))>0 if name in accepted else None for name in FACE_NAMES]
    for key,path in paths.items():assert digest(path)==signature['inputs'][key]==result['inputs'][key],key
    for name,sha in result['files'].items():assert digest(out/name)==sha,name
    point_filter=verify_point_filter(out,signature)
    rgb=np.asarray(Image.open(paths['rgb']).convert('RGB'));radial=np.load(paths['depth'],allow_pickle=False)
    mask=np.asarray(Image.open(paths['mask']).convert('L'))>0
    valid=np.isfinite(radial)&(radial>1e-3)
    if bundle:valid &= np.asarray(Image.open(paths['support']))>0
    base=PlyData.read(paths['base'],mmap='r')['vertex'].data
    final=PlyData.read(out/'merged.ply',mmap='r')['vertex'].data
    patch=PlyData.read(out/'patch.ply',mmap='r')['vertex'].data
    assert base.dtype==final.dtype==patch.dtype
    assert len(final)==len(base)+len(patch)==result['count']
    assert len(base)==result['base_count'] and len(patch)==result['added_count']
    for offset in range(0,len(base),65536):
        assert base[offset:offset+65536].tobytes()==final[offset:min(offset+65536,len(base))].tobytes()
    assert patch.tobytes()==final[len(base):].tobytes()
    for field in patch.dtype.names:assert np.isfinite(patch[field]).all(),field
    cursor=0;faces=[];render_rows=[]
    for i,name in enumerate(FACE_NAMES):
        face=out/name;cam=read(face/'camera.json');saved=next(r for r in result['acceptance']['face_results'] if r['face']==name)
        actual_rgb=np.asarray(Image.open(face/'rgb.png'));actual_hole=np.asarray(Image.open(face/'mask.png'))>0
        local,rays=view_rays(camera(i,size=512,fov=100))
        expected_rgb=np.rint(sample_erp(rgb,rays)).clip(0,255).astype(np.uint8)
        np.testing.assert_array_equal(actual_rgb,expected_rgb)
        expected_hole=sample_erp(mask.astype(np.float32),rays)>0
        projected=sample_erp(np.where(valid,radial,0),rays)*local[...,2]
        supported=sample_erp(valid.astype(np.float32),rays)>.999999
        if bundle:
            supported &= np.isfinite(projected)&(projected>1e-3)
            if valid_faces[i] is not None:supported &= valid_faces[i]
            supported &= name in accepted
            expected_hole &= supported
            np.testing.assert_array_equal(np.asarray(Image.open(face/'support.png'))>0,supported)
        projected[~supported]=0
        np.testing.assert_array_equal(actual_hole,expected_hole)
        np.testing.assert_array_equal(np.load(face/'depth-projected.npy'),projected)
        dense=np.load(face/'depth-input.npy');extra=np.asarray(Image.open(face/'extrapolated.png'))>0
        np.testing.assert_array_equal(extra,~supported)
        np.testing.assert_array_equal(dense[supported],projected[supported]);assert np.all(np.isfinite(dense)&(dense>1e-3))
        patch_depth=np.load(out/f'patch-{name}-z.npy');patch_alpha=np.load(out/f'patch-{name}-alpha.npy')
        render_valid=actual_hole&np.isfinite(patch_depth)&(patch_depth>1e-3)&(patch_alpha>=.2)
        relative=np.abs(patch_depth[render_valid]-dense[render_valid])/dense[render_valid]
        if actual_hole.any():render_rows.append(dict(face=name,coverage=float(render_valid.sum()/actual_hole.sum()),
            relative_error_median=float(np.median(relative)) if len(relative) else None,relative_error_p90=float(np.quantile(relative,.9)) if len(relative) else None))
        np.testing.assert_allclose(np.asarray(cam['c2w'])[:3,:3],np.asarray(camera(i)['c2w'])[:3,:3],atol=1e-10)
        np.testing.assert_allclose(np.asarray(cam['c2w'])[:3,3],result['position_world_m'],atol=1e-10)
        np.testing.assert_allclose(np.asarray(cam['w2c'])@np.asarray(cam['c2w']),np.eye(4),atol=1e-10)
        world=PlyData.read(face/'patch-world.ply',mmap=False)['vertex'].data
        assert patch[cursor:cursor+len(world)].tobytes()==world.tobytes();cursor+=len(world)
        if saved['skipped']:
            assert not actual_hole.any() and not len(world);continue
        sharp=PlyData.read(face/'sharp/sharp_hard_layer0_camera.ply',mmap='r')['vertex'].data
        layer_z=np.load(face/'sharp/sharp_hard_layer0_z.npy');expected=expected_target_grid(dense)
        if signature['settings'].get('depth_geometry') == 'shared_panorama_at_final_ray':
            xyz=np.column_stack([sharp[k][:768*768] for k in ('x','y','z')]).astype(np.float64)
            rays=xyz@np.asarray(cam['c2w'])[:3,:3].T
            coverage=sample_erp(valid.astype(np.float32),rays)
            ranges=sample_erp(np.where(valid,radial,0),rays)
            target_range=np.divide(ranges,coverage,out=np.linalg.norm(rays,axis=1),where=coverage>0)
            expected_z=target_range*xyz[:,2]/np.linalg.norm(rays,axis=1)
            expected=np.where(coverage.reshape(768,768)>0,expected_z.reshape(768,768),expected)
        assert len(sharp)==2*768*768 and layer_z.shape==(768,768)
        # Float32 Torch interpolation vs CPU bilinear differs by a few ulps.
        np.testing.assert_allclose(layer_z,expected,atol=5e-5,rtol=2e-5)
        np.testing.assert_array_equal(sharp['z'][:768*768].reshape(768,768),layer_z)
        ids=np.load(face/'selected-rows.npy');weights=np.load(face/'selected-weights.npy')
        assert len(ids)==len(world)==saved['count'] and np.all((ids>=0)&(ids<768*768))
        assert len(np.unique(ids))==len(ids) and np.all((weights>0)&(weights<=1))
        # Reconstruct the six-ring selection independently from the authority
        # image, then check every selected lattice row and overlap weight.
        settings=signature['settings']
        assert settings['blend_reference_pixels']==6 and settings['blend_reference_short_side']==512
        radius=round(settings['blend_reference_pixels']*min(actual_hole.shape)/settings['blend_reference_short_side'])
        distance=distance_transform_edt(~actual_hole)
        labels=np.zeros(actual_hole.shape,np.uint8);labels[actual_hole]=1
        band=(~actual_hole)&(distance<=radius)
        labels[band]=np.ceil(distance[band]/(radius/6)).clip(1,6).astype(np.uint8)+1
        if bundle:labels[~supported]=0
        np.testing.assert_array_equal(np.asarray(Image.open(face/'ring-labels.png')),labels)
        axis=np.arange(768)*512//768;label_grid=labels[np.ix_(axis,axis)].reshape(-1)
        u=(np.arange(6)+.5)/6;ring_weights=1-u*u*(3-2*u)
        native_weights=np.r_[0.,1.,ring_weights][label_grid]
        full_xyz=np.column_stack([sharp[n][:768*768] for n in ('x','y','z')])
        patch_rays=full_xyz.astype(np.float64)@np.asarray(cam['c2w'])[:3,:3].T
        overlap=angular_weights(patch_rays,i,fov=100)
        if valid_faces is not None:
            face_weights=[]
            for other,face_valid in enumerate(valid_faces):
                if face_valid is None:face_weights.append(np.zeros(len(patch_rays)));continue
                other_cam=camera(other,size=512,fov=100);local_rays=patch_rays@np.asarray(other_cam['c2w'])[:3,:3]
                z=local_rays[:,2];f=other_cam['K_pixel_edge'][0][0]
                x=local_rays[:,0]/np.maximum(z,1e-8)*f+255.5;y=local_rays[:,1]/np.maximum(z,1e-8)*f+255.5
                visible=(z>0)&(x>=0)&(x<=511)&(y>=0)&(y<=511)
                covered=map_coordinates(face_valid.astype(np.float32),[y,x],order=1,mode='nearest',prefilter=False)>.999999
                face_weights.append(angular_weights(patch_rays,other,fov=100)*visible*covered)
            total=np.sum(face_weights,axis=0)
            overlap=np.divide(face_weights[i],total,out=np.zeros_like(total),where=total>1e-6)
        point_keep=np.load(face/'point-keep.npy')
        before_ids=np.flatnonzero((label_grid>0)&(overlap>1e-4))
        expected_ids=before_ids[point_keep[before_ids]]
        np.testing.assert_array_equal(ids,expected_ids)
        np.testing.assert_allclose(weights,native_weights[ids]*overlap[ids],atol=1e-7)
        camera_patch=PlyData.read(face/'patch-camera.ply',mmap=False)['vertex'].data
        for field in sharp.dtype.names:
            if field!='opacity':np.testing.assert_array_equal(camera_patch[field],sharp[field][ids])
        alpha=1/(1+np.exp(-np.clip(sharp['opacity'][ids].astype(np.float64),-60,60)))
        weighted=np.clip(alpha*weights,1e-6,1-1e-6)
        np.testing.assert_allclose(camera_patch['opacity'],np.log(weighted)-np.log1p(-weighted),atol=1e-6)
        xyz=np.column_stack([world[n] for n in ('x','y','z')]);camera_xyz=np.c_[xyz,np.ones(len(xyz))]@np.asarray(cam['w2c']).T
        expected_xyz=np.column_stack([camera_patch[n] for n in ('x','y','z')])
        transform_error=float(np.max(np.abs(camera_xyz[:,:3]-expected_xyz)))
        assert transform_error<1e-5
        z_error=float(np.max(np.abs(camera_xyz[:,2]-expected.reshape(-1)[ids])))
        assert z_error<1e-4
        rot=np.asarray(cam['c2w'])[:3,:3]
        expected_cov=rot[None]@covariance(camera_patch)@rot.T[None]
        world_cov=covariance(world)
        cov_relative=np.linalg.norm(world_cov-expected_cov,axis=(1,2))/np.maximum(np.linalg.norm(expected_cov,axis=(1,2)),1e-12)
        assert float(cov_relative.max())<1e-5
        sharp_report=read(face/'sharp/sharp_hard_acceptance.json')
        assert sharp_report['status']==('diagnostic_only' if diagnostic_only else 'accepted')
        assert sharp_report['alignment_unet']=='bypassed'
        if diagnostic_only:assert sharp_report['quality_gate_enabled'] is False and 'thresholds' not in sharp_report
        diag=sharp_report['center_diagnostics'];assert diag['surface_covariance']['applied']
        assert diag['surface_covariance']['mode']=='surface_jacobian' and diag['disabled_unsupervised_layers']==1
        assert diag['surface_covariance']['tangent_sigma_factor']==signature['settings']['surface_tangent_sigma']
        assert diag['preserved_ray_error_max']<1e-5
        selection=saved['selection'];assert selection['blend_ring_count']==6 and selection['blend_reference_pixels']==6
        assert selection['point_filter_applied'] and selection['before_point_filter']==len(before_ids)
        assert selection['point_filter_removed']==len(before_ids)-len(ids)
        assert selection['blend_reference_short_side']==512 and selection['blend_radius_pixels']==radius
        assert selection['core_opacity_mode']=='full' and not selection['posthoc_depth_fit']
        # Recompute the reference's SHARP image metrics from the saved render.
        import cv2
        rendered=np.load(face/'sharp/sharp_hard_render_depth.npy');alpha_map=np.load(face/'sharp/sharp_hard_render_alpha.npy')
        rh,rw=rendered.shape;target=cv2.resize(dense,(rw,rh),interpolation=cv2.INTER_LINEAR)
        hmask=cv2.resize(actual_hole.astype(np.uint8),(rw,rh),interpolation=cv2.INTER_NEAREST)>0
        rvalid=hmask&np.isfinite(rendered)&(rendered>1e-3)&np.isfinite(alpha_map)&(alpha_map>=.2)
        errors=np.abs(rendered[rvalid]-target[rvalid])/target[rvalid]
        measured=dict(relative_error_median=float(np.median(errors)) if len(errors) else None,
            relative_error_p90=float(np.quantile(errors,.9)) if len(errors) else None,
            coverage=float(rvalid.sum()/max(1,hmask.sum())))
        for key,value in measured.items():
            if value is None:assert sharp_report['render_depth_metrics_in_hole'][key] is None
            else:np.testing.assert_allclose(value,sharp_report['render_depth_metrics_in_hole'][key],atol=1e-7)
        if not diagnostic_only:
            assert np.median(errors)<=.03 and np.quantile(errors,.9)<=.08
            assert alpha_map[hmask].mean()>=.95 and np.quantile(alpha_map[hmask],.1)>=.85
        faces.append(dict(face=name,added=len(world),extrapolated_pixels=int(extra.sum()),center_depth_error_max_m=z_error,
            camera_roundtrip_max_m=transform_error,covariance_relative_error_max=float(cov_relative.max())))
    assert cursor==len(patch)
    render=read(out/'render-acceptance.json');failures=[]
    assert len(render_rows)==len(render['patch'])
    for row,saved in zip(render_rows,render['patch']):
        assert row['face']==saved['face']
        for key in ('coverage','relative_error_median','relative_error_p90'):
            if row[key] is None:assert saved[key] is None
            else:np.testing.assert_allclose(row[key],saved[key],atol=1e-7)
        if not diagnostic_only:
            if row['relative_error_median']>.03:failures.append(f"{row['face']}: patch median depth error > 0.03")
            if row['relative_error_p90']>.08:failures.append(f"{row['face']}: patch p90 depth error > 0.08")
    assert render.get('coverage_policy')=='diagnostic_only'
    if diagnostic_only:
        assert signature['settings']['patch_quality_gate'] is False
        assert render['status']=='diagnostic_only' and render['quality_gate_enabled'] is False
        assert 'thresholds' not in render and 'failures' not in render
        assert result['acceptance']['status']=='accepted' and result['acceptance']['quality_gate_enabled'] is False
        assert result['warnings']==[]
    else:
        assert 'min_patch_coverage' not in render['thresholds']
        assert render['failures']==failures and render['status']==('rejected' if failures else 'accepted')
        assert result['acceptance']['status']==('review_required' if failures else 'accepted')
        assert bool(result.get('warnings'))==bool(failures)
    return dict(passed=True,camera=result['camera_id'],base_prefix_exact=True,source_rgb_depth_mask_unchanged=True,
        reference_sources_and_checkpoint_verified=True,patch_revision=result['patch_revision'],
        base_count=len(base),added_count=len(patch),final_count=len(final),artifact_hashes_checked=len(result['files']),
        all_six_face_inputs_match_current_panoramas=True,point_filter=point_filter,faces=faces,render=render['patch'],
        quality_gate_enabled=not diagnostic_only,render_quality_passed=None if diagnostic_only else not failures,render_failures=failures)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('job',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    report=verify(args.job);text=json.dumps(report,ensure_ascii=False,indent=2)
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)
