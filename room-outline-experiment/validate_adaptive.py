"""Full adaptive sampling + per-camera artifact audit, including cache provenance."""
from pathlib import Path
from collections import Counter
import json
import time
import numpy as np
from PIL import Image
from batch import digest,write_json,HERE
from cameras import sample_cameras,SamplingError,square_order,MIN_NEIGHBORS,OFFSET_STOP,MASK_THRESHOLD
from shapely.geometry import shape,Point
from adaptive_sampling import SOURCE,TARGET,read
from candidate_selection import VERSION as SELECTION_VERSION


def expected_component(pool):
    # Independent transitive closure (not the production flood-fill function).
    eligible={tuple(c['grid']) for c in pool if c['coincides_with_original'] or c['mask_stats']['mask_fraction']<=.15}
    reached={(0,0)}
    while True:
        next_reached=reached|{p for p in eligible if any(abs(p[0]-q[0])+abs(p[1]-q[1])==1 for q in reached)}
        if next_reached==reached:break
        reached=next_reached
    return [c['id'] for c in pool if tuple(c['grid']) in reached]


def validate(run=TARGET,source=SOURCE):
    started=time.perf_counter();manifest=read(run/'adaptive-sampling.json');catalog=read(run/'catalog.json')
    status=read(run/'render-status.json');assert status['complete'] and not status['errors'],status
    assert digest(source/'catalog.json')==manifest['source_catalog_sha256']
    equivalence=read(run/'implementation/valid-depth-equivalence.json')
    current=digest(HERE/'occlusion.py')
    assert equivalence['passed'] and current==equivalence['current_sha256']
    assert digest(run/'implementation/occlusion-valid-depth.py')==equivalence['baseline_sha256']
    totals=Counter();rows=[]
    for item in read(run/'inputs.json'):
        name=item['stem'];job=run/name;old=source/name;info=read(job/'sample-cameras.json');prior=read(old/'sample-cameras.json')
        record=manifest['scenes'][name]
        for filename,key in [('sample-cameras.json','source_metadata_sha256'),('cameras.png','source_map_sha256'),('outline.json','source_outline_sha256')]:
            assert digest(old/filename)==record[key],(name,filename,'source changed')
        for filename in ['input.png','outline.json','cameras.json']:
            assert digest(job/filename)==digest(old/filename),(name,filename)
        for path in old.rglob('*'):
            rel=path.relative_to(old)
            if path.is_file() and rel.parts[0]!='panoramas' and path.suffix in {'.npy','.npz','.bin','.las'}:
                assert path.samefile(job/rel),(name,rel)
        try:expected=sample_cameras(read(job/'outline.json')['polygon_m'])
        except SamplingError as error:expected=error.info
        expected=json.loads(json.dumps(expected))  # GeoJSON tuples serialize as JSON arrays.
        for key in ['inward_offset_m','neighbor_count','offset_attempts','sampling_error','sampling_warning','inset_geometry']:
            assert info[key]==expected[key],(name,key)
        pool=info['all_cameras']
        assert info['geometric_camera_count']==expected['count']==len(pool)
        assert pool==sorted(pool,key=square_order)
        assert [c['id'] for c in pool]==[f'C{i:03d}' for i in range(len(pool))]
        assert [c['grid'] for c in pool]==[c['grid'] for c in expected['cameras']]
        assert info['cameras'][0]['position_world_m']==[0.,0.,0.]
        assert info['neighbor_count']>=MIN_NEIGHBORS or info['inward_offset_m']<OFFSET_STOP
        if len(pool)==1:assert info['sampling_error']
        assert bool(info['selection_error'])==(info['count']==1)
        assert all(a['neighbor_count']<MIN_NEIGHBORS and a['offset_m']>=OFFSET_STOP for a in info['offset_attempts'][:-1])
        room=shape(info['room_geometry']);inner=shape(info['inset_geometry'])
        previous={c['id']:c for c in prior['cameras']}
        assert info['render_complete'] and len(info['cameras'])==info['count']
        valid=None;mask_kept=[];reused=0
        before=read(run/'selection-history/before-mask15-connected'/name/'sample-cameras.json')
        assert [{k:v for k,v in c.items() if k!='selection_state'} for c in pool]==before['cameras'],(name,'camera payload or pose changed')
        for camera in pool:
            x,y,z=camera['position_world_m'];assert y==0 and x/.5==round(x/.5) and z/.5==round(z/.5)
            if not camera['reference_only']:
                assert room.contains(Point(x,z)) and inner.covers(Point(x,z))
                assert camera['boundary_distance_m']+1e-9>=info['inward_offset_m']
            fp=camera['render_fingerprint']
            assert fp['rgb']==info['render_fingerprint']['rgb'] and fp['depth']==info['render_fingerprint']['depth']
            assert fp['output_implementation']==digest(HERE/'render_unknown.py')
            if fp['implementation']!=current:
                if valid is None:
                    depth=np.load(job/'depth.npy',mmap_mode='r');valid=bool((np.isfinite(depth)&(depth>0)).all())
                assert valid and fp['implementation']==equivalence['baseline_sha256']
            cached=previous.get(camera.get('reused_camera_id'))
            if cached:
                reused+=1
                assert cached['position_world_m']==camera['position_world_m']
                assert cached['mask_stats']==camera['mask_stats'] and fp==prior['render_fingerprint']
                for kind,path in camera['views'].items():assert (job/path).samefile(old/cached['views'][kind])
                assert (job/f'panoramas/{camera["id"]}-depths.npz').samefile(old/f'panoramas/{cached["id"]}-depths.npz')
            arrays={}
            for kind,path in camera['views'].items():
                with Image.open(job/path) as im:
                    assert im.size==(1024,512)
                    arrays[kind]=np.asarray(im)
            mask=arrays['mask']>0;bits=arrays['reason'];assert np.all(bits<=15)
            np.testing.assert_array_equal(mask,bits!=0)
            np.testing.assert_array_equal(arrays['coverage'],(~mask).astype(np.uint8)*255)
            np.testing.assert_array_equal(arrays['unknown'],(bits&2!=0).astype(np.uint8)*255)
            np.testing.assert_array_equal(arrays['uncertain'],(bits&12!=0).astype(np.uint8)*255)
            np.testing.assert_array_equal(arrays['holes'],(bits&1!=0).astype(np.uint8)*255)
            assert abs(float(mask.mean())-camera['mask_stats']['mask_fraction'])<1e-12
            assert abs(camera['coverage']-(1-float(mask.mean())))<1e-12
            np.testing.assert_array_equal(arrays['masked'][~mask],arrays['raw'][~mask])
            if camera['coincides_with_original']:
                # Cache identity above also pins this to the earlier pixelwise origin audit.
                assert cached is not None
            with np.load(job/f'panoramas/{camera["id"]}-depths.npz') as depths:
                for key in ['surface_range_m','unknown_entry_range_m']:
                    values=depths[key];assert values.shape==(512,1024) and values.dtype==np.float32
                    assert ((values>=0)&~np.isnan(values)).all()
            if camera['coincides_with_original'] or float(mask.mean())<=MASK_THRESHOLD:mask_kept.append(camera['id'])
        retained=expected_component(pool)
        assert retained==info['default_retained_camera_ids']==[c['id'] for c in info['cameras']]
        assert info['selection_version']==SELECTION_VERSION and info['selection_complete']
        assert info['selection']['connectivity']==4 and info['selection']['mask_threshold']==.15
        for c in pool:
            state='origin' if c['coincides_with_original'] else 'retained' if c['id'] in retained else 'disconnected' if c['id'] in mask_kept else 'excluded'
            assert c['selection_state']==state,(name,c['id'])
        assert info['selection']['counts']==dict(Counter(c['selection_state'] for c in pool),**{k:sum(c['selection_state']==k for c in pool) for k in ['origin','retained','excluded','disconnected','pending']})
        exported=read(job/'candidate-cameras.json')
        assert exported['cameras']==info['cameras'] and exported['selection']==info['selection']
        assert exported['selection_error']==info['selection_error']
        assert reused==info['reused_render_count']==record['reused']
        for filename in ['cameras.png','camera-map-base.png']:
            with Image.open(job/filename) as image:assert list(image.size)==info['camera_map_projection']['image_size']
        for camera in pool:
            x,_,z=camera['position_world_m'];p=info['camera_map_projection']
            assert p['xlim'][0]<=x<=p['xlim'][1] and p['zlim'][0]<=z<=p['zlim'][1]
        published=next(r for r in catalog['scenes'] if r['id']==name)
        assert published['sample']==info and published['cameras']==info['cameras'] and published['camera_map_base']
        totals.update(cameras=len(pool),reused=reused,new=len(pool)-reused,retained=len(retained),
            mask_excluded=len(pool)-len(mask_kept),disconnected=len(mask_kept)-len(retained),
            relaxed=int(info['inward_offset_m']<.5),sampling_errors=int(bool(info['sampling_error'])),selection_errors=int(bool(info['selection_error'])))
        rows.append(dict(scene=name,offset_m=info['inward_offset_m'],neighbors=info['neighbor_count'],geometric_count=len(pool),retained=len(retained)))
        print(f'VALIDATED_ADAPTIVE {name}: {info["count"]}',flush=True)
    assert len(rows)==catalog['total']==catalog['render_complete']==47
    assert totals['retained']==catalog['camera_count']==status['camera_count']
    assert totals['cameras']==catalog['geometric_camera_count']==status['geometric_camera_count']
    assert catalog['rules']['connectivity']==4 and catalog['rules']['mask_threshold']==.15
    assert len(status['selection_errors'])==totals['selection_errors']
    assert not list(run.glob('*/panoramas/*.partial'))
    result=dict(passed=True,scenes=len(rows),totals=dict(totals),default_mask_threshold=MASK_THRESHOLD,
        rows=rows,seconds=time.perf_counter()-started,
        checks=['Exact eight-neighbor stopping rule','Only-origin explicit error','All poses within actual inset (except origin reference)',
        'Old scene metadata/outline/preview preserved','Per-camera provenance','Every PNG and depth payload checked',
        'Mask fractions match actual masks','Camera numbering and map coordinates','Published catalog equals artifacts',
        'Fixed <=15% then independent four-connected origin closure','No diagonal/gap bridges',
        'Canonical candidates, export and state labels agree','All original geometric camera records preserved'])
    write_json(run/'sampling-validation.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':validate()
