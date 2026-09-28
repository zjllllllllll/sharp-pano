"""Full output audit for render-only unknown-space revisions."""
from pathlib import Path
import argparse
import json
import time
from collections import Counter

import numpy as np
from PIL import Image
from batch import digest,write_json,HERE,ROOT
from occlusion import VERSION,BACKGROUND,REASONS
from cameras import square_order, NUMBERING


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def validate(run,source):
    started=time.perf_counter();inputs=read(run/'inputs.json');catalog=read(run/'catalog.json')
    status=read(run/'render-status.json')
    assert status['complete'] and not status['errors'],status
    assert not list(run.glob('*/panoramas/*.partial')),'Unfinished camera artifacts'
    manifest=read(run/'rerender.json')
    equivalence_path=run/'implementation/valid-depth-equivalence.json'
    equivalence=read(equivalence_path) if equivalence_path.exists() else None
    current_hash=digest(HERE/'occlusion.py')
    if equivalence:
        assert equivalence['passed'] and equivalence['fully_valid_only']
        assert equivalence['current_sha256']==current_hash
        assert equivalence['baseline_sha256']==digest(run/'implementation/occlusion-valid-depth.py')
    assert manifest['plan_sha256']==digest(ROOT/'novel-view-occlusion-mask-plan.md')
    # Protected model/geometry/mask inference code is not in scope for edits.
    original=read(source/'source-fingerprints.json')
    protected={p:s for p,s in original.items() if p.startswith(('sharp-sixface/','DepthMaster/')) or
               (p.startswith('pano-web/static/') and not Path(p).name.startswith('camera-lab') and Path(p).name!='index.html') or
               (p.startswith('pano-web/') and not p.startswith('pano-web/static/') and Path(p).name not in {'server.py','camera_lab.py','test_camera_lab.py'})}
    for p,s in protected.items():assert digest(ROOT/p)==s,p
    for p,s in read(run/'previous-render-fingerprints.json').items():assert digest(source/p)==s,p
    for relative in manifest['immutable_hardlinks']:
        assert (run/relative).samefile(source/relative),relative
    rows=[];states=Counter();reasons=Counter();masks=[];all_count=0;origin_checks=[]
    for item in inputs:
        name=item['stem'];job=run/name;old=source/name
        assert digest(Path(item['source']))==item['sha256']
        assert digest(job/'input.png')==item['rgb_sha256']
        for filename in ['outline.json','cameras.json']:
            assert digest(job/filename)==digest(old/filename),(name,filename)
        previous=read(old/'sample-cameras.json');info=read(job/'sample-cameras.json')
        assert info['render_version']==VERSION and info['render_complete']
        for key in ['spacing_m','inward_offset_m','height_world_y_m','origin_world_m','room_geometry','inset_geometry','empty_reason']:
            assert previous[key]==info[key],(name,key)
        assert info['numbering']==NUMBERING
        assert [c['id'] for c in info['cameras']]==[f'C{i:03d}' for i in range(info['count'])]
        assert info['cameras'][0]['position_world_m']==[0,0,0]
        assert info['cameras']==sorted(info['cameras'],key=square_order)
        assert {c['id']:c['position_world_m'] for c in previous['cameras']}=={
            c['previous_id']:c['position_world_m'] for c in info['cameras'] if c['previous_id'] is not None}
        assert all(c['position_world_m']==[0,0,0] and c['reference_only'] for c in info['cameras'] if c['previous_id'] is None)
        fp=info['render_fingerprint']
        assert fp['depth']==digest(job/'depth.npy') and fp['rgb']==item['rgb_sha256']
        source_depth=np.load(job/'depth.npy',mmap_mode='r')
        source_valid=np.isfinite(source_depth)&(source_depth>0)
        if (job/'depth-valid.npy').exists():source_valid &= np.load(job/'depth-valid.npy')
        if fp['implementation']!=current_hash:
            assert equivalence and source_valid.all() and fp['implementation']==equivalence['baseline_sha256'],name
        assert fp['output_implementation']==digest(HERE/'render_unknown.py')
        scene_mask=[]
        for camera in info['cameras']:
            arrays={}
            for kind,path in camera['views'].items():
                with Image.open(job/path) as im:
                    assert im.size==(1024,512),(name,camera['id'],kind)
                    assert im.mode==('RGB' if kind in {'raw','masked','overlay','reason_color'} else 'L')
                    arrays[kind]=np.asarray(im)
            bits=arrays['reason'];masked=bits!=0
            assert np.all(bits<=15)
            np.testing.assert_array_equal(arrays['mask'],masked.astype(np.uint8)*255)
            np.testing.assert_array_equal(arrays['coverage'],(~masked).astype(np.uint8)*255)
            np.testing.assert_array_equal(arrays['unknown'],(bits&2!=0).astype(np.uint8)*255)
            np.testing.assert_array_equal(arrays['uncertain'],(bits&12!=0).astype(np.uint8)*255)
            np.testing.assert_array_equal(arrays['holes'],(bits&1!=0).astype(np.uint8)*255)
            assert np.all(arrays['masked'][masked]==BACKGROUND)
            np.testing.assert_array_equal(arrays['masked'][~masked],arrays['raw'][~masked])
            stats=camera['mask_stats'];fraction=float(masked.mean())
            assert abs(stats['mask_fraction']-fraction)<1e-9
            assert abs(camera['coverage']-(1-fraction))<1e-9
            for bit,reason in REASONS.items():
                assert abs(stats['reasons'][reason]-float((bits&bit!=0).mean()))<1e-9
                reasons[reason]+=int(np.count_nonzero(bits&bit!=0))
            if stats['camera_state']!='known_free':assert np.all(bits&8!=0)
            if camera['coincides_with_original']:
                extra=float(((bits&2!=0)&(bits&1==0)).mean())
                assert extra<.001,(name,'unexpected source-camera self-occlusion',extra)
                if info['volume']['source_valid_fraction']==1:
                    assert not masked.any(),(name,'valid source footprint lost',fraction)
                h,w=source_valid.shape
                expected=~source_valid.reshape(512,h//512,1024,w//1024).all(axis=(1,3))
                np.testing.assert_array_equal(masked,expected,err_msg=f'{name}: source validity footprint mismatch')
                origin_checks.append(dict(scene=name,mask_fraction=fraction,
                    source_invalid_pixels=int((~source_valid).sum()),expected_mask_pixels=int(expected.sum()),
                    additional_mask_pixels=int((masked&~expected).sum()),
                    additional_unknown_over_reliable_surface_fraction=extra))
            states[stats['camera_state']]+=1
            with np.load(job/f'panoramas/{camera["id"]}-depths.npz',allow_pickle=False) as depths:
                for key in ['surface_range_m','unknown_entry_range_m']:
                    d=depths[key];assert d.shape==(512,1024) and d.dtype==np.float32
                    # A camera exactly on a boundary can legitimately hit at 0.
                    assert ((d>=0)&~np.isnan(d)).all()
            scene_mask.append(fraction);all_count+=1
        masks.extend(scene_mask)
        rows.append(dict(scene=name,count=info['count'],mask_mean=float(np.mean(scene_mask)) if scene_mask else None))
        print(f'VALIDATED_MASK {name}: {info["count"]}',flush=True)
    assert catalog['total']==len(inputs)==catalog['render_complete']
    assert all_count==catalog['camera_count']==sum(r['count'] for r in rows)
    result=dict(passed=True,scenes=len(inputs),cameras=all_count,version=VERSION,
        camera_states=dict(states),mask_fraction_mean=float(np.mean(masks)),
        zero_camera_scenes=[r['scene'] for r in rows if not r['count']],
        added_origin_references=all_count-sum(read(source/item['stem']/'sample-cameras.json')['count'] for item in inputs),
        preserved_original_files=len(read(run/'previous-render-fingerprints.json')),
        preserved_immutable_links=len(manifest['immutable_hardlinks']),
        reason_pixel_counts=dict(reasons),rows=rows,seconds=time.perf_counter()-started,
        original_camera_checks=origin_checks,
        implementation_equivalence=equivalence,
        note='All files audited; geometric mask is based on estimated RGB-D, not ground truth. No AI inpainting or SHARP changes.')
    write_json(run/'mask-validation.json',result)
    lines=['# 新视角未知空间遮挡 Mask：全量结果','',
           f'已完成 {len(inputs)} 张输入 / {all_count} 个相机，1024×512。全部文件校验通过。',
           '原相机 C000，其余按方形圈从正上方顺时针编号；旧采样位置不变，缺少原视点时补参考位。轮廓、RGB-D、SHARP 点云与旧预览保留。未进行 AI 修补。',
           f'原视点回归验证：{len(origin_checks)} 张，新增误删像素总数 {sum(r["additional_mask_pixels"] for r in origin_checks)}；原始无效深度对应的 Mask 保留。逐像素核对源有效性，不再忽略缺失表面。',
           f'相机状态：{dict(states)}。unknown / unconfirmed 相机按方案整图保守 Mask，不代表确认位于实体内部。','',
           f'[完整校验]({(run/"mask-validation.json").as_posix()}) · [实现与局限]({(HERE/"UNKNOWN-MASK.md").as_posix()})','',
           '## 示例：0.1 / C001','',
           f'![Mask 原因叠加]({(run/"0.1/panoramas/C001-overlay.png").as_posix()})','',
           '红色表示未知空间先于可靠表面遮挡。颜色叠加是调试显示，不是生成补图。','',
           '## 全部场景','',
           '| 场景 | 相机数 | 平均 Mask | 第一个相机结果 | 全部相机元数据 |',
           '| --- | ---: | ---: | --- | --- |']
    for row in rows:
        name=row['scene'];folder=run/name
        fraction=f'{row["mask_mean"]:.1%}' if row['count'] else '—'
        views=' · '.join(f'[{label}]({(folder/"panoramas"/filename).as_posix()})' for label,filename in [
            ('结果','C001.png'),('遮挡前','C001-raw.png'),('Mask','C001-mask.png'),('叠加','C001-overlay.png')]) if row['count'] else '内缩区域无格点'
        lines.append(f'| {name} | {row["count"]} | {fraction} | {views} | [相机列表]({(folder/"sample-cameras.json").as_posix()}) |')
    (run/'mask-report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'runs/20260922-unknown-v2')
    p.add_argument('--source',type=Path,default=HERE/'runs/20260921-unknown-v1');a=p.parse_args();validate(a.run,a.source)
