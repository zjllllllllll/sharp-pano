"""Check the revised camera output and preservation of the previous version."""
import argparse
import json
from pathlib import Path
from batch import HERE, digest, write_json
from validate_run import validate


def verify(previous, current):
    result=validate(current,expected_inset=.5)
    assert result['scene_count']==47 and result['camera_count']==3263
    assert result['zero_camera_scenes']==['0.4']
    fingerprints=json.loads((current/'previous-camera-fingerprints.json').read_text(encoding='utf-8'))
    for relative,expected in fingerprints.items():
        assert digest(previous/relative)==expected, f'Old camera artifact changed: {relative}'
    manifest=json.loads((current/'resample.json').read_text(encoding='utf-8'))
    for relative in manifest['immutable_hardlinks']:
        assert (previous/relative).samefile(current/relative), relative
    inputs=json.loads((current/'inputs.json').read_text(encoding='utf-8'))
    reused=0
    for item in inputs:
        old,new=previous/item['stem'],current/item['stem']
        assert digest(old/'outline.json')==digest(new/'outline.json')
        old_info=json.loads((old/'sample-cameras.json').read_text(encoding='utf-8'))
        new_info=json.loads((new/'sample-cameras.json').read_text(encoding='utf-8'))
        cached={c['id']:c for c in old_info['cameras']}
        old_positions={tuple(c['position_world_m']) for c in cached.values()}
        new_positions={tuple(c['position_world_m']) for c in new_info['cameras']}
        assert old_positions<=new_positions
        for camera in new_info['cameras']:
            if 'reused_camera_id' not in camera:
                continue
            original=cached[camera['reused_camera_id']]
            assert camera['position_world_m']==original['position_world_m']
            assert digest(new/camera['panorama'])==digest(old/original['panorama'])
            assert digest(new/'panoramas'/f'{camera["id"]}-coverage.png')==digest(old/'panoramas'/f'{original["id"]}-coverage.png')
            reused+=1
    assert reused==794
    summary=dict(passed=True,scenes=47,cameras=3263,previous_cameras_preserved=794,
                 renders_reused_by_position=reused,new_renders=3263-reused,
                 spacing_m=.5,inset_m=.5,height_world_y_m=0.,previous_directory=previous.name,
                 immutable_source_files=len(manifest['immutable_hardlinks']))
    write_json(current/'revision-validation.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--previous',type=Path,default=HERE/'runs/20260921-all')
    parser.add_argument('--current',type=Path,default=HERE/'runs/20260921-inset-0p5')
    args=parser.parse_args()
    verify(args.previous,args.current)
