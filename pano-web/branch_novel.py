"""Create an independent stage-6 experiment from completed local stages 1–5."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time
import uuid
from novel_pipeline import read,write_json,digest,source_ply


def create(source):
    source=source.resolve();before=digest(source/'job.json');prior=read(source/'job.json')
    if prior.get('busy'):raise ValueError('Source job is running')
    if any(str(i) not in prior['stages'] for i in range(1,6)):raise ValueError('Stages 1–5 required')
    frozen=source_ply(source,prior.get('novel_scene_variant','masked'))
    target=source.parent/uuid.uuid4().hex;target.mkdir()
    files=[]
    for path in sorted(source.iterdir()):
        if not path.is_file():continue
        include=(path.suffix in ('.ply','.bin','.npy','.npz') or
            path.name in ('input.png','depth.png','depth-gray.png','cameras.json','model.json') or
            path.name.startswith(('gaussian-dense-','point-')) or
            path.name in {f'stage-{i}.{ext}' for i in range(2,6) for ext in ('json','log')})
        if include:
            # Real copies keep subsequent edits to either experiment isolated.
            shutil.copy2(path,target/path.name)
            files.append(dict(file=path.name,bytes=path.stat().st_size))
    if digest(target/frozen['file'])!=frozen['sha256']:raise ValueError('Base copy hash differs')
    if digest(source/'job.json')!=before:raise ValueError('Source changed during branch creation')
    branch=dict(name='最大 Mask 候选分支',source_job=source.name,start_stage=6,created=time.time(),
        source_state_sha256=before,source_scene=frozen,selection_policy='max-mask',
        ranking='Existing candidate geometry mask_fraction, descending; ties by numeric camera ID',
        copied_files=files)
    state=dict(id=target.name,created=time.time(),busy=False,running_stage=None,error=None,
        message='已独立复制前五步，等待最大 Mask 候选分支',
        stages={str(i):deepcopy(prior['stages'][str(i)]) for i in range(1,6)},
        novel_scene_variant=frozen['variant'],camera_selection_policy='max-mask',rgb_repair_model='gpt-image-2',
        branch={k:v for k,v in branch.items() if k!='copied_files'})
    write_json(target/'branch.json',branch);write_json(target/'job.json',state)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path)
    print(create(parser.parse_args().source),flush=True)
