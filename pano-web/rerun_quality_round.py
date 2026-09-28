"""Explicit administrative rerun: preserve old outputs and reuse verified GPT returns.

Only prepares state and dispatches the existing server/verified GPU launchers.
"""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
import time
import uuid

from face_quality import PARAMETERS, POLICY_HASH, VERSION
from rerun_novel import api, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--round', type=int, choices=(2,), default=2)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / 'runs'
    parent = (root / args.job).resolve()
    if parent.parent != root.resolve():
        raise ValueError('Invalid job path')
    base = f'http://127.0.0.1:{args.port}'
    online = api(base, '/api/state')
    state = read(parent / 'job.json')
    child = root / state['iteration_jobs']['2']
    child_state = read(child / 'job.json')
    if online['gpu_busy'] or state['busy'] or child_state['busy']:
        raise ValueError('Wait for the active task')
    if child_state['iteration_parent'] != parent.name:
        raise ValueError('Iteration owner mismatch')
    if state.get('quality_run', {}).get('status') in ('running', 'awaiting_capture'):
        raise ValueError('An existing quality attempt must finish first')
    published = child_state['stages']['7']
    if state['stages']['11']['capture'] != published['capture']:
        raise ValueError('Parent and child RGB publications differ')
    folder = child / 'novel' / published['revision']
    preview = published['capture']
    old = (folder / preview['path']).resolve()
    if not old.is_relative_to(folder.resolve()):
        raise ValueError('Invalid capture path')
    selection_path = folder / 'selection.json'
    selection = read(selection_path)
    if selection['current']['camera_id'] != published['camera_id']:
        raise ValueError('Confirmed camera changed')
    for name, expected in [('raw.png', preview['raw_sha256']), ('mask.png', preview['mask_sha256'])]:
        if sha(old / name) != expected:
            raise ValueError('Capture changed: ' + name)
    if sha(child / preview['source']['file']) != preview['source']['sha256']:
        raise ValueError('Frozen Gaussian source changed')
    reused = {}
    for face in preview['faces']:
        receipt = read(old / 'repair' / (face + '.json'))
        if receipt.get('skipped'):
            continue
        model = old / 'repair' / (face + '-model.png')
        if sha(model) != receipt['model_sha256']:
            raise ValueError('GPT output changed: ' + face)
        reused[face] = receipt['model_sha256']
    watched = (parent / 'job.json', child / 'job.json', selection_path)
    before = {str(p): sha(p) for p in watched}
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    backup = parent / 'reruns' / ('quality-round-2-' + stamp)
    backup.mkdir(parents=True)
    for role, job in [('parent', parent), ('child', child)]:
        out = backup / role
        out.mkdir()
        for p in job.iterdir():
            if p.is_file() and (p.name == 'job.json' or p.name.startswith('stage-')):
                shutil.copy2(p, out / p.name)
    shutil.copy2(selection_path, backup / 'selection.json')
    new = old.parent / uuid.uuid4().hex[:16]
    new.mkdir()
    for p in old.iterdir():
        if p.is_file():
            shutil.copy2(p, new / p.name)
    shutil.copytree(old / 'repair', new / 'repair')
    replacement = deepcopy(preview)
    replacement.update(path=new.relative_to(folder).as_posix(), rerun_of=preview['path'],
                       rerun_created=time.time(), quality_policy_hash=POLICY_HASH)
    write(new / 'capture.json', replacement)
    run = dict(id=uuid.uuid4().hex, round_start=10, target_stage=13, next_stage=11,
               execution_job=child.name, source_sha256=preview['source']['sha256'],
               status='ready', max_attempts=3, policy_version=VERSION, policy_hash=POLICY_HASH,
               attempts=[dict(id=uuid.uuid4().hex, camera_id=published['camera_id'], status='running', results={})])
    manifest = dict(parent=parent.name, child=child.name, camera_id=published['camera_id'],
                    old_capture=preview['path'], new_capture=replacement['path'],
                    policy_version=VERSION, policy_hash=POLICY_HASH, parameters=PARAMETERS,
                    reused_gpt_sha256=reused, backup=str(backup), run_id=run['id'],
                    previous_scene_sha256=state['stages']['9']['sha256'], created=time.time())
    write(backup / 'rerun.json', manifest)
    if any(sha(Path(p)) != h for p, h in before.items()) or api(base, '/api/state')['gpu_busy']:
        raise ValueError('State changed; live job left untouched')
    selection['current']['preview'] = replacement
    selection['previews'][published['camera_id']] = replacement
    for key in ('selected_ids', 'completed_ids', 'failed_ids'):
        if key in selection:
            selection[key] = [c for c in selection[key] if c != published['camera_id']]
    for n in (7, 8, 9):
        child_state['stages'].pop(str(n), None)
    for n in (11, 12, 13):
        state['stages'].pop(str(n), None)
    state.update(quality_run=run, busy=False, running_stage=None, error=None,
                 message='按固定质量规则重建第二轮，旧结果已保留')
    child_state.update(busy=False, running_stage=None, error=None)
    write(selection_path, selection)
    write(child / 'job.json', child_state)
    write(parent / 'job.json', state)
    # Exactly one dispatch; an uncertain HTTP response must be inspected, never blindly repeated.
    result = api(base, f'/api/jobs/{parent.name}/next', token=online['token'])
    print(__import__('json').dumps(dict(**manifest, running_stage=result['running_stage']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
