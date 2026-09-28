"""Offline, resumable experiment using the unmodified workbench stages 2/3."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def source_fingerprints():
    paths = list((ROOT / 'pano-web').glob('*.py'))
    paths += list((ROOT / 'pano-web' / 'static').rglob('*'))
    paths += list((ROOT / 'sharp-sixface').glob('*.py'))
    paths += [ROOT / 'third_party/DepthMaster' / 'infer_panoramas.py']
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
            for p in sorted(paths) if p.is_file()}


def prepare(run):
    import shutil
    from PIL import Image, ImageOps
    if (run / 'inputs.json').exists():
        raise FileExistsError('Run already prepared; use another output directory.')
    run.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in (ROOT / 'pano-tests').iterdir()
                   if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'})
    inputs = []
    for path in paths:
        job = run / path.stem
        job.mkdir()
        with Image.open(path) as raw:
            im = ImageOps.exif_transpose(raw).convert('RGB')
            if im.width != im.height * 2:
                raise ValueError(f'Expected 2:1 panorama: {path}')
            im.save(job / 'input.png')
            size = list(im.size)
        inputs.append(dict(name=path.name, stem=path.stem, source=str(path),
                           sha256=digest(path), rgb_sha256=digest(job / 'input.png'), size=size))
    write_json(run / 'inputs.json', inputs)
    write_json(run / 'source-fingerprints.json', source_fingerprints())
    write_json(run / 'experiment.json', dict(
        created=time.strftime('%Y-%m-%d %H:%M:%S'), inputs=len(inputs),
        depth='Unmodified pano-web.worker.depth_stage: DepthMaster joint panorama, 518, FOV95',
        cloud='Unmodified pano-web.worker.cloud_stage: native SHARP shallow alignment, 6 FOV100 faces, 2 initializer layers',
        source_cloud='Raw step-3 cloud; no densification, no final Gaussian forward, no semantic models',
        contour='Geometric bird-eye evidence only; experimental, not validated architecture',
        coordinates='Cloud X right, Y down, Z forward; planning XY=(X,Z), height=-Y',
        web_modified=True, web_scope='Independent read-only camera lab; existing five stages unchanged'))
    print(f'PREPARED {len(inputs)} panoramas at {run}', flush=True)


def run_stage(run, stage, only=None):
    sys.path.insert(0, str(ROOT / 'pano-web'))
    import worker
    import torch
    # Only cache loading in this isolated process. The canonical stage function,
    # tensor operations, seeds and output formats are called without changes.
    if stage == 2:
        sys.path.insert(0, str(ROOT / 'third_party/DepthMaster'))
        import infer_panoramas as backend
        original = backend.load_verified_model
        loaded = []
        def load_once(*args, **kwargs):
            if not loaded:
                loaded.append(original(*args, **kwargs))
            return loaded[0]
        backend.load_verified_model = load_once
        function = worker.depth_stage
    else:
        import sharp_model as backend
        original = backend.load_model
        loaded = []
        def load_once(*args, **kwargs):
            if not loaded:
                loaded.append(original(*args, **kwargs))
            return loaded[0]
        backend.load_model = load_once
        function = worker.cloud_stage
    inputs = json.loads((run / 'inputs.json').read_text(encoding='utf-8'))
    selected = [row for row in inputs if only is None or row['stem'] in only]
    for i, row in enumerate(selected, 1):
        job = run / row['stem']
        marker = job / f'stage-{stage}.json'
        if marker.exists():
            required = ['depth.npy', 'conditioning.npy', 'depth.png'] if stage == 2 else [
                'cloud.las', 'cloud-full.bin', 'cameras.json', *[f'base/{f}.npz' for f in range(6)]]
            if all((job / p).is_file() for p in required):
                print(f'SKIP {i}/{len(selected)} {row["name"]} stage={stage}', flush=True)
                continue
        print(f'BEGIN {i}/{len(selected)} {row["name"]} stage={stage}', flush=True)
        start = time.perf_counter()
        result = function(job)
        result['seconds'] = time.perf_counter() - start
        result['source_sha256'] = row['sha256']
        result['canonical_stage'] = f'pano-web.worker.{function.__name__}'
        write_json(marker, result)
        torch.cuda.empty_cache()
        print(f'DONE {i}/{len(selected)} {row["name"]} stage={stage} seconds={result["seconds"]:.2f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'depth', 'cloud'])
    p.add_argument('--run', type=Path, default=HERE / 'runs' / '20260921-all')
    p.add_argument('--only', nargs='+')
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.run)
    else:
        run_stage(a.run, 2 if a.mode == 'depth' else 3, a.only)


if __name__ == '__main__':
    main()
