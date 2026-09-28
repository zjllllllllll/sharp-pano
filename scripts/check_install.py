"""Offline installation and source-origin audit; optional CUDA backend check."""
from pathlib import Path
import argparse
import importlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pano-web'))
import project_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--skip-models', action='store_true',
                        help='Check the runtime before downloading model weights')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = {'root': str(ROOT), 'python': sys.executable, 'modules': {}, 'models': {}}
    if not Path(sys.executable).resolve().is_relative_to(ROOT/'.envs/runtime'):
        raise RuntimeError('Use the checkout-local Python runtime')
    names = ['numpy', 'scipy', 'PIL', 'cv2', 'torch', 'torchvision', 'gsplat',
             'plyfile', 'laspy', 'embreex', 'shapely', 'utils3d', 'sharp.models',
             'depthmaster.model', 'mdm.model.v2', 'studio.image_api',
             'studio.company_image_api', 'alibabacloud_oss_v2', 'fuse_and_validate_depth']
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError(f'{name} unexpectedly loaded from outside the checkout: {path}')
        result['modules'][name] = str(path.relative_to(ROOT))
    result['models_skipped'] = args.skip_models
    if not args.skip_models:
        for row in json.loads((ROOT/'models/manifest.json').read_text(encoding='utf-8')):
            path = ROOT/'models'/row['file']
            if not path.is_file() or path.stat().st_size != row['bytes']:
                raise RuntimeError(f'Missing or truncated weight: {path.name}')
            result['models'][row['name']] = row['file']
    from image_provider import status
    result['image_provider'] = status()
    if args.gpu:
        import torch
        from gsplat.cuda._backend import _C
        if _C is None:
            raise RuntimeError('gsplat CUDA backend did not load')
        backend = Path(_C.__file__).resolve()
        if not backend.is_relative_to(ROOT):
            raise RuntimeError(f'CUDA extension loaded outside checkout: {backend}')
        result['gpu'] = {'name': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                         'backend': str(backend.relative_to(ROOT)),
                         'tensor_sum': float(torch.ones(2,device='cuda').sum())}
    result['passed'] = True
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
