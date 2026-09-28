"""One checkout-local source of paths; no sibling-project fallbacks."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT/'third_party'
SHARP = THIRD_PARTY/'ml-sharp'
DEPTHMASTER = THIRD_PARTY/'DepthMaster'
LINGBOT = THIRD_PARTY/'lingbot-depth'
REPAIR = ROOT/'repair'
PROVIDERS = ROOT/'providers'
MODELS = ROOT/'models'
SECRETS = ROOT/'secrets'
ENVIRONMENT = ROOT/'.envs/runtime'
PYTHON = ENVIRONMENT/('python.exe' if os.name == 'nt' else 'bin/python')
CUDA_LAUNCHER = ROOT/'run_cuda_python.cmd' if os.name == 'nt' else ROOT/'scripts/run_python.sh'
CPU_LAUNCHER = ROOT/'run_python.cmd' if os.name == 'nt' else ROOT/'scripts/run_python.sh'


def configuration():
    path = ROOT/'config.local.json'
    settings = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    provider = os.environ.get('PANO_IMAGE_PROVIDER', settings.get('image_provider', 'grsai'))
    if provider not in {'company', 'grsai'}:
        raise ValueError('image_provider must be company or grsai')
    return {**settings, 'image_provider': provider}


def activate_sources():
    # Insert all authoritative checkouts ahead of stale editable installations.
    paths = [SHARP/'src', DEPTHMASTER, DEPTHMASTER/'third_party/utils3d',
             LINGBOT, REPAIR, PROVIDERS, ROOT/'sharp-sixface', ROOT/'room-outline-experiment']
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


activate_sources()
