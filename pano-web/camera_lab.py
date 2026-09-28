"""Read-only access to the independent offline camera sampling experiment."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / 'room-outline-experiment' / 'runs' / '20260922-adaptive-sampling'
SCENE = re.compile(r'^\d+\.\d+$')
CAMERA = re.compile(r'^C\d{3,6}(?:(?:-(?:coverage|raw|mask|overlay|reasons|unknown|uncertain|holes|reason-bits))?\.png|-depths\.npz)$')
FILES = {'input.png', 'depth.png', 'bev.png', 'bev-full.png', 'overlay.png',
         'overview.png', 'evidence.png', 'cameras.png', 'camera-map-base.png', 'outline.json', 'sample-cameras.json'}


def catalog():
    path = RUN / 'catalog.json'
    if not path.is_file():
        return dict(version=1, total=0, scenes=[], outline_ready=0, render_complete=0,
                    camera_count=0, message='离线实验正在准备，稍后自动刷新。')
    return json.loads(path.read_text(encoding='utf-8'))


def artifact(relative):
    parts=relative.split('/')
    valid=(len(parts)==2 and SCENE.fullmatch(parts[0]) and parts[1] in FILES)
    valid=valid or (len(parts)==3 and SCENE.fullmatch(parts[0]) and parts[1]=='panoramas' and CAMERA.fullmatch(parts[2]))
    if not valid:
        raise FileNotFoundError('不可访问此相机实验资源')
    path=(RUN/relative).resolve()
    if not path.is_relative_to(RUN.resolve()):
        raise FileNotFoundError('非法资源路径')
    return path
