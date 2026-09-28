"""Read-only diagnostics API with an explicit artifact allowlist."""
import re
from face_quality_io import read
from novel_geometry import FACE_NAMES


def latest(job):
    reports=sorted((job/'diagnostics'/'face-quality').glob('*/report.json'))
    return reports[-1] if reports else None


def public_report(job):
    state=read(job/'job.json')
    if state.get('iteration_parent'):job=job.parent/state['iteration_parent']
    path=latest(job)
    if not path:return dict(status='not_evaluated',rows=[],faces=[])
    r=read(path);root=f'/quality-artifacts/{job.name}/{path.parent.name}/'
    return dict(status='ready',version=r['version'],policy_hash=r['policy_hash'],applied=False,
        report_url=root+'report.html',rows=[{**row,'maps':{k:root+v for k,v in row.get('maps',{}).items()}} for row in r['rows']],
        faces=r['faces'],immutability=dict(status=r['immutability']['status'],files=r['immutability']['files']))


def artifact(job,relative):
    parts=relative.split('/')
    if not parts or not re.fullmatch(r'\d{8}-\d{6}-[a-f0-9]{8}',parts[0]):raise FileNotFoundError('Invalid quality revision')
    allowed=len(parts)==2 and parts[1] in ('report.json','report.csv','report.html')
    if len(parts)==3 and re.fullmatch(r'round-[1-9]\d*',parts[1]):
        names={f'{f}-{kind}-{name}.png' for f in FACE_NAMES for kind in ('rgb','depth') for name in ('anchors','metric_support','difference','bad')}
        allowed=parts[2] in names
    if not allowed:raise FileNotFoundError('Invalid quality artifact')
    root=(job/'diagnostics'/'face-quality').resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):raise FileNotFoundError('Invalid quality path')
    return path
