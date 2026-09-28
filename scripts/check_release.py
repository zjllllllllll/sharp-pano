"""Audit the actual Git source set without staging or publishing files.

Checks known local credential values without printing them; this is not a
replacement for a complete secret scanner or license review.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    raw=subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    names=sorted(set(raw.decode('utf-8').split('\0'))-{''})
    violations=[]
    credentials=set()
    def collect(value):
        if isinstance(value,dict):
            for item in value.values():collect(item)
        elif isinstance(value,list):
            for item in value:collect(item)
        elif isinstance(value,str) and len(value.strip())>=12:
            credentials.add(value.strip().encode())
    for path in (ROOT/'secrets').glob('*'):
        if path.is_file():
            value=path.read_text(encoding='utf-8-sig').strip()
            if path.suffix=='.json':collect(json.loads(value))
            else:collect(value)
    for name,value in os.environ.items():
        if name in {'GRSAI_API_KEY','CLICKGS_IMAGE_API_KEY','ARK_API_KEY','OSS_ACCESS_KEY_ID','OSS_ACCESS_KEY_SECRET','OSS_SESSION_TOKEN'}:collect(value)
    size=0
    for name in names:
        path=ROOT/name
        if not path.is_file():
            violations.append({'path':name,'reason':'not a regular file'})
            continue
        if path.is_symlink():violations.append({'path':name,'reason':'source symlink'})
        parts=set(Path(name).parts)
        if parts & {'secrets','.envs','.cache','.runtime','runs','node_modules','__pycache__'} or Path(name).suffix in {'.pt','.safetensors','.ply','.npy','.npz','.las'} or Path(name).name.startswith('config.local.'):
            violations.append({'path':name,'reason':'local installation or generated data included'})
        data=path.read_bytes();size+=len(data)
        if any(value in data for value in credentials):
            violations.append({'path':name,'reason':'known local credential value present'})
    manifest=json.loads((ROOT/'third_party/SOURCE_SNAPSHOT.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=ROOT/row['path']
        data=path.read_bytes()
        if row.get('hash_mode')=='utf8-lf':data=data.decode('utf-8').replace('\r\n','\n').replace('\r','\n').encode('utf-8')
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            violations.append({'path':row['path'],'reason':'vendored snapshot hash mismatch'})
    result={'passed':not violations,'source_files':len(names),'source_bytes':size,'vendored_files_verified':len(manifest['files']),'known_local_credentials_checked':len(credentials),'violations':violations}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__=='__main__':main()
