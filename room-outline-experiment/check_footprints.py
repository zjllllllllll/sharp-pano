"""Production-resolution regression: source view + small/large translations."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from occlusion import RGBDVolume
from batch import write_json

HERE=Path(__file__).resolve().parent


def check():
    job=HERE/'runs/20260921-unknown-v1/0.1'
    output=HERE/'edge-fix-checks';output.mkdir(exist_ok=True)
    v=RGBDVolume.from_job(job)
    rows=[]
    for name,position in [('origin',[0,0,0]),('nearby',[.02,0,0]),('C001',[-1.5,0,-2])]:
        result=v.render(position)
        for kind in ['masked','raw','mask','overlay']:
            Image.fromarray(result[kind]).save(output/f'{name}-{kind}.png')
        row=dict(name=name,position=position,**result['stats'])
        rows.append(row);print(json.dumps(row),flush=True)
        if name=='origin':
            assert not np.any(result['mask']),'Valid source observation lost'
            assert np.all(np.isfinite(result['surface_depth']))
    write_json(output/'checks.json',dict(volume=v.stats,checks=rows))


if __name__=='__main__':check()
