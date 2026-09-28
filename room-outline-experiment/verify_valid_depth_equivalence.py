"""Audit the invalid-only numerical fix without falsifying producer hashes."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
from occlusion import RGBDVolume,Config
from batch import digest,write_json

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/20260922-unknown-v2'
BASELINE=RUN/'implementation/occlusion-valid-depth.py'
BASELINE_HASH='82039742009a98f48401cef02e676483f398c3c87b39501300479c1d03d3383c'


def verify():
    assert digest(BASELINE)==BASELINE_HASH
    spec=importlib.util.spec_from_file_location('valid_depth_baseline',BASELINE)
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    rng=np.random.default_rng(20260922)
    smooth=np.full((64,128),5,np.float32)
    foreground=smooth.copy();foreground[15:45,50:75]=2;foreground[20:40,:8]=1;foreground[20:40,-8:]=1
    noisy=rng.uniform(.3,20,(64,128)).astype(np.float32)
    count=0
    for depth in [smooth,foreground,noisy]:
        rgb=rng.integers(0,256,(*depth.shape,3),dtype=np.uint8)
        new=RGBDVolume(depth,rgb,config=Config(proxy_max_width=128,supersampling=2))
        old=baseline.RGBDVolume(depth,rgb,config=baseline.Config(proxy_max_width=128,supersampling=2))
        assert new.validation_lookup is None  # new recovery path is not used
        for position in [[0,0,0],[.02,0,0],[1.5,.2,0],[0,0,7]]:
            a=new.render(position,128,64);b=old.render(position,128,64)
            for key in a:
                if key!='stats':np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            count+=1
    write_json(RUN/'implementation/valid-depth-equivalence.json',dict(
        passed=True,fully_valid_only=True,baseline_sha256=BASELINE_HASH,
        current_sha256=digest(HERE/'occlusion.py'),camera_cases=count,
        note='Current changes only execute when invalid source depth exists. Fully-valid outputs are bitwise equal in regression cases. Existing scenes retain their ACTUAL producing implementation hash, not relabelled as rerendered by a different file.'))
    print(f'VALID_DEPTH_EQUIVALENCE {count} cases: bitwise equal',flush=True)


if __name__=='__main__':verify()
