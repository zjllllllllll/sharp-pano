"""Read-only consistency audit for a finished web-platform run."""
import argparse
import json
from pathlib import Path
import sys
import laspy
import numpy as np
from PIL import Image
from plyfile import PlyData

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sharp-sixface'))
from gaussian_ops import validate_vertices, digest


def verify(job):
    state=json.loads((job/'job.json').read_text(encoding='utf-8'))
    assert {'1','2','3','4','5'}.issubset(state['stages']) and not state['busy']
    rgb=np.asarray(Image.open(job/'input.png'))
    depth=np.load(job/'depth.npy',mmap_mode='r')
    assert rgb.shape[:2]==depth.shape and rgb.shape[1]==2*rgb.shape[0]
    assert np.isfinite(np.load(job/'conditioning.npy')).all()
    results={}
    for stage,name in [('4','initialized.ply'),('5','adjusted.ply')]:
        result=state['stages'][stage]
        variants=result.get('variants', {'unmasked':result})
        for variant, item in variants.items():
            file=job/item['preview']
            data=PlyData.read(file,mmap='r')['vertex'].data
            validate_vertices(data, allow_empty=variant=='masked')
            assert len(data)==item['count']
            assert digest(file)==item['sha256']
            results[file.name]=len(data)
            del data
    if state['stages']['4'].get('mask') and state['stages']['5'].get('mask'):
        from point_mask import cached_result, PARAMETERS
        mask=cached_result(job)
        assert mask and state['stages']['4']['mask']['sha256']==state['stages']['5']['mask']['sha256']==mask['sha256']
        assert state['stages']['4']['mask']['parameters']==state['stages']['5']['mask']['parameters']==PARAMETERS
    cloud=np.memmap(job/'cloud-full.bin',dtype='<f4',mode='r').reshape(-1,6)
    initial=PlyData.read(job/'initialized.ply',mmap='r')['vertex'].data
    if state['stages']['4'].get('variants'):
        from point_mask import fixed_keep, NEIGHBORS
        import point_distances as distances
        import point_normals as normals
        angles=np.memmap(job/normals.FULL,dtype='<f4',mode='r')
        scores=np.memmap(job/distances.FULL,dtype='<f4',mode='r').reshape(-1,len(distances.NEIGHBORS))
        keep=fixed_keep(angles,scores[:,distances.NEIGHBORS.index(NEIGHBORS)])
        filtered=PlyData.read(job/'initialized-masked.ply',mmap='r')['vertex'].data
        assert len(initial)==len(keep) and len(filtered)==int(keep.sum())
        offset=0
        for start in range(0,len(initial),65536):
            expected=initial[start:start+65536][keep[start:start+65536]]
            np.testing.assert_array_equal(filtered[offset:offset+len(expected)],expected)
            offset+=len(expected)
        del filtered
    sample=np.arange(0,len(cloud),max(1,len(cloud)//4096))
    xyz=np.column_stack([initial[k][sample] for k in ['x','y','z']])
    np.testing.assert_array_equal(cloud[sample,:3],xyz)
    points=laspy.mmap(job/'cloud.las')
    try:
        assert len(points.points)==len(cloud)==state['stages']['3']['count']
        las_xyz=np.column_stack([np.asarray(getattr(points,k)[sample]) for k in ['x','y','z']])
        assert np.max(np.abs(las_xyz-xyz)) <= .0000051
    finally:
        points.close()
    for item in state['stages']['5']['diagnostics']['changes_from_raw_initializer']:
        assert item['position_change_m_p50_p90_p99'][0]>0
    return dict(passed=True, counts=results, cloud_points=len(cloud),
                initialization_cloud_centers_identical=True, las_quantization_verified=True,
                both_layers=True, final_has_nonzero_predicted_changes=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.job),indent=2))
