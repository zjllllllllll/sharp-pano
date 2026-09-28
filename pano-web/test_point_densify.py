import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import point_densify as density
import point_mask as mask


class DensityTests(unittest.TestCase):
    @staticmethod
    def plane(z=3., step=.02, size=12):
        axis=(np.arange(size)-(size-1)/2)*step
        x,y=np.meshgrid(axis,axis)
        return np.stack((x,y,np.full_like(x,z)),axis=-1)

    def test_distance_squared_near_and_dense_exclusion(self):
        p=self.plane(z=1.)
        keep=np.ones(p.shape[:2],dtype=bool)
        self.assertFalse(density.cell_weights(p,keep).any())
        near=density.cell_weights(self.plane(2.2),keep)
        far=density.cell_weights(self.plane(4.),keep)
        self.assertGreater(float(far.sum()),float(near.sum()))
        self.assertLessEqual(float(far.sum(axis=-1).max()),3)
        self.assertFalse(density.cell_weights(self.plane(4.,.004),keep).any())
        offset=np.array([1.,2,3])
        np.testing.assert_allclose(density.cell_weights(self.plane(4.)+offset,keep,offset),far)

    def test_mask_holes_depth_edges_invalid_and_large_gaps_are_not_bridged(self):
        original=self.plane(3.,size=2)
        keep=np.ones((2,2),dtype=bool)
        self.assertTrue(density.cell_weights(original,keep).any())
        keep[1,1]=False
        self.assertFalse(density.cell_weights(original,keep).any())
        keep[:]=True
        bad=original.copy();bad[1,1,2]+=1
        self.assertFalse(density.cell_weights(bad,keep).any())
        bad=original.copy();bad[1,1,0]=np.nan
        self.assertFalse(density.cell_weights(bad,keep).any())
        self.assertFalse(density.cell_weights(self.plane(3.,.2,2),keep).any())

    def test_spacing_only_refines_sparse_axis(self):
        p=self.plane(4.)
        p[...,0]*=.2
        weights=density.cell_weights(p,np.ones(p.shape[:2],dtype=bool))
        self.assertFalse(weights[...,0].any())
        self.assertFalse(weights[...,2].any())
        self.assertTrue(weights[...,1].any())

    def test_default_7_5mm_refines_gaps_below_old_10mm_target(self):
        self.assertEqual(density.PARAMETERS['target_spacing_m'], .0075)
        points = self.plane(4., .009)
        before = points.copy()
        keep = np.ones(points.shape[:2], dtype=bool)
        with patch.dict(density.PARAMETERS, target_spacing_m=.01):
            self.assertFalse(density.cell_weights(points, keep).any())
        weights = density.cell_weights(points, keep)
        self.assertTrue((weights > 0).all())
        np.testing.assert_array_equal(points, before)

    def test_budget_seed_source_ids_and_far_priority(self):
        weights=np.ones((30,30,3),dtype=np.float32)
        weights[:15]*=.1
        args=(weights,0,31)
        a,ids=density.candidate_keys(*args,np.random.default_rng(42))
        b,ids2=density.candidate_keys(*args,np.random.default_rng(42))
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(ids,ids2)
        _,selected=density.best_candidates(a,ids,200)
        self.assertEqual(len(selected),200)
        self.assertEqual(len(np.unique(selected)),200)
        rows=selected//3//31
        self.assertGreater(int((rows>=15).sum()),150)
        self.assertEqual(len(density.best_candidates(a,ids,0)[1]),0)

    def test_perspective_interpolation_and_camera_transform(self):
        from scipy.spatial.transform import Rotation
        uv=np.array([[-.01,-.01],[.01,-.01],[-.01,.01],[.01,.01]])
        z=3/(1-.3*uv[:,0]-.2*uv[:,1])
        local=np.column_stack((uv,np.ones(4)))*z[:,None]
        transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('xyz',[.2,.4,.1]).as_matrix()
        transform[:3,3]=[1,2,3]
        means=local@transform[:3,:3].T+transform[:3,3]
        colors=np.array([[0.,.1,.2],[1,.1,.2],[0,.1,.2],[1,.1,.2]])
        result=density.interpolate_children(means,colors,np.array([0,1,2]),{'c2w':transform.tolist()},2)
        camera=(result[:,:3]-transform[:3,3])@transform[:3,:3]
        np.testing.assert_allclose(camera[:,2]-.3*camera[:,0]-.2*camera[:,1],3,atol=1e-6)
        np.testing.assert_allclose(camera[:,:2]/camera[:,2:],[[0,-.01],[-.01,0],[0,0]],atol=1e-6)
        self.assertFalse(np.any(np.all(result[0,:3]==means,axis=1)))
        self.assertTrue(((result[:,3:]>=0)&(result[:,3:]<=1)).all())

    def fixture(self,job):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sharp-sixface'))
        from geometry import camera,angular_weights
        (job/'base').mkdir()
        cams,full,sample=[],[],[]
        for i in range(6):
            cam=camera(i,size=24)
            local=np.stack((self.plane(3),self.plane(5))).reshape(-1,3)
            means=(local@np.array(cam['c2w'])[:3,:3].T).astype('float32')
            colors=np.full(means.shape,.25,dtype='float32')
            np.savez(job/'base'/f'{i}.npz',means=means,colors=colors)
            domain=angular_weights(means,i,100)>1e-4
            points=np.column_stack((means[domain],np.full((domain.sum(),3),.5370987))).astype('<f4')
            indices=np.random.default_rng(42+i).choice(len(points),len(points)//16,replace=False)
            full.append(points);sample.append(points[indices]);cams.append(cam)
        full=np.concatenate(full);sample=np.concatenate(sample)
        full.tofile(job/'cloud-full.bin');sample.tofile(job/'cloud-preview.bin')
        (job/'cameras.json').write_text(json.dumps(cams))
        (job/'stage-3.json').write_text(json.dumps(dict(layers=2,diagnostics={'initializer_grid':[12,12]},count=len(full),preview_count=len(sample))))
        for name in ('job.json','cloud.las','initialized.ply','adjusted.ply','initialized-masked.ply','adjusted-masked.ply'):
            (job/name).write_bytes(b'unchanged source '+name.encode())
        mask.build_mask(job,lambda _:None)

    def test_build_cache_mapping_sampling_and_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);self.fixture(job)
            original={p:p.read_bytes() for p in job.rglob('*') if p.is_file()}
            info=density.build_densification(job,lambda _:None)
            self.assertGreater(info['added'],0)
            self.assertLessEqual(info['added'],int(info['before']*.25))
            self.assertEqual(info['after'],info['before']+info['added'])
            self.assertFalse(info['downstream'])
            points=np.fromfile(job/density.FULL,dtype='<f4').reshape(-1,6)
            sampled=np.fromfile(job/density.SAMPLE,dtype='<f4').reshape(-1,6)
            ids=np.fromfile(job/density.INDEX,dtype='<u4')
            self.assertEqual(len(ids),info['added']);self.assertEqual(len(np.unique(ids)),len(ids))
            take=np.sort(np.random.default_rng(density.PARAMETERS['seed']).choice(len(ids),len(ids)//16,replace=False))
            np.testing.assert_array_equal(sampled,points[take])
            cams=json.loads((job/'cameras.json').read_text())
            for i,cam in enumerate(cams):
                face=(ids//(288*3))==i
                local=ids[face]-i*288*3
                parent=local//3
                with np.load(job/'base'/f'{i}.npz') as base:
                    expected=density.interpolate_children(base['means'],base['colors'],local,cam,12)
                np.testing.assert_array_equal(points[face],expected)
                kept=mask.load_face(job,i,288)
                self.assertTrue(kept[parent[:,None]+[0,1,12,13]].all())
                self.assertTrue((parent%12<11).all());self.assertTrue((parent//12%12<11).all())
            self.assertEqual(info,density.cached_result(job))
            stamp=(job/density.FULL).stat().st_mtime_ns
            density.build_densification(job,lambda _:None)
            self.assertEqual(stamp,(job/density.FULL).stat().st_mtime_ns)
            for path,data in original.items():self.assertEqual(path.read_bytes(),data)
            (job/density.SAMPLE).write_bytes(b'bad')
            self.assertIsNone(density.cached_result(job))
            restored=density.build_densification(job,lambda _:None)
            self.assertEqual(restored['files'][density.FULL]['sha256'],info['files'][density.FULL]['sha256'])

    def test_empty_budget_and_failure_do_not_publish_partial_result(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);self.fixture(job)
            with patch.object(density,'interpolate_children',side_effect=RuntimeError('controlled')):
                with self.assertRaisesRegex(RuntimeError,'controlled'):
                    density.build_densification(job,lambda _:None)
            self.assertFalse((job/density.MANIFEST).exists())
            self.assertEqual(list(job.glob('*.partial')),[])
            with patch.dict(density.PARAMETERS,max_extra_fraction=0):
                result=density.build_densification(job,lambda _:None)
                self.assertEqual(result['added'],0)
                self.assertEqual((job/density.FULL).stat().st_size,0)
                self.assertIsNotNone(density.cached_result(job))


if __name__=='__main__':unittest.main()
