"""Depth projection/fusion checks: no CUDA, model requests, or real-run writes."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from novel_depth import REFERENCE,write_depth_outputs
from novel_geometry import camera, view_rays, erp_directions
from novel_depth_geometry import merge_depth_faces, fuse_face
from novel_pipeline import artifact

sys.path.insert(0, str(REFERENCE))


def sphere_faces(radius=3., size=128):
    return [view_rays(camera(i, size=size))[0][..., 2]*radius for i in range(6)]


class DepthGeometryTests(unittest.TestCase):
    def test_camera_z_to_radial_sphere(self):
        output, valid = merge_depth_faces(sphere_faces(), 256, 128)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(output, 3., atol=4e-4)

    def test_asymmetric_scene_preserves_all_six_directions(self):
        faces=[]
        for i in range(6):
            local,world=view_rays(camera(i,size=128))
            radius=3+world[...,0]*.2+world[...,1]*.4+world[...,2]*.7
            faces.append(radius*local[...,2])
        output,valid=merge_depth_faces(faces,256,128)
        rays=erp_directions(256,128)
        expected=3+rays[...,0]*.2+rays[...,1]*.4+rays[...,2]*.7
        self.assertTrue(valid.all())
        np.testing.assert_allclose(output,expected,atol=6e-4)

    def test_invalid_samples_are_not_averaged_as_zero(self):
        faces=sphere_faces();faces[0][:]=0;faces[1][:]=np.nan
        output,valid=merge_depth_faces(faces,256,128)
        self.assertTrue((~valid).any())
        self.assertTrue(valid.any())
        np.testing.assert_allclose(output[valid],3.,atol=4e-4)
        self.assertTrue(np.all(output[~valid]==0))

    def test_panorama_uses_model_everywhere_independent_of_mask_and_original(self):
        mask=np.zeros((128,256),bool);mask[30:80,60:100]=True
        model=sphere_faces(3.)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            metrics=write_depth_outputs(out,sphere_faces(2.),sphere_faces(4.),model,mask,[0,0,0])
            final=np.load(out/'panorama-completed.npy')
            np.testing.assert_array_equal(final,merge_depth_faces(model,256,128)[0])
            np.testing.assert_allclose(final,3.,atol=4e-4)
            self.assertEqual(metrics['outside_changed_pixels'],int((~mask).sum()))
            # Changing both original/composite values and the Mask cannot alter
            # any completed panorama pixel: only the model contributes.
            write_depth_outputs(out,sphere_faces(6.),sphere_faces(7.),model,~mask,[0,0,0])
            np.testing.assert_array_equal(final,np.load(out/'panorama-completed.npy'))

    def test_uncovered_model_pixels_remain_invalid_instead_of_copying_original(self):
        model=sphere_faces();model[0][:]=np.inf
        expected,valid=merge_depth_faces(model,256,128)
        self.assertTrue((~valid).any())
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            metrics=write_depth_outputs(out,sphere_faces(2.),sphere_faces(4.),model,np.zeros((128,256),bool),[0,0,0])
            np.testing.assert_array_equal(np.load(out/'panorama-completed.npy'),expected)
            self.assertEqual(metrics['invalid_completed_pixels'],int((~valid).sum()))
            with np.load(out/'depths.npz') as data:
                np.testing.assert_array_equal(data['completed_valid'],valid)
                np.testing.assert_array_equal(data['face_lingbot_z_m'],model)
                self.assertTrue(np.all(data['completed_radial_m'][~valid]==0))
            with self.assertRaises(ValueError):
                write_depth_outputs(out,sphere_faces(),sphere_faces(),[np.zeros((128,128))]*6,np.zeros((128,256),bool),[0,0,0])

    def test_depth_artifact_allowlist_and_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);prefix='a'*16+'/previews/C007/'+'b'*16+'/depth/'+'c'*16+'/'
            for filename in ['depths.npz','panorama-completed.npy','front-rendered.png','up-camera.json']:
                self.assertEqual(artifact(job,prefix+filename),(job/'novel'/prefix/filename).resolve())
            for filename in ['inputs.json','../../job.json','private.env','front.npy']:
                with self.assertRaises(FileNotFoundError):artifact(job,prefix+filename)


class DepthFusionTests(unittest.TestCase):
    def setUp(self):
        y,x=np.mgrid[:160,:160]
        self.source=(2+x*.008+y*.004).astype(np.float32)
        self.predicted=((self.source-.15)/1.1).astype(np.float32)
        self.hole=np.zeros_like(self.source,bool);self.hole[55:105,55:105]=True
        self.valid=np.ones_like(self.hole)

    def test_calibration_and_outside_exact(self):
        result,record=fuse_face(self.source,self.predicted,self.valid,self.hole)
        self.assertEqual(result[~self.hole].tobytes(),self.source[~self.hole].tobytes())
        self.assertAlmostEqual(record['scale'],1.1,places=5)
        self.assertAlmostEqual(record['offset'],.15,places=5)
        np.testing.assert_allclose(result[73:87,73:87],self.source[73:87,73:87],atol=1e-5)
        self.assertTrue(np.all(result>0))

    def test_model_invalid_hole_pixels_are_explicitly_reported(self):
        self.valid[75:80,75:80]=False;self.predicted[75:80,75:80]=np.inf
        result,record=fuse_face(self.source,self.predicted,self.valid,self.hole)
        self.assertEqual(record['extrapolated_hole_pixels'],25)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result[~self.hole],self.source[~self.hole])

    def test_no_hole_skips_and_bad_inputs_fail_closed(self):
        result,record=fuse_face(self.source,self.predicted,self.valid,np.zeros_like(self.hole))
        self.assertTrue(record['skipped']);np.testing.assert_array_equal(result,self.source)
        with self.assertRaises(ValueError):fuse_face(self.source,self.predicted,self.valid,np.ones_like(self.hole))
        with self.assertRaises(ValueError):fuse_face(self.source,self.source/10,self.valid,self.hole)
        self.source[0,0]=0
        with self.assertRaises(ValueError):fuse_face(self.source,self.predicted,self.valid,self.hole)

    def test_failed_affine_can_only_fall_back_for_comparison(self):
        predicted=6-self.source
        before=predicted.copy()
        with self.assertRaises(ValueError):fuse_face(self.source,predicted,self.valid,self.hole)
        result,record=fuse_face(self.source,predicted,self.valid,self.hole,comparison_only=True)
        self.assertTrue(record['comparison_only_uncalibrated'])
        self.assertLess(record['rejected_calibration']['scale'],0)
        self.assertEqual(record['scale'],1.)
        self.assertEqual(record['offset'],0.)
        np.testing.assert_array_equal(result[~self.hole],self.source[~self.hole])
        np.testing.assert_array_equal(predicted,before)
        np.testing.assert_allclose(result[73:87,73:87],predicted[73:87,73:87],atol=1e-6)


if __name__=='__main__':unittest.main()
