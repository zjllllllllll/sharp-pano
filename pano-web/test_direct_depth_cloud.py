"""Geometry regressions for direct depth backprojection, without model weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import worker
from geometry import camera


class DirectDepthCloudTests(unittest.TestCase):
    def test_constant_range_has_same_radius_on_every_face_and_layer(self):
        rgb = np.full((128, 256, 3), 128, dtype=np.uint8)
        depth = np.full(rgb.shape[:2], 2.5, dtype=np.float32)
        for i in range(6):
            scene, _ = worker.direct_depth_scene(rgb, depth, camera(i), 24)
            xyz = scene['means'].reshape(2, 24, 24, 3)
            np.testing.assert_allclose(np.linalg.norm(xyz, axis=-1), 2.5, atol=5e-7)
            np.testing.assert_array_equal(xyz[0], xyz[1])
            np.testing.assert_allclose(worker.to_srgb(scene['colors']), 128/255, atol=2e-7)
            self.assertTrue(np.all(scene['scales'] > 0))
            np.testing.assert_allclose(scene['alpha'], .5)

    def test_front_and_down_reconstruct_one_continuous_plane(self):
        h, w = 512, 1024
        lat = ((np.arange(h)+.5)/h-.5)*np.pi
        depth = np.broadcast_to((1/np.maximum(np.sin(lat), .05))[:, None], (h,w)).astype('f4')
        rgb = np.zeros((h,w,3), dtype='u1')
        for i in [0,5]:
            scene, _ = worker.direct_depth_scene(rgb, depth, camera(i), 96)
            xyz = scene['means']
            rays = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
            bed = (rays[:,1]>.60)&(rays[:,1]<.78)&(rays[:,2]>.60)
            self.assertGreater(int(bed.sum()), 100)
            np.testing.assert_allclose(xyz[bed,1], 1, atol=3e-5)

    def test_initial_fields_match_native_conventions_without_depth_inference(self):
        import torch
        from sharp.models.initializer import MultiLayerInitializer
        from gaussian_ops import ndc_to_camera, world_transform
        h,w=512,1024
        lon=((np.arange(w)+.5)/w-.5)*2*np.pi
        lat=((np.arange(h)+.5)/h-.5)*np.pi
        radial=(2/np.maximum(np.cos(lat)[:,None]*np.cos(lon)[None,:], .05)).astype('f4')
        rgb=np.full((h,w,3),128,dtype='u1');cam=camera(0,size=48)
        actual,_=worker.direct_depth_scene(rgb,radial,cam,24)
        init=MultiLayerInitializer(num_layers=2,stride=2,base_depth=10,scale_factor=1,
            disparity_factor=1,color_option='all_layers',normalize_depth=False)
        expected=worker.base_scene(init(torch.full((1,3,48,48),128/255),torch.full((1,1,48,48),2.)))
        expected,_=ndc_to_camera(expected,cam);expected=world_transform(expected,cam)
        for name in ['means','scales','colors','alpha']:
            np.testing.assert_allclose(actual[name],expected[name],rtol=5e-5,atol=2e-5,err_msg=name)

    def test_cloud_stage_reads_depth_npy_without_loading_sharp(self):
        import laspy
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder)
            Image.fromarray(np.full((64,128,3),100,dtype='u1')).save(job/'input.png')
            np.save(job/'depth.npy',np.full((64,128),2.,dtype='f4'))
            np.save(job/'conditioning.npy',np.full((64,128),900.,dtype='f4'))
            with patch.object(worker,'CLOUD_GRID_SIZE',16), patch('sharp_model.load_model',side_effect=AssertionError('Must not load SHARP')):
                result=worker.cloud_stage(job)
            cloud=np.fromfile(job/'cloud-full.bin',dtype='<f4').reshape(-1,6)
            self.assertEqual(len(cloud),result['count'])
            np.testing.assert_allclose(np.linalg.norm(cloud[:,:3],axis=1),2.,atol=5e-7)
            self.assertEqual(result['diagnostics']['initializer_grid'],[16,16])
            self.assertFalse(result['diagnostics']['learned_alignment'])
            self.assertFalse(json.loads((job/'model.json').read_text())['learned_depth'])
            las=laspy.read(job/'cloud.las')
            np.testing.assert_allclose(np.stack([las.x,las.y,las.z],axis=1),cloud[:,:3],atol=5.1e-6)
            for i in range(6):
                with np.load(job/'base'/f'{i}.npz') as f:self.assertEqual(f['means'].shape,(512,3))


if __name__=='__main__':
    unittest.main()
