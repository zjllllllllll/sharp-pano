import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from gaussian_alignment import align_short_axes
from point_normals import estimate_angles


class AlignmentTests(unittest.TestCase):
    def fixture(self, curved=False):
        x, z = np.meshgrid(np.linspace(-.2, .2, 17), np.linspace(1.8, 2.2, 17))
        y = .5 + ((z-2)**2 if curved else np.zeros_like(x))
        grids = np.stack([np.stack((x, y, z), axis=-1), np.stack((x, y+.2, z), axis=-1)])
        means = grids.reshape(-1, 3).astype('f4')
        n = len(means)
        rng = np.random.default_rng(4)
        scales = np.array([[.001, .004, .008], [.004, .001, .008], [.004, .008, .001]], dtype='f4')[np.arange(n)%3]
        scene = dict(means=means+rng.uniform(-.01,.01,means.shape).astype('f4'),
                     scales=scales, quats=Rotation.random(n, random_state=rng).as_quat()[:,[3,0,1,2]].astype('f4'),
                     colors=rng.random((n,3)).astype('f4'), alpha=rng.random(n).astype('f4'))
        return scene, dict(means=means), dict(c2w=np.eye(4).tolist())

    def test_only_rotation_changes_and_shortest_axis_is_normal(self):
        scene, base, cam = self.fixture()
        before = {k:v.copy() for k,v in scene.items()}
        result, info = align_short_axes(scene,base,cam,17)
        shortest = np.argmin(result['scales'],axis=1)
        old = Rotation.from_quat(scene['quats'][:,[1,2,3,0]])
        new = Rotation.from_quat(result['quats'][:,[1,2,3,0]])
        axes = np.take_along_axis(new.as_matrix(),shortest[:,None,None],axis=2)[:,:,0]
        np.testing.assert_allclose(np.abs(axes[:,1]),1,atol=1e-6)
        self.assertTrue(np.all((new*old.inv()).magnitude()<=np.pi/2+1e-6))
        for key in scene:
            np.testing.assert_array_equal(scene[key],before[key])
            if key!='quats':np.testing.assert_array_equal(result[key],before[key])
        self.assertEqual(info['reliable'],len(scene['means']))
        self.assertLess(info['max_alignment_error_degrees'],1e-4)

    def test_unreliable_normals_leave_rotation_untouched(self):
        scene,base,cam=self.fixture();base['means'][:]=0
        result,info=align_short_axes(scene,base,cam,17)
        for key in scene:np.testing.assert_array_equal(result[key],scene[key])
        self.assertEqual(info['reliable'],0)
        self.assertEqual(info['unchanged_unreliable'],len(scene['means']))

    def test_curved_surface_uses_local_normals_and_keeps_invalid_row(self):
        scene,base,cam=self.fixture(curved=True)
        base['means'][8*17+8]=np.nan
        result,info=align_short_axes(scene,base,cam,17)
        np.testing.assert_array_equal(result['quats'][8*17+8],scene['quats'][8*17+8])
        for layer,grid in enumerate(base['means'].reshape(2,17,17,3)):
            angles,normals=estimate_angles(grid,return_normals=True)
            np.testing.assert_array_equal(angles,estimate_angles(grid))
            valid=angles.ravel()>=0;ids=np.flatnonzero(valid)+layer*17*17
            matrices=Rotation.from_quat(result['quats'][ids][:,[1,2,3,0]]).as_matrix()
            axes=np.take_along_axis(matrices,np.argmin(result['scales'][ids],axis=1)[:,None,None],axis=2)[:,:,0]
            dots=np.abs(np.sum(axes*normals.reshape(-1,3)[valid],axis=1))
            np.testing.assert_allclose(dots,1,atol=1e-6)
            self.assertGreater(np.ptp(normals[...,2]),.1)
        self.assertEqual(info['unchanged_unreliable'],1)

    def test_already_aligned_axes_do_not_rotate(self):
        scene,base,cam=self.fixture()
        scene['scales'][:]=[.004,.001,.008];scene['quats'][:]=[1,0,0,0]
        result,_=align_short_axes(scene,base,cam,17)
        np.testing.assert_allclose(result['quats'],scene['quats'],atol=1e-7)


if __name__=='__main__':unittest.main()
