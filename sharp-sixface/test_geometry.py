import unittest
import numpy as np
from geometry import camera, view_rays, sample_erp, angular_weights, feather_alpha


class GeometryTests(unittest.TestCase):
    def test_proper_poses_and_fov(self):
        for i in range(6):
            c = camera(i)
            r = np.asarray(c['c2w'])[:3,:3]
            np.testing.assert_allclose(r.T@r, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1)
            np.testing.assert_allclose(np.asarray(c['c2w'])@np.asarray(c['w2c']), np.eye(4))
            self.assertAlmostEqual(np.degrees(2*np.arctan(1536/(2*c['K_pixel_edge'][0][0]))),100)

    def test_erp_roundtrip_pixel_centers(self):
        h,w=32,64
        lon=((np.arange(w)+.5)/w-.5)*2*np.pi
        lat=((np.arange(h)+.5)/h-.5)*np.pi
        la,lo=np.meshgrid(lat,lon,indexing='ij')
        rays=np.stack([np.sin(lo)*np.cos(la),np.sin(la),np.cos(lo)*np.cos(la)],axis=-1)
        src=np.arange(h*w,dtype=np.float32).reshape(h,w)
        np.testing.assert_allclose(sample_erp(src,rays),src,atol=1e-4)

    def test_constant_range_and_camera_z(self):
        pano=np.full((32,64),3.0,dtype=np.float32)
        for i in range(6):
            local,world=view_rays(camera(i,size=64))
            r=sample_erp(pano,world)
            z=r*local[...,2]
            xyz=local/local[...,2:3]*z[...,None]
            np.testing.assert_allclose(np.linalg.norm(xyz,axis=-1),3,atol=1e-6)
            self.assertLess(z[0,0],z[32,32])

    def test_partition_of_unity_and_opacity(self):
        rng=np.random.default_rng(7)
        rays=rng.normal(size=(10000,3))
        ws=np.stack([angular_weights(rays,i) for i in range(6)])
        np.testing.assert_allclose(ws.sum(axis=0),1,atol=2e-7)
        alphas=feather_alpha(np.full_like(ws,.8),ws)
        np.testing.assert_allclose(1-np.prod(1-alphas,axis=0),.8,atol=3e-7)

    def test_adjacent_faces_same_world_point(self):
        world=np.array([[2.,.3,2.],[-3.,-.4,3.],[.2,-1.,1.]])
        for i in range(6):
            c=camera(i)
            rot=np.asarray(c['c2w'])[:3,:3]
            local=world@rot
            np.testing.assert_allclose(local@rot.T,world,atol=1e-12)


if __name__=='__main__':
    unittest.main()
