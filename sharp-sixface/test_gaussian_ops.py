import tempfile
from pathlib import Path
import unittest
import numpy as np
from geometry import camera,view_rays
from gaussian_ops import covariance,world_transform,ndc_to_camera,to_vertices,save_vertices


class GaussianTests(unittest.TestCase):
    def scene(self,n):
        local,_=view_rays(camera(0,size=n))
        pos=(local*2).reshape(-1,3)
        count=len(pos)
        return {'means':pos,'scales':np.full((count,3),.01,dtype=np.float32),
                'quats':np.tile([1.,0.,0.,0.],(count,1)).astype(np.float32),
                'colors':np.full((count,3),.5,dtype=np.float32),'alpha':np.full(count,.8,dtype=np.float32)}

    def test_rigid_covariance_rotation(self):
        scene=self.scene(8); scene['scales'][:,1]=.02
        cam=camera(1,size=8); r=np.asarray(cam['c2w'])[:3,:3]
        world=world_transform(scene,cam)
        np.testing.assert_allclose(covariance(world),r@covariance(scene)@r.T,atol=1e-9)

    def test_ndc_covariance_affine(self):
        scene=self.scene(8); cam=camera(0,size=8)
        out,info=ndc_to_camera(scene,cam)
        factor=8/(2*cam['K_pixel_edge'][0][0]); d=np.diag([factor,factor,1])
        np.testing.assert_allclose(covariance(out),d@covariance(scene)@d.T,atol=1e-9)

    def test_ply_roundtrip_and_refuse_overwrite(self):
        scene=self.scene(8); v,info=to_vertices(scene,0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'test.ply'
            result=save_vertices(path,v)
            self.assertTrue(result['bit_exact_readback'])
            with self.assertRaises(FileExistsError): save_vertices(path,v)


if __name__=='__main__': unittest.main()
