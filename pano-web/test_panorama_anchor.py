"""Regression: final Gaussian rays must read the shared surface, not old pixels."""
import unittest
import numpy as np
from geometry import camera, view_rays
from panorama_anchor import anchor_points, anchor_ndc_scene
from gaussian_ops import ndc_to_camera, world_transform, covariance


class PanoramaAnchorTests(unittest.TestCase):
    def test_all_faces_and_layers_stay_on_one_sphere(self):
        radial=np.full((128,256),2.5,dtype='f4')
        for index in range(6):
            cam=camera(index,size=24)
            local,_=view_rays(cam)
            points=np.concatenate([(local*.7).reshape(-1,3),(local*4).reshape(-1,3)])
            actual,ratio=anchor_points(points,radial,rotation=np.asarray(cam['c2w'])[:3,:3])
            np.testing.assert_allclose(np.linalg.norm(actual,axis=1),2.5,atol=5e-7)
            np.testing.assert_allclose(actual/actual[:,2:],points/points[:,2:],atol=3e-7)
            np.testing.assert_allclose(ratio[:24*24],2.5/.7,rtol=3e-7)

    def test_front_down_learned_rays_land_on_the_same_bed_plane(self):
        # y=1 plane, sampled as radial ERP. A lateral network shift must read
        # the new ray's depth; retaining the original pixel depth fails this.
        h,w=1024,2048
        lat=((np.arange(h)+.5)/h-.5)*np.pi
        radial=np.broadcast_to(1/np.maximum(np.sin(lat),.01)[:,None],(h,w)).astype('f4')
        for face in (0,5):
            cam=camera(face,size=48)
            local,world=view_rays(cam)
            local=local.copy();local[...,0]+=.035;local[...,1]-=.025
            pts=(local*1.3).reshape(-1,3)
            anchored,_=anchor_points(pts,radial,rotation=np.asarray(cam['c2w'])[:3,:3])
            xyz=anchored@np.asarray(cam['c2w'])[:3,:3].T
            rays=xyz/np.linalg.norm(xyz,axis=1,keepdims=True)
            bed=(rays[:,1]>.60)&(rays[:,1]<.78)&(rays[:,2]>.60)
            self.assertGreater(bed.sum(),50)
            np.testing.assert_allclose(xyz[bed,1],1.,atol=1e-5)

    def test_scale_transport_preserves_shape_rotation_and_appearance(self):
        cam=camera(0,size=8);local,_=view_rays(cam);means=(local*.8).reshape(-1,3)
        scale=np.array([8/(2*cam['K_pixel_edge'][0][0])]*2+[1])
        scene=dict(means=(means/scale).astype('f4'),scales=np.tile([.01,.02,.03],(64,1)).astype('f4'),
            quats=np.tile([1.,0,0,0],(64,1)).astype('f4'),colors=np.full((64,3),.3,dtype='f4'),alpha=np.full(64,.7,dtype='f4'))
        anchored=anchor_ndc_scene(scene,np.full((32,64),2.4,dtype='f4'),cam)
        for key in ('quats','colors','alpha'):np.testing.assert_array_equal(anchored[key],scene[key])
        np.testing.assert_allclose(covariance(anchored),covariance(scene)*9,atol=3e-9)
        world=world_transform(ndc_to_camera(anchored,cam)[0],cam)
        np.testing.assert_allclose(np.linalg.norm(world['means'],axis=1),2.4,atol=5e-7)

    def test_missing_panorama_does_not_introduce_zero_depth(self):
        points=np.array([[0,0,2.],[1,0,2.]],dtype='f4')
        actual,ratio=anchor_points(points,np.zeros((16,32),dtype='f4'))
        np.testing.assert_array_equal(actual,points);np.testing.assert_array_equal(ratio,1)



if __name__=='__main__':unittest.main()
