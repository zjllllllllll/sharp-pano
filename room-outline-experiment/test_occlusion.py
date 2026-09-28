import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from occlusion import RGBDVolume,Config,erp_rays,EmbreeScene,TriangleMesh
from render_unknown import render_job
from cameras import sample_cameras


class VolumeTests(unittest.TestCase):
    def volume(self,depth,**kwargs):
        rgb=np.full((*depth.shape,3),[120,180,220],np.uint8)
        return RGBDVolume(depth,rgb,config=Config(proxy_max_width=256,supersampling=1),**kwargs)

    def test_origin_sphere_no_self_occlusion_including_seam_poles(self):
        v=self.volume(np.full((64,128),5,np.float32))
        result=v.render([0,0,0],128,64)
        self.assertEqual(result['stats']['mask_fraction'],0)
        self.assertEqual(result['stats']['camera_state'],'known_free')
        self.assertTrue(np.all(result['raw']==[120,180,220]))
        self.assertEqual(v.query([0,0,0],np.array([[0,1,0],[0,-1,0],[0,0,-1]],np.float32))[1].sum(),0)

    def test_wall_first_unknown_behind_wall_does_not_mask(self):
        v=self.volume(np.full((64,128),5,np.float32))
        r=v.render([1,0,0],64,32)
        self.assertEqual(r['stats']['mask_fraction'],0)

    def test_foreground_cone_masks_visible_background(self):
        depth=np.full((128,256),6,np.float32)
        depth[40:88,116:140]=2
        v=self.volume(depth)
        r=v.render([1.5,0,0],256,128)
        # Crucial regression: mask a ray WITH a reliable background endpoint.
        # A count tuned to v1's dilated silhouettes would reward the old bug.
        self.assertGreater(np.count_nonzero((r['reason']&2!=0)&(r['reason']&1==0)),0)
        self.assertGreater(np.count_nonzero(r['reason']==0),1000)
        # Straight to the real front surface must still be kept.
        ray=np.array([[-1.5,0,2]],np.float32);ray/=np.linalg.norm(ray,axis=1,keepdims=True)
        self.assertEqual(int(v.query([1.5,0,0],ray)[1][0]),0)

    def test_invalid_and_filtered_depth_is_not_free(self):
        depth=np.full((64,128),5,np.float32);depth[25:40,60:68]=np.nan
        v=self.volume(depth)
        result=v.query([0,0,0],np.array([[0,0,1]],np.float32))
        self.assertNotEqual(int(result[1][0]),0)
        self.assertTrue(int(result[1][0])&4)
        valid=np.ones((64,128),bool);valid[25:40,60:68]=False
        v2=self.volume(np.full((64,128),5,np.float32),valid=valid)
        self.assertTrue(int(v2.query([0,0,0],np.array([[0,0,1]],np.float32))[1][0])&4)

    def test_camera_outside_does_not_fake_valid_observation(self):
        v=self.volume(np.full((64,128),2,np.float32))
        r=v.render([3,0,0],32,16)
        self.assertEqual(r['stats']['camera_state'],'unknown')
        self.assertTrue(np.all(r['reason']&8!=0))

    def test_missing_all_surface_is_mask_not_crash(self):
        v=self.volume(np.full((16,32),np.nan,np.float32))
        r=v.render([0,0,0],32,16)
        self.assertEqual(r['stats']['mask_fraction'],1)

    def test_periodic_depth_discontinuity_and_no_fake_bridge(self):
        d=np.full((64,128),6,np.float32);d[20:44,:8]=2;d[20:44,-8:]=2
        v=self.volume(d)
        ray=np.array([[0,0,-1]],np.float32)
        self.assertEqual(int(v.query([0,0,0],ray)[1][0]),0)
        self.assertLess(v.stats['surface_triangles'],v.stats['boundary_triangles'])

    def test_reduction_preserves_thin_foreground_unknown_guard(self):
        d=np.full((128,256),6,np.float32);d[40:88,128]=2
        cfg=Config(proxy_max_width=128,supersampling=1)
        v=RGBDVolume(d,np.full((128,256,3),128,np.uint8),config=cfg)
        r=v.render([1,0,0],256,128)
        self.assertGreater(np.count_nonzero(r['reason']&2!=0),0)

    def test_origin_keeps_every_valid_pixel_across_discontinuities(self):
        d=np.full((64,128),6,np.float32)
        d[15:45,50:75]=2;d[20:40,:8]=1;d[20:40,-8:]=1
        d[:6,20:60]=3;d[-6:,70:100]=4  # discontinuous polar caps
        v=self.volume(d)
        self.assertEqual(v.stats['surface_triangles'],2*d.size-2*d.shape[1])
        # All pixels, not only rays whose triangles happened to survive.
        for size in ((128,64),(256,128),(117,59)):
            r=v.render([0,0,0],*size)
            self.assertEqual(np.count_nonzero(r['mask']),0,size)
        rng=np.random.default_rng(42)
        rays=rng.normal(size=(10000,3)).astype(np.float32)
        rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        self.assertEqual(np.count_nonzero(v.query([0,0,0],rays)[1]),0)

    def test_nearby_camera_keeps_foreground_silhouette_surface(self):
        d=np.full((64,128),6,np.float32);d[16:48,48:80]=2
        v=self.volume(d)
        points=erp_rays(128,64)[16:48,48:80].reshape(-1,3)*2
        camera=np.array([.02,0,0],np.float32)
        rays=points-camera;rays/=np.linalg.norm(rays,axis=1,keepdims=True)
        # Includes the full outer ring formerly removed. No origin shortcut.
        self.assertEqual(np.count_nonzero(v.query(camera,rays)[1]),0)

    def test_invalid_pixel_does_not_erase_valid_neighbours(self):
        d=np.full((32,64),5,np.float32);d[16,32]=np.nan
        r=self.volume(d).render([0,0,0],64,32)
        expected=np.zeros_like(d,dtype=np.uint8);expected[16,32]=255
        np.testing.assert_array_equal(r['mask'],expected)

    def test_near_coplanar_wall_cannot_report_impossible_hit(self):
        # Captured float32 triangle from scene 0.4: Embree can claim a hit at
        # 0.03m although the triangle is ~1m away. Validate physical location.
        tri=np.array([[[-.48299843,-.60936636,.63887376],
                       [-.35196733,-.44405329,.46555573],
                       [-.48103613,-.60936636,.64035261]]],np.float32)
        scene=EmbreeScene(robust=True)
        TriangleMesh(scene,tri.reshape(-1,3),np.array([[0,1,2]],np.int32))
        rays=erp_rays(2048,1024)[300,787].reshape(1,3)
        hit=RGBDVolume.intersect(scene,np.zeros_like(rays),rays,tri,0)
        self.assertEqual(int(hit['primID'][0]),-1)

    def test_partial_run_resumes_and_preserves_completed_camera(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp)
            np.save(job/'depth.npy',np.full((32,64),5,np.float32))
            Image.fromarray(np.full((32,64,3),160,np.uint8)).save(job/'input.png')
            info=sample_cameras([[-1,-1],[1,-1],[1,1],[-1,1]])
            (job/'sample-cameras.json').write_text(json.dumps(info),encoding='utf-8')
            cfg=Config(proxy_max_width=64,supersampling=1)
            first=render_job(job,32,16,cfg,limit=2)
            self.assertFalse(first['render_complete'])
            path=job/first['cameras'][0]['panorama'];stamp=path.stat().st_mtime_ns
            second=render_job(job,32,16,cfg)
            self.assertTrue(second['render_complete'])
            self.assertEqual(path.stat().st_mtime_ns,stamp)
            self.assertEqual([(c['id'],c['position_world_m']) for c in info['cameras']],
                             [(c['id'],c['position_world_m']) for c in second['cameras']])
            with patch('render_unknown.RGBDVolume.from_job') as builder:
                third=render_job(job,32,16,cfg)
                builder.assert_not_called()
                self.assertTrue(third['render_complete'])
            for camera in third['cameras']:
                mask=np.asarray(Image.open(job/camera['views']['mask']))
                coverage=np.asarray(Image.open(job/camera['views']['coverage']))
                np.testing.assert_array_equal(mask.astype(int)+coverage,255)

    def test_zero_camera_scene_is_complete_without_building_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp)
            np.save(job/'depth.npy',np.full((16,32),5,np.float32))
            Image.fromarray(np.full((16,32,3),160,np.uint8)).save(job/'input.png')
            info=sample_cameras([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]],adaptive=False)
            # Renderer still accepts an explicitly empty external camera list.
            info.update(cameras=[],count=0)
            (job/'sample-cameras.json').write_text(json.dumps(info),encoding='utf-8')
            with patch('render_unknown.RGBDVolume.from_job') as builder:
                result=render_job(job)
                builder.assert_not_called()
                self.assertTrue(result['render_complete'])
                self.assertEqual(result['count'],0)


if __name__=='__main__':unittest.main()
