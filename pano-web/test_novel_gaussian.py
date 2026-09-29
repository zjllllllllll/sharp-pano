"""Stage-9 contracts; run via the reference CUDA launcher, no model inference."""
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import json
from PIL import Image
from pathlib import Path
import numpy as np
from novel_gaussian_geometry import REFERENCE,project_inputs,select_patch
from novel_geometry import camera,view_rays,FACE_NAMES
from novel_pipeline import artifact

sys.path.insert(0,str(REFERENCE))


class PatchGeometryTests(unittest.TestCase):
    def test_single_anchor_keeps_original_grid_depth_after_learned_ray_crosses_edge(self):
        import torch
        import run_sharp_hard_depth as hard
        from sharp.utils.gaussians import Gaussians3D
        # The left cell belongs to the rear surface (3 m), but its learned
        # lateral offset points towards the right, foreground surface (1 m).
        target=np.ones((8,8),np.float32);target[:,:4]=3.
        base_x=torch.tensor([[-.5,.5],[-.5,.5]])[None,None,None].expand(1,1,2,2,2)
        base_y=torch.tensor([[-.5,-.5],[.5,.5]])[None,None,None].expand(1,1,2,2,2)
        base=SimpleNamespace(mean_x_ndc=base_x,mean_y_ndc=base_y)
        init=SimpleNamespace(gaussian_base_values=base,feature_input=None,global_scale=None)
        means=torch.ones(1,3,2,2,2)*2.
        scales=torch.ones_like(means)*.1
        quats=torch.zeros(1,4,2,2,2);quats[:,0]=1.
        colors=torch.full_like(means,.25);alpha=torch.ones(1,2,2,2)*.8
        grid=Gaussians3D(means,scales,quats,colors,alpha)
        predictor=SimpleNamespace(
            monodepth_model=lambda _:SimpleNamespace(disparity=torch.ones(1,1,2,2),output_features=None),
            init_model=lambda *args:init,
            feature_model=lambda *args,**kwargs:None,
            prediction_head=lambda _:None,
            gaussian_composer=lambda **kwargs:grid)
        predictor.init_model.num_layers=2
        surfaces=[]
        def covariance(gaussians,surface,*args):
            surfaces.append(surface.clone());return gaussians,{}
        with patch.object(hard,'unproject_gaussians',side_effect=lambda g,*args:g), \
             patch.object(hard,'get_unprojection_matrix',return_value=torch.eye(4)), \
             patch.object(hard,'surface_jacobian_covariance',side_effect=covariance), \
             patch.object(hard.torch.cuda,'synchronize'):
            result,z,diag,_=hard.hard_predict(predictor,np.zeros((8,8,3),np.uint8),target,4.,
                torch.device('cpu'),'surface_jacobian',1.5,.05,.35,4.)
        expected=torch.tensor([[3.,1.],[3.,1.]])
        np.testing.assert_array_equal(z,expected.numpy())
        # Same learned ray and color; only the original-grid depth controls range.
        torch.testing.assert_close(result.mean_vectors[0,:4],expected.reshape(-1,1).expand(4,3))
        torch.testing.assert_close(result.colors,torch.full_like(result.colors,.25))
        torch.testing.assert_close(result.singular_values[0,:4],expected.reshape(-1,1).expand(4,3)*.05)
        self.assertTrue(torch.all(result.opacities[0,4:]==0))
        self.assertEqual(diag['center_update'],'preserve_learned_ray_then_set_exact_camera_z')
        # The reference covariance uses the regular target grid, not displaced centers.
        torch.testing.assert_close(surfaces[0][0,...,0],expected*base_x[0,0,0])
        torch.testing.assert_close(surfaces[0][0,...,2],expected)

    def test_single_face_writes_diagnostics_even_with_large_errors_and_low_alpha(self):
        import torch
        import run_sharp_hard_depth as hard
        for alpha_value in (.4,0.):
            with self.subTest(alpha=alpha_value),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);image=np.full((8,8,3),100,np.uint8)
                Image.fromarray(image).save(root/'rgb.png');Image.fromarray(np.full((8,8),255,np.uint8)).save(root/'mask.png')
                np.save(root/'depth.npy',np.full((8,8),2,np.float32))
                (root/'camera.json').write_text(json.dumps(dict(intrinsic_K_opencv=[[4,0,4],[0,4,4],[0,0,1]])))
                (root/'weights.pt').write_bytes(b'test')
                args=['sharp','--image',str(root/'rgb.png'),'--depth',str(root/'depth.npy'),'--hole-mask',str(root/'mask.png'),
                    '--camera-json',str(root/'camera.json'),'--checkpoint',str(root/'weights.pt'),'--output-dir',str(root/'out')]
                fake=SimpleNamespace(mean_vectors=torch.zeros(1,3,3))
                with patch.object(hard.sys,'argv',args),patch.object(hard,'load_predictor',return_value=object()),patch.object(hard,'patch_gpu_svd'),patch.object(hard,'save_ply'),patch.object(hard.torch.cuda,'is_available',return_value=True),patch.object(hard.torch.cuda,'reset_peak_memory_stats'),patch.object(hard.torch.cuda,'max_memory_allocated',return_value=0),patch.object(hard,'hard_predict',return_value=(fake,np.ones((8,8)),dict(layer0_center_absolute_error_p99=1.),0.)),patch.object(hard,'render',return_value=(image,np.full((8,8),4,np.float32),np.full((8,8),alpha_value,np.float32),0.)):
                    hard.main()
                result=json.loads((root/'out/sharp_hard_acceptance.json').read_text())
                self.assertEqual(result['status'],'diagnostic_only');self.assertFalse(result['quality_gate_enabled'])
                self.assertNotIn('thresholds',result);self.assertNotIn('failures',result)
                error=result['render_depth_metrics_in_hole']['relative_error_median']
                self.assertEqual(error,1. if alpha_value else None)

    def test_final_patch_render_records_high_error_or_no_depth_without_rejection(self):
        from novel_gaussian import render_diagnostics
        for alpha_value in (.4,0.):
            with self.subTest(alpha=alpha_value),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);rgb=np.full((8,8,3),100,np.uint8);depth=np.full((8,8),4,np.float32)
                alpha=np.full((8,8),alpha_value,np.float32)
                for name in FACE_NAMES:
                    face=root/name;face.mkdir()
                    (face/'camera.json').write_text(json.dumps(dict(w2c=np.eye(4).tolist(),intrinsic_K_opencv=np.eye(3).tolist())))
                    Image.fromarray(rgb).save(face/'rgb.png');Image.fromarray(np.full((8,8),255,np.uint8)).save(face/'mask.png')
                    np.save(face/'depth-input.npy',np.full((8,8),2,np.float32))
                fake=SimpleNamespace();fake.to=lambda device:fake
                with patch('sharp.utils.gaussians.load_ply',return_value=(fake,None)),patch('merge_hard_patch.render_gaussians',return_value=(rgb,depth,alpha)):
                    report=render_diagnostics(root,root/'base.ply',[dict(face=n) for n in FACE_NAMES])
                self.assertEqual(report['status'],'diagnostic_only');self.assertFalse(report['quality_gate_enabled'])
                self.assertNotIn('thresholds',report);self.assertNotIn('failures',report)
                self.assertEqual(len(report['patch']),6)
                self.assertEqual(report['patch'][0]['relative_error_median'],1. if alpha_value else None)

    def test_quality_support_cannot_be_restored_by_extrapolation(self):
        rgb=np.full((128,256,3),87,np.uint8);radial=np.full((128,256),3.,np.float32)
        support=np.ones((128,256),bool);support[58:70,115:140]=False
        result=project_inputs(rgb,radial,np.ones(support.shape,bool),0,[0,0,0],size=128,support=support)
        self.assertTrue(result['extrapolated'].any())
        self.assertFalse((result['mask']&result['extrapolated']).any())
        self.assertFalse((result['support']&result['extrapolated']).any())
        disabled=project_inputs(rgb,radial,np.ones(support.shape,bool),0,[0,0,0],size=128,support=support,face_enabled=False)
        self.assertFalse(disabled['mask'].any());self.assertFalse(disabled['support'].any())

    def test_too_few_valid_neighbors_are_not_relaxed(self):
        from novel_gaussian_filter import filter_cloud
        from quality_retry import QualityNoContribution
        with self.assertRaises(QualityNoContribution):filter_cloud(np.zeros((64,3)),np.zeros(64),strict_quality=True)

    def test_radial_to_camera_z_and_fixed_pose_all_faces(self):
        radial=np.full((128,256),3.,np.float32);rgb=np.full((128,256,3),87,np.uint8)
        mask=np.zeros(radial.shape,bool);mask[40:60,100:130]=True
        for index in range(6):
            projected=project_inputs(rgb,radial,mask,index,[-.5,.2,.3],size=64)
            expected=view_rays(camera(index,size=64,fov=100))[0][...,2]*3
            np.testing.assert_allclose(projected['depth_input'],expected,atol=1e-6)
            np.testing.assert_array_equal(projected['camera']['c2w'],np.linalg.inv(projected['camera']['w2c']))
            np.testing.assert_array_equal(np.asarray(projected['camera']['c2w'])[:3,3],[-.5,.2,.3])
            self.assertFalse(projected['extrapolated'].any())
            self.assertTrue((projected['rgb']==87).all())

    def test_invalid_lingbot_is_filled_without_modifying_supported_depth(self):
        radial=np.full((128,256),3.,np.float32);radial[58:68,122:133]=0
        result=project_inputs(np.zeros((128,256,3),np.uint8),radial,np.ones(radial.shape,bool),0,[0,0,0],size=128)
        missing=result['extrapolated'];self.assertTrue(missing.any())
        self.assertTrue(np.all(result['depth_input']>0))
        np.testing.assert_array_equal(result['depth_input'][~missing],result['depth_projected'][~missing])
        self.assertTrue(np.all(result['depth_projected'][missing]==0))

    def test_reference_selection_only_layer0_rigid_covariance_and_append_only(self):
        from gaussian_patch_io import append_patch_to_base
        from plyfile import PlyData
        side=32;n=side*side
        names=['x','y','z','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
        vertices=np.zeros(2*n,dtype=[(name,'<f4') for name in names])
        vertices['z']=2;vertices['rot_0']=1
        for key in ('scale_0','scale_1','scale_2'):vertices[key]=-4
        local,_=view_rays(camera(1,size=side,fov=100));xyz=local/local[...,2,None]*2
        for i,key in enumerate(('x','y','z')):vertices[key][:n]=xyz[...,i].reshape(-1)
        hole=np.zeros((128,128),bool);hole[40:88,40:88]=True
        cam=camera(1,size=128,fov=100);pose=np.asarray(cam['c2w']);pose[:3,3]=[-.5,.2,.3]
        cam.update(c2w=pose.tolist(),w2c=np.linalg.inv(pose).tolist())
        tail,transform,ids,labels,weights,record=select_patch(vertices,np.full((side,side),2.,np.float32),hole,1,cam)
        self.assertTrue(np.all(ids<n));self.assertTrue(np.all(labels[hole]==1))
        self.assertEqual(record['blend_ring_count'],6);self.assertEqual(record['core_opacity_mode'],'full')
        self.assertTrue(all(a>b>0 for a,b in zip(record['ring_weights'],record['ring_weights'][1:])))
        self.assertLess(record['world_camera_roundtrip_max_m'],1e-5)
        self.assertEqual(record['center_z_error_max_m'],0.)
        self.assertTrue(np.all(weights>0));self.assertTrue(np.all(weights<=1))
        # Propagate initializer identity even when network-moved positions match.
        point_keep=np.zeros(2*n,bool);point_keep[ids[::2]]=True
        filtered,_,filtered_ids,_,filtered_weights,filtered_record=select_patch(
            vertices,np.full((side,side),2.,np.float32),hole,1,cam,point_keep)
        np.testing.assert_array_equal(filtered_ids,ids[::2])
        np.testing.assert_array_equal(filtered,tail[::2])
        np.testing.assert_array_equal(filtered_weights,weights[::2])
        self.assertEqual(filtered_record['point_filter_removed'],len(ids)-len(filtered_ids))
        np.testing.assert_array_equal(transform['scale_logs'],np.full((len(tail),3),-4,np.float32))
        self.assertFalse(np.allclose(transform['quaternion_wxyz'],np.column_stack([tail[f'rot_{i}'] for i in range(4)])))
        base=vertices[:10].copy();base['x']=np.arange(10)*.03
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'merged.ply';count=append_patch_to_base(base,tail,transform,path)
            merged=PlyData.read(path,mmap=False)['vertex'].data
            self.assertEqual(count,len(base)+len(tail));self.assertEqual(merged[:10].tobytes(),base.tobytes())

    def test_artifact_allowlist_and_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            job=Path(temp);prefix='a'*16+'/previews/C007/'+'b'*16+'/gaussians/'+'c'*16+'/'
            for name in ('merged.ply','patch.ply','acceptance.json','front/depth-input.npy','front/sharp/sharp_hard_acceptance.json'):
                self.assertEqual(artifact(job,prefix+name),(job/'novel'/prefix/name).resolve())
            for name in ('inputs.json','front/cache.json','../../job.json','front/../../private.env'):
                with self.assertRaises(FileNotFoundError):artifact(job,prefix+name)

    def test_point_filter_normal_first_full_cloud_and_outlier(self):
        from novel_gaussian_filter import filter_cloud
        cluster=np.random.default_rng(21).normal(0,.005,(80,3))
        outlier=np.array([[10.,0,0]])
        invalid_normals=np.repeat(outlier,80,axis=0)
        points=np.concatenate([cluster,outlier,invalid_normals])
        angles=np.r_[np.full(80,82.),-1.,np.full(80,83.)].astype(np.float32)
        keep,scores=filter_cloud(points,angles,lambda _:None)
        self.assertTrue(keep[:80].all());self.assertFalse(keep[80:].any())
        self.assertGreater(scores[80],.15);self.assertTrue((scores[81:]==-1).all())
        # The isolated patch row has neighbors in the complete scene, before crop.
        self.assertTrue(keep[:1].all())

    def test_capture_raw_initializer_without_mutation_and_new_camera_origin(self):
        import torch
        from types import SimpleNamespace
        from novel_gaussian_filter import capture_initializer
        inverse=torch.full((1,1,2,4,4),.5)
        base=SimpleNamespace(mean_inverse_z_ndc=inverse,mean_x_ndc=torch.ones_like(inverse)*.25,
            mean_y_ndc=torch.ones_like(inverse)*-.5)
        output=SimpleNamespace(gaussian_base_values=base,global_scale=torch.tensor([3.]))
        cam=camera(1,size=512,fov=100);pose=np.asarray(cam['c2w']);pose[:3,3]=[-.5,.2,.3]
        cam.update(c2w=pose.tolist(),intrinsic_K_opencv=cam['K_pixel_edge'])
        with tempfile.TemporaryDirectory() as temp:
            capture_initializer(output,cam,Path(temp))
            ndc=np.load(Path(temp)/'initializer-ndc.npy');world=np.load(Path(temp)/'initializer-world.npy')
        self.assertEqual(ndc.shape,(2,4,4,3))
        np.testing.assert_allclose(ndc,np.broadcast_to([1.5,-3.,6.],ndc.shape))
        camera_xyz=(world-pose[:3,3])@pose[:3,:3]
        np.testing.assert_allclose(camera_xyz,ndc*np.array([512/(2*cam['K_pixel_edge'][0][0])]*2+[1]),atol=1e-6)
        self.assertTrue(torch.equal(inverse,torch.full_like(inverse,.5)))


if __name__=='__main__':unittest.main()
