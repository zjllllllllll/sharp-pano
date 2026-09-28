"""Mock inference integration; use the reference launcher, no real model/GPU calls."""
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
import cv2  # Load native extension before temporary sys.modules mocking.
import test_novel_pipeline as fixtures
from face_quality import POLICY_HASH
from face_quality_io import read,write,digest,assess_rgb
from gpt_rgb_experiment import assemble
from novel_geometry import FACE_NAMES,camera,six_faces
from quality_depth import complete_depth


class QualityDepthTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.NovelPipelineTests('runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.job=fixture.job;self.folder=fixture.folder
        import novel_pipeline as n
        self.preview=n.capture(self.job,'C002',fixture.payload)
        self.capture=self.folder/self.preview['path'];out=self.capture/'repair';out.mkdir()
        rows=[];qualities=[]
        for name in FACE_NAMES:
            base=np.asarray(Image.open(self.capture/f'{name}.png'));mask=np.asarray(Image.open(self.capture/f'{name}-mask.png'))>0
            generated=base.copy();generated[mask]=[100,100,100];Image.fromarray(generated).save(out/f'{name}-model.png')
            rows.append(dict(face=name,skipped=not mask.any()))
            qualities.append(assess_rgb(self.capture,name,out))
        self.rgb=assemble(self.folder,self.preview,dict(revision=fixture.rev,camera_id='C002'),'mock',{},rows,0.,qualities)
        state=read(self.job/'job.json');state['stages']['7']=self.rgb;write(self.job/'job.json',state)
        self.checkpoint=self.job/'test-checkpoint';self.checkpoint.write_bytes(b'no model')
        self.calls=[];self.render_calls=[];self.tensor=torch.tensor
        self.bad=False
        owner=self
        class Model:
            @classmethod
            def from_pretrained(cls,p):return cls()
            def eval(self):return self
            def to(self,device):return self
            def infer(self,rgb,**kw):
                owner.calls.append(rgb.cpu().numpy().copy())
                return dict(depth=owner.tensor(np.full((512,512),.5 if owner.bad else 2.,np.float32)),
                            mask=owner.tensor(np.ones((512,512),bool)))
        self.module=types.ModuleType('mdm.model.v2');self.module.MDMModel=Model

    def render(self,ply,position,out):
        self.render_calls.append(1);arrays=[]
        for i,name in enumerate(FACE_NAMES):
            z=np.full((512,512),2.,np.float32);arrays.append(z)
            np.save(out/f'{name}-rendered-z.npy',z);np.save(out/f'{name}-alpha.npy',np.ones_like(z))
            cam=camera(i,size=512,fov=100);pose=np.asarray(cam['c2w']);pose[:3,3]=position
            write(out/f'{name}-camera.json',dict(c2w=pose.tolist(),intrinsic_K_opencv=cam['K_pixel_edge'],semantics='camera_z_m'))
        return arrays

    def run_depth(self):
        def cpu_tensor(*a,**kw):kw['device']='cpu';return self.tensor(*a,**kw)
        with patch.dict('sys.modules',{'mdm.model.v2':self.module}),patch('novel_depth.CHECKPOINT',self.checkpoint),\
             patch('novel_depth.render_faces',self.render),patch('torch.tensor',side_effect=cpu_tensor),patch('torch.cuda.empty_cache'):
            return complete_depth(self.job)

    def test_fused_panorama_input_joint_bundle_and_rescore_reuses_raw_inference(self):
        first=self.run_depth();self.assertEqual(len(self.calls),1)
        panorama=np.asarray(Image.open(self.folder/self.rgb['image']))
        expected=six_faces(panorama,np.zeros(panorama.shape[:2],bool))[0]['rgb']/255
        own=np.asarray(Image.open(self.capture/'repair/front-repaired.png'))/255
        self.assertFalse(np.array_equal(expected,own))
        self.assertEqual(first['accepted_bundle']['guidance'],'fused_panorama_reprojected')
        np.testing.assert_allclose(self.calls[0][0].transpose(1,2,0),expected,atol=1e-7)
        self.assertGreater(first['accepted_bundle']['effective_pixels'],0)
        state=read(self.job/'job.json');state['stages']['7']['quality_faces'][0]['diagnostic_note']='rescored'
        write(self.job/'job.json',state)
        second=self.run_depth()
        self.assertNotEqual(first['depth_revision'],second['depth_revision'])
        self.assertEqual(len(self.calls),1);self.assertEqual(len(self.render_calls),1)

    def test_bad_depth_excludes_rgb_and_rgb_rejection_skips_inference(self):
        self.bad=True;result=self.run_depth()
        front=next(r for r in result['accepted_bundle']['faces'] if r['face']=='front')
        self.assertFalse(front['accepted']);self.assertEqual(front['depth_status'],'rejected')
        valid=np.asarray(Image.open(self.folder/result['path']/'front-accepted-valid.png'))
        self.assertFalse(valid.any())
        state=read(self.job/'job.json');state['stages']['7']['quality_faces'][0]['status']='rejected'
        write(self.job/'job.json',state);self.calls.clear()
        result=self.run_depth();self.assertEqual(len(self.calls),0)
        self.assertIn('upstream_rejected',result['quality_faces'][0]['reason_codes'])


if __name__=='__main__':unittest.main()
