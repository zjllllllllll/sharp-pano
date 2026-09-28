import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from face_quality import evaluate_rgb,evaluate_depth,joint,record
from quality_fusion import merge_accepted,compose_accepted


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((512,512),bool);self.mask[220:290,220:290]=True
        self.geo=self.mask.copy();self.alpha=np.ones((512,512),np.float32)
        rng=np.random.default_rng(42)
        self.rgb=np.clip(gaussian_filter(rng.random((512,512,3)),(2,2,0))*255,0,255).astype(np.uint8)
        self.z=np.broadcast_to(np.linspace(1,4,512),(512,512)).copy()

    def rgb_score(self,p):return evaluate_rgb(self.rgb,p,self.mask,self.geo,self.alpha)[0]
    def depth_score(self,p,valid=None):return evaluate_depth(self.z,p,self.alpha.astype(bool) if valid is None else valid,self.mask,self.geo,self.alpha)[0]

    def test_identity_and_hole_only_edit_pass(self):
        self.assertEqual(self.rgb_score(self.rgb)['status'],'passed')
        p=self.rgb.copy();p[self.mask]=[255,0,255]
        self.assertEqual(self.rgb_score(p)['score'],100)
        self.assertEqual(self.depth_score(self.z)['score'],100)

    def test_removed_object_and_shift_fail(self):
        p=self.rgb.copy();p[35:220,35:220]=0
        self.assertEqual(self.rgb_score(p)['status'],'rejected')
        self.assertEqual(self.rgb_score(np.roll(self.rgb,20,axis=0))['status'],'rejected')

    def test_plane_collapse_and_legitimate_flat(self):
        r=self.depth_score(np.full_like(self.z,2.5))
        self.assertIn('depth_collapse',r['reason_codes'])
        self.z[:]=2.5
        self.assertEqual(self.depth_score(self.z)['status'],'passed')

    def test_invalids_stay_in_denominator(self):
        valid=np.ones_like(self.mask);valid[:200]=False
        r=self.depth_score(self.z,valid)
        self.assertLess(r['metrics']['valid_fraction'],.7)
        p=self.z.copy();p[:200]=np.nan
        self.assertEqual(self.depth_score(p)['status'],'rejected')

    def test_small_or_clustered_anchors_insufficient(self):
        self.alpha[:]=0;self.alpha[:220,:220]=1
        self.assertEqual(self.rgb_score(self.rgb)['status'],'insufficient_evidence')

    def test_either_stage_rejects_pair(self):
        good=record('rgb','passed');bad=record('depth','rejected')
        self.assertFalse(joint(good,bad)['accepted']);self.assertFalse(joint(bad,good)['accepted'])

    def test_rejected_faces_have_no_contribution(self):
        rgbs=[np.full((512,512,3),i*40,np.uint8) for i in range(6)]
        valid=[np.ones((512,512),bool)]+[None]*5
        out=merge_accepted(rgbs,[None]*6,valid,128,64)
        self.assertTrue((out['provenance'][out['support']]==1).all())
        self.assertTrue((out['rgb'][out['support']]==0).all())
        none=merge_accepted(rgbs,[None]*6,[None]*6,128,64)
        self.assertFalse(none['support'].any())

    def test_unsupported_holes_stay_holes_original_outside(self):
        original=np.full((64,128,3),88,np.uint8);generated=np.full_like(original,200)
        mask=np.zeros((64,128),bool);mask[20:40,40:70]=True
        result,effective,unresolved=compose_accepted(original,generated,mask,np.zeros(mask.shape,bool))
        self.assertFalse(effective.any());np.testing.assert_array_equal(unresolved,mask)
        self.assertTrue((result[~mask]==88).all());self.assertTrue((result[mask]==[0,255,0]).all())

    def test_local_depth_deviation_warns_but_whole_face_failure_rejects(self):
        p=self.z.copy();p[30:145]*=1.35
        r=self.depth_score(p)
        self.assertEqual(r['status'],'passed');self.assertTrue(r['warnings'])
        self.assertEqual(self.depth_score(self.z*.45)['status'],'rejected')

    def test_overlapping_depths_are_averaged_without_cross_face_rejection(self):
        from novel_geometry import camera,view_rays
        faces=[]
        for i in range(6):faces.append(3*view_rays(camera(i,size=512,fov=100))[0][...,2])
        valid=[np.ones((512,512),bool)]*6;rgb=[self.rgb]*6
        good=merge_accepted(rgb,faces,valid,256,128)
        self.assertTrue(good['support'].all());self.assertFalse(good['conflict'].any())
        faces[0]=faces[0]*2
        bad=merge_accepted(rgb,faces,valid,256,128)
        self.assertFalse(bad['conflict'].any());self.assertTrue(bad['support'].all())
        overlap=(bad['provenance']&1!=0)&(bad['provenance']!=1)
        self.assertTrue(overlap.any())
        self.assertTrue(np.all(bad['depth'][overlap]>3.))
        self.assertTrue(np.all(bad['depth'][overlap]<6.))
        valid[0]=None
        sparse=merge_accepted(rgb,faces,valid,256,128)
        self.assertFalse(sparse['conflict'].any());self.assertTrue((sparse['provenance']&1==0).all())


if __name__=='__main__':unittest.main()
