import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from multi_view_volume import covered_segments
from novel_iteration import ensure_workspace,trusted_patch_pixels,compose_observation_depth
from novel_pipeline import write_json,read,digest


class Interval:
    def __init__(self,start,end):self.start,self.end=start,end
    def contains(self,points):return (points[:,0]>=self.start)&(points[:,0]<self.end)
    def boundary(self,points,directions):return (self.end-points[:,0])/directions[:,0]


class IterationTests(unittest.TestCase):
    def test_union_bridges_two_individually_incomplete_observations(self):
        ray=np.array([[1,0,0]],np.float32)
        a,b=Interval(-1,2),Interval(1,4)
        self.assertFalse(covered_segments([a],[0,0,0],ray,[3])[0])
        self.assertFalse(covered_segments([b],[0,0,0],ray,[3])[0])
        self.assertTrue(covered_segments([a,b],[0,0,0],ray,[3])[0])

    def test_gap_and_missing_endpoint_stay_unknown(self):
        rays=np.array([[1,0,0],[1,0,0]],np.float32)
        actual=covered_segments([Interval(-1,1),Interval(2,4)],[0,0,0],rays,[3,np.inf])
        np.testing.assert_array_equal(actual,[False,False])

    def test_no_fixed_step_can_skip_a_thin_gap(self):
        self.assertFalse(covered_segments([Interval(-1,1),Interval(1.001,4)],[0,0,0],np.array([[1,0,0]],np.float32),[3])[0])

    def test_extrapolated_transparent_and_patch_target_mismatch_not_promoted(self):
        depth=np.array([2,2,2,2,4,np.nan],np.float32)
        alpha=np.array([1,1,.69,1,1,1],np.float32)
        patch_alpha=np.array([1,1,1,.69,1,1],np.float32)
        extra=np.array([False,True,False,False,False,False])
        actual=trusted_patch_pixels(depth,alpha,patch_alpha,np.full(6,2),np.ones(6,bool),extra)
        np.testing.assert_array_equal(actual,[True,False,False,False,False,False])

    def test_accepted_patch_is_retained_when_old_geometry_disagrees(self):
        # A nearer door, a surface behind old geometry, and a patch overlapping
        # old known coverage all retain the accepted patch's own depth.
        rendered=np.array([2.49,3.4,1.9,0,0],np.float32)
        valid=np.array([True,True,True,False,False])
        reason=np.array([3,4,0,0,1],np.uint8)
        old_surface=np.array([np.inf,2.,3.,4.,np.inf],np.float32)
        depth,accepted,old_known=compose_observation_depth(rendered,valid,reason,old_surface)
        np.testing.assert_array_equal(depth,np.array([2.49,3.4,1.9,4.,0],np.float32))
        np.testing.assert_array_equal(accepted,[True,True,True,True,False])
        np.testing.assert_array_equal(old_known,[False,False,True,True,False])

    def test_workspace_retains_previous_result_and_copies_base(self):
        with tempfile.TemporaryDirectory() as root:
            parent=Path(root)/('a'*32);parent.mkdir()
            source=parent/'novel'/'abc'/'merged.ply';source.parent.mkdir(parents=True);source.write_bytes(b'previous Gaussian bytes')
            previous=dict(revision='abc',download='merged.ply',sha256=digest(source),count=7,patch_revision='p',acceptance={'status':'accepted'})
            stages={str(i):{'old':i} for i in range(1,9)};stages['9']=previous
            write_json(parent/'job.json',dict(stages=stages,busy=True))
            child=ensure_workspace(parent)
            self.assertEqual(read(parent/'job.json')['stages'],stages)
            self.assertEqual((child/'iteration-base.ply').read_bytes(),source.read_bytes())
            self.assertNotEqual((child/'iteration-base.ply').stat().st_ino,source.stat().st_ino)
            self.assertEqual(ensure_workspace(parent),child)
            self.assertEqual(read(child/'job.json')['iteration_parent'],parent.name)

    def test_rejected_round_cannot_create_next_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            parent=Path(root)/('a'*32);parent.mkdir()
            write_json(parent/'job.json',dict(stages={'9':{'acceptance':{'status':'review_required'}}}))
            with self.assertRaises(ValueError):ensure_workspace(parent)


if __name__=='__main__':unittest.main()
