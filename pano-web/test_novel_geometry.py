import unittest
import numpy as np
from novel_geometry import next_camera,six_faces,merge_faces,composite_inside_mask,erp_directions,outward_blend_weights,composite_outward,clean_mask,smooth_mask,repair_masks,repair_faces


def point(i,x,z,m):return dict(id=f'C{i:03d}',grid=[x,z],mask_stats=dict(mask_fraction=m))


class NovelGeometryTests(unittest.TestCase):
    def test_global_max_includes_distant_candidates_and_excludes_selected(self):
        pool=[point(0,0,0,0),point(1,0,1,.1),point(3,3,3,.14),point(2,-3,3,.14)]
        self.assertEqual(next_camera(pool,['C000']),('C001',['C001']))
        self.assertEqual(next_camera(pool,['C000'],'max-mask'),('C002',['C002','C003','C001']))
        self.assertEqual(next_camera(pool,['C000','C002'],'max-mask')[0],'C003')
        self.assertEqual(next_camera(pool,[c['id'] for c in pool],'max-mask'),(None,[]))
        with self.assertRaises(ValueError):next_camera(pool,[],'unknown')

    def test_repair_masks_use_only_cleaned_geometry(self):
        raw=np.zeros((64,128),bool);raw[12:44,26:62]=True
        raw[3,3]=True
        masks=repair_masks(raw)
        np.testing.assert_array_equal(masks['combined'],clean_mask(raw,periodic=True))
        np.testing.assert_array_equal(masks['geometry'],masks['combined'])
        self.assertFalse(masks['combined'][3,3])
        self.assertEqual(set(masks),{'geometry','combined'})

    def test_face_masks_do_not_add_transparent_or_black_holes(self):
        panorama=np.zeros((64,128,3),np.uint8)
        self.assertTrue(all(not face['mask'].any() for face in repair_faces(panorama,np.zeros((64,128),bool))))
        mask=np.zeros((64,128),bool);mask[20:44,55:73]=True
        actual=repair_faces(panorama,mask)
        projected=six_faces(panorama,mask)
        for face,expected in zip(actual,projected,strict=True):
            np.testing.assert_array_equal(face['mask'],clean_mask(expected['mask']))

    def test_cleanup_removes_specks_and_thin_lines_but_keeps_real_holes(self):
        m=np.zeros((32,64),bool);m[4:14,4:14]=True;m[20:23,20:23]=True;m[20:24,40:44]=True;m[28,5:30]=True
        cleaned=clean_mask(m,smoothing=False)
        self.assertEqual(cleaned.sum(),100+16)
        self.assertTrue(cleaned[20:24,40:44].all())
        self.assertFalse(cleaned[~m].any())
        np.testing.assert_array_equal(clean_mask(cleaned,smoothing=False),cleaned)

    def test_cleanup_joins_components_across_erp_seam_and_preserves_borders(self):
        m=np.zeros((16,32),bool);m[5:9,:2]=True;m[5:9,-2:]=True
        np.testing.assert_array_equal(clean_mask(m,periodic=True,smoothing=False),m)
        self.assertFalse(clean_mask(m,periodic=False).any())
        self.assertTrue(clean_mask(np.ones_like(m),True).all())
        self.assertFalse(clean_mask(np.zeros_like(m),True).any())

    def test_feather_threshold_rounds_convex_and_concave_edges(self):
        m=np.zeros((40,80),bool);m[5:35,10:60]=True;m[5,30]=False
        result=smooth_mask(m)
        self.assertEqual(result.dtype,np.bool_)
        self.assertFalse(result[5,10])  # Trim a convex corner.
        self.assertTrue(result[5,30])  # Fill the one-pixel concave notch.
        self.assertTrue(result[7:33,12:58].all())
        self.assertFalse(result[:4].any())

    def test_feather_is_periodic_at_erp_seam_and_handles_uniform_masks(self):
        m=np.zeros((32,64),bool);m[5:25,:6]=True;m[5:25,-6:]=True;m[5,0]=False
        result=smooth_mask(m,True)
        np.testing.assert_array_equal(np.roll(result,17,axis=1),smooth_mask(np.roll(m,17,axis=1),True))
        self.assertTrue(smooth_mask(np.ones_like(m),True).all())
        self.assertFalse(smooth_mask(np.zeros_like(m),True).any())

    def test_frontier_uses_all_selected_anchors(self):
        pool=[point(0,0,0,0),point(1,0,1,.1),point(2,1,0,.14),point(3,0,2,.12),point(4,0,3,.15)]
        self.assertEqual(next_camera(pool,['C000'])[0],'C002')
        # After selecting front, the right neighbor of ORIGINAL is still eligible.
        self.assertEqual(next_camera(pool,['C000','C001'])[0],'C002')
        self.assertNotIn('C004',next_camera(pool,['C000','C001'])[1])

    def test_no_diagonals_repeats_or_unknown_anchors(self):
        pool=[point(0,0,0,0),point(1,1,1,.14),point(2,0,1,.1)]
        self.assertEqual(next_camera(pool,['C000']),('C002',['C002']))
        self.assertEqual(next_camera(pool,['C000','C001','C002']),(None,[]))
        with self.assertRaises(ValueError):next_camera(pool,['C099'])

    def test_tie_break_is_numbering_not_input_order(self):
        pool=[point(2,1,0,.1),point(0,0,0,0),point(1,0,1,.1)]
        self.assertEqual(next_camera(pool,['C000'])[0],'C001')

    def test_outside_mask_is_byte_exact_even_with_total_model_redraw(self):
        rng=np.random.default_rng(4)
        original=rng.integers(0,256,(32,64,3),dtype=np.uint8)
        generated=255-original;mask=np.zeros((32,64),bool);mask[4:25,5:35]=True
        for periodic in [False,True]:
            output=composite_inside_mask(original,generated,mask,4,periodic)
            np.testing.assert_array_equal(output[~mask],original[~mask])
            np.testing.assert_array_equal(output[10:20,12:25],generated[10:20,12:25])
            self.assertTrue(np.any(output[4,5:35]!=original[4,5:35]))

    def test_empty_full_thin_and_erp_seam_masks(self):
        original=np.zeros((16,32,3),np.uint8);generated=np.full_like(original,240)
        np.testing.assert_array_equal(composite_inside_mask(original,generated,np.zeros((16,32),bool)),original)
        np.testing.assert_array_equal(composite_inside_mask(original,generated,np.ones((16,32),bool)),generated)
        mask=np.zeros((16,32),bool);mask[3:13,:3]=True;mask[3:13,-3:]=True
        output=composite_inside_mask(original,generated,mask,4,True)
        np.testing.assert_array_equal(output[:,0],output[:,-1])
        np.testing.assert_array_equal(output[~mask],original[~mask])

    def test_projection_axes_overlap_and_final_erp_preservation(self):
        rays=erp_directions(256,128);original=np.rint((rays+1)*127.5).astype(np.uint8)
        mask=np.zeros((128,256),bool);mask[25:70,110:150]=True
        faces=six_faces(original,mask,128)
        self.assertEqual([f['name'] for f in faces],['front','right','back','left','up','down'])
        self.assertTrue(all(f['camera']['hfov_deg']==100 for f in faces))
        fused=merge_faces([f['rgb'] for f in faces],256,128)
        self.assertLess(np.abs(fused.astype(float)-original).mean(),.4)
        output=composite_inside_mask(original,fused,mask,4,True)
        np.testing.assert_array_equal(output[~mask],original[~mask])

    def test_outward_feather_keeps_single_pixel_hole_fully_generated(self):
        original=np.zeros((17,17,3),np.uint8);generated=np.full_like(original,200)
        mask=np.zeros((17,17),bool);mask[8,8]=True
        output=composite_outward(original,generated,mask,4)
        self.assertEqual(output[8,8,0],200)
        self.assertEqual(output[8,9,0],150)
        self.assertEqual(output[8,10,0],100)
        self.assertEqual(output[8,11,0],50)
        self.assertEqual(output[8,12,0],0)
        weight=outward_blend_weights(mask,4)
        np.testing.assert_array_equal(output[weight==0],original[weight==0])

    def test_outward_feather_empty_full_zero_width_and_erp_seam(self):
        original=np.zeros((16,32,3),np.uint8);generated=np.full_like(original,240)
        empty=np.zeros((16,32),bool);full=~empty
        np.testing.assert_array_equal(composite_outward(original,generated,empty),original)
        np.testing.assert_array_equal(composite_outward(original,generated,full),generated)
        mask=empty.copy();mask[4:12,0]=True
        output=composite_outward(original,generated,mask,4,True)
        np.testing.assert_array_equal(output[mask],generated[mask])
        np.testing.assert_array_equal(output[4:12,-1],output[4:12,1])
        self.assertTrue(np.all(output[6,-1]==180))
        sharp=composite_outward(original,generated,mask,0,True)
        np.testing.assert_array_equal(sharp[~mask],original[~mask])
        for width in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):composite_outward(original,generated,mask,width)


if __name__=='__main__':unittest.main()
