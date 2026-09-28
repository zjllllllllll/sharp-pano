from copy import deepcopy
import unittest
from candidate_selection import select, apply_selection


def camera(i,x,z,mask=.1):
    return dict(id=f'C{i:03d}',grid=[x,z],position_world_m=[x*.5,0.,z*.5],
        coincides_with_original=(x==0 and z==0),mask_stats=dict(mask_fraction=mask))


class SelectionTests(unittest.TestCase):
    def test_threshold_equality_and_origin_exemption(self):
        pool=[camera(0,0,0,1),camera(1,0,1,.15),camera(2,1,0,.15000001)]
        self.assertEqual([c['id'] for c in select(pool)['kept']],['C000','C001'])

    def test_diagonal_is_not_connected(self):
        result=select([camera(0,0,0),camera(1,1,1)])
        self.assertEqual(result['states']['C001'],'disconnected')
        self.assertTrue(result['only_origin'])

    def test_threshold_before_connectivity_no_bridge_or_jump(self):
        result=select([camera(0,0,0),camera(1,1,0,.2),camera(2,2,0),camera(3,-2,0)])
        self.assertEqual(result['states'],dict(C000='origin',C001='excluded',C002='disconnected',C003='disconnected'))

    def test_can_reach_diagonal_by_valid_cardinal_path(self):
        pool=[camera(0,0,0),camera(1,1,1),camera(2,1,0),camera(3,0,-1),camera(4,-1,-1)]
        self.assertEqual(select(pool)['kept'],pool)

    def test_invalid_masks_do_not_bridge_and_pending_is_not_final_error(self):
        for value in [None,float('nan'),float('inf'),-.1,1.1,'0.1',True]:
            result=select([camera(0,0,0),camera(1,1,0,value),camera(2,2,0)])
            self.assertEqual(result['states']['C001'],'pending')
            self.assertEqual(result['states']['C002'],'disconnected')
            self.assertFalse(result['only_origin'])

    def test_idempotent_finalization_preserves_source_identity_and_order(self):
        info=dict(spacing_m=.5,cameras=[camera(0,0,0),camera(1,0,1,.2),camera(2,1,0),camera(3,0,2)],
            inward_offset_m=.3,offset_attempts=[dict(offset_m=.3)],sampling_error=None)
        original=deepcopy(info);output=apply_selection(info)
        self.assertEqual(info,original)
        self.assertEqual(output,apply_selection(output))
        self.assertEqual([c['id'] for c in output['cameras']],['C000','C002'])
        self.assertEqual(output['geometric_camera_count'],4)
        self.assertEqual(output['offset_attempts'],original['offset_attempts'])
        for a,b in zip(output['all_cameras'],original['cameras']):
            self.assertEqual({k:v for k,v in a.items() if k!='selection_state'},b)

    def test_only_origin_is_explicit_error_after_selection(self):
        result=apply_selection(dict(spacing_m=.5,cameras=[camera(0,0,0),camera(1,1,0,.2)]))
        self.assertTrue(result['selection_error'])
        self.assertTrue(result['selection_complete'])

    def test_malformed_grid_poses_rejected(self):
        for pool in [[camera(1,1,0)], [camera(0,0,0),camera(1,0,0)]]:
            with self.assertRaises(ValueError):select(pool)
        point=camera(1,1,0);point['position_world_m'][0]=1
        with self.assertRaises(ValueError):select([camera(0,0,0),point])


if __name__=='__main__':unittest.main()
