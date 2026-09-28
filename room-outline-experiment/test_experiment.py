import unittest
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from PIL import Image
from cameras import sample_cameras, render_points, process, number_cameras, square_order, SamplingError
from shapely.geometry import Polygon, Point
from outline import fit


class CameraTests(unittest.TestCase):
    def test_default_renderer_rejects_legacy_point_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            old,new=Path(temp)/'old',Path(temp)/'new'
            old.mkdir();new.mkdir();(old/'panoramas').mkdir()
            np.array([[0,0,1,1,0,0]],np.float32).tofile(old/'cloud-full.bin')
            os.link(old/'cloud-full.bin',new/'cloud-full.bin')
            (new/'outline.json').write_text(json.dumps({'polygon_m':[[-1,-1],[1,-1],[1,1],[-1,1]]}),encoding='utf-8')
            cached=np.full((16,32,3),[40,80,120],np.uint8)
            Image.fromarray(cached).save(old/'panoramas/C900.png')
            Image.fromarray(np.full((16,32),255,np.uint8)).save(old/'panoramas/C900-coverage.png')
            previous=dict(render_resolution=[32,16],render_source='Native SHARP stage-3 point cloud, both layers; not final Gaussians',
                          cameras=[dict(id='C900',position_world_m=[0.,0.,0.],panorama='panoramas/C900.png',coverage=1.)])
            (old/'sample-cameras.json').write_text(json.dumps(previous),encoding='utf-8')
            def fake_renderer(job,**kwargs):
                result=json.loads((job/'sample-cameras.json').read_text(encoding='utf-8'))
                result['render_complete']=True
                for camera in result['cameras']:camera['mask_stats']={'mask_fraction':.1}
                return result
            with patch('cameras.draw_camera_map'),patch('cameras.WIDTH',32),patch('cameras.HEIGHT',16),patch('render_unknown.render_job',side_effect=fake_renderer) as renderer,patch('cameras.render_points') as legacy:
                result=process(new,inset=.5,reuse_from=old)
            self.assertEqual(result['count'],9)
            self.assertEqual(renderer.call_count,1)
            legacy.assert_not_called()
            from occlusion import VERSION
            self.assertEqual(result['render_version'],VERSION)
            center=next(c for c in result['cameras'] if c['position_world_m']==[0.,0.,0.])
            self.assertEqual(center['id'],'C000')
            self.assertNotIn('reused_camera_id',center)
            self.assertEqual(json.loads((old/'sample-cameras.json').read_text(encoding='utf-8')),previous)

    def test_regular_lattice_exact_distance_height(self):
        polygon=[[-3,-3],[3,-3],[3,3],[-3,3]]
        info=sample_cameras(polygon)
        self.assertEqual(info['count'],121)
        self.assertEqual(sample_cameras(polygon,inset=1.5)['count'],49)
        room=Polygon(polygon)
        for row in info['cameras']:
            x,y,z=row['position_world_m']
            self.assertEqual(y,0.)
            self.assertEqual(x/.5,round(x/.5));self.assertEqual(z/.5,round(z/.5))
            self.assertGreaterEqual(Point(x,z).distance(room.boundary)+1e-9,.5)
    def test_empty_inset_does_not_relax_rule(self):
        info=sample_cameras([[-.4,-4],[.4,-4],[.4,4],[-.4,4]],adaptive=False)
        self.assertEqual(info['count'],1)
        self.assertEqual(info['sampled_camera_count'],0)
        self.assertTrue(info['cameras'][0]['reference_only'])
        self.assertEqual(info['empty_reason'],'offset_erased_room')
    def test_concave_room_not_convex_hull(self):
        vertices=[[-5,-5],[5,-5],[5,0],[0,0],[0,5],[-5,5]]
        info=sample_cameras(vertices,adaptive=False)
        room=Polygon(vertices)
        for row in info['cameras']:
            if row['reference_only']:continue
            x,_,z=row['position_world_m']
            self.assertTrue(room.contains(Point(x,z)))
            self.assertGreaterEqual(room.boundary.distance(Point(x,z))+1e-9,.5)
            self.assertFalse(x>0 and z>0)

    def test_square_rings_begin_at_top_and_run_clockwise(self):
        info=sample_cameras([[-2,-2],[2,-2],[2,2],[-2,2]])
        self.assertEqual([c['grid'] for c in info['cameras'][:9]],
            [[0,0],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]])
        self.assertEqual(info['cameras'][9]['grid'],[0,2])
        self.assertEqual([c['id'] for c in info['cameras']], [f'C{i:03d}' for i in range(info['count'])])

    def test_adaptive_stops_at_first_sufficient_offset(self):
        info=sample_cameras([[-.8,-.8],[.8,-.8],[.8,.8],[-.8,.8]])
        self.assertEqual(info['inward_offset_m'],.3)
        self.assertEqual(info['neighbor_count'],8)
        self.assertEqual([a['offset_m'] for a in info['offset_attempts']],[.5,.45,.4,.35,.3])
        self.assertEqual(info['requested_offset_m'],.5)
        self.assertTrue(all(a['neighbor_count']<4 for a in info['offset_attempts'][:-1]))

    def test_many_distant_points_do_not_satisfy_near_neighbor_target(self):
        info=sample_cameras([[-.3,-4],[.3,-4],[.3,4],[-.3,4]])
        self.assertGreater(info['count'],4)
        self.assertEqual(info['neighbor_count'],2)
        self.assertEqual(info['inward_offset_m'],.05)
        self.assertIn(.1,[a['offset_m'] for a in info['offset_attempts']])
        self.assertEqual(info['offset_stop_reason'],'below_min_offset')
        self.assertTrue(info['sampling_warning'])

    def test_only_origin_is_explicit_sampling_error(self):
        with self.assertRaises(SamplingError) as caught:
            sample_cameras([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]])
        info=caught.exception.info
        self.assertEqual(info['count'],1)
        self.assertEqual(info['inward_offset_m'],.05)
        self.assertEqual(info['cameras'][0]['id'],'C000')
        self.assertTrue(info['sampling_error'])

    def test_process_persists_only_origin_error_for_frontend(self):
        with tempfile.TemporaryDirectory() as folder:
            job=Path(folder)
            (job/'outline.json').write_text(json.dumps({'polygon_m':[[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2]]}),encoding='utf-8')
            with patch('cameras.draw_camera_map'):
                process(job,render=False)
            saved=json.loads((job/'sample-cameras.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['count'],1)
            self.assertTrue(saved['sampling_error'])
            self.assertEqual(saved['inward_offset_m'],.05)

    def test_adaptive_nodes_respect_final_boundary_and_height(self):
        info=sample_cameras([[-.8,-.8],[.8,-.8],[.8,.8],[-.8,.8]])
        for c in info['cameras']:
            self.assertEqual(c['position_world_m'][1],0.)
            if not c['reference_only']:
                self.assertGreaterEqual(c['boundary_distance_m']+1e-9,info['inward_offset_m'])

    def test_bad_sampling_parameters_fail(self):
        for options in [dict(spacing=0),dict(inset=-1),dict(spacing=float('nan'))]:
            with self.assertRaises(ValueError):sample_cameras([[-1,-1],[1,-1],[1,1],[-1,1]],**options)

    def test_renumber_preserves_positions_and_adds_only_origin(self):
        info=sample_cameras([[-2,-2],[2,-2],[2,2],[-2,2]])
        info['cameras']=[c for c in info['cameras'] if c['grid'][0]>0]
        old={tuple(c['position_world_m']) for c in info['cameras']}
        result=number_cameras(info)
        self.assertEqual({tuple(c['position_world_m']) for c in result['cameras']},old|{(0.,0.,0.)})
        self.assertTrue(result['cameras'][0]['reference_only'])
        self.assertEqual(result['cameras'],sorted(result['cameras'],key=square_order))
    def test_nearest_point_wins_and_unknown_stays_empty(self):
        cloud=np.array([[0,0,1,1,0,0],[0,0,2,0,1,0]],dtype=np.float32)
        rgb,valid=render_points(cloud,[0,0,0],width=64,height=32)
        np.testing.assert_array_equal(rgb[16,32],[255,0,0])
        self.assertLess(valid.mean(),.01)
    def test_panorama_axes(self):
        cloud=np.array([[1,0,0,1,0,0],[-1,0,0,0,1,0],[0,-1,0,0,0,1]],dtype=np.float32)
        rgb,_=render_points(cloud,[0,0,0],width=64,height=32)
        np.testing.assert_array_equal(rgb[16,48],[255,0,0])
        np.testing.assert_array_equal(rgb[16,16],[0,255,0])
        np.testing.assert_array_equal(rgb[0,32],[0,0,255])
    def test_synthetic_rectangular_wall_outline(self):
        heights=np.linspace(-1.3,1.3,40)
        xyz=[]
        for x in np.linspace(-3,3,180):
            for z in [-2.5,2.5]:
                xyz.extend((x,h,z) for h in heights)
        for z in np.linspace(-2.5,2.5,180):
            for x in [-3,3]:
                xyz.extend((x,h,z) for h in heights)
        result,_=fit(np.asarray(xyz,np.float32))
        polygon=Polygon(result['polygon_m'])
        self.assertTrue(polygon.is_valid)
        self.assertTrue(polygon.contains(Point(0,0)))
        self.assertAlmostEqual(polygon.area,30,delta=2)


if __name__=='__main__':unittest.main()
