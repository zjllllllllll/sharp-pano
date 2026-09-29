import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData

import gaussian_densify as gd
sys.path.insert(0, str(gd.ROOT/'third_party/ml-sharp/src'))
from sharp.models.initializer import MultiLayerInitializer
from worker import base_scene
from gaussian_ops import ndc_to_camera, world_transform, to_vertices, DTYPE
from geometry import camera
from point_densify import interpolate_children


class DenseTests(unittest.TestCase):
    def fixture(self):
        init=MultiLayerInitializer(num_layers=2,stride=2,base_depth=10,scale_factor=1,
                                  disparity_factor=1,color_option='all_layers')
        initial=init(torch.full((1,3,8,8),.5),torch.full((1,2,8,8),3.))
        cam=camera(2);scene,_=ndc_to_camera(base_scene(initial),cam);base=world_transform(scene,cam)
        vertices,_,source=to_vertices(base,2,cam['hfov_deg'],return_indices=True)
        return cam,base,vertices,source,np.ones(32,dtype=bool)

    def test_stencil_separate_layers(self):
        idx,weights=gd.stencil([0,2,9*3],3)
        np.testing.assert_array_equal(idx,[[0,1,3,4],[0,1,3,4],[9,10,12,13]])
        np.testing.assert_allclose(weights.sum(1),1)

    def test_initial_exact_parent_fields_and_point_centers(self):
        cam,base,vertices,source,keep=self.fixture()
        ids=np.array([5*3,5*3+1,5*3+2,(16+5)*3+2])
        # Every parent has distinctive values; averaging even two parents fails.
        for field in vertices.dtype.names:
            if field not in ('x','y','z') and not field.startswith('rot_'):
                vertices[field]=np.arange(len(vertices),dtype='f4')*.013
        before=vertices.copy()
        children,stats=gd.inherit_children(vertices,source,keep,base,ids,cam,4,False)
        rows=np.searchsorted(source,ids//3)
        for field in vertices.dtype.names:
            if field not in ('x','y','z'):
                np.testing.assert_array_equal(children[field],vertices[field][rows])
        points=interpolate_children(base['means'],base['colors'],ids,cam,4)
        np.testing.assert_array_equal(np.column_stack([children[k] for k in ('x','y','z')]),points[:,:3])
        np.testing.assert_array_equal(vertices,before)
        self.assertEqual(stats['added'],4)

    def test_adjusted_inherits_single_parents_world_offset(self):
        cam,base,vertices,source,keep=self.fixture()
        ids=np.array([15,16,17,63])
        offsets=np.column_stack((source*.002,source*.003,source*-.004)).astype('f4')
        for axis,key in enumerate(('x','y','z')):vertices[key]+=offsets[:,axis]
        before=vertices.copy()
        children,_=gd.inherit_children(vertices,source,keep,base,ids,cam,4,True)
        rows=np.searchsorted(source,ids//3)
        parent_xyz=np.column_stack([vertices[k][rows] for k in ('x','y','z')])
        points=interpolate_children(base['means'],base['colors'],ids,cam,4)[:,:3]
        expected=points+(parent_xyz-base['means'][ids//3])
        np.testing.assert_array_equal(np.column_stack([children[k] for k in ('x','y','z')]),expected)
        for field in vertices.dtype.names:
            if field not in ('x','y','z'):np.testing.assert_array_equal(children[field],vertices[field][rows])
        np.testing.assert_array_equal(vertices,before)

    def test_missing_parent_mask_holes_and_no_spatial_reassignment(self):
        cam,base,vertices,source,keep=self.fixture()
        # Remove anchor 5 from native output; never borrow another parent's fields.
        retained=source!=5
        children,stats=gd.inherit_children(vertices[retained],source[retained],keep,base,np.array([15]),cam,4,True)
        self.assertEqual(len(children),0);self.assertEqual(stats['missing_native_parent'],1)
        keep[6]=False
        children,stats=gd.inherit_children(vertices,source,keep,base,np.array([15]),cam,4,False)
        self.assertEqual(len(children),0);self.assertEqual(stats['masked_parent_cells'],1)

    def test_invalid_ids_and_empty_input(self):
        cam,base,vertices,source,keep=self.fixture()
        for ids in ([-1],[3*3],[12*3],[32*3]):
            with self.assertRaises(ValueError):gd.inherit_children(vertices,source,keep,base,np.array(ids),cam,4,False)
        children,stats=gd.inherit_children(vertices,source,keep,base,np.array([],dtype='i8'),cam,4,False)
        self.assertEqual(len(children),0);self.assertEqual(stats['added'],0)

    def test_shared_parts_keep_original_rows_unchanged(self):
        cam,base,vertices,source,keep=self.fixture();keep[10]=False
        children,stats=gd.inherit_children(vertices,source,keep,base,np.array([15]),cam,4,True)
        before=vertices.copy()
        with tempfile.TemporaryDirectory() as root:
            writer=gd.ComparisonWriter(Path(root),5)
            writer.append(vertices,source,keep,children,stats)
            with patch.object(PlyData,'read',side_effect=AssertionError('Export must not read back PLY')):
                views,_=writer.finish()
            arrays={k:PlyData.read(Path(root)/v['preview'],mmap=False)['vertex'].data for k,v in views.items()}
            np.testing.assert_array_equal(arrays['baseline'],before)
            np.testing.assert_array_equal(arrays['common'],before[keep[source]])
            np.testing.assert_array_equal(arrays['masked'],arrays['common'])
            np.testing.assert_array_equal(arrays['dense'],np.concatenate((arrays['common'],children)))
            self.assertEqual(set(gd.PARTS),{'common','removed','children'})
        np.testing.assert_array_equal(vertices,before)

    def test_new_initialized_stage_publishes_all_variants(self):
        from worker import gaussian_stage
        init=MultiLayerInitializer(num_layers=2,stride=2,base_depth=10,scale_factor=1,disparity_factor=1)
        raw=base_scene(init(torch.full((1,3,8,8),.5),torch.full((1,2,8,8),3.)))
        with tempfile.TemporaryDirectory() as root:
            job=Path(root);(job/'base').mkdir()
            (job/'stage-3.json').write_text(json.dumps(dict(diagnostics=dict(initializer_grid=[4,4]))))
            np.array([i*32*3+15 for i in range(6)],dtype='<u4').tofile(job/gd.INDEX)
            for i in range(6):
                cam=camera(i);scene,_=ndc_to_camera(raw,cam);scene=world_transform(scene,cam)
                np.savez(job/'base'/f'{i}.npz',**scene)
            mask=dict(parameters={},sha256='test',count=192,kept=192,removed=0,index='test',outside_cloud_policy='keep')
            density=dict(added=6,mask_sha256='test')
            with patch('point_mask.build_mask',return_value=mask),patch('point_mask.load_face',return_value=np.ones(32,dtype=bool)),patch('gaussian_densify.build_densification',return_value=density):
                result=gaussian_stage(job,False)
            self.assertEqual(result['densification']['version'],3)
            self.assertEqual(set(result['variants']),{'masked','unmasked'})
            self.assertEqual(result['count'],result['densification']['baseline_variants']['masked']['count'])
            self.assertEqual(result['densification']['added'],6)
            common=PlyData.read(job/result['densification']['parts']['common']['preview'],mmap=False)['vertex'].data
            original=PlyData.read(job/result['variants']['masked']['preview'],mmap=False)['vertex'].data
            np.testing.assert_array_equal(common,original)


if __name__ == '__main__':
    unittest.main()
