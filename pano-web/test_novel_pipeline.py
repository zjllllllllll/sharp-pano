"""No network, no model billing, no edits to real run artifacts."""
import base64
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import novel_pipeline as n
from novel_geometry import erp_directions,six_faces,outward_blend_weights


class NovelPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.job=Path(self.temp.name);self.rev='a'*16;self.folder=self.job/'novel'/self.rev;self.folder.mkdir(parents=True)
        (self.job/'source.ply').write_bytes(b'fixture-only-not-a-real-ply')
        self.source=dict(file='source.ply',sha256=n.digest(self.job/'source.ply'),count=123,variant='masked')
        self.mask=np.zeros((512,1024),bool);self.mask[180:330,410:620]=True
        Image.fromarray(self.mask.astype(np.uint8)*255).save(self.job/'mask.png')
        self.rgb=np.rint((erp_directions()+1)*127.5).astype(np.uint8)
        Image.fromarray(self.rgb).save(self.job/'input.png')
        self.cameras=[dict(id=f'C{i:03d}',grid=[x,z],position_world_m=[x*.5,0,z*.5],
            coincides_with_original=i==0,mask_stats=dict(mask_fraction=value),views=dict(mask='mask.png'))
            for i,x,z,value in [(0,0,0,0),(1,0,1,.1),(2,1,0,.14),(3,0,2,.12)]]
        n.write_json(self.job/'job.json',dict(stages={'6':dict(revision=self.rev)}))
        n.write_json(self.folder/'pipeline.json',dict(revision=self.rev,source=self.source,mask_source=str(self.job),mask_version='unknown-volume-v2-footprints'))
        n.write_json(self.folder/'sample-cameras.json',dict(cameras=self.cameras,inward_offset_m=.5,neighbor_count=8))
        n.write_json(self.folder/'selection.json',dict(selected_ids=['C000'],current=None,completed_ids=[],previews={}))
        frames=six_faces(self.rgb,self.mask);uris=[]
        for f in frames:
            data=io.BytesIO();Image.fromarray(f['rgb']).save(data,format='PNG')
            uris.append('data:image/png;base64,'+base64.b64encode(data.getvalue()).decode())
        data=io.BytesIO();Image.fromarray(np.full((512,512),255,np.uint8)).save(data,format='PNG')
        opaque='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
        self.payload=dict(revision=self.rev,source_sha256=self.source['sha256'],faces=uris,
            alpha_faces=[opaque]*6,capture_version=n.CAPTURE_VERSION)

    def test_capture_preserves_raw_mask_and_does_not_select(self):
        before=n.public_state(self.job)
        result=n.capture(self.job,'C001',self.payload)
        after=n.public_state(self.job)
        self.assertEqual(before['selected_ids'],after['selected_ids'])
        self.assertEqual(after['recommended'],'C002')
        self.assertEqual(result['raw_geometry_sha256'],n.digest(self.job/'mask.png'))
        self.assertNotEqual(result['mask_sha256'],result['raw_geometry_sha256'])
        self.assertEqual(result['mask_cleanup']['smoothing']['threshold'],.5)
        self.assertTrue((self.folder/result['path']/'up-masked.png').exists())
        self.assertLess(np.abs(np.asarray(Image.open(self.folder/result['path']/'raw.png')).astype(float)-self.rgb).mean(),.6)

    def test_global_policy_is_used_for_confirmation(self):
        manifest=n.read(self.folder/'pipeline.json');manifest['selection_policy']='max-mask'
        n.write_json(self.folder/'pipeline.json',manifest)
        info=n.read(self.folder/'sample-cameras.json');info['cameras'][-1]['mask_stats']['mask_fraction']=.15
        n.write_json(self.folder/'sample-cameras.json',info)
        self.assertEqual(n.public_state(self.job)['recommended'],'C003')
        n.capture(self.job,'C003',self.payload);n.capture(self.job,'C001',self.payload)
        self.assertEqual(n.begin_repair(self.job)['camera_id'],'C003')

    def test_confirmation_uses_recommended_not_last_preview_and_is_idempotent(self):
        n.capture(self.job,'C002',self.payload);n.capture(self.job,'C001',self.payload)
        result=n.begin_repair(self.job)
        self.assertEqual(result['camera_id'],'C002')
        self.assertEqual(n.begin_repair(self.job),result)
        self.assertEqual(n.public_state(self.job)['selected_ids'],['C000'])
        self.assertEqual(n.public_state(self.job)['highlighted'],'C002')

    def test_transparency_never_expands_panorama_or_face_masks(self):
        baseline=n.capture(self.job,'C001',self.payload)
        baseline_dir=self.folder/baseline['path']
        data=io.BytesIO();Image.fromarray(np.zeros((512,512,3),np.uint8)).save(data,format='PNG')
        black='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
        alpha=np.zeros((512,512),np.uint8)
        data=io.BytesIO();Image.fromarray(alpha).save(data,format='PNG')
        transparent='data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
        payload={**self.payload,'faces':[black]*6,'alpha_faces':[transparent]*6}
        result=n.capture(self.job,'C001',payload);capture=self.folder/result['path']
        final_mask=np.asarray(Image.open(capture/'mask.png'))>0
        geometry=np.asarray(Image.open(capture/'geometry-mask.png'))>0
        np.testing.assert_array_equal(final_mask,geometry)
        np.testing.assert_array_equal(final_mask,np.asarray(Image.open(baseline_dir/'mask.png'))>0)
        for face in n.FACE_NAMES:
            np.testing.assert_array_equal(np.asarray(Image.open(capture/f'{face}-mask.png')),
                np.asarray(Image.open(baseline_dir/f'{face}-mask.png')))
        masked=np.asarray(Image.open(capture/'masked.png'))
        self.assertTrue(np.all(masked[final_mask]==n.MASK_COLOR))
        self.assertTrue(np.all(masked[~final_mask]==0))
        self.assertFalse((capture/'alpha-mask.png').exists())
        self.assertFalse((capture/'alpha-mask-raw.png').exists())
        self.assertNotIn('alpha_threshold',result)
        self.assertNotIn('alpha_added_pixels',result)
        # Existing quality evaluation still gets the true render alpha.
        np.testing.assert_array_equal(np.asarray(Image.open(capture/'front-render-alpha.png')),alpha)

    def test_capture_without_quality_alpha_is_rejected(self):
        payload={**self.payload};payload.pop('alpha_faces')
        with self.assertRaises(ValueError):n.capture(self.job,'C001',payload)

    def test_bad_capture_and_artifact_paths_rejected(self):
        for cid,payload in [('C999',self.payload),('C001',{**self.payload,'revision':'b'*16}),
                            ('C001',{**self.payload,'faces':['not-png']*6}),
                            ('C001',{**self.payload,'capture_version':'unknown-plus-alpha070-open3-smooth1-green-v3'})]:
            with self.assertRaises(ValueError):n.capture(self.job,cid,payload)
        for path in [f'{self.rev}/pipeline.json',f'{self.rev}/selection.json','../job.json',f'{self.rev}/../../job.json']:
            with self.assertRaises(FileNotFoundError):n.artifact(self.job,path)
        (self.job/'source.ply').write_bytes(b'changed')
        with self.assertRaises(ValueError):n.capture(self.job,'C001',self.payload)

    def test_mock_model_preserves_outside_outer_band_and_retry_is_cached(self):
        preview=n.capture(self.job,'C002',self.payload);n.begin_repair(self.job)
        # No original-camera reference image may be read for the single-image API.
        (self.job/'input.png').unlink()
        count=[];sizes=[]
        module=types.ModuleType('click_gaussian.image_repair')
        module.RepairConfig=lambda **kw:types.SimpleNamespace(model='mock-seedream',**kw)
        module.request_repair=lambda images,**kw:(count.append((images,kw['config'])) or ['mock-result'],{})
        module.fetch_image=lambda url:np.ones((512,512,3),np.float32)*.8
        def encode(image):
            data=io.BytesIO()
            Image.fromarray(np.rint(image*255).astype(np.uint8)).save(data,format='PNG')
            return 'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()
        module.encode_data_uri=encode
        module.minimum_seedream_output_size=lambda *args:(sizes.append(args) or '1920x1920')
        with patch.dict('sys.modules',{'click_gaussian.image_repair':module}):
            result=n.repair(self.job);calls=len(count);again=n.repair(self.job)
            self.assertEqual(len(count),calls)
            # Legacy receipts have no request fingerprint: never reuse them under
            # the new prompt/input settings. This affects this temp fixture only.
            face=next(row['face'] for row in result['faces'] if not row['skipped'])
            receipt=self.folder/preview['path']/'repair'/f'{face}.json'
            legacy=n.read(receipt);legacy.pop('request_fingerprint');n.write_json(receipt,legacy)
            with self.assertRaisesRegex(RuntimeError,'uncertain'):n.repair(self.job)
            self.assertEqual(len(count),calls)
        self.assertEqual(sizes,[(1024,1024)]*3)
        for images,config in count:
            self.assertEqual(len(images),1)
            with Image.open(io.BytesIO(base64.b64decode(images[0].split(',',1)[1]))) as image:
                self.assertEqual(image.size,(1024,1024))
            self.assertEqual(config.prompt,n.REPAIR_PROMPT)
            self.assertEqual(config.output_size,'1920x1920')
        self.assertEqual(result['request']['image_count'],1)
        self.assertEqual(result['request']['input_size'],[1024,1024])
        self.assertGreater(calls,0)
        self.assertEqual(result['outside_feather_changed_pixels'],0)
        self.assertEqual(result['inside_generated_mismatch_pixels'],0)
        raw=np.asarray(Image.open(self.folder/preview['path']/'raw.png'))
        final=np.asarray(Image.open(self.folder/result['image']))
        final_mask=np.asarray(Image.open(self.folder/preview['path']/'mask.png'))>0
        weight=outward_blend_weights(final_mask,4,True)
        np.testing.assert_array_equal(final[weight==0],raw[weight==0])
        fused=np.asarray(Image.open(self.folder/preview['path']/'repair/fused-before-composite.png'))
        np.testing.assert_array_equal(final[final_mask],fused[final_mask])
        self.assertTrue(np.any(final[final_mask]!=raw[final_mask]))
        self.assertEqual(result['image_sha256'],again['image_sha256'])
        self.assertTrue(all(row['outside_feather_changed_pixels']==0 for row in result['faces']))

    def test_single_image_input_resizes_then_applies_binary_mask(self):
        base=np.zeros((512,512,3),np.uint8);base[:]=[180,150,100]
        mask=np.zeros((512,512),bool);mask[100:200,200:300]=True
        image=n.repair_input(base,mask)
        expected_mask=np.repeat(np.repeat(mask,2,axis=0),2,axis=1)
        self.assertEqual(image.shape,(1024,1024,3))
        self.assertTrue(np.all(image[expected_mask]==[0,255,0]))
        self.assertTrue(np.all(image[~expected_mask]==[180,150,100]))
        # Input preparation must not mutate the native face or its mask.
        self.assertTrue(np.all(base==[180,150,100]))
        self.assertEqual(int(mask.sum()),10000)


if __name__=='__main__':unittest.main()
