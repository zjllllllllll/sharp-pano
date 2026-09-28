import io
import unittest
import numpy as np
from PIL import Image
import novel_pipeline as n
from gpt_green_repair import generate,validate_capture
from test_novel_pipeline import NovelPipelineTests


class GreenRequestTests(unittest.TestCase):
    def setUp(self):
        fixture=NovelPipelineTests('runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.job=fixture.job;self.folder=fixture.folder
        self.preview=n.capture(self.job,'C001',fixture.payload)
        self.baseline=dict(revision=fixture.rev,camera_id='C001',capture=self.preview,
            image='old-image.png',comparison_variants=[])

    def test_green_requests_and_successful_faces_are_not_billed_twice(self):
        calls=[]
        def mock_request(path,out,model,prompt,key,**kw):
            with Image.open(path) as image:
                self.assertEqual(image.size,(1024,1024))
                pixels=np.asarray(image)
                self.assertTrue(np.any(np.all(pixels==[0,255,0],axis=-1)))
            self.assertIn('#00FF00',prompt);calls.append(path)
            Image.new('RGB',(2048,2048),(123,100,80)).save(out/'api_original.image',format='PNG')
        decode=lambda data:Image.open(io.BytesIO(data))
        first=generate(self.job,self.preview,self.baseline,mock_request,decode,'not-a-key',ConnectionError)
        count=len(calls);self.assertGreater(count,0)
        second=generate(self.job,self.preview,self.baseline,mock_request,decode,'not-a-key',ConnectionError)
        self.assertEqual(len(calls),count)
        self.assertEqual(first['image_sha256'],second['image_sha256'])
        self.assertEqual(first['inside_generated_mismatch_pixels'],0)
        self.assertEqual(first['outside_feather_changed_pixels'],0)
        fresh=generate(self.job,self.preview,{k:self.baseline[k] for k in ('revision','camera_id')},mock_request,decode,'not-a-key',ConnectionError)
        self.assertNotIn('comparison_variants',fresh)
        self.assertEqual(fresh['camera_id'],'C001')
        self.assertEqual(len(calls),count)
        def unavailable_provider():raise AssertionError('Cached images must not require live credentials')
        generate(self.job,self.preview,self.baseline,mock_request,decode,None,ConnectionError,unavailable_provider)
        self.assertEqual(len(calls),count)

    def test_uncertain_post_is_never_repeated_and_tampered_alpha_fails(self):
        count=[]
        def fail(*args,**kw):count.append(1);raise RuntimeError('network failed after submission')
        with self.assertRaises(RuntimeError):generate(self.job,self.preview,self.baseline,fail,None,'not-a-key',ConnectionError)
        with self.assertRaisesRegex(RuntimeError,'uncertain'):
            generate(self.job,self.preview,self.baseline,fail,None,'not-a-key',ConnectionError)
        self.assertEqual(len(count),1)
        Image.new('L',(512,512),0).save(self.folder/self.preview['path']/'front-render-alpha.png')
        with self.assertRaisesRegex(ValueError,'Alpha changed'):validate_capture(self.folder,self.preview)


if __name__=='__main__':unittest.main()
