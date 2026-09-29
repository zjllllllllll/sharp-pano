"""Small regression checks; no model weights/GPU needed."""
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
import numpy as np
from PIL import Image
import server


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.original = server.DATA
        server.DATA = Path(cls.temp.name)
        cls.http = server.ThreadingHTTPServer(('127.0.0.1',0), server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        server.DATA = cls.original
        cls.temp.cleanup()

    def request(self, path, data=None, headers=None):
        req = urllib.request.Request(self.url+path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def image(self, size):
        raw = io.BytesIO()
        Image.new('RGB',size,(120,60,20)).save(raw,format='PNG')
        return raw.getvalue()

    def test_mode_assets_are_served(self):
        for path,mime in [('/workflow-mode.mjs','text/javascript'),('/mode.css','text/css')]:
            with urllib.request.urlopen(self.url+path) as response:
                self.assertEqual(response.status,200)
                self.assertIn(mime,response.headers['Content-Type'])
                self.assertGreater(len(response.read()),0)

    def test_upload_validation_and_token(self):
        self.assertEqual(self.request('/api/upload',self.image((512,256)))[0],403)
        headers={'X-Local-Token':server.TOKEN,'Content-Type':'image/png'}
        self.assertEqual(self.request('/api/upload',self.image((512,512)),headers)[0],400)
        self.assertEqual(self.request('/api/upload',b'not an image',headers)[0],400)
        code,job=self.request('/api/upload',self.image((512,256)),headers)
        self.assertEqual(code,201)
        self.assertEqual(list(job['stages']),['1'])
        self.assertEqual(self.request('/api/jobs/'+job['id'])[1]['id'],job['id'])
        self.assertEqual(self.request('/api/jobs/'+job['id']+'/debug/1')[0],200)

    def test_workflow_mode_and_stale_next_do_not_run_another_stage(self):
        headers={'X-Local-Token':server.TOKEN, 'X-Workflow-Mode':'auto'}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        self.assertEqual(job['execution_mode'],'auto')
        folder=server.job_path(job['id']);endpoint='/api/jobs/'+job['id']+'/next'
        with patch('server.run_stage') as worker:
            code,state=self.request(endpoint,b'{"expected_stage":2}',headers)
            self.assertEqual(code,202)
            worker.assert_called_once_with(folder,2)
        # Simulate completion before a second tab submits its stale request.
        server.GPU.release()
        state.update(busy=False,running_stage=None)
        state['stages']['2']={'kind':'fixture'}
        server.write_job(folder,state)
        with patch('server.run_stage') as worker:
            code,repeated=self.request(endpoint,b'{"expected_stage":2}',headers)
            self.assertEqual(code,200)
            worker.assert_not_called()
        self.assertEqual(list(repeated['stages']),['1','2'])
        self.assertFalse(server.GPU.locked())
        _,manual=self.request('/api/upload',self.image((512,256)),{'X-Local-Token':server.TOKEN})
        self.assertEqual(manual['execution_mode'],'debug')

    def test_novel_routes_fail_closed_and_keep_gpu_lock_healthy(self):
        headers={'X-Local-Token':server.TOKEN}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        root='/api/jobs/'+job['id'];folder=server.job_path(job['id'])
        self.assertEqual(self.request(root+'/novel')[0],404)
        self.assertEqual(self.request(root+'/novel/preview/C000',b'{}')[0],403)
        self.assertEqual(self.request(root+'/novel/preview/C000',b'{}',headers)[0],400)
        self.assertEqual(self.request(root+'/next',b'[]',headers)[0],400)
        state=server.read_job(folder);state['stages']={str(i):{'kind':'test'} for i in range(1,6)};server.write_job(folder,state)
        self.assertEqual(self.request(root+'/next',b'{"gaussian_variant":"invalid"}',headers)[0],400)
        self.assertFalse(server.GPU.locked())
        with patch('server.run_stage') as worker:
            code,value=self.request(root+'/next',b'{"gaussian_variant":"dense"}',headers)
            self.assertEqual(code,202);self.assertEqual(value['running_stage'],6)
            self.assertEqual(value['novel_scene_variant'],'dense');worker.assert_called_once_with(folder,6)
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled candidate failure')):
            server.run_stage(folder,6)
        self.assertFalse(server.GPU.locked());self.assertFalse(server.NORMALS.locked())
        self.assertEqual(len(server.read_job(folder)['stages']),5)

    def test_repair_worker_uses_reference_launcher_and_cwd(self):
        _,job=self.request('/api/upload',self.image((512,256)),{'X-Local-Token':server.TOKEN})
        folder=server.job_path(job['id'])
        server.GPU.acquire()
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled preflight')) as process:
            server.run_stage(folder,7)
        args,kwargs=process.call_args
        self.assertEqual(args[0][0],str(server.novel_pipeline.LAUNCHER))
        self.assertEqual(kwargs['cwd'],server.novel_pipeline.REFERENCE)
        self.assertFalse(server.GPU.locked());self.assertFalse(server.NORMALS.locked())

    def test_depth_stage_api_launcher_and_completion_limit(self):
        headers={'X-Local-Token':server.TOKEN}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        folder=server.job_path(job['id']);endpoint='/api/jobs/'+job['id']+'/next'
        state=server.read_job(folder)
        state['stages']={str(i):{'kind':'fixture'} for i in range(1,8)}
        server.write_job(folder,state)
        with patch('server.run_stage') as worker, patch('quality_retry.prepare_run'):
            code,result=self.request(endpoint,b'{}',headers)
            self.assertEqual(code,202);self.assertEqual(result['running_stage'],8)
            worker.assert_called_once_with(folder,8)
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled LingBot preflight')) as process:
            server.run_stage(folder,8)
        args,kwargs=process.call_args
        self.assertEqual(args[0][0],str(server.novel_depth.LAUNCHER))
        self.assertEqual(kwargs['cwd'],server.novel_depth.REFERENCE)
        self.assertFalse(server.GPU.locked());self.assertEqual(len(server.read_job(folder)['stages']),7)
        state=server.read_job(folder);state['stages']['8']={'kind':'depth-completion'};server.write_job(folder,state)
        self.assertEqual(self.request('/api/jobs/'+job['id']+'/debug/8')[0],200)
        with patch('server.run_stage') as worker, patch('quality_retry.prepare_run'):
            code,result=self.request(endpoint,b'{}',headers)
            self.assertEqual(code,202);self.assertEqual(result['running_stage'],9)
            worker.assert_called_once_with(folder,9)
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled SHARP repair preflight')) as process:
            server.run_stage(folder,9)
        args,kwargs=process.call_args
        self.assertEqual(args[0][0],str(server.novel_depth.LAUNCHER))
        self.assertEqual(kwargs['cwd'],server.novel_depth.REFERENCE)
        self.assertFalse(server.GPU.locked())
        state=server.read_job(folder);state['stages']['9']={'kind':'gaussian-repair'};server.write_job(folder,state)
        self.assertEqual(self.request('/api/jobs/'+job['id']+'/debug/9')[0],200)
        state['stages'].update({str(i):{'kind':'fixture'} for i in range(10,14)})
        server.write_job(folder,state)
        with patch('server.run_stage') as worker:
            code,result=self.request(endpoint,b'{}',headers)
            self.assertEqual(code,202);self.assertEqual(result['running_stage'],14)
            worker.assert_called_once_with(folder,14)
        server.GPU.release()
        state.update(busy=False,running_stage=None,iteration_status='stopped')
        server.write_job(folder,state)
        self.assertEqual(self.request(endpoint,b'{}',headers)[0],400)
        self.assertFalse(server.GPU.locked())
        state.pop('iteration_status');state['stages'].update({str(i):{'kind':'fixture'} for i in range(14,26)})
        server.write_job(folder,state)
        self.assertEqual(self.request(endpoint,b'{}',headers)[0],400)
        self.assertFalse(server.GPU.locked())

    def test_reject_untrusted_host_and_traversal(self):
        self.assertEqual(self.request('/api/state',headers={'Host':'external.invalid'})[0],403)
        self.assertEqual(self.request('/artifacts/../../server.py')[0],404)
        self.assertEqual(self.request('/vendor/../server.py')[0],404)

    def test_gaussian_density_requires_token_stage_and_releases_locks(self):
        headers={'X-Local-Token':server.TOKEN}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        endpoint='/api/jobs/'+job['id']+'/gaussian-densify/4'
        self.assertEqual(self.request(endpoint,b'')[0],403)
        self.assertEqual(self.request(endpoint,b'',headers)[0],409)
        folder=server.job_path(job['id'])
        before=(folder/'job.json').read_bytes()
        self.assertTrue(server.GPU.acquire(blocking=False))
        self.assertTrue(server.NORMALS.acquire(blocking=False))
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled dense failure')):
            server.run_gaussian_dense(folder,4)
        self.assertFalse(server.GPU.locked());self.assertFalse(server.NORMALS.locked())
        self.assertEqual(self.request(endpoint)[1]['status'],'error')
        self.assertEqual((folder/'job.json').read_bytes(),before)

    def test_normals_requires_token_and_cloud_and_preserves_pipeline(self):
        headers={'X-Local-Token':server.TOKEN}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        endpoint='/api/jobs/'+job['id']+'/point-normals'
        self.assertEqual(self.request(endpoint,b'')[0],403)
        self.assertEqual(self.request(endpoint,b'',headers)[0],409)
        self.assertEqual(self.request(endpoint)[1]['status'],'missing')
        folder=server.job_path(job['id'])
        before=(folder/'job.json').read_bytes()
        self.assertTrue(server.NORMALS.acquire(blocking=False))
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled normal failure')):
            server.run_normals(folder)
        self.assertEqual(self.request(endpoint)[1]['status'],'error')
        self.assertFalse(server.NORMALS.locked())
        self.assertEqual((folder/'job.json').read_bytes(),before)

    def test_stale_normals_ready_state_becomes_missing(self):
        _,job=self.request('/api/upload',self.image((512,256)),{'X-Local-Token':server.TOKEN})
        server.NORMAL_STATES[job['id']]={'status':'ready'}
        self.assertEqual(server.normals_state(server.job_path(job['id']))['status'],'missing')

    def test_distances_require_token_cloud_and_preserve_pipeline_on_failure(self):
        headers={'X-Local-Token':server.TOKEN}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        endpoint='/api/jobs/'+job['id']+'/point-distances'
        self.assertEqual(self.request(endpoint,b'')[0],403)
        self.assertEqual(self.request(endpoint,b'',headers)[0],409)
        self.assertEqual(self.request(endpoint)[1]['status'],'missing')
        folder=server.job_path(job['id'])
        before=(folder/'job.json').read_bytes()
        self.assertTrue(server.NORMALS.acquire(blocking=False))
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled distance failure')):
            server.run_distances(folder)
        self.assertEqual(self.request(endpoint)[1]['status'],'error')
        self.assertFalse(server.NORMALS.locked())
        self.assertEqual((folder/'job.json').read_bytes(),before)
        server.DISTANCE_STATES[job['id']]={'status':'ready'}
        self.assertEqual(server.distances_state(folder)['status'],'missing')

    def test_model_failure_preserves_completed_stage_and_unlocks(self):
        _,job=self.request('/api/upload',self.image((512,256)),{'X-Local-Token':server.TOKEN})
        folder=server.job_path(job['id'])
        state=server.read_job(folder)
        state.update(busy=True,running_stage=2)
        server.write_job(folder,state)
        self.assertTrue(server.GPU.acquire(blocking=False))
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled failure')):
            server.run_stage(folder,2)
        result=server.read_job(folder)
        self.assertFalse(result['busy'])
        self.assertEqual(list(result['stages']),['1'])
        self.assertIn('controlled failure',result['error'])
        self.assertFalse(server.GPU.locked())

    def test_densification_route_locks_and_failure_preserve_pipeline(self):
        headers={'X-Local-Token':server.TOKEN}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        folder=server.job_path(job['id']);endpoint='/api/jobs/'+job['id']+'/point-densify'
        self.assertEqual(self.request(endpoint,b'')[0],403)
        self.assertEqual(self.request(endpoint,b'',headers)[0],409)
        self.assertEqual(self.request(endpoint)[1]['status'],'missing')
        state=server.read_job(folder);state['stages']['3']={'kind':'points'};server.write_job(folder,state)
        original=(folder/'job.json').read_bytes()
        self.assertTrue(server.NORMALS.acquire(blocking=False))
        try:
            self.assertEqual(self.request(endpoint,b'',headers)[0],409)
            self.assertFalse(server.GPU.locked())
        finally:server.NORMALS.release()
        with patch('server.run_densification') as worker:
            code,value=self.request(endpoint,b'',headers)
            self.assertEqual(code,202);self.assertEqual(value['status'],'running')
            worker.assert_called_once_with(folder)
            self.assertEqual(self.request(endpoint,b'',headers)[0],202)
            worker.assert_called_once()
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled densification failure')):
            server.run_densification(folder)
        self.assertFalse(server.GPU.locked());self.assertFalse(server.NORMALS.locked())
        self.assertEqual(self.request(endpoint)[1]['status'],'error')
        self.assertEqual(original,(folder/'job.json').read_bytes())

    def test_gaussian_mask_route_requires_token_stage_and_serializes_workers(self):
        headers={'X-Local-Token':server.TOKEN}
        _,job=self.request('/api/upload',self.image((512,256)),headers)
        folder=server.job_path(job['id'])
        endpoint='/api/jobs/'+job['id']+'/gaussian-mask/4'
        self.assertEqual(self.request(endpoint,b'')[0],403)
        self.assertEqual(self.request(endpoint,b'',headers)[0],409)
        state=server.read_job(folder)
        state['stages']['4']={'kind':'gaussians','preview':'initialized.ply','count':10}
        server.write_job(folder,state)
        self.assertTrue(server.GPU.acquire(blocking=False))
        try:
            self.assertEqual(self.request(endpoint,b'',headers)[0],409)
        finally:
            server.GPU.release()
        with patch('server.run_stage') as worker:
            code,response=self.request(endpoint,b'',headers)
            self.assertEqual(code,202)
            self.assertTrue(response['busy'])
            self.assertEqual(response['running_stage'],4)
            self.assertEqual(response['stages']['4']['count'],10)
            worker.assert_called_once_with(folder,4)
        with patch('server.subprocess.Popen',side_effect=RuntimeError('controlled mask failure')):
            server.run_stage(folder,4)
        self.assertFalse(server.GPU.locked())
        self.assertFalse(server.NORMALS.locked())
        self.assertEqual(server.read_job(folder)['stages']['4']['count'],10)


class BaseTests(unittest.TestCase):
    def test_exact_initializer_depth_not_zero_delta_composer(self):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/ml-sharp/src'))
        import torch
        from sharp.models.initializer import MultiLayerInitializer
        from worker import base_scene
        init=MultiLayerInitializer(num_layers=2,stride=2,base_depth=10,scale_factor=1,
                                  disparity_factor=1,color_option='all_layers',
                                  first_layer_depth_option='surface_min',rest_layer_depth_option='surface_max')
        depth=torch.full((1,2,8,8),3.0)
        image=torch.full((1,3,8,8),.5)
        result=base_scene(init(image,depth))
        self.assertEqual(result['means'].shape,(32,3))
        np.testing.assert_allclose(result['means'][:,2],3,rtol=1e-6)
        np.testing.assert_allclose(result['alpha'],.5)
        np.testing.assert_allclose(np.linalg.norm(result['quats'],axis=1),1)
        self.assertTrue(np.all(result['scales']>0))


if __name__=='__main__':
    unittest.main()
