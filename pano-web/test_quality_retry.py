import unittest
from unittest.mock import patch
from test_novel_pipeline import NovelPipelineTests
import novel_pipeline as n
import quality_retry as q


class RetryTests(unittest.TestCase):
    def setUp(self):
        fixture=NovelPipelineTests('runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.job,self.folder,self.source=fixture.job,fixture.folder,fixture.source
        manifest=n.read(self.folder/'pipeline.json');manifest['selection_policy']='max-mask';n.write_json(self.folder/'pipeline.json',manifest)
        selection=n.read(self.folder/'selection.json')
        selection['previews']={f'C{i:03d}':dict(capture_version=n.CAPTURE_VERSION,source=self.source,path=f'fake-{i}') for i in (1,2,3)}
        n.write_json(self.folder/'selection.json',selection)
        n.begin_repair(self.job);q.prepare_run(self.job,self.job,7)

    def reject_and_resume(self,stage=7):
        state=q.reject_attempt(self.job,self.job,stage,dict(quality_no_contribution=True))
        if state['quality_run']['status']=='awaiting_capture':
            state,started=q.resume_run(self.job,state['quality_run']['pending_id']);self.assertTrue(started)
        return state

    def test_three_distinct_cameras_then_exhausted(self):
        self.reject_and_resume();self.reject_and_resume()
        state=self.reject_and_resume();run=state['quality_run']
        self.assertEqual(run['status'],'exhausted');self.assertEqual(len(run['attempts']),3)
        self.assertEqual([a['camera_id'] for a in run['attempts']],['C002','C003','C001'])
        self.assertEqual(n.read(self.folder/'selection.json')['selected_ids'],['C000'])
        with self.assertRaises(ValueError):q.prepare_run(self.job,self.job,7)

    def test_resume_is_idempotent_and_rejects_stale_ids(self):
        state=q.reject_attempt(self.job,self.job,7,{})
        attempt=state['quality_run']['pending_id']
        with self.assertRaises(ValueError):q.resume_run(self.job,'wrong')
        self.assertTrue(q.resume_run(self.job,attempt)[1]);self.assertFalse(q.resume_run(self.job,attempt)[1])
        self.assertEqual(len(n.read(self.job/'job.json')['quality_run']['attempts']),2)

    def test_partial_success_does_not_retry_and_target_is_bounded(self):
        result=dict(quality_hash='test',effective_mask_pixels=1)
        self.assertFalse(q.no_contribution(result,7))
        self.assertIsNone(q.publish_attempt(self.job,self.job,7,result))
        q.prepare_run(self.job,self.job,8)
        self.reject_and_resume(8)
        self.assertEqual(q.publish_attempt(self.job,self.job,7,result),8)
        self.assertIsNone(q.publish_attempt(self.job,self.job,8,dict(accepted_bundle=dict(effective_pixels=1))))

    def test_execution_error_keeps_camera_and_attempt_count(self):
        q.fail_execution(self.job);state=n.read(self.job/'job.json')
        self.assertEqual(state['quality_run']['status'],'error');self.assertEqual(len(state['quality_run']['attempts']),1)
        self.assertEqual(n.read(self.folder/'selection.json')['failed_ids'] if 'failed_ids' in n.read(self.folder/'selection.json') else [],[])

    def test_trusted_only_after_accepted_patch(self):
        q.prepare_run(self.job,self.job,9)
        q.publish_attempt(self.job,self.job,9,dict(acceptance=dict(status='accepted')))
        self.assertEqual(n.read(self.folder/'selection.json')['selected_ids'],['C000','C002'])

    def test_server_resumes_only_to_authorized_depth_stage(self):
        import server
        q.prepare_run(self.job,self.job,8)
        calls=[]
        def worker(parent,execution,stage,local):
            calls.append(local)
            if local==7:return dict(quality_hash='fixture',effective_mask_pixels=42)
            return dict(accepted_bundle=dict(effective_pixels=0 if len(calls)==2 else 42))
        with patch('server.execute_stage',side_effect=worker):
            server.GPU.acquire();server.run_stage(self.job,7)
            state=n.read(self.job/'job.json');self.assertEqual(state['quality_run']['status'],'awaiting_capture')
            self.assertNotIn('7',state['stages']);self.assertFalse(server.GPU.locked())
            q.resume_run(self.job,state['quality_run']['pending_id'])
            server.GPU.acquire();server.run_stage(self.job,7)
        state=n.read(self.job/'job.json')
        self.assertEqual(calls,[7,8,7,8]);self.assertEqual(state['quality_run']['status'],'ready')
        self.assertNotIn('9',state['stages']);self.assertEqual(len(state['quality_run']['attempts']),2)
        self.assertFalse(server.GPU.locked())


if __name__=='__main__':unittest.main()
