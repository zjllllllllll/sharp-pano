import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from iteration_stop import decide,remaining_rows,evaluate
from novel_iteration import ensure_workspace
from novel_pipeline import read,write_json,digest


class StopTests(unittest.TestCase):
    def test_thresholds_use_unrounded_values_and_population_std(self):
        self.assertEqual(decide([5,0],2)['reason'],'strict_max')
        self.assertEqual(decide([7,5],2)['reason'],'relaxed_max_and_std')
        self.assertEqual(decide([7,4.9999],2)['reason'],'continue')
        self.assertEqual(decide([7,4],2)['reason'],'continue')
        self.assertEqual(decide([7.00001,7],2)['reason'],'continue')
        result=decide([6,6],2)
        self.assertEqual(result['statistics']['std_percentage_points'],0)
        self.assertEqual(result['reason'],'relaxed_max_and_std')

    def test_cap_counts_completed_repairs_and_exhaustion_is_separate(self):
        self.assertFalse(decide([10],4)['stop'])
        self.assertEqual(decide([10],5)['reason'],'max_rounds')
        self.assertEqual(decide([0],5)['reason'],'max_rounds')
        self.assertEqual(decide([],2)['reason'],'no_remaining_candidates')
        self.assertIsNone(decide([],2)['statistics']['max_percent'])

    def test_excludes_processed_and_uses_pixel_fraction(self):
        cameras=[dict(id='C000',mask_stats=dict(mask_fraction=0,raw_mask_fraction=0)),
                 dict(id='C001',mask_stats=dict(mask_fraction=.06,raw_mask_fraction=.08,spherical_fraction=.99)),
                 dict(id='C002',mask_stats=dict(mask_fraction=.04,raw_mask_fraction=.05))]
        rows=remaining_rows(cameras,['C000','C002'])
        self.assertEqual(rows,[dict(camera='C001',pixel_percent=6,raw_pixel_percent=8)])

    def test_workspace_never_creates_sixth_round(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError,'5'):
                ensure_workspace(Path(root),6)

    def test_round_three_base_is_round_two_and_root_history_is_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            parent=Path(root)/('a'*32);second=Path(root)/('b'*32)
            parent.mkdir();second.mkdir();source=second/'novel/r2/merged.ply';source.parent.mkdir(parents=True);source.write_bytes(b'round two merged')
            result=dict(execution_job=second.name,revision='r2',download='merged.ply',sha256=digest(source),
                count=123,patch_revision='patch2',acceptance=dict(status='accepted'))
            stages={str(i):dict(old=i) for i in range(1,14)};stages['13']=result
            write_json(parent/'job.json',dict(stages=stages,iteration_jobs={'2':second.name}))
            child=ensure_workspace(parent,3);state=read(child/'job.json')
            self.assertEqual(state['iteration_index'],3)
            self.assertEqual(state['iteration_source']['sha256'],result['sha256'])
            self.assertEqual((child/'iteration-base.ply').read_bytes(),b'round two merged')
            self.assertEqual(read(parent/'job.json')['stages'],stages)
            self.assertEqual(ensure_workspace(parent,3),child)
            main=read(parent/'job.json');main['iteration_evaluation']=dict(after_round=2,source_sha256=result['sha256'],stop=True);write_json(parent/'job.json',main)
            with self.assertRaisesRegex(ValueError,'停止'):ensure_workspace(parent,3)

    def test_server_stops_before_creating_or_rendering_next_round(self):
        import server
        with tempfile.TemporaryDirectory() as root:
            parent=Path(root);write_json(parent/'job.json',dict(stages={},busy=True))
            with patch('server.evaluate_iteration',return_value=dict(stop=True)),patch('server.novel_iteration.ensure_workspace') as workspace,patch('server.execute_stage') as worker:
                server.GPU.acquire();server.run_stage(parent,14)
                workspace.assert_not_called();worker.assert_not_called()
            self.assertFalse(server.GPU.locked());self.assertFalse(read(parent/'job.json')['busy'])

    def test_evaluation_keeps_all_previous_observations_and_pixel_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            parent=Path(temp)/('a'*32);original=parent/'novel/base';original.mkdir(parents=True)
            raw=original/'raw';raw.mkdir()
            cameras=[]
            for i in range(5):
                cid=f'C{i:03d}';Image.fromarray(np.full((2,4),255,np.uint8)).save(raw/f'{cid}.png')
                cameras.append(dict(id=cid,position_world_m=[i,0,0],views=dict(mask=f'{cid}.png'),mask_stats={}))
            write_json(original/'sample-cameras.json',dict(cameras=cameras,all_cameras=cameras))
            write_json(original/'pipeline.json',dict(mask_source=str(raw)))
            stages={'6':dict(revision='base')}
            for i in range(1,4):
                job=parent if i==1 else Path(temp)/(str(i)*32);folder=job/'novel/base';folder.mkdir(parents=True,exist_ok=True)
                source=folder/'merged.ply';source.write_bytes(f'round{i}'.encode())
                write_json(folder/'selection.json',dict(selected_ids=['C000',f'C{i:03d}']))
                result=dict(camera_id=f'C{i:03d}',execution_job=job.name,revision='base',download='merged.ply',sha256=digest(source),
                    patch_revision=str(i),patch_sha256=str(i),acceptance=dict(status='accepted'))
                stages[str(5+4*i)]=result
                if i>1:write_json(job/'job.json',dict(stages={'8':dict(files={'panorama-completed.npy':str(i)})}))
            stages['8']=dict(files={'panorama-completed.npy':'1'});write_json(parent/'job.json',dict(stages=stages))
            observed=[]
            def observation(job,result,initial,out):
                observed.append(result['camera_id']);return object(),dict(camera_id=result['camera_id'],position_world_m=[len(observed),0,0])
            mask=np.zeros((2,4),bool);mask[0,0]=True
            with patch('multi_view_volume.RGBDVolume.from_job',return_value=object()),patch('novel_iteration.accepted_observation',side_effect=observation),patch('multi_view_volume.AccumulatedVolume') as accumulated,patch('iteration_stop.clean_mask',side_effect=lambda value,**kw:value):
                accumulated.return_value.mask.return_value=mask
                report=evaluate(parent,3)
            self.assertEqual(observed,['C001','C002','C003'])
            self.assertEqual(len(accumulated.call_args.args[0]),4)
            self.assertEqual(report['statistics']['count'],1)
            self.assertEqual(report['rows'][0]['camera'],'C004')
            self.assertEqual(report['rows'][0]['pixel_percent'],12.5)
            self.assertEqual(report['reason'],'continue')
            saved=np.asarray(Image.open(parent/report['path']/'geometry-masks/C004.png'))>0
            np.testing.assert_array_equal(saved,mask)


if __name__=='__main__':unittest.main()
