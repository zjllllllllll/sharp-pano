"""Bounded, persistent camera attempts. Call mutations under the server LOCK."""
from copy import deepcopy
import uuid
from face_quality_io import read,write


class QualityNoContribution(ValueError):
    """Only semantic no-contribution failures may advance to another camera."""


def bounds(stage):
    start=6+4*((stage-6)//4)
    return start,6+(stage-start)


def prepare_run(parent,execution,stage):
    from novel_pipeline import work_folder,public_state
    state=read(parent/'job.json');folder=work_folder(execution);view=public_state(execution,folder)
    start,_=bounds(stage);run=state.get('quality_run')
    if run and run['round_start']==start and run['source_sha256']==view['source']['sha256']:
        if run['status'] in ('awaiting_capture','exhausted','cancelled'):
            raise ValueError('质量尝试等待采集或已结束，请使用对应的恢复操作')
        run['target_stage']=max(stage,run['target_stage']);run['next_stage']=stage;run['status']='running'
    else:
        camera=view['highlighted']
        if not camera:raise ValueError('没有剩余候选相机')
        run=dict(id=uuid.uuid4().hex,round_start=start,target_stage=stage,next_stage=stage,
            execution_job=execution.name,source_sha256=view['source']['sha256'],status='running',max_attempts=3,
            attempts=[dict(id=uuid.uuid4().hex,camera_id=camera,status='running',results={})])
    state['quality_run']=run;write(parent/'job.json',state);return run


def no_contribution(result,local_stage):
    if result.get('quality_no_contribution'):return True
    if local_stage==7 and result.get('quality_hash'):return result.get('effective_mask_pixels',0)==0
    if local_stage==8 and result.get('accepted_bundle'):return result['accepted_bundle']['effective_pixels']==0
    return False


def reject_attempt(parent,execution,stage,result):
    from novel_pipeline import work_folder,public_state
    state=read(parent/'job.json');run=state['quality_run'];attempt=run['attempts'][-1]
    start,local=bounds(stage);attempt['results'][str(local)]=deepcopy(result)
    attempt.update(status='rejected',failed_stage=stage,reason=result.get('reason','没有通过检查的新增待补区域'))
    folder=work_folder(execution);selection=read(folder/'selection.json')
    failed=selection.setdefault('failed_ids',[])
    if attempt['camera_id'] not in failed:failed.append(attempt['camera_id'])
    selection['current']=None;write(folder/'selection.json',selection)
    # Archive metadata in the attempt before removing active publications. Files stay intact.
    for n in range(start+1,start+4):
        previous=state['stages'].pop(str(n),None)
        if previous:attempt['results'].setdefault(str(6+n-start),deepcopy(previous))
    if execution!=parent:
        child=read(execution/'job.json')
        for n in (7,8,9):child['stages'].pop(str(n),None)
        child.update(busy=False,running_stage=None,error=None);write(execution/'job.json',child)
    recommended=public_state(execution,folder)['recommended']
    if len(run['attempts'])>=3 or not recommended:
        run.update(status='exhausted',pending_camera=None)
        message='本轮未找到可用修补：三台相机均失败或没有剩余候选'
    else:
        run.update(status='awaiting_capture',pending_camera=recommended,next_stage=start+1,pending_id=uuid.uuid4().hex)
        message=f'质量检查未通过，等待采集下一台相机 {recommended}（{len(run["attempts"])+1}/3）'
    state.update(quality_run=run,busy=False,running_stage=None,error=None,message=message);write(parent/'job.json',state)
    return state


def resume_run(parent,attempt_id):
    from novel_pipeline import work_folder,begin_repair
    state=read(parent/'job.json');run=state.get('quality_run')
    if not run or run.get('pending_id')!=attempt_id:raise ValueError('过期的相机尝试编号')
    if run['status']!='awaiting_capture':return state,False
    if len(run['attempts'])>=3:raise ValueError('本轮最多三台相机')
    execution=parent.parent/run['execution_job'];folder=work_folder(execution)
    if read(folder/'pipeline.json')['source']['sha256']!=run['source_sha256']:raise ValueError('基础场景已经改变')
    selection=read(folder/'selection.json');preview=selection['previews'].get(run['pending_camera'])
    if not preview or preview['source']['sha256']!=run['source_sha256']:raise ValueError('请先完成当前候选的六面采集')
    current=begin_repair(execution)
    if current['camera_id']!=run['pending_camera']:raise ValueError('采集相机与待重试相机不一致')
    run['attempts'].append(dict(id=attempt_id,camera_id=run['pending_camera'],status='running',results={}))
    run['status']='running'
    state.update(quality_run=run,busy=True,running_stage=run['next_stage'],error=None,message='继续质量检查尝试')
    write(parent/'job.json',state);return state,True


def publish_attempt(parent,execution,stage,result):
    """Called after normal stage publication. Returns the next authorized stage."""
    from novel_pipeline import work_folder
    state=read(parent/'job.json');run=state.get('quality_run')
    if not run:return None
    start,local=bounds(stage)
    if start!=run['round_start']:return None
    attempt=run['attempts'][-1];attempt['results'][str(local)]=deepcopy(result)
    attempt['status']='patch_generated' if local==9 else 'rgb_passed' if local==7 else 'rgbd_passed'
    if local==9:
        if result['acceptance']['status']=='accepted':
            attempt['status']='trusted';folder=work_folder(execution);selection=read(folder/'selection.json')
            for key in ('selected_ids','completed_ids'):
                if attempt['camera_id'] not in selection[key]:selection[key].append(attempt['camera_id'])
            write(folder/'selection.json',selection)
        run['status']='complete';next_stage=None
    else:
        next_stage=stage+1 if stage<run['target_stage'] else None
        run['status']='running' if next_stage else 'ready';run['next_stage']=stage+1
    state.update(quality_run=run,busy=next_stage is not None,running_stage=next_stage)
    if next_stage:state['message']='相机检查通过，继续已请求的下一阶段'
    write(parent/'job.json',state);return next_stage


def fail_execution(parent):
    state=read(parent/'job.json');run=state.get('quality_run')
    if run and run['status']=='running':run['status']='error';write(parent/'job.json',state)
