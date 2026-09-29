"""Local-only, dependency-light web API; GPU workers run in existing environments."""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit, unquote
import argparse
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from PIL import Image, ImageOps, UnidentifiedImageError
from point_normals import cached_result
from point_distances import cached_result as cached_distances
from point_mask import cached_result as cached_mask
from point_densify import cached_result as cached_densification
from gaussian_densify import cached_result as cached_gaussian_dense
import camera_lab
import novel_pipeline
import novel_depth
import novel_iteration
from iteration_stop import MAX_ROUNDS, POLICY as ITERATION_POLICY
from image_provider import selected_provider, status as provider_status

APP = Path(__file__).resolve().parent
ROOT = APP.parent
DATA = APP / 'runs'
from project_paths import PYTHON, MODELS
PYTHONS = {stage: PYTHON for stage in range(2,7)}
TOKEN = secrets.token_urlsafe(32)
LOCK = threading.RLock()
GPU = threading.Lock()
NORMALS = threading.Lock()
NORMAL_STATES = {}
DISTANCE_STATES = {}
DENSIFY_STATES = {}
GAUSSIAN_DENSE_STATES = {}
JOB_ID = re.compile(r'^[a-f0-9]{32}$')
ARTIFACTS = {'input.png', 'depth.png', 'depth-gray.png', 'depth.npy',
             'iteration-base.ply',
             'cloud-preview.bin', 'cloud-full.bin', 'cloud.las', 'initialized.ply', 'adjusted.ply',
             'cloud-angles-full.bin', 'cloud-angles-preview.bin',
             'cloud-distances-full.bin', 'cloud-distances-preview.bin',
             'initialized-masked.ply', 'adjusted-masked.ply',
             'cloud-dense-added.bin', 'cloud-dense-added-preview.bin'}
ARTIFACTS.update(f'{stem}-{suffix}.ply' for stem in ('initialized', 'adjusted')
                 for suffix in ('dense','compare-baseline','compare-masked', *(f'view-{k}' for k in ('common','removed','parents','refined','children'))))
Image.MAX_IMAGE_PIXELS = 40_000_000
MAX_UPLOAD = 80 * 1024 * 1024


def job_path(jid):
    if not JOB_ID.fullmatch(jid):
        raise ValueError('无效任务编号')
    path = DATA / jid
    if not (path / 'job.json').is_file():
        raise FileNotFoundError('任务不存在')
    return path


def read_job(path):
    with LOCK:
        return json.loads((path / 'job.json').read_text(encoding='utf-8'))


def write_job(path, data):
    with LOCK:
        temp = path / 'job.tmp'
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(path / 'job.json')


def execute_stage(path, execution, stage, local_stage):
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'}
    worker=APP/('novel_worker.py' if stage>=6 else 'worker.py')
    reference=local_stage in (8,9)
    executable=novel_depth.LAUNCHER if reference else novel_pipeline.LAUNCHER if local_stage==7 else PYTHONS[local_stage]
    cwd=novel_depth.REFERENCE if reference else novel_pipeline.REFERENCE if local_stage==7 else ROOT
    with (path/f'stage-{stage}.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([str(executable),str(worker),str(local_stage),str(execution)],cwd=cwd,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        tail=[]
        for line in process.stdout:
            log.write(line);log.flush();tail=(tail+[line.strip()])[-12:]
            with LOCK:
                state=read_job(path);state['message']=line.strip()[:400];write_job(path,state)
        code=process.wait()
    if code:raise RuntimeError('\n'.join(tail)[-2500:])
    return json.loads((execution/f'stage-{local_stage}.json').read_text(encoding='utf-8'))


def evaluate_iteration(path,stage):
    """Run the post-merge decision before any next-round capture or paid call."""
    with LOCK:
        state=read_job(path);state.update(busy=True,running_stage=stage,error=None,message='正在评估剩余候选几何 Mask 像素占比…')
        write_job(path,state)
    env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1'}
    with (path/'iteration-evaluation.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen([str(PYTHON),str(APP/'iteration_stop.py'),str(path)],cwd=ROOT,env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        tail=[]
        for line in process.stdout:
            log.write(line);log.flush();tail=(tail+[line.strip()])[-12:]
            with LOCK:
                state=read_job(path);state['message']=line.strip()[:400];write_job(path,state)
        if process.wait():raise RuntimeError('停止评估失败：'+'\n'.join(tail)[-2000:])
    report=json.loads((path/'iteration-evaluation.json').read_text(encoding='utf-8'))
    with LOCK:
        state=read_job(path);state['iteration_policy']=ITERATION_POLICY;state['iteration_evaluation']=report
        state['iteration_status']='stopped' if report['stop'] else 'ready'
        reasons=dict(max_rounds='达到最多 5 轮',no_remaining_candidates='没有剩余候选相机',
            strict_max='最大几何 Mask 像素占比 ≤5%',relaxed_max_and_std='最大几何 Mask 像素占比 ≤7%，且标准差 ≤1 个百分点',
            continue_='尚未满足停止条件')
        reason=reasons.get(report['reason'],reasons['continue_']);stats=report['statistics']
        state['message']=f'第 {report["after_round"]} 轮完成；'+('迭代已停止：' if report['stop'] else '')+reason
        if stats['count']:
            state['message']+=f'。剩余 {stats["count"]} 个视角，最大值 {stats["max_percent"]:.4f}%，标准差 {stats["std_percentage_points"]:.4f} 个百分点。'
        write_job(path,state)
    return report


def run_stage(path, stage):
    import quality_retry
    mask_lock=False;execution=path
    try:
        round_index=(stage-6)//4+1 if stage>=6 else 0
        if stage>=10:
            if (stage-6)%4==0:
                report=evaluate_iteration(path,stage)
                if report['stop']:
                    with LOCK:
                        state=read_job(path);state.update(busy=False,running_stage=None);write_job(path,state)
                    return
            with LOCK:execution=novel_iteration.ensure_workspace(path,round_index)
        if stage in (4,5):NORMALS.acquire();mask_lock=True
        while True:
            local_stage=6+(stage-6)%4 if stage>=6 else stage
            if execution!=path:
                with LOCK:
                    child=read_job(execution);child.update(busy=True,running_stage=local_stage,error=None,message='正在运行本轮')
                    write_job(execution,child)
            result=execute_stage(path,execution,stage,local_stage)
            with LOCK:
                if local_stage in (7,8,9) and quality_retry.no_contribution(result,local_stage):
                    quality_retry.reject_attempt(path,execution,stage,result)
                    return
                if execution!=path:
                    child=read_job(execution);child['stages'][str(local_stage)]=result
                    child.update(busy=False,running_stage=None,error=None,message='本步骤已完成');write_job(execution,child)
                    result={**result,'execution_job':execution.name,'iteration_index':round_index}
                    (path/f'stage-{stage}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                state=read_job(path);state['stages'][str(stage)]=result
                if stage==7:state.pop('mask_update_pending',None)
                state.update(busy=False,running_stage=None,error=None,message='本步骤已完成');write_job(path,state)
                next_stage=quality_retry.publish_attempt(path,execution,stage,result) if local_stage in (7,8,9) else None
                assess=local_stage==9 and result.get('acceptance',{}).get('status')=='accepted'
                if assess:
                    state=read_job(path);state.update(busy=True,running_stage=stage,message='正在评估迭代停止条件…');write_job(path,state)
            if assess:
                evaluate_iteration(path,stage)
                with LOCK:
                    state=read_job(path);state.update(busy=False,running_stage=None);write_job(path,state)
            if next_stage is None:break
            stage=next_stage
    except Exception as exc:
        with LOCK:
            if execution!=path:
                child=read_job(execution);child.update(busy=False,running_stage=None,error=str(exc),message='本轮生成失败');write_job(execution,child)
            state=read_job(path);state.update(busy=False,running_stage=None,error=str(exc),message='执行失败；未自动更换相机或重发付费请求')
            write_job(path,state);quality_retry.fail_execution(path)
    finally:
        if mask_lock:NORMALS.release()
        GPU.release()


def normals_state(path):
    with LOCK:
        current = NORMAL_STATES.get(path.name)
        if current and current['status'] == 'running':
            return dict(current)
        result = cached_result(path)
        if result:
            return {'status': 'ready', 'result': result}
        return dict(current) if current and current['status'] == 'error' else {'status': 'missing'}


def run_normals(path):
    try:
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        with (path / 'point-normals.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(PYTHONS[3]), str(APP/'point_normals.py'), str(path)],
                                       cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                       errors='replace',
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            tail = []
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail = (tail + [line.strip()])[-12:]
                with LOCK:
                    NORMAL_STATES[path.name] = {'status': 'running', 'message': line.strip()[:400]}
            if process.wait():
                raise RuntimeError('\n'.join(tail)[-2000:])
        if not cached_result(path):
            raise RuntimeError('法向文件校验失败')
        with LOCK:
            NORMAL_STATES[path.name] = {'status': 'ready'}
    except Exception as exc:
        with LOCK:
            NORMAL_STATES[path.name] = {'status': 'error', 'message': str(exc)}
    finally:
        NORMALS.release()


def distances_state(path):
    with LOCK:
        current = DISTANCE_STATES.get(path.name)
        if current and current['status'] == 'running':
            return dict(current)
        result = cached_distances(path)
        if result:
            return {'status': 'ready', 'result': result}
        return dict(current) if current and current['status'] == 'error' else {'status': 'missing'}


def run_distances(path):
    try:
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        with (path/'point-distances.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(PYTHONS[3]), str(APP/'point_distances.py'), str(path)],
                                       cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                       errors='replace',
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            tail = []
            for line in process.stdout:
                log.write(line)
                log.flush()
                tail = (tail + [line.strip()])[-12:]
                with LOCK:
                    DISTANCE_STATES[path.name] = {'status': 'running', 'message': line.strip()[:400]}
            if process.wait():
                raise RuntimeError('\n'.join(tail)[-2000:])
        if not cached_distances(path):
            raise RuntimeError('距离文件校验失败')
        with LOCK:
            DISTANCE_STATES[path.name] = {'status': 'ready'}
    except Exception as exc:
        with LOCK:
            DISTANCE_STATES[path.name] = {'status': 'error', 'message': str(exc)}
    finally:
        NORMALS.release()


def densification_state(path):
    with LOCK:
        current = DENSIFY_STATES.get(path.name)
        if current and current['status'] == 'running':
            return dict(current)
        result = cached_densification(path)
        if result:
            return {'status': 'ready', 'result': result}
        return dict(current) if current and current['status'] == 'error' else {'status': 'missing'}


def run_densification(path):
    try:
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        with (path/'point-densify.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(PYTHONS[3]), str(APP/'point_densify.py'), str(path)],
                                       cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                       errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            tail = []
            for line in process.stdout:
                log.write(line); log.flush(); tail = (tail+[line.strip()])[-12:]
                with LOCK:
                    DENSIFY_STATES[path.name] = {'status': 'running', 'message': line.strip()[:400]}
            if process.wait():
                raise RuntimeError('\n'.join(tail)[-2200:])
        if not cached_densification(path):
            raise RuntimeError('点云加密预览校验失败')
        with LOCK:
            DENSIFY_STATES[path.name] = {'status': 'ready'}
    except Exception as exc:
        with LOCK:
            DENSIFY_STATES[path.name] = {'status': 'error', 'message': str(exc)}
    finally:
        NORMALS.release()
        GPU.release()


def gaussian_dense_state(path, stage):
    key = (path.name, stage)
    current = GAUSSIAN_DENSE_STATES.get(key, {})
    if current.get('status') == 'running':
        return dict(current)
    result = cached_gaussian_dense(path, stage)
    if result:
        return {'status': 'ready', 'result': result}
    return dict(current) if current.get('status') == 'error' else {'status': 'missing'}


def run_gaussian_dense(path, stage):
    key = (path.name, stage)
    try:
        env = {**os.environ, 'PYTHONUTF8':'1', 'PYTHONUNBUFFERED':'1', 'HF_HUB_OFFLINE':'1'}
        with (path/f'gaussian-dense-{stage}.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen([str(PYTHONS[stage]), str(APP/'gaussian_densify.py'), str(path), str(stage)],
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding='utf-8', errors='replace', creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            tail = []
            for line in process.stdout:
                log.write(line); log.flush(); tail = (tail+[line.strip()])[-12:]
                with LOCK:
                    GAUSSIAN_DENSE_STATES[key] = dict(status='running', message=line.strip()[:400])
            if process.wait():
                raise RuntimeError('\n'.join(tail)[-2500:])
        info = cached_gaussian_dense(path, stage)
        if not info:
            raise RuntimeError('加密高斯缓存校验失败')
        with LOCK:
            state = read_job(path)
            state['stages'][str(stage)]['densification'] = info
            write_job(path, state)
            GAUSSIAN_DENSE_STATES[key] = dict(status='ready')
    except Exception as exc:
        with LOCK:
            GAUSSIAN_DENSE_STATES[key] = dict(status='error', message=str(exc))
    finally:
        NORMALS.release(); GPU.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if self.command != 'GET' or self.path not in ['/api/state']:
            super().log_message(fmt, *args)

    def local_request(self):
        host = self.headers.get('Host', '').lower().split(':')[0]
        return host in ('localhost', '127.0.0.1')

    def json(self, value, code=200):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def file(self, path, download=False):
        if not path.is_file():
            raise FileNotFoundError('文件尚未生成')
        mime = {'.js':'text/javascript', '.mjs':'text/javascript', '.wasm':'application/wasm',
                '.bin':'application/octet-stream', '.ply':'application/octet-stream'}.get(path.suffix)
        mime = mime or mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(path.stat().st_size))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-cache')
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{path.name}"')
        self.end_headers()
        with path.open('rb') as stream:
            shutil.copyfileobj(stream, self.wfile, 1024*1024)

    def do_GET(self):
        try:
            if not self.local_request():
                return self.json({'error':'只允许本机访问'}, 403)
            url = urlsplit(self.path)
            path = unquote(url.path)
            if path == '/api/camera-lab':
                return self.json(camera_lab.catalog())
            if path.startswith('/camera-lab-artifacts/'):
                return self.file(camera_lab.artifact(path.removeprefix('/camera-lab-artifacts/')))
            if path in ('/camera-lab', '/camera-lab/'):
                return self.file(APP/'static/camera-lab.html')
            if path in ('/camera-lab.js','/camera-lab.css','/camera-lab-filter.mjs'):
                return self.file(APP/'static'/path[1:])
            if path == '/api/state':
                jobs = sorted((p for p in DATA.glob('*/job.json') if not read_job(p.parent).get('iteration_parent')), key=lambda p:p.stat().st_mtime, reverse=True)
                health = {
                    'depthmaster': PYTHONS[2].is_file() and (MODELS/'depthmaster.pt').is_file(),
                    'sharp': PYTHONS[3].is_file() and (MODELS/'sharp_2572gikvuh.pt').is_file(),
                    'lingbot': PYTHON.is_file() and (MODELS/'lingbot-depth.pt').is_file(),
                    'repair_launcher': novel_depth.LAUNCHER.is_file(),
                    'image_provider': provider_status()}
                return self.json({'token':TOKEN, 'job':read_job(jobs[0].parent) if jobs else None,
                                  'health':health, 'gpu_busy':GPU.locked()})
            if path.startswith('/api/jobs/'):
                match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})/face-quality',path)
                if match:
                    from quality_report_api import public_report
                    return self.json(public_report(job_path(match[1])))
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/novel', path)
                if match:
                    with LOCK:return self.json(novel_pipeline.public_state(job_path(match[1])))
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/gaussian-densify/([45])', path)
                if match:
                    return self.json(gaussian_dense_state(job_path(match[1]), int(match[2])))
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/point-densify', path)
                if match:
                    return self.json(densification_state(job_path(match[1])))
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/point-distances', path)
                if match:
                    return self.json(distances_state(job_path(match[1])))
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/point-normals', path)
                if match:
                    return self.json(normals_state(job_path(match[1])))
                match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/debug/([1-9]|1[0-9]|2[0-5])', path)
                if match:
                    job = job_path(match[1])
                    log = job/f'stage-{match[2]}.log'
                    density_log = job/'point-densify.log'
                    return self.json({'log':log.read_text(encoding='utf-8')[-20000:] if log.exists() else '此步骤没有模型运行日志。',
                                      'densification':densification_state(job) if match[2] == '3' else None,
                                      'densification_log':density_log.read_text(encoding='utf-8')[-12000:] if match[2] == '3' and density_log.exists() else '',
                                      'result':read_job(job)['stages'].get(match[2]),
                                      'gaussian_densification':gaussian_dense_state(job,int(match[2])) if match[2] in ('4','5') else None,
                                      'pipeline':{'depth_model':'DepthMaster', 'sharp':'original FP32',
                                      'injection':'panorama_initializer', 'faces':6, 'fov_degrees':100, 'layers':2}})
                return self.json(read_job(job_path(path.removeprefix('/api/jobs/'))))
            match = re.fullmatch(r'/novel-artifacts/([a-f0-9]{32})/(.+)',path)
            if match:
                return self.file(novel_pipeline.artifact(job_path(match[1]),match[2]),url.query=='download=1')
            match=re.fullmatch(r'/quality-artifacts/([a-f0-9]{32})/(.+)',path)
            if match:
                from quality_report_api import artifact
                return self.file(artifact(job_path(match[1]),match[2]))
            if path.startswith('/artifacts/'):
                _, _, jid, name = path.split('/')
                if name not in ARTIFACTS:
                    raise FileNotFoundError('不可访问此文件')
                return self.file(job_path(jid)/name, url.query == 'download=1')
            if path == '/':
                return self.file(APP/'static/index.html')
            if path in ('/app.js','/workflow-mode.mjs','/mode.css','/point-filter.mjs','/view-cache.mjs','/novel-view.js','/novel.css','/style.css','/debug.css','/favicon.svg'):
                return self.file(APP/'static'/path[1:])
            if path.startswith('/vendor/'):
                relative = path.removeprefix('/vendor/')
                vendor = (APP/'node_modules').resolve()
                file = (vendor/relative).resolve()
                if not file.is_relative_to(vendor) or file.suffix not in ('.js','.wasm','.mjs'):
                    raise FileNotFoundError('非法资源路径')
                return self.file(file)
            raise FileNotFoundError('页面不存在')
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except (ValueError, FileNotFoundError) as exc:
            self.json({'error':str(exc)}, 404)

    def do_POST(self):
        try:
            if not self.local_request() or not secrets.compare_digest(self.headers.get('X-Local-Token',''), TOKEN):
                return self.json({'error':'请刷新本地页面后重试'}, 403)
            path = urlsplit(self.path).path
            match=re.fullmatch(r'/api/jobs/([a-f0-9]{32})/quality/resume',path)
            if match:
                import quality_retry
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=4096:raise ValueError('Invalid resume request')
                options=json.loads(self.rfile.read(length));job=job_path(match[1])
                with LOCK:
                    state=read_job(job);run=state.get('quality_run',{})
                    if run.get('pending_id')!=options.get('attempt_id'):raise ValueError('过期的尝试编号')
                    if run.get('status')!='awaiting_capture':return self.json(state,200)
                    if not GPU.acquire(blocking=False):return self.json({'error':'GPU 忙，请等待'},409)
                    try:
                        state,started=quality_retry.resume_run(job,options['attempt_id'])
                        if not started:GPU.release();return self.json(state)
                        threading.Thread(target=run_stage,args=(job,state['quality_run']['next_stage']),daemon=True).start()
                    except Exception:GPU.release();raise
                return self.json(state,202)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/novel/preview/(C\d{3,6})',path)
            if match:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=48*1024*1024:return self.json({'error':'截图数据过大或为空'},413)
                payload=json.loads(self.rfile.read(length))
                with LOCK:
                    job=job_path(match[1])
                    if read_job(job).get('busy'):return self.json({'error':'请等待当前步骤完成'},409)
                    return self.json(novel_pipeline.capture(job,match[2],payload),201)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/gaussian-densify/([45])', path)
            if match:
                job, stage = job_path(match[1]), int(match[2])
                with LOCK:
                    if str(stage) not in read_job(job)['stages']:
                        return self.json({'error':'请先生成本步骤高斯'},409)
                    current = gaussian_dense_state(job, stage)
                    if current['status'] in ('ready','running'):
                        return self.json(current,200 if current['status']=='ready' else 202)
                    if not GPU.acquire(blocking=False):
                        return self.json({'error':'请等待当前生成步骤完成'},409)
                    if not NORMALS.acquire(blocking=False):
                        GPU.release()
                        return self.json({'error':'请等待点云筛选完成'},409)
                    GAUSSIAN_DENSE_STATES[(job.name,stage)] = dict(status='running',message='准备加密高斯与快速对比缓存…')
                    try:
                        threading.Thread(target=run_gaussian_dense,args=(job,stage),daemon=True).start()
                    except Exception:
                        GAUSSIAN_DENSE_STATES.pop((job.name,stage),None); NORMALS.release(); GPU.release(); raise
                    return self.json(dict(GAUSSIAN_DENSE_STATES[(job.name,stage)]),202)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/point-densify', path)
            if match:
                job = job_path(match[1])
                with LOCK:
                    state = read_job(job)
                    if '3' not in state['stages']:
                        return self.json({'error':'请先生成第三步点云'},409)
                    current = densification_state(job)
                    if current['status'] in ('ready','running'):
                        return self.json(current,200 if current['status'] == 'ready' else 202)
                    if not GPU.acquire(blocking=False):
                        return self.json({'error':'请等待当前生成步骤完成'},409)
                    if not NORMALS.acquire(blocking=False):
                        GPU.release()
                        return self.json({'error':'正在计算法向或距离，完成后可重试加密'},409)
                    DENSIFY_STATES[job.name] = {'status':'running','message':'准备距离自适应点云加密…'}
                    try:
                        threading.Thread(target=run_densification,args=(job,),daemon=True).start()
                    except Exception:
                        DENSIFY_STATES.pop(job.name,None); NORMALS.release(); GPU.release(); raise
                    return self.json(dict(DENSIFY_STATES[job.name]),202)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/gaussian-mask/([45])', path)
            if match:
                job, stage = job_path(match[1]), int(match[2])
                with LOCK:
                    state = read_job(job)
                    result = state['stages'].get(str(stage))
                    if not result:
                        return self.json({'error': '请先生成本步骤的高斯'}, 409)
                    mask = cached_mask(job)
                    if (mask and result.get('mask', {}).get('sha256') == mask['sha256']
                            and all((job/v['preview']).is_file() for v in result.get('variants', {}).values())
                            and set(result.get('variants', {})) == {'masked', 'unmasked'}):
                        return self.json(state)
                    if not GPU.acquire(blocking=False):
                        return self.json({'error': '请等待当前生成步骤完成'}, 409)
                    state.update(busy=True, running_stage=stage, error=None, message='正在补生成 Mask 前后对比…')
                    write_job(job, state)
                threading.Thread(target=run_stage, args=(job, stage), daemon=True).start()
                return self.json(state, 202)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/point-distances', path)
            if match:
                job = job_path(match[1])
                with LOCK:
                    if '3' not in read_job(job)['stages']:
                        return self.json({'error': '请先生成第三步点云'}, 409)
                    current = distances_state(job)
                    if current['status'] in ('ready', 'running'):
                        return self.json(current, 200 if current['status'] == 'ready' else 202)
                    if not NORMALS.acquire(blocking=False):
                        return self.json({'error': '正在计算法向或距离，请稍后重试'}, 409)
                    DISTANCE_STATES[job.name] = {'status': 'running', 'message': '准备在 82° 筛选后计算近邻距离…'}
                    try:
                        threading.Thread(target=run_distances, args=(job,), daemon=True).start()
                    except Exception:
                        DISTANCE_STATES.pop(job.name, None)
                        NORMALS.release()
                        raise
                    return self.json(dict(DISTANCE_STATES[job.name]), 202)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/point-normals', path)
            if match:
                job = job_path(match[1])
                with LOCK:
                    state = read_job(job)
                    if '3' not in state['stages']:
                        return self.json({'error': '请先生成第三步点云'}, 409)
                    normal = normals_state(job)
                    if normal['status'] in ('ready', 'running'):
                        return self.json(normal, 200 if normal['status'] == 'ready' else 202)
                    if not NORMALS.acquire(blocking=False):
                        return self.json({'error': '另一个任务正在计算法向，请稍后重试'}, 409)
                    NORMAL_STATES[job.name] = {'status': 'running', 'message': '准备估计点云法向…'}
                    try:
                        threading.Thread(target=run_normals, args=(job,), daemon=True).start()
                    except Exception:
                        NORMAL_STATES.pop(job.name, None)
                        NORMALS.release()
                        raise
                    return self.json(dict(NORMAL_STATES[job.name]), 202)
            if path == '/api/upload':
                length = int(self.headers.get('Content-Length','0'))
                if length < 1 or length > MAX_UPLOAD:
                    return self.json({'error':'图片不能为空，且不得超过 80 MB'}, 413)
                if GPU.locked():
                    return self.json({'error':'请等待当前生成步骤完成再上传新图'}, 409)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError('上传中断，请重新上传')
                with Image.open(io.BytesIO(raw)) as original:
                    if original.format not in ('PNG', 'JPEG', 'WEBP'):
                        raise ValueError('请上传 PNG、JPG 或 WebP 图片')
                    if original.width * original.height > 40_000_000:
                        raise ValueError('图片不能超过 4000 万像素')
                    rgb = ImageOps.exif_transpose(original).convert('RGB')
                w,h = rgb.size
                if w != h*2 or w < 512:
                    raise ValueError('需要完整 2:1 等距柱状全景图（宽度至少 512 像素）')
                jid = uuid.uuid4().hex
                job = DATA/jid
                job.mkdir(parents=True)
                rgb.save(job/'input.png')
                state = dict(id=jid, created=time.time(), busy=False, running_stage=None, error=None,
                             rgb_repair_model='gpt-image-2', image_provider=selected_provider(),
                             camera_selection_policy='max-mask',
                             execution_mode='auto' if self.headers.get('X-Workflow-Mode')=='auto' else 'debug',
                             message='全景图已上传', stages={'1':dict(kind='image', image='input.png',
                             download='input.png', width=w, height=h, description='原始 2:1 全景图。前六步本地计算；第七步补洞调用所配置的图片服务。')})
                write_job(job, state)
                return self.json(state, 201)
            match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/next', path)
            if match:
                job = job_path(match[1])
                length=int(self.headers.get('Content-Length','0'))
                if not 0<=length<=4096:raise ValueError('Invalid request size')
                options=json.loads(self.rfile.read(length)) if length else {}
                if not isinstance(options,dict):raise ValueError('Expected request object')
                if not GPU.acquire(blocking=False):
                    return self.json({'error':'GPU 正在处理一个步骤'}, 409)
                try:
                    with LOCK:
                        state = read_job(job)
                        if state.get('iteration_parent'):raise ValueError('请从主流程启动迭代步骤')
                        if state.get('busy'):raise ValueError('已有步骤正在运行')
                        if state.get('quality_run',{}).get('status') in ('awaiting_capture','exhausted','cancelled'):
                            raise ValueError('本轮质量尝试等待采集或已结束，请查看质量尝试状态')
                        stage = len(state['stages'])+1
                        # A second automatic tab must not advance a later stage on a stale request.
                        if 'expected_stage' in options and options['expected_stage'] != stage:
                            GPU.release()
                            return self.json(state)
                        if state.get('iteration_status')=='stopped':raise ValueError('迭代已按停止规则结束')
                        if stage > 5+4*MAX_ROUNDS:raise ValueError('已达到最多 5 轮修补')
                        local_stage=6+(stage-6)%4 if stage>=6 else stage
                        round_index=(stage-6)//4+1 if stage>=6 else 0
                        if stage==6:
                            variant=options.get('gaussian_variant','masked')
                            if variant not in ('masked','unmasked','dense'):raise ValueError('Invalid Gaussian variant')
                            state['novel_scene_variant']=variant
                        if local_stage in (7,8,9):
                            import quality_retry
                            execution=novel_iteration.ensure_workspace(job,round_index) if stage>=10 else job
                            # Confirm only after the browser has supplied the exact capture.
                            if local_stage==7:novel_pipeline.begin_repair(execution)
                            quality_retry.prepare_run(job,execution,stage)
                            state=read_job(job)
                        state.update(busy=True, running_stage=stage, error=None, message='正在启动本步骤…')
                        write_job(job, state)
                    threading.Thread(target=run_stage, args=(job,stage), daemon=True).start()
                except Exception:
                    GPU.release();raise
                return self.json(state, 202)
            raise ValueError('不支持此操作')
        except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, FileNotFoundError) as exc:
            self.json({'error':str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    for path in DATA.glob('*/job.json'):
        state = read_job(path.parent)
        if state.get('busy'):
            state.update(busy=False, running_stage=None, error='上次服务已中断，请重试未完成的步骤。')
            write_job(path.parent, state)
    server = ThreadingHTTPServer(('127.0.0.1',args.port), Handler)
    print(f'SHARP panorama workbench: http://127.0.0.1:{args.port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
