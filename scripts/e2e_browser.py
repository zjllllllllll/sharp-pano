"""Drive an existing fresh browser job through the production iteration stop decision.

Uses the normal Next button, including the browser's real Spark RGB/alpha
capture before each repair. No cached model outputs or fake API responses.
Requires a running server and agent-browser session; --job resumes that same job.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    parser.add_argument('--port',type=int,default=8771)
    parser.add_argument('--session',default='pano-e2e')
    args=parser.parse_args()
    browser=ROOT/'pano-web/node_modules/agent-browser/bin/agent-browser-win32-x64.exe'
    out=ROOT/'.runtime/e2e'/args.job
    out.mkdir(parents=True,exist_ok=True)

    def action(*commands):
        result=subprocess.run([str(browser),'--session',args.session,*commands],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        if result.returncode:raise RuntimeError(result.stdout+'\n'+result.stderr)
        return result.stdout

    def state():
        with urlopen(f'http://127.0.0.1:{args.port}/api/jobs/{args.job}',timeout=15) as response:
            return json.load(response)

    journal=[]
    previous=out/'progress.json'
    if previous.exists():journal=json.loads(previous.read_text(encoding='utf-8'))
    end=time.monotonic()+7200
    while time.monotonic()<end:
        job=state()
        if job.get('error'):raise RuntimeError(job['error'])
        count=len(job['stages'])
        if job.get('iteration_status')=='stopped' and not job['busy']:
            (out/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2),encoding='utf-8')
            action('screenshot',str(out/'completed.png'))
            print('COMPLETED',args.job,flush=True)
            return
        if job['busy'] or job.get('quality_run',{}).get('status')=='awaiting_capture':
            print(json.dumps({'completed':count,'running':job.get('running_stage'),'message':job.get('message','')[:220]},ensure_ascii=False),flush=True)
            time.sleep(15)
            continue
        action('snapshot','-i')
        action('wait','#next:not([disabled])')
        next_stage=count+1
        print('CLICK NEXT',next_stage,flush=True)
        action('click','#next')
        journal.append({'stage':next_stage,'clicked_at':time.time(),'completed_before':count})
        previous.write_text(json.dumps(journal,ensure_ascii=False,indent=2),encoding='utf-8')
        # The click awaits a potentially expensive six-face browser capture.
        # Do not click again while the UI is completing that request.
        sent=time.monotonic()
        while time.monotonic()-sent<180:
            fresh=state()
            if fresh.get('error'):raise RuntimeError(fresh['error'])
            if fresh['busy'] or len(fresh['stages'])>count or fresh.get('iteration_status')=='stopped':break
            visible=action('eval',"JSON.stringify({error:document.getElementById('error').hidden?null:document.getElementById('error').textContent,nextDisabled:document.getElementById('next').disabled})")
            if 'error' in visible and '"error":null' not in visible.replace('\\"','"'):
                print('UI',visible[:1200],flush=True)
            time.sleep(4)
        else:
            raise RuntimeError(f'Stage {next_stage} never started; inspect browser before resuming')
    raise TimeoutError('E2E deadline reached; job preserved, inspect before resuming')


if __name__=='__main__':
    main()
