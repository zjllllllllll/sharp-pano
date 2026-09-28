"""Offline audit only. No model imports, network requests or stage publication."""
import argparse
import csv
import html
import json
from pathlib import Path
import time
import uuid
from face_quality import PARAMETERS,VERSION,POLICY_HASH,joint
from face_quality_io import read,write,digest,assess_rgb,assess_depth
from novel_geometry import FACE_NAMES

LABELS=dict(zip(FACE_NAMES,['前','右','后','左','上','下']))
STATUS=dict(passed='通过',rejected='拒收',insufficient_evidence='证据不足',not_needed='无需修补')


def locations(job,rounds):
    state=read(job/'job.json');items=[]
    for n in rounds:
        start=6+4*(n-1);rgb=state['stages'][str(start+1)];depth=state['stages'][str(start+2)]
        execution=job.parent/rgb.get('execution_job',job.name)
        if (rgb['camera_id']!=depth['camera_id'] or rgb['revision']!=depth['revision'] or
            rgb['capture']['source']!=depth['source']):raise ValueError('Mixed historical RGB-D provenance')
        folder=execution/'novel'/rgb['revision']
        items.append(dict(round=n,execution=execution,folder=folder,rgb=rgb,depth=depth,
            capture=folder/rgb['capture']['path'],depth_folder=folder/depth['path']))
    return items


def protected_files(job,items):
    """Hash all published stage artifacts, plus the exact model/capture inputs."""
    paths=set()
    jobs={job}|{i['execution'] for i in items}
    for execution in jobs:
        paths.add(execution/'job.json');paths.update(execution.glob('stage-*.json'))
        state=read(execution/'job.json')
        for result in state['stages'].values():
            owner=execution.parent/result.get('execution_job',execution.name)
            novel=owner/'novel'/result['revision'] if result.get('revision') else owner
            for key in ('image','preview','download','patch','before_image','patch_image'):
                value=result.get(key)
                if isinstance(value,str):
                    for root in (owner,novel):
                        p=root/value
                        if p.is_file():paths.add(p)
            source=result.get('source',{})
            if isinstance(source,dict) and source.get('file') and (owner/source['file']).is_file():paths.add(owner/source['file'])
            if result.get('path'):
                for name in result.get('files',{}):
                    p=novel/result['path']/name
                    if p.is_file():paths.add(p)
    for item in items:
        cap=item['capture'];paths.update(cap.glob('*.png'));paths.update((cap/'repair').glob('*.png'))
        paths.update(item['depth_folder'].glob('*.npy'));paths.update(item['depth_folder'].glob('*.json'))
        paths.update(item['depth_folder'].glob('*.png'))
    return sorted(paths)


def render_report(report,out):
    columns=['round','camera','face','kind','score','status','joint_accepted','disposition','reason']
    with (out/'report.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader()
        for r in report['rows']:writer.writerow(r)
    esc=lambda v:html.escape(str(v));body=[]
    for r in report['rows']:
        score='—' if r['score'] is None else f"{r['score']:.2f}"
        images=''.join(f'<figure><img loading="lazy" src="{esc(path)}"><figcaption>{esc(name)}</figcaption></figure>' for name,path in r.get('maps',{}).items())
        body.append(f'<tr class="{"pass" if r["joint_accepted"] else "reject"}"><td>{r["round"]}</td><td>{esc(r["camera"])}</td>'
            f'<td>{LABELS[r["face"]]} · {r["face"]}</td><td>{r["kind"].upper()}</td><td>{score}</td>'
            f'<td>{STATUS.get(r["status"],r["status"])}</td><td>{esc(r["disposition"])}</td><td>{esc(r["reason"])}</td></tr>'
            f'<tr><td colspan="8"><details><summary>指标与评估区域</summary><pre>{esc(json.dumps(r["assessment"],ensure_ascii=False,indent=2))}</pre><div class="maps">{images}</div></details></td></tr>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>六面质量离线诊断</title>
<style>body{background:#101512;color:#e9f2ec;font:15px/1.6 system-ui;margin:32px}h1{font-size:26px}a{color:#b8f777}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #354235;text-align:left}.reject td:first-child{border-left:3px solid #f5b75f}.pass td:first-child{border-left:3px solid #b8f777}pre{white-space:pre-wrap;font-size:12px}.maps{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:240px}summary{cursor:pointer;color:#b8f777}</style>
<h1>六面 RGB / 深度质量离线诊断</h1><p>face-quality-v2-obvious-failure · 仅明显失效弃用 · 通过线 80 · 质量分表示一致性，不是生成正确概率。轻微偏差仅提示。</p>
<p>仅诊断历史输出。未重新推理，未更新原全景、深度或 PLY；以下“使用/弃用”表示按新规则的判定，尚未应用到已生成场景。</p>
<p>历史 LingBot 使用混合全景 RGB 引导；本报告不代表改用本面 RGB 后的新推理结果。</p>
<p><a href="report.json">完整 JSON</a> · <a href="report.csv">CSV 表格</a></p>'''
    page+=f'<p>共 {len(report["rows"])} 条独立评估；发布文件前后哈希校验：{report["immutability"]["status"]}。</p>'
    page+='<table><thead><tr>'+''.join('<th>'+s+'</th>' for s in ['轮次','相机','面','类型','质量分','自身判定','联合判定','原因'])+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></html>'
    (out/'report.html').write_text(page,encoding='utf-8')


def audit(job,rounds=(1,2)):
    job=Path(job).resolve();items=locations(job,rounds);paths=protected_files(job,items)
    print(f'校验 {len(paths)} 个既有发布及输入文件…',flush=True)
    before={str(p):digest(p) for p in paths}
    for item in items:
        rgb,depth,cap=item['rgb'],item['depth'],item['capture'];preview=rgb['capture']
        declared={item['execution']/depth['source']['file']:depth['source']['sha256'],
            item['folder']/rgb['image']:rgb['image_sha256'],cap/'raw.png':preview['raw_sha256'],
            cap/'mask.png':preview['mask_sha256'],cap/'geometry-mask-raw.png':preview['raw_geometry_sha256']}
        declared.update({cap/f'{name}-render-alpha.png':data['sha256'] for name,data in preview['alpha_faces'].items()})
        declared.update({item['depth_folder']/name:sha for name,sha in depth['files'].items()})
        declared.update({cap/'repair'/f'{r["face"]}-model.png':r['model_sha256'] for r in rgb['faces'] if r.get('model_sha256')})
        for path,expected in declared.items():
            if path.exists() and before.get(str(path),digest(path))!=expected:raise ValueError('Historical input hash mismatch: '+str(path))
    revision=time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    out=job/'diagnostics'/'face-quality'/revision;out.mkdir(parents=True)
    rows=[];pairs=[]
    for item in items:
        n=item['round'];camera_id=item['rgb']['camera_id'];relative=f'round-{n}'
        for name in FACE_NAMES:
            rgb=assess_rgb(item['capture'],name,out/relative)
            depth=assess_depth(item['capture'],item['depth_folder'],name,out/relative,item['depth']['position_world_m'])
            decision=joint(rgb,depth);pairs.append(dict(round=n,camera=camera_id,face=name,**decision))
            for kind,assessment in [('rgb',rgb),('depth',depth)]:
                intrinsic=assessment['status'] in ('passed','not_needed')
                disposition='使用' if decision['accepted'] else '连带弃用' if intrinsic else '自身拒收'
                reason='；'.join(assessment['reasons']) or ('同面的另一阶段未通过' if not decision['accepted'] else '所有拒收检查通过')
                if assessment.get('warnings'):reason+='；提示：'+'、'.join(assessment['warnings'])
                rows.append(dict(round=n,camera=camera_id,face=name,kind=kind,score=assessment['score'],
                    source_revision=item['rgb']['revision'],capture_path=item['rgb']['capture']['path'],depth_revision=item['depth']['depth_revision'],
                    status=assessment['status'],joint_accepted=decision['accepted'],disposition=disposition,reason=reason,
                    legacy_mixed_rgb_input=True,assessment=assessment,maps={k:f'{relative}/{v}' for k,v in assessment.get('maps',{}).items()}))
            print(f'第 {n} 轮 {camera_id} {name}: RGB {rgb["status"]} / 深度 {depth["status"]}',flush=True)
    after={str(p):digest(p) for p in paths};changed=[p for p in before if before[p]!=after[p]]
    if changed:raise RuntimeError('Published artifacts changed during audit: '+str(changed))
    report=dict(version=VERSION,policy_hash=POLICY_HASH,parameters=PARAMETERS,mode='audit',applied=False,
        revision=revision,job_id=job.name,rows=rows,faces=pairs,immutability=dict(status='unchanged',files=len(paths),hashes=before))
    write(out/'report.json',report);render_report(report,out)
    print(str(out/'report.html'),flush=True);return out,report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('job',type=Path);p.add_argument('--rounds',default='1,2');p.add_argument('--mode',choices=['audit'],default='audit')
    a=p.parse_args();audit(a.job,tuple(map(int,a.rounds.split(','))))
