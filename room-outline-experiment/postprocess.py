"""Consume completed native cloud stages; publish static, read-only lab data."""
from pathlib import Path
import argparse
import json
import os
import time
import traceback

from PIL import Image, ImageDraw, ImageFont
import outline
import cameras
from candidate_selection import VERSION as SELECTION_VERSION, apply_selection, write_candidates
from batch import write_json

HERE=Path(__file__).resolve().parent


def read_json(path):
    # Windows readers/virus scanners can briefly deny a concurrent metadata
    # replacement. Retry that transient condition without losing batch progress.
    for attempt in range(6):
        try:return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            if attempt==5:raise
            time.sleep(.2)


def publish(run):
    inputs=read_json(run/'inputs.json')
    rows=[]
    for item in inputs:
        job=run/item['stem']
        row=dict(id=item['stem'],name=item['name'],input=f'{item["stem"]}/input.png',
                 depth_ready=(job/'stage-2.json').exists(),cloud_ready=(job/'stage-3.json').exists(),
                 outline_ready=(job/'outline.json').exists(),cameras=[],render_complete=False)
        if row['outline_ready']:
            result=read_json(job/'outline.json')
            row.update(diagnostics=result['diagnostics'],polygon=result['polygon_m'],
                       overview=f'{item["stem"]}/overview.png',overlay=f'{item["stem"]}/overlay.png',
                       bev=f'{item["stem"]}/bev.png',full_bev=f'{item["stem"]}/bev-full.png',
                       evidence=f'{item["stem"]}/evidence.png')
        if (job/'sample-cameras.json').exists():
            data=read_json(job/'sample-cameras.json')
            row.update(sample=data,cameras=data['cameras'],render_complete=data['render_complete'] and (job/'cameras.png').exists(),
                       camera_map=f'{item["stem"]}/cameras.png' if (job/'cameras.png').exists() else None,
                       camera_map_base=f'{item["stem"]}/camera-map-base.png' if (job/'camera-map-base.png').exists() else None,
                       sampling_error=data.get('sampling_error'),selection_error=data.get('selection_error'))
        if (job/'analysis-error.json').exists():
            row['error']=read_json(job/'analysis-error.json')['error']
        rows.append(row)
    write_json(run/'catalog.json',dict(version=1,updated=time.time(),run=run.name,
        total=len(rows),outline_ready=sum(r['outline_ready'] for r in rows),
        render_complete=sum(r['render_complete'] for r in rows),
        camera_count=sum(len(r['cameras']) for r in rows),
        geometric_camera_count=sum(r.get('sample',{}).get('geometric_camera_count',len(r['cameras'])) for r in rows),
        rules=dict(spacing_m=cameras.SPACING,inset_m=cameras.INSET,height='same_as_source_camera',
            adaptive_offset=any(r.get('sample',{}).get('adaptive_offset') for r in rows),
            mask_threshold=cameras.MASK_THRESHOLD,connectivity=4,selection_version=SELECTION_VERSION,
            offset_step_m=cameras.OFFSET_STEP,offset_stop_below_m=cameras.OFFSET_STOP),
        revision=run.name+'-'+SELECTION_VERSION+'-'+next((r['sample'].get('sampling_version','fixed')+'-'+r['sample'].get('render_version','point-splat')+'-'+r['sample'].get('render_fingerprint',{}).get('implementation','')[:12] for r in rows if 'sample' in r),'pending'),scenes=rows))
    return rows


def make_report(run,rows):
    inset=next((r['sample']['inward_offset_m'] for r in rows if 'sample' in r),cameras.INSET)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    pages=[]
    for start in range(0,len(rows),12):
        chosen=rows[start:start+12]
        canvas=Image.new('RGB',(2400,4*620+85),'#101716')
        draw=ImageDraw.Draw(canvas)
        draw.text((25,18),f'点云俯视轮廓与相机采样  |  {start+1}–{start+len(chosen)} / {len(rows)}',font=font,fill='#e3eee6')
        for k,row in enumerate(chosen):
            x,y=(k%3)*800,(k//3)*620+70
            folder=run/row['id']
            draw.text((x+15,y),f'{row["id"]} · {len(row["cameras"])} 个采样相机',font=font,fill='#d4f2ae')
            for i,filename in enumerate(['overlay.png','cameras.png']):
                path=folder/filename
                if path.exists():
                    with Image.open(path) as im:
                        im=im.convert('RGB');im.thumbnail((396,550))
                        canvas.paste(im,(x+i*400+(400-im.width)//2,y+40+(550-im.height)//2))
            scene_inset=row.get('sample',{}).get('inward_offset_m',inset)
            draw.text((x+20,y+590),f'左：轮廓叠加 / 右：内缩 {scene_inset:g}m、0.5m 点阵',font=small,fill='#9db0a4')
        filename=f'contact-sheet-{start//12+1}.jpg'
        canvas.save(run/filename,quality=93)
        pages.append(filename)
    lines=['# 全景点云轮廓与相机采样实验','',
           f'共 {len(rows)} 张全景图；DepthMaster → 原版 SHARP 第 3 步双层初始化中心。',
           '轮廓仅用俯视几何与多高度支撑，无语义模型、无高斯加密。',
           '采样相机与原相机同高，0.5 m 格网；自适应版本从 offset 0.5 m 每次减 0.05 m，直到原点八邻域至少 4 点或 offset < 0.1 m。每场景实际 offset 记录在 sample-cameras.json。C000 单独保留；仅剩 C000 报采样错误。',
           '固定先排除全景 Mask 像素比例 >15% 的点（等于保留，C000 参考豁免），再仅保留原点所在四邻接连通分量。连接步长 0.5m，仅前后左右，不跨对角、缺失点或被排除点；筛选后不再放宽 offset。没有滑块或浏览器参数覆盖。图中绿点最终候选、红叉 Mask 超限、橙叉不连通；排除点的旧预览仅用于检查，不参与候选列表及导出。',
           '编号：原相机 C000，方形圈由内向外、每圈从俯视图正上方 (+Z) 顺时针，跳过不可采样格点。',
           f'共 {sum(len(r["cameras"]) for r in rows)} 个最终候选（含原视点参考），几何点池 {sum(r.get("sample",{}).get("geometric_camera_count",len(r["cameras"])) for r in rows)} 个；{sum(len(r["cameras"])<=1 for r in rows)} 张无新增可采样位置。',
           ('预览为原始 RGB-D 可靠表面与未知空间首次遮挡判定；暗色区域是待修补 Mask，不是最终高斯，也没有 AI 修补。'
            if any(r.get('sample',{}).get('render_version') for r in rows) else
            '预览为点云球面深度缓冲渲染，不是最终高斯渲染；暗色空洞未修补。'),
           '轮廓是实验性的可见几何包络，不等于建筑真值；相机尚未通过家具碰撞/未知空域验证。','',
           '## 总览','']
    for page in pages:lines += [f'![{page}]({(run/page).as_posix()})','']
    lines += ['## 每张图','']
    for row in rows:
        folder=run/row['id']
        lines += [f'### {row["id"]} — {len(row["cameras"])} 个相机','',
                  f'[原始深度]({(folder/"depth.npy").as_posix()}) · [完整点云 LAS]({(folder/"cloud.las").as_posix()}) · '
                  f'[俯视点云]({(folder/"bev.png").as_posix()}) · [轮廓叠加]({(folder/"overlay.png").as_posix()}) · '
                  f'[编号相机图]({(folder/"cameras.png").as_posix()})','',
                  f'![{row["id"]} 对照]({(folder/"overview.png").as_posix()})','']
    (run/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=HERE/'runs/20260922-adaptive-sampling')
    p.add_argument('--watch-pid',type=int);p.add_argument('--no-render',action='store_true')
    p.add_argument('--only',nargs='+');p.add_argument('--force',action='store_true')
    a=p.parse_args()
    inputs=json.loads((a.run/'inputs.json').read_text(encoding='utf-8'))
    selected=[x for x in inputs if not a.only or x['stem'] in a.only]
    done=set()
    while True:
        for row in selected:
            if row['stem'] in done:continue
            job=a.run/row['stem']
            if not (job/'stage-3.json').exists():continue
            try:
                if a.force or not (job/'outline.json').exists():outline.process(job)
                ready=False
                if (job/'sample-cameras.json').exists() and not a.force:
                    from occlusion import VERSION
                    cached=json.loads((job/'sample-cameras.json').read_text(encoding='utf-8'))
                    ready=cached['render_complete'] and cached.get('render_version')==VERSION
                    if ready and (cached.get('selection_version')!=SELECTION_VERSION or not (job/'candidate-cameras.json').exists()):
                        cached=apply_selection(cached)
                        cameras.draw_camera_map(job,cached)
                        write_json(job/'sample-cameras.json',cached)
                        write_candidates(job,cached)
                    if a.no_render:ready=True
                if not ready:cameras.process(job,render=not a.no_render)
                print(f'POSTPROCESS {row["stem"]} OK',flush=True)
            except Exception as error:
                traceback.print_exc()
                write_json(job/'analysis-error.json',dict(error=str(error),traceback=traceback.format_exc()))
            done.add(row['stem'])
            publish(a.run)
        if len(done)==len(selected) or not a.watch_pid:break
        import psutil
        if not psutil.pid_exists(a.watch_pid):
            # Recheck stage markers on one more iteration after producer exits.
            pending=[r for r in selected if r['stem'] not in done and (a.run/r['stem']/'stage-3.json').exists()]
            if not pending:break
        time.sleep(2)
    rows=publish(a.run);make_report(a.run,rows)
    print(f'POSTPROCESS FINISHED {len(done)}/{len(selected)}',flush=True)


if __name__=='__main__':main()
