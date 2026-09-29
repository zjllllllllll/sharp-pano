import * as THREE from 'three';
import {ViewCache} from './view-cache.mjs';
import {visibleStep, displayStep, latestVisibleStep, nextAutomaticStage} from './workflow-mode.mjs';
import {NovelView} from './novel-view.js';
import {FIXED_NORMAL_ANGLE, DISTANCE_NEIGHBORS, angleStatistics, visibleAt,
  FIXED_DISTANCE_NEIGHBORS, FIXED_DISTANCE_CM, distanceStatistics, distanceCounts, attachAngleShader} from './point-filter.mjs';

const $ = id => document.getElementById(id);
const mode=new URLSearchParams(location.search).get('mode')==='debug'?'debug':'auto';
const automatic=mode==='auto';
document.body.dataset.mode=mode;$('mode').value=mode;
$('mode-title').textContent=automatic?'默认模式':'调试模式';
document.title=`SHARP · ${automatic?'全景生成':'调试模式'}`;
$('empty').querySelector('p').textContent=automatic?'上传全景图，自动生成完整场景':'上传后，逐步查看深度、点云和高斯的变化';
$('mode').onchange=()=>{const url=new URL(location.href);url.searchParams.set('mode',$('mode').value);location.assign(url);};
let advancing=false, automaticFailed=null, initializing=true, viewSelections=0;

const names = ['全景图', '全景深度图', '反投影点云', '初始化高斯', '调整后的高斯','候选相机','六面补洞','深度补全','高斯补洞'];
const subtitles = ['RGB 输入', 'DepthMaster · 径向距离', '全景深度 · 直接反投影', 'SHARP · 未加预测修正', 'SHARP · 最终预测','高斯场景 · 全局四邻域','RGB 补洞 · 外羽化','高斯深度 · LingBot','SHARP · 深度锚定补丁'];
names.push('候选相机 · 第二轮','六面补洞 · 第二轮','深度补全 · 第二轮','高斯补洞 · 第二轮');
subtitles.push('累计可信空间 · 最大 Mask','RGB 补洞 · 外羽化','高斯深度 · LingBot','SHARP · 深度锚定补丁');
const repairRound=step=>step<6?0:Math.floor((step-6)/4)+1;
const roundStart=round=>6+(round-1)*4;
const localStep=step=>step<6?step:6+(step-6)%4;
const stepName=step=>names[step-1]||`${names[localStep(step)-1]} · 第 ${repairRound(step)} 轮`;
const executionJob=step=>step>=10?job?.stages[String(roundStart(repairRound(step)))]?.execution_job:job?.id;
const roundStages=step=>Object.fromEntries([6,7,8,9].map(i=>[i,job?.stages[String(i+4*Math.max(0,repairRound(step)-1))]]));
const roundDescription=(text,step)=>step<10?text:['第七步','第八步','第九步'].reduce((value,label,i)=>value?.replaceAll(label,`第 ${7+i+4*(repairRound(step)-1)} 步`),text);
let repairMode='merged';
let token = '', job = null, selected = 1, loadingUpload = false, viewSerial = 0, polling = false;
let navigationJob=null, navigationSelected=null, navigationRound=1;
let renderer, scene, camera, spark, object, pointMaterial;
let yaw = 0, pitch = 0, lastTime = performance.now(), imgScale = 1, imgX = 0, imgY = 0, gray = false;
const keys = new Set();
let drag = null;
let savedPose = null, debugData = null;
let normalStats = null, normalPending = false, pointPreviewFull = true;
let normalAngles=null, distanceStats=null, distanceBuffer=null, distancePending=false;
let gaussianVariant='masked';
let pointMasked=true;
let denseObject=null, denseLoadingSerial=null;
const denseRecords=new Map(), denseEnabled=new Set(), denseRequests=new Map();
let gaussianDense=false, gaussianDensePending=false, activeEntry=null, cacheJob=null, preloadJob=null;
let lastSwitch=null;
const viewCache=new ViewCache(entry=>{
  scene?.remove(entry.root);
  entry.root.traverse(node=>{if(node.isPoints){node.geometry.dispose();node.material.dispose();}else if(node.splats)node.dispose();});
});
const imageCache=new Map();
const normalThreshold = {value:FIXED_NORMAL_ANGLE};
const distanceUniforms={threshold:{value:FIXED_DISTANCE_CM/100},enabled:{value:0}};
const novel=new NovelView({api,context:()=>({renderer,scene,camera,spark,aim:direction=>{
  camera.lookAt(camera.position.clone().add(direction));pitch=camera.rotation.x;yaw=camera.rotation.y;
}}),error:error=>showError(error.message),onBusy:()=>renderState()});

function updateNormalStatus() {
  if (!normalStats) return;
  const visible = visibleAt(normalStats, FIXED_NORMAL_ANGLE), hidden = normalStats.total-visible;
  $('normal-status').textContent = `${pointPreviewFull?'全量':'1/16 抽样'}筛选统计：法向删除 ${hidden.toLocaleString()} 点 · 剩余 ${visible.toLocaleString()} / ${normalStats.total.toLocaleString()} 点 · 法向不可靠 ${normalStats.unreliable.toLocaleString()} 点也进入距离筛选`;
}

function updatePointMask() {
  normalThreshold.value=pointMasked?FIXED_NORMAL_ANGLE:90;
  $('point-mask-before').setAttribute('aria-pressed',String(!pointMasked));
  $('point-mask-after').setAttribute('aria-pressed',String(pointMasked));
  updateFixedDistance();
  updateDensifyPanel();
}

function updateFixedDistance() {
  const index=DISTANCE_NEIGHBORS.indexOf(FIXED_DISTANCE_NEIGHBORS);
  distanceUniforms.enabled.value=distanceStats && pointMasked?1:0;
  if(!distanceStats) return;
  if(object?.isPoints && distanceBuffer) {
    object.geometry.setAttribute('meanNeighborDistance',new THREE.InterleavedBufferAttribute(distanceBuffer,1,index));
  }
  const counts=distanceCounts(distanceStats,index,FIXED_DISTANCE_CM);
  $('distance-status').textContent=`当前：${pointMasked?'Mask 后 · 已筛选':'Mask 前 · 原始'} · 显示 ${(pointMasked?counts.visible:distanceStats.total).toLocaleString()} 点。筛选统计：距离额外删除 ${counts.hidden.toLocaleString()} 点，其中法向不可靠点 ${counts.unknownRemoved.toLocaleString()} 个。${pointMasked?'':'当前未应用两轮删除。'}`;
  updateDensifyPanel();
}

async function loadPointDistances(jid,serial,useFull) {
  distancePending=true;$('distance-retry').hidden=true;
  $('distance-status').textContent='正在准备剩余点的近邻距离；首次计算会缓存…';
  const current=()=>serial===viewSerial && job?.id===jid && selected===3 && object?.isPoints;
  try {
    let status=await api(`/api/jobs/${jid}/point-distances`,{method:'POST'});
    while(status.status==='running') {
      if(!current()) return;
      $('distance-status').textContent=status.message || '正在搜索近邻…';
      await new Promise(resolve=>setTimeout(resolve,1000));
      if(!current()) return;
      status=await api(`/api/jobs/${jid}/point-distances`);
    }
    if(!current()) return;
    if(status.status!=='ready' || !status.result) throw new Error(status.message || '距离数据未就绪');
    const info=status.result;
    if(info.normal_angle_degrees!==FIXED_NORMAL_ANGLE || JSON.stringify(info.neighbors)!==JSON.stringify(DISTANCE_NEIGHBORS)) throw new Error('距离缓存的法向阈值或 K 档位不一致，请刷新页面');
    $('distance-status').textContent='正在载入距离缓存…';
    const response=await fetch(`/artifacts/${jid}/${useFull?info.full:info.preview}`);
    if(!response.ok) throw new Error('距离缓存读取失败');
    const data=new Float32Array(await response.arrayBuffer());
    if(!current()) return;
    distanceStats=distanceStatistics(normalAngles,data);
    distanceBuffer=new THREE.InterleavedBuffer(data,DISTANCE_NEIGHBORS.length);
    updateFixedDistance();
    attachAngleShader(pointMaterial,normalThreshold,distanceUniforms);
  } catch(error) {
    if(current()) {
      distanceUniforms.enabled.value=0;
      $('distance-status').textContent=`距离筛选暂不可用：${error.message}。当前仅应用固定 82° 法向筛选。`;
      $('distance-retry').hidden=false;
    }
  } finally {if(serial===viewSerial) {distancePending=false;updateDensifyPanel();}}
}

async function loadPointNormals(jid, serial, useFull) {
  normalPending=true;
  $('normal-retry').hidden=true;
  $('normal-status').textContent='正在准备法向夹角；首次计算会缓存，不需要重新运行模型…';
  const current = () => serial===viewSerial && job?.id===jid && selected===3 && object?.isPoints;
  try {
    let status=await api(`/api/jobs/${jid}/point-normals`,{method:'POST'});
    while (status.status==='running') {
      if (!current()) return;
      $('normal-status').textContent=status.message || '正在估计法向…';
      await new Promise(resolve=>setTimeout(resolve,1000));
      if (!current()) return;
      status=await api(`/api/jobs/${jid}/point-normals`);
    }
    if (!current()) return;
    if (status.status!=='ready' || !status.result) throw new Error(status.message || '法向数据尚未就绪');
    const name=useFull?status.result.full:status.result.preview;
    const response=await fetch(`/artifacts/${jid}/${name}`);
    if (!response.ok) throw new Error('无法读取法向夹角文件');
    const angles=new Float32Array(await response.arrayBuffer());
    if (!current()) return;
    if (angles.length!==object.geometry.attributes.position.count) throw new Error('法向与当前点云数量不匹配');
    normalStats=angleStatistics(angles);
    normalAngles=angles;
    object.geometry.setAttribute('normalAngle',new THREE.BufferAttribute(angles,1));
    attachAngleShader(pointMaterial,normalThreshold);
    updateNormalStatus();
    void loadPointDistances(jid,serial,useFull);
  } catch(error) {
    if (current()) {
      $('normal-status').textContent=`法向筛选暂不可用：${error.message}。点云仍完整显示。`;
      $('normal-retry').hidden=false;
    }
  } finally { if(serial===viewSerial) {normalPending=false;updateDensifyPanel();} }
}

function denseShown() {
  return selected===3 && pointMasked && !!distanceStats && denseEnabled.has(job?.id) && !!denseObject;
}
function updateDensifyPanel() {
  if(denseObject) denseObject.visible=denseShown();
  if(selected!==3 || !job) return;
  const status=denseRecords.get(job.id), info=status?.result, enabled=denseEnabled.has(job.id);
  const ready=!!distanceStats && !normalPending && !distancePending && object?.isPoints && $('view-loading').hidden;
  $('densify-before').disabled=!object?.isPoints;
  $('densify-after').disabled=!ready || !!job.busy || denseRequests.has(job.id) || denseLoadingSerial===viewSerial;
  $('densify-before').setAttribute('aria-pressed',String(!enabled));
  $('densify-after').setAttribute('aria-pressed',String(enabled));
  if(status?.status==='error') $('densify-status').textContent=`加密预览暂不可用：${status.message}。原点云未改动，可点击“加密后”重试。`;
  else if(status?.status==='running') $('densify-status').textContent=`${status.message||'正在生成加密预览…'}；当前暂显加密前点云。`;
  else if(denseLoadingSerial===viewSerial) $('densify-status').textContent='正在载入新增点；当前暂显加密前点云…';
  else if(info) {
    const base=distanceStats?(pointMasked?distanceCounts(distanceStats,DISTANCE_NEIGHBORS.indexOf(FIXED_DISTANCE_NEIGHBORS),FIXED_DISTANCE_CM).visible:distanceStats.total):null;
    const added=denseShown()?denseObject.geometry.attributes.position.count:0;
    $('densify-status').textContent=`当前：${denseShown()?'加密后':pointMasked?'加密前':'Mask 前 · 未加密'}${base!==null?` · 显示 ${(base+added).toLocaleString()} 点（新增 ${added.toLocaleString()}）`:''}。全量 Mask 后：${info.before.toLocaleString()} → ${info.after.toLocaleString()}，新增 ${info.added.toLocaleString()} 点。${pointPreviewFull?'':'当前为 1/16 抽样；新增点也抽取同一全量加密结果的 1/16。'}`;
  } else $('densify-status').textContent=ready?'点击“加密后”生成预览；新增点索引接入后续高斯。':'等待点云与两轮 Mask 载入完成后，可生成加密预览。';
}
function ensureDensification(jid) {
  if(denseRequests.has(jid)) return denseRequests.get(jid);
  const task=(async()=>{
    let status=await api(`/api/jobs/${jid}/point-densify`,{method:'POST'});
    denseRecords.set(jid,status);updateDensifyPanel();
    while(status.status==='running') {
      await new Promise(resolve=>setTimeout(resolve,1000));
      status=await api(`/api/jobs/${jid}/point-densify`);
      denseRecords.set(jid,status);updateDensifyPanel();
    }
    if(status.status!=='ready' || !status.result) throw new Error(status.message||'加密预览未就绪');
    return status.result;
  })().catch(error=>{denseRecords.set(jid,{status:'error',message:error.message});throw error;})
    .finally(()=>{denseRequests.delete(jid);renderState();updateDensifyPanel();});
  denseRequests.set(jid,task);renderState();updateDensifyPanel();
  return task;
}
async function loadDensePoints(jid,serial,useFull) {
  if(denseLoadingSerial===serial || denseObject) return;
  denseLoadingSerial=serial;updateDensifyPanel();
  const current=()=>serial===viewSerial && job?.id===jid && selected===3 && object?.isPoints;
  try {
    const status=denseRecords.get(jid);
    const info=status?.status==='ready'?status.result:await ensureDensification(jid);
    if(!current() || !denseEnabled.has(jid)) return;
    const response=await fetch(`/artifacts/${jid}/${useFull?info.full:info.preview}`);
    if(!response.ok) throw new Error('新增点文件读取失败');
    const data=new Float32Array(await response.arrayBuffer());
    if(!current() || !denseEnabled.has(jid)) return;
    const count=useFull?info.added:info.preview_added;
    if(data.length!==count*6) throw new Error('新增点数量与记录不一致');
    for(let i=0;i<data.length;i+=6) for(let c=3;c<6;c++) data[i+c]=data[i+c]<=.04045?data[i+c]/12.92:((data[i+c]+.055)/1.055)**2.4;
    const buffer=new THREE.InterleavedBuffer(data,6),geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));
    geometry.setAttribute('color',new THREE.InterleavedBufferAttribute(buffer,3,3));
    const material=new THREE.PointsMaterial({vertexColors:true,size:Number($('point-size').value),sizeAttenuation:false});
    denseObject=new THREE.Points(geometry,material);
    activeEntry.dense=denseObject;activeEntry.root.add(denseObject);
  } catch(error) {
    if(current()) denseRecords.set(jid,{status:'error',message:error.message});
  } finally {if(serial===viewSerial){denseLoadingSerial=null;updateDensifyPanel();}}
}
async function restoreDensification(jid,serial,useFull) {
  try {
    const status=await api(`/api/jobs/${jid}/point-densify`);
    if(serial!==viewSerial || job?.id!==jid) return;
    denseRecords.set(jid,status);updateDensifyPanel();
    if(status.status==='running') await ensureDensification(jid);
    if(serial===viewSerial && job?.id===jid && denseEnabled.has(jid)) await loadDensePoints(jid,serial,useFull);
  } catch(error) {if(serial===viewSerial){denseRecords.set(jid,{status:'error',message:error.message});updateDensifyPanel();}}
}

function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }
async function api(url, options = {}) {
  const response = await fetch(url, {...options, headers: {...options.headers, 'X-Local-Token': token}});
  const value = await response.json();
  if (!response.ok) {const error=new Error(value.error || `请求失败 (${response.status})`);error.status=response.status;throw error;}
  return value;
}
function artifact(name) { return `/artifacts/${job.id}/${name}`; }
function pipelineStepCount(){
  const done=Math.max(0,...Object.keys(job?.stages||{}).map(Number));
  const next=job?.iteration_status==='stopped'?done:Math.min(25,done+1);
  const last=Math.max(9,next,job?.running_stage||0);
  return 5+4*Math.ceil((last-5)/4);
}
function canViewStep(step){if(automatic)return !!job?.stages[String(step)];return !!job?.stages[String(step)]||(localStep(step)>=7&&job?.running_stage===step)||(step===7&&job?.mask_update_pending);}
function setSelectedStep(step){
  selected=step;document.body.classList.remove('auto-processing');
  const url=new URL(location.href);url.searchParams.set('job',job.id);url.searchParams.set('step',String(step));
  history.replaceState(null,'',url);
}
function renderStepList(container,steps){
  steps=steps.filter(step=>visibleStep(step,mode));
  const key=steps.join(',');
  if(container.dataset.steps!==key){
    container.dataset.steps=key;
    container.replaceChildren(...steps.map(step=>{
      const button=document.createElement('button');button.type='button';button.dataset.step=String(step);
      const num=document.createElement('span');num.className='num';num.textContent=String(displayStep(step,mode)).padStart(2,'0');
      const text=document.createElement('span');text.className='name';text.append(document.createTextNode(automatic&&step>=6?'修补结果':names[localStep(step)-1]),document.createElement('small'));
      const status=document.createElement('span');status.className='step-status';
      button.append(num,text,status);button.onclick=()=>selectStep(step);return button;
    }));
  }
  for(const button of container.children){
    const step=Number(button.dataset.step),done=!!job?.stages[String(step)],running=job?.busy&&job.running_stage===step;
    button.className=`step${selected===step?' active':''}${done?' done':''}${running?' running':''}`;
    button.disabled=advancing||qualityResuming||novel.busy||!canViewStep(step);
    button.setAttribute('aria-current',selected===step?'step':'false');
    button.setAttribute('aria-label',`第 ${displayStep(step,mode)} 步 ${names[localStep(step)-1]}`);
    button.querySelector('small').textContent=running?'正在生成…':step===3&&done&&job.stages['3'].diagnostics?.learned_alignment!==false?'SHARP · 初始中心':step>=10&&localStep(step)===6?'累计可信空间 · 最大 Mask':subtitles[localStep(step)-1];
    const status=button.querySelector('.step-status');status.textContent=running?'…':done?'✓':'';
    status.setAttribute('aria-label',running?'正在生成':done?'已生成':'尚未生成');
  }
}
function renderNavigation(){
  const jid=job?.id||null,initial=$('initial-stage-group');
  if(navigationJob!==jid){navigationJob=jid;navigationSelected=null;navigationRound=repairRound(selected)||1;initial.open=selected<=5;}
  if(navigationSelected!==selected){
    if(selected>=6){navigationRound=repairRound(selected);if(navigationSelected===null||navigationSelected<=5)initial.open=false;}
    else initial.open=true;
    navigationSelected=selected;
  }
  const count=pipelineStepCount(),roundCount=Math.ceil((count-5)/4);
  const latest=Math.max(1,repairRound(Math.max(job?.running_stage||0,...Object.keys(job?.stages||{}).map(Number))));
  $('pipeline-summary').textContent=automatic?'自动生成':`${count} 个步骤 · ${roundCount} 轮修补`;
  const initialDone=[1,2,3,4,5].filter(step=>job?.stages[String(step)]).length;
  $('initial-stage-status').textContent=automatic?(!job?'未开始':initialDone===5?'已完成':'生成中'):`01—05 · ${initialDone===5?'已完成':`${initialDone}/5`}`;
  renderStepList($('initial-steps'),[1,2,3,4,5]);
  const picker=$('repair-round');
  if(picker.options.length!==roundCount)picker.replaceChildren(...Array.from({length:roundCount},(_,i)=>{const option=document.createElement('option');option.value=String(i+1);return option;}));
  for(const option of picker.options){
    const round=Number(option.value),start=roundStart(round),stages=[0,1,2,3].map(i=>job?.stages[String(start+i)]);
    const camera=stages[3]?.camera_id||stages[2]?.camera_id||stages[1]?.camera_id||stages[0]?.current?.camera_id||stages[0]?.recommended;
    const generated=stages.filter(Boolean).length,review=stages[3]?.acceptance?.status==='review_required';
    const running=job?.busy&&repairRound(job.running_stage)===round;
    option.textContent=automatic?`第 ${round} 轮 · ${generated===4?'已完成':running||generated?'生成中':'未开始'}`:`第 ${round} 轮${camera?` · ${camera}`:''} · ${running?'生成中':review?'待复核':generated===4?'已完成':generated?`${generated}/4`:'未开始'}`;
    option.disabled=automatic?!canViewStep(start+3):![0,1,2,3].some(i=>canViewStep(start+i));
  }
  picker.value=String(navigationRound);picker.disabled=advancing||qualityResuming||novel.busy||![...picker.options].some(option=>!option.disabled);
  const start=roundStart(navigationRound),completed=[0,1,2,3].filter(i=>job?.stages[String(start+i)]).length;
  const review=job?.stages[String(start+3)]?.acceptance?.status==='review_required';
  $('repair-round-status').textContent=automatic?(completed===4?'修补结果已生成':'等待修补结果'):`${navigationRound<latest?'历史轮次':navigationRound===latest?'最新一轮':'待开始轮次'} · ${review?'已生成，补丁覆盖率待复核':completed===4?'4/4 已完成':`${completed}/4 已完成`}`;
  renderStepList($('repair-steps'),[start,start+1,start+2,start+3]);
  $('pipeline-running').hidden=!job?.busy;
  if(job?.busy)$('pipeline-running').textContent=`正在运行：${repairRound(job.running_stage)?`第 ${repairRound(job.running_stage)} 轮 · `:''}${names[localStep(job.running_stage)-1]}`;
  $('pipeline-current-step').textContent=`STEP ${String(displayStep(selected,mode)).padStart(2,'0')} · ${selected<=5?'首次生成':`第 ${repairRound(selected)} 轮`} · ${names[localStep(selected)-1]}`;
}
$('repair-round').onchange=async()=>{
  const start=roundStart(Number($('repair-round').value)),target=start+(selected>=6?localStep(selected)-6:0);
  const available=[start,start+1,start+2,start+3].filter(step=>visibleStep(step,mode)&&canViewStep(step));
  if(available.length)await selectStep(available.includes(target)?target:available.at(-1));
  else renderNavigation();
};
function renderState() {
  const done = job ? Object.keys(job.stages).length : 0;
  const total=pipelineStepCount(),stopped=job?.iteration_status==='stopped',atLimit=done>=25;
  const gpt=job?.rgb_repair_model==='gpt-image-2',globalMask=job?.camera_selection_policy==='max-mask';
  subtitles[5]=globalMask?'高斯场景 · 最大 Mask':'高斯场景 · 全局四邻域';
  const branch=$('branch-info');branch.replaceChildren();branch.hidden=!job?.branch;
  if(job?.branch){
    branch.append(document.createTextNode(`${job.branch.name} · 从第六步分支 · `));
    const source=document.createElement('a');source.textContent='查看原流程';source.href=`/?job=${job.branch.source_job}&step=9`;branch.append(source);
  }
  renderNavigation();
  $('next').disabled = advancing || !job || job.busy || novel.busy || loadingUpload || stopped || atLimit || denseRequests.size>0 || gaussianDensePending;
  const phase=localStep(done);
  $('next').textContent = !job ? '请先上传全景图' : job.busy ? '正在生成…' : novel.busy?'正在渲染…':stopped ? '迭代已停止 ✓' : atLimit ? '已达到 5 轮上限'  : phase===6?`确认推荐相机 · ${gpt?'GPT Image 2':'Seedream'} 补洞 →`:phase===7?'渲染深度 · LingBot 补全 →':phase===8?'SHARP 生成补丁 · 合并高斯 →':`生成${stepName(done+1)} →`;
  if(job?.quality_run?.status==='awaiting_capture'){
    $('next').disabled=qualityResuming||novel.busy;$('next').textContent=qualityResuming?'正在采集下一台相机…':'继续采集下一台候选相机 →';
  }
  if(['exhausted','cancelled'].includes(job?.quality_run?.status)){
    $('next').disabled=true;$('next').textContent='本轮未找到可用修补';
  }
  $('file').disabled = advancing || qualityResuming || !!job?.busy || novel.busy || loadingUpload || denseRequests.size>0 || gaussianDensePending;
  $('progress').max=total;$('progress').value = done;
  document.body.classList.toggle('busy', !!job?.busy);
  $('progress-label').textContent = automatic ? !job?'等待上传全景图':
    job.error||automaticFailed===job.id?'生成已暂停，请切换调试模式查看':
    ['exhausted','cancelled'].includes(job.quality_run?.status)?'本轮修补已结束':
    stopped||atLimit?'生成完成':job.busy||advancing||qualityResuming?
    ((job.running_stage||done+1)>=6?`正在进行第 ${repairRound(job.running_stage||done+1)} 轮修补…`:'正在生成场景…'):
    job.execution_mode==='auto'?'自动生成中…':'已有结果' : job ? `${done} / ${total} 已完成 · ${job.message}` : '等待上传全景图';
  $('novel-render').disabled=novel.busy||!!job?.busy;
  if (job?.error) showError(job.error);
  $('compare').disabled = !job?.stages['5'];
  $('compare').hidden = localStep(selected)===9;
  $('running-debug').disabled = !job?.busy;
  $('mask-build').disabled=!!job?.busy;
}

function gaussianResult(result) {
  if(result.kind!=='gaussians') return result;
  if(result.densification?.baseline_variants)result={...result,variants:result.densification.baseline_variants,
    mask:{...result.mask,gaussian_removed:result.densification.parts.removed.count}};
  const hasVariants=!!(result.variants?.masked && result.variants?.unmasked);
  const variant=hasVariants?gaussianVariant:'unmasked';
  $('mask-before').disabled=!hasVariants;
  $('mask-after').disabled=!hasVariants;
  $('mask-before').setAttribute('aria-pressed',String(variant==='unmasked'));
  $('mask-after').setAttribute('aria-pressed',String(variant==='masked'));
  $('mask-build').hidden=hasVariants;
  let visible=hasVariants?{...result,...result.variants[variant]}:result;
  if(gaussianDense && variant==='masked' && result.densification) visible={...visible,...result.densification.dense};
  $('mask-status').textContent=hasVariants?
    `当前：${variant==='masked'?'Mask 后 · 已筛选':'Mask 前 · 原始'} · ${visible.count.toLocaleString()} 个高斯 · Mask 剔除 ${result.mask.gaussian_removed.toLocaleString()} 个（${(100*result.mask.gaussian_removed/Math.max(1,result.variants.unmasked.count)).toFixed(2)}%）`:
    '当前为旧版未筛选结果。点击“生成本步 Mask 对比”补充筛选版，原始 PLY 会保留。';
  return visible;
}
function updateGaussianDensePanel() {
  if(selected<4 || !job) return;
  const info=job.stages[String(selected)]?.densification;
  $('gaussian-dense-before').setAttribute('aria-pressed',String(!gaussianDense));
  $('gaussian-dense-after').setAttribute('aria-pressed',String(gaussianDense));
  $('gaussian-dense-before').disabled=gaussianDensePending;
  $('gaussian-dense-after').disabled=!info || gaussianDensePending;
  $('gaussian-dense-build').hidden=!!info;
  $('gaussian-dense-build').disabled=!!job.busy || gaussianDensePending;
  if(!gaussianDensePending) $('gaussian-dense-status').textContent=info?
    `当前：${gaussianDense?'加密后':'加密前'} · 新增 ${info.added.toLocaleString()} 个高斯 · 加密后 ${info.dense.count.toLocaleString()} 个。仅追加：原高斯完全不变，新增高斯除位置外继承原像素父高斯。切回 Mask 前会关闭加密。`:
    '此旧任务尚未生成加密高斯。点击生成；初始化不运行网络，调整版需运行一次 SHARP。';
}
function metadata(result) {
  const values = [];
  if (result.width) values.push(['分辨率',`${result.width} × ${result.height}`]);
  if (result.count) values.push([result.kind === 'points' ? '点数' : '高斯数',result.count.toLocaleString()]);
  if (result.layers) values.push(['层数',result.layers]);
  if (result.seconds) values.push(['本步用时',`${result.seconds.toFixed(1)} 秒`]);
  if (result.display_p01_p99_m) values.push(['显示范围',`${result.display_p01_p99_m.map(x=>x.toFixed(2)).join(' — ')} m`]);
  if (result.filled_pixels) values.push(['注入补齐的无效像素',result.filled_pixels]);
  $('metrics').replaceChildren(...values.map(([label,value])=>{
    const item=document.createElement('span'), name=document.createElement('b'); name.textContent=label;
    item.append(name,document.createTextNode(String(value))); return item;
  }));
}
function disposeObject() {
  // Detach the view, not its resources. Preloaded variants remain reusable.
  if(activeEntry) activeEntry.root.visible=false;
  activeEntry=null;denseObject=null;
  denseLoadingSerial=null;
  normalStats=null;normalPending=false;
  normalAngles=null;distanceStats=null;distanceBuffer=null;distancePending=false;
  distanceUniforms.enabled.value=0;
  object = null; pointMaterial = null;
}

async function readFloats(jid,name) {
  const response=await fetch(`/artifacts/${jid}/${name}`);
  if(!response.ok) throw new Error(`无法读取 ${name}`);
  return new Float32Array(await response.arrayBuffer());
}
function pointsMesh(data) {
  for(let i=0;i<data.length;i+=6) for(let c=3;c<6;c++) data[i+c]=data[i+c]<=.04045?data[i+c]/12.92:((data[i+c]+.055)/1.055)**2.4;
  const buffer=new THREE.InterleavedBuffer(data,6), geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.InterleavedBufferAttribute(buffer,3,0));
  geometry.setAttribute('color',new THREE.InterleavedBufferAttribute(buffer,3,3));
  return new THREE.Points(geometry,new THREE.PointsMaterial({vertexColors:true,size:Number($('point-size').value),sizeAttenuation:false}));
}
async function readySidecar(jid,name) {
  let state=await api(`/api/jobs/${jid}/${name}`,{method:'POST'});
  while(state.status==='running') {
    await new Promise(resolve=>setTimeout(resolve,1000));
    if(job?.id!==jid) throw new Error('任务已切换');
    state=await api(`/api/jobs/${jid}/${name}`);
  }
  if(state.status!=='ready') throw new Error(state.message||`${name} 未就绪`);
  return state.result;
}
function pointEntry(jid,result,full) {
  const key=`${jid}:points:${full}`;
  return viewCache.get(key,async()=>{
    const root=new THREE.Group();root.rotation.x=Math.PI;root.visible=false;
    try {
      const points=pointsMesh(await readFloats(jid,full?result.full_preview:result.preview));root.add(points);
      const normal=await readySidecar(jid,'point-normals');
      const distance=await readySidecar(jid,'point-distances');
      const [angles,distances]=await Promise.all([readFloats(jid,full?normal.full:normal.preview),readFloats(jid,full?distance.full:distance.preview)]);
      if(angles.length!==points.geometry.attributes.position.count)throw new Error('点云 / 法向数量不匹配');
      const stats=distanceStatistics(angles,distances), db=new THREE.InterleavedBuffer(distances,DISTANCE_NEIGHBORS.length);
      points.geometry.setAttribute('normalAngle',new THREE.BufferAttribute(angles,1));
      points.geometry.setAttribute('meanNeighborDistance',new THREE.InterleavedBufferAttribute(db,1,DISTANCE_NEIGHBORS.indexOf(FIXED_DISTANCE_NEIGHBORS)));
      attachAngleShader(points.material,normalThreshold,distanceUniforms);
      const entry={root,points,angles,normalStats:angleStatistics(angles),distanceStats:stats,distanceBuffer:db,dense:null,full};
      const status=await api(`/api/jobs/${jid}/point-densify`);denseRecords.set(jid,status);
      if(status.status==='ready') {
        entry.dense=pointsMesh(await readFloats(jid,full?status.result.full:status.result.preview));
        entry.dense.visible=false;root.add(entry.dense);
      }
      scene.add(root);return entry;
    } catch(error) {root.traverse(n=>{if(n.isPoints){n.geometry.dispose();n.material.dispose();}});throw error;}
  });
}
function gaussianEntry(jid,step,result) {
  const info=result.densification;
  const key=`${jid}:gaussian:${step}:${info?.dense.sha256 || result.variants?.unmasked?.sha256 || result.sha256}`;
  for(const old of viewCache.entries.keys())if(old.startsWith(`${jid}:gaussian:${step}:`) && old!==key)viewCache.delete(old);
  return viewCache.get(key,async()=>{
    const {SparkRenderer,SplatMesh}=await import('@sparkjsdev/spark');
    if(!spark){spark=new SparkRenderer({renderer});scene.add(spark);}
    const root=new THREE.Group();root.rotation.x=Math.PI;root.visible=false;
    const meshes={};
    try {
      const parts=info?info.parts:(result.variants || {unmasked:result});
      // Sequential decoding bounds peak CPU memory; all decoded parts stay warm.
      for(const [name,part] of Object.entries(parts)) {
        if(!part.count) continue;
        const mesh=new SplatMesh({url:`/artifacts/${jid}/${part.preview}?v=${part.sha256||''}`,lod:false});
        meshes[name]=mesh;root.add(mesh);await mesh.initialized;
        mesh.splats.prepareFetchSplat();
        if(mesh.packedSplats) renderer.initTexture(mesh.packedSplats.getTexture());
      }
      scene.add(root);return {root,meshes,shared:!!info,counts:parts};
    } catch(error) {for(const mesh of Object.values(meshes))mesh.dispose();throw error;}
  });
}
function showGaussians(entry) {
  const keys=entry.shared?(gaussianDense?['common','children']:
    gaussianVariant==='masked'?['common']:['common','removed']):[gaussianVariant];
  for(const [name,mesh] of Object.entries(entry.meshes)) mesh.visible=keys.includes(name);
}
function ensureCacheJob() {
  if(cacheJob===job.id) return;
  disposeObject();viewCache.clear();imageCache.clear();cacheJob=job.id;
  denseRecords.clear();denseEnabled.clear();gaussianDense=false;
}
function warmImage(jid,name) {
  const key=`${jid}:${name}`;
  if(!imageCache.has(key)) {
    const image=new Image();image.id='result-image';image.alt='当前步骤结果';image.draggable=false;
    image.src=`/artifacts/${jid}/${name}`;
    imageCache.set(key,{image,ready:image.decode().then(()=>image)});
  }
  return imageCache.get(key).ready;
}
async function resolveDenseInfo(jid,step) {
  const result=job.stages[String(step)];
  if(result.densification?.version===3 || result.denseChecked) return;
  delete result.densification;
  const state=await api(`/api/jobs/${jid}/gaussian-densify/${step}`);
  if(job?.id!==jid) return;
  if(state.status==='ready' && state.result.version===3)result.densification=state.result;
  result.denseChecked=true;
}
async function preloadAll() {
  const jid=job?.id;
  if(automatic || !jid || preloadJob===jid || job.busy || gaussianDensePending) return;
  preloadJob=jid;
  try {
    init3D();
    for(const result of Object.values(job.stages))if(result.kind==='image') {
      await warmImage(jid,result.image);if(result.gray)await warmImage(jid,result.gray);
    }
    // Prioritize the other Gaussian step during comparisons; one task at a time.
    const order=selected>=6?[5]:selected>=4?[selected,selected===4?5:4,3]:[3,4,5];
    for(const step of order) {
      if(job?.id!==jid)return;
      const result=job.stages[String(step)];if(!result)continue;
      $('preload-status').textContent=`正在预加载：${stepName(step)}及对比版本…`;
      if(step===3){await pointEntry(jid,result,true);await pointEntry(jid,result,false);}
      else {await resolveDenseInfo(jid,step);await gaussianEntry(jid,step,result);}
    }
    if(job?.id===jid)$('preload-status').textContent='预加载完成 · 已完成步骤和对比版本驻留缓存；切换无需重新下载或解析。';
  } catch(error) {
    if(job?.id===jid)$('preload-status').textContent=`部分预加载未完成：${error.message}。点击目标步骤可重试。`;
  } finally {if(preloadJob===jid)preloadJob=null;}
}
function init3D() {
  if (renderer) return;
  renderer = new THREE.WebGLRenderer({antialias:false});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio,1.5));
  renderer.setClearColor(0x090c0c);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  $('three-view').append(renderer.domElement);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(75,1,.01,1000);
  camera.rotation.order = 'YXZ';
  new ResizeObserver(resize).observe($('viewport'));
  resize();
  renderer.setAnimationLoop(now => {
    const delta = Math.min((now-lastTime)/1000,.1); lastTime=now;
    if ($('three-view').hidden || novel.busy) return;
    if (document.activeElement === $('three-view')) {
      const speed=delta*(keys.has('Shift') ? 3 : 1);
      if(keys.has('w')||keys.has('W')) camera.translateZ(-speed);
      if(keys.has('s')||keys.has('S')) camera.translateZ(speed);
      if(keys.has('a')||keys.has('A')) camera.translateX(-speed);
      if(keys.has('d')||keys.has('D')) camera.translateX(speed);
      if(keys.has('q')||keys.has('Q')) camera.position.y-=speed;
      if(keys.has('e')||keys.has('E')) camera.position.y+=speed;
    }
    renderer.render(scene,camera);
    novel.tick();
  });
}
function resize() {
  if (!renderer) return;
  const {width,height}=$('viewport').getBoundingClientRect();
  renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();
}
function resetView() {
  imgScale=1;imgX=0;imgY=0;transformImage();
  if(camera) {const p=localStep(selected)===9?job?.stages[String(selected)]?.position_world_m:null;camera.position.set(p?.[0]||0,-(p?.[1]||0),-(p?.[2]||0));yaw=pitch=0;camera.rotation.set(0,0,0);camera.fov=75;camera.updateProjectionMatrix();$('fov').value=75;$('direction').value='front';}
}
function transformImage() {$('result-image').style.transform=`translate(${imgX}px,${imgY}px) scale(${imgScale})`;}

async function selectStep(step) {
  viewSelections++;
  try {return await loadSelectedStep(step);}
  finally {viewSelections--;}
}
async function loadSelectedStep(step) {
  if(novel.busy)return;
  if(!visibleStep(step,mode))step=latestVisibleStep(job,mode,step);
  if(step>=6)return selectNovelStep(step);
  if (!job?.stages[String(step)]) return;
  novel.hide();$('gaussian-repair-control').hidden=true;$('viewport').hidden=false;
  ensureCacheJob();
  const switchedAt=performance.now();
  setSelectedStep(step); gray=false;
  const serial=++viewSerial;
  if(step>=4 && job.stages[String(step)].densification?.version!==3 && !job.stages[String(step)].denseChecked) {
    try {await resolveDenseInfo(job.id,step);}catch(error){showError(error.message);}
    if(serial!==viewSerial)return;
  }
  const result=gaussianResult(job.stages[String(step)]);
  renderState();showError(job.error);
  $('title').textContent=stepName(step);$('step-index').textContent=`STEP ${String(displayStep(step,mode)).padStart(2,'0')}`;
  $('description').textContent=step===1?'原始 2:1 全景图。前六步在本机计算；第七步逐面补洞，每次仅向 Seedream 发送一张 1024×1024 待补图，不发送参考图。':result.description;metadata(result);
  $('empty').hidden=true;$('download').hidden=false;
  $('download').href=artifact(result.download)+'?download=1';
  $('download').textContent=`下载 ${result.download.split('.').pop().toUpperCase()} ↗`;
  $('view-tools').hidden=false;$('gray').hidden=step!==2;$('gray').textContent='切换灰度';
  $('point-control').hidden=result.kind!=='points';
  $('normal-control').hidden=result.kind!=='points';
  $('densify-control').hidden=result.kind!=='points';
  $('gaussian-mask-control').hidden=result.kind!=='gaussians';
  $('gaussian-densify-control').hidden=result.kind!=='gaussians';
  if(result.kind==='gaussians')updateGaussianDensePanel();
  if(result.kind==='points') {
    $('download').textContent='下载原始 LAS ↗';
    $('distance-retry').hidden=true;
    $('normal-retry').hidden=true;
    $('normal-status').textContent='正在载入点云…';
    $('distance-status').textContent='等待法向筛选完成…';
  }
  $('image-view').hidden=result.kind!=='image';$('three-view').hidden=result.kind==='image';
  $('view-loading').hidden=true;
  $('compare-tools').hidden=result.kind==='image';
  if($('debug-panel').open) refreshDebug();
  disposeObject();
  if(result.kind==='points') updatePointMask();
  if(result.kind==='image') {
    imgScale=1;imgX=imgY=0;transformImage();
    try {
      const image=await warmImage(job.id,result.image);
      if(serial!==viewSerial)return;
      if($('result-image')!==image)$('result-image').replaceWith(image);
      transformImage();
    } catch(error){if(serial===viewSerial)showError(`图片载入失败：${error.message}`);}
    $('view-hint').textContent='滚轮放大 · 拖动查看细节';
    lastSwitch={step,milliseconds:performance.now()-switchedAt};void preloadAll();return;
  }
  $('view-loading').hidden=false;$('view-loading').textContent='正在载入完整结果…';
  if(result.kind==='gaussians' && result.count>0) {
    $('mask-before').disabled=true;$('mask-after').disabled=true;
  }
  $('view-hint').textContent='左键拖动环视 · 右键平移 · 滚轮前后移动 · 点击画面后 WASD / QE 移动';
  try {
    init3D();resize();
    if(result.kind==='points') {
      pointPreviewFull=$('point-density').value==='full' && !!result.full_preview;
      const entry=await pointEntry(job.id,result,pointPreviewFull);
      if(serial!==viewSerial) return;
      activeEntry=entry;entry.root.visible=true;object=entry.points;pointMaterial=object.material;
      pointMaterial.size=Number($('point-size').value);
      normalStats=entry.normalStats;normalAngles=entry.angles;distanceStats=entry.distanceStats;distanceBuffer=entry.distanceBuffer;
      denseObject=entry.dense;if(denseObject)denseObject.material.size=pointMaterial.size;
      updateNormalStatus();updatePointMask();
      if(denseEnabled.has(job.id) && !denseObject)void loadDensePoints(job.id,serial,pointPreviewFull);
    } else {
      const entry=await gaussianEntry(job.id,step,job.stages[String(step)]);
      if(serial!==viewSerial)return;
      activeEntry=entry;showGaussians(entry);entry.root.visible=true;object=entry.root;
    }
    $('view-loading').hidden=true;
    lastSwitch={step,milliseconds:performance.now()-switchedAt};
    void preloadAll();
  } catch(error) {
    if(serial===viewSerial) {$('view-loading').hidden=false;$('view-loading').textContent='三维显示失败，详情见下方错误信息。结果仍可下载。';showError(`三维显示失败：${error.message}。结果仍可下载；请使用支持 WebGL2 的 Chrome / Edge。`);}
  } finally {
    if(serial===viewSerial && result.kind==='points') updateDensifyPanel();
    if(serial===viewSerial && result.kind==='gaussians') {
      const ready=!!(result.variants?.masked && result.variants?.unmasked);
      $('mask-before').disabled=!ready;$('mask-after').disabled=!ready;
    }
  }
}

async function selectNovelStep(step){
  const phase=localStep(step),stages=roundStages(step),jid=executionJob(step);
  if(phase===9)return selectGaussianRepair(step);
  $('gaussian-repair-control').hidden=true;
  if(!stages[6] || (phase>=7&&!job.stages[String(step)]&&job.running_stage!==step&&!(step===7&&job.mask_update_pending)))return;
  ensureCacheJob();setSelectedStep(step);const serial=++viewSerial;novel.hide();disposeObject();renderState();showError(job.error);
  $('title').textContent=stepName(step);$('step-index').textContent=`STEP ${String(displayStep(step,mode)).padStart(2,'0')}`;
  for(const id of ['empty','image-view','normal-control','densify-control','gaussian-mask-control','gaussian-densify-control','point-control','gray'])$(id).hidden=true;
  $('view-loading').hidden=true;$('viewport').hidden=phase>=7;$('three-view').hidden=phase!==6;
  $('compare-tools').hidden=phase!==6;$('view-tools').hidden=phase!==6;$('download').hidden=true;
  $('description').textContent=roundDescription(phase===8?(stages[8]?.description||'在本轮确认的相机位置渲染高斯深度，用补好的 RGB 引导 LingBot。'):
    phase===6?stages[6].description:(stages[7]?.description||'正在补洞。Mask 内完整采用新图，仅向外羽化；外侧过渡带之外保持原图。'),step);
  $('metrics').textContent='';$('preload-status').textContent='';
  try{
    if(phase===8){
      await novel.showDepth(jid,stages[8],stages[7]);if(serial!==viewSerial)return;
      if(stages[8]){$('download').hidden=false;$('download').href=novel.url(stages[8].download)+'?download=1';$('download').textContent='下载深度数据 NPZ ↗';}
      return;
    }
    if(phase===7){
      await novel.showRepair(jid,stages[7]);
      if(serial!==viewSerial)return;
      if(stages[7]){$('download').hidden=false;$('download').href=novel.url((novel.repairResult||stages[7]).image)+'?download=1';$('download').textContent='下载当前补洞全景 PNG ↗';}
      return;
    }
    init3D();resize();$('view-loading').hidden=false;$('view-loading').textContent='载入同一高斯场景和候选相机…';
    const state=await api(`/api/jobs/${jid}/novel`);
    if(step<10)await resolveDenseInfo(job.id,5);
    const source=step>=10?{sha256:state.source.sha256,variants:{masked:{preview:state.source.file,count:state.source.count,sha256:state.source.sha256}}}:job.stages['5'];
    const entry=await gaussianEntry(jid,step>=10?10:5,source);if(serial!==viewSerial)return;
    gaussianVariant=state.source.variant==='unmasked'?'unmasked':'masked';gaussianDense=state.source.variant==='dense';
    activeEntry=entry;showGaussians(entry);entry.root.visible=true;object=entry.root;
    await novel.show(jid,state);$('view-loading').hidden=true;
    $('view-hint').textContent='点击相机编号现场渲染 · 拖动环视 · 右键平移 · WASD / QE 移动';
  }catch(error){$('view-loading').hidden=true;showError(error.message);}
}

function repairDisplayParts(currentJob,step){
  // Each merge only appends rows, so display the first source plus all saved patches.
  // Keep the initial scene separate without loading a duplicate previous-round PLY.
  const first=currentJob.stages['9'];
  if(!first?.source)throw new Error('首次高斯来源缺失，无法显示原始对照');
  const original={count:first.base_count,url:`/artifacts/${first.execution_job||currentJob.id}/${first.source.file}`};
  const parts=[{name:'base',url:original.url}];
  let count=original.count;
  for(let stage=9;stage<=step;stage+=4){
    const result=currentJob.stages[String(stage)];
    if(!result||result.base_count!==count||result.count!==count+result.added_count)
      throw new Error('历轮补丁与合并数量不一致，无法显示累计对照');
    const jid=result.execution_job||currentJob.stages[String(stage-3)]?.execution_job||currentJob.id;
    parts.push({name:stage===step?'patch':`history-${stage}`,url:`/novel-artifacts/${jid}/${result.revision}/${result.patch}`});
    count=result.count;
  }
  return {original,parts,key:`${first.source.sha256}:${parts.map(part=>part.url).join('|')}`};
}

function showRepairMode(){
  const result=job?.stages[String(selected)];if(localStep(selected)!==9||!result||!activeEntry?.repair)return;
  const original=activeEntry.original;
  for(const [name,mesh] of Object.entries(activeEntry.meshes))mesh.visible=repairMode==='merged'||name===repairMode;
  for(const mode of ['base','patch','merged'])$(`repair-${mode}`).setAttribute('aria-pressed',String(mode===repairMode));
  const count=repairMode==='base'?original.count:repairMode==='patch'?result.added_count:result.count;
  $('gaussian-repair-status').textContent=`当前显示 ${count.toLocaleString()} 个高斯 · 首次生成 ${original.count.toLocaleString()} + 累计新增 ${(result.count-original.count).toLocaleString()}（本轮 ${result.added_count.toLocaleString()}）· 首次高斯逐字节保留`;
  $('download').href=(repairMode==='base'?original.url:novel.url(repairMode==='patch'?result.patch:result.download))+'?download=1';
  $('download').textContent=`下载${repairMode==='base'?'原始':repairMode==='patch'?'补丁':'合并后'} PLY ↗`;
}
for(const mode of ['base','patch','merged'])$(`repair-${mode}`).onclick=()=>{repairMode=mode;showRepairMode();};

async function selectGaussianRepair(step=9){
  const stages=roundStages(step),jid=executionJob(step);
  if(!stages[8]||(!stages[9]&&job.running_stage!==step))return;
  ensureCacheJob();setSelectedStep(step);const serial=++viewSerial;novel.hide();disposeObject();renderState();showError(job.error);
  const result=stages[9];
  $('title').textContent=stepName(step);$('step-index').textContent=`STEP ${String(displayStep(step,mode)).padStart(2,'0')}`;
  for(const id of ['empty','image-view','normal-control','densify-control','gaussian-mask-control','gaussian-densify-control','point-control','gray'])$(id).hidden=true;
  $('viewport').hidden=!result;$('three-view').hidden=!result;$('compare-tools').hidden=!result;$('view-tools').hidden=!result;
  $('gaussian-repair-control').hidden=!result;$('download').hidden=!result;$('view-loading').hidden=true;
  $('description').textContent=roundDescription(result?.description||'正在用当前 RGB 和 LingBot 深度运行参考 SHARP 补洞流程；原始高斯保留。',step);
  $('metrics').textContent='';$('preload-status').textContent='';
  if(automatic){repairMode='merged';novel.jid=jid;novel.state={revision:result?.revision};}
  else novel.showGaussianRepair(jid,result,stages[8]);
  if(!result)return;
  metadata(result);$('view-loading').hidden=false;$('view-loading').textContent='载入首次高斯与历轮补丁…';
  for(const mode of ['base','patch','merged'])$(`repair-${mode}`).disabled=true;
  try{
    init3D();resize();
    const display=repairDisplayParts(job,step),key=`${job.id}:repair:${display.key}`;
    const entry=await viewCache.get(key,async()=>{
      const {SparkRenderer,SplatMesh}=await import('@sparkjsdev/spark');
      if(!spark){spark=new SparkRenderer({renderer});scene.add(spark);}
      const root=new THREE.Group();root.rotation.x=Math.PI;root.visible=false;const meshes={};
      try{
        for(const {name,url} of display.parts){
          const mesh=new SplatMesh({url,lod:false});meshes[name]=mesh;root.add(mesh);await mesh.initialized;
          mesh.splats.prepareFetchSplat();if(mesh.packedSplats)renderer.initTexture(mesh.packedSplats.getTexture());
        }
        scene.add(root);return {root,meshes,repair:true,original:display.original};
      }catch(error){for(const mesh of Object.values(meshes))mesh.dispose();throw error;}
    });
    if(serial!==viewSerial)return;
    activeEntry=entry;object=entry.root;entry.root.visible=true;showRepairMode();resetView();
    $('view-hint').textContent='同一视角切换首次高斯 / 本轮补丁 / 累计合并 · 拖动环视 · WASD / QE 移动';
    for(const mode of ['base','patch','merged'])$(`repair-${mode}`).disabled=false;
  }catch(error){if(serial===viewSerial)showError(error.message);}
  finally{if(serial===viewSerial)$('view-loading').hidden=true;}
}

async function upload(file) {
  if(!file||advancing||qualityResuming||job?.busy||novel.busy||loadingUpload||denseRequests.size>0||gaussianDensePending) return;
  if(file.size>80*1024*1024) return showError('图片不能超过 80 MB');
  loadingUpload=true;renderState();showError(null);$('upload-status').textContent='正在上传并检查全景图…';
  try {
    job=await api('/api/upload',{method:'POST',body:file,headers:{'Content-Type':file.type||'application/octet-stream','X-Workflow-Mode':mode}});
    automaticFailed=null;qualityResumeFailed=null;savedPose=null;$('restore-pose').disabled=true;resetView();await selectStep(1);$('upload-status').textContent=file.name;
  } catch(error) {showError(error.message);$('upload-status').textContent='上传失败，请重新选择图片';return;}
  finally {loadingUpload=false;renderState();$('file').value='';}
  await advanceAutomatically();
}
$('file').addEventListener('change',event=>upload(event.target.files[0]));
for(const type of ['dragenter','dragover']) $('drop').addEventListener(type,event=>{event.preventDefault();$('drop').classList.add('over');});
for(const type of ['dragleave','drop']) $('drop').addEventListener(type,event=>{event.preventDefault();$('drop').classList.remove('over');});
$('drop').addEventListener('drop',event=>upload(event.dataTransfer.files[0]));
async function advanceStep(expectedStage=null){
  if(!job||job.busy||advancing) return;
  if(job.quality_run?.status==='awaiting_capture'){qualityResumeFailed=null;await resumeQualityCapture();return;}
  advancing=true;renderState();showError(null);
  try {
    const next=expectedStage??Object.keys(job.stages).length+1;
    if(localStep(next)===7){
      if(automatic)await prepareAutomaticCapture(next-1);
      else {if(selected!==next-1)await selectStep(next-1);await novel.ensureRecommended();}
    }
    job=await api(`/api/jobs/${job.id}/next`,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({expected_stage:next,gaussian_variant:automatic?'masked':gaussianDense?'dense':gaussianVariant})});renderState();
    if(automatic){
      // The completed result remains visible while the next stage runs.
      viewCache.retain(activeEntry);
      if(!job.busy&&visibleStep(Object.keys(job.stages).length,mode))await selectStep(Object.keys(job.stages).length);
      return;
    }
    if(localStep(next)===8||localStep(next)===9){disposeObject();viewCache.clear();await selectStep(next);return;}
    if(localStep(next)===7){await selectStep(next);return;}
    if(localStep(next)===6)return;
    // Do not keep multiple GPU previews resident while the model is generating.
    disposeObject();viewCache.clear();
    $('preload-status').textContent='模型生成期间已释放三维缓存，完成后自动重新预加载。';
  }
  catch(error) {
    if(automatic&&error.status===409){
      try {job=await api(`/api/jobs/${job.id}`);}
      catch(refreshError){automaticFailed=job.id;showError(refreshError.message);}
    }
    else {if(automatic)automaticFailed=job.id;showError(error.message);}
  } finally {advancing=false;renderState();}
}
$('next').onclick=()=>advanceStep();
async function advanceAutomatically(){
  if(!automatic||viewSelections||advancing||polling||qualityResuming||loadingUpload||automaticFailed===job?.id)return;
  const next=nextAutomaticStage(job);if(next!==null)await advanceStep(next);
}
async function prepareAutomaticCapture(step){
  const jid=executionJob(step),previousStep=selected;
  const pose=camera?{position:camera.position.clone(),yaw,pitch,fov:camera.fov}:null;
  ensureCacheJob();++viewSerial;disposeObject();
  document.body.classList.add('auto-processing');$('viewport').hidden=false;
  $('view-loading').textContent=`正在采集第 ${repairRound(step)} 轮修补视图…`;
  try {
    init3D();resize();
    const state=await api(`/api/jobs/${jid}/novel`);
    if(step<10)await resolveDenseInfo(job.id,5);
    const source=step>=10?{sha256:state.source.sha256,variants:{masked:{preview:state.source.file,count:state.source.count,sha256:state.source.sha256}}}:job.stages['5'];
    const entry=await gaussianEntry(jid,step>=10?10:5,source);
    gaussianVariant=state.source.variant==='unmasked'?'unmasked':'masked';gaussianDense=state.source.variant==='dense';
    activeEntry=entry;showGaussians(entry);entry.root.visible=true;object=entry.root;
    await novel.show(jid,state);
    await novel.ensureRecommended();
    return state;
  } finally {
    novel.hide();disposeObject();
    await selectStep(previousStep);
    if(pose&&camera){camera.position.copy(pose.position);yaw=pose.yaw;pitch=pose.pitch;camera.rotation.set(pitch,yaw,0);camera.fov=pose.fov;camera.updateProjectionMatrix();}
    viewCache.retain(activeEntry);
  }
}
$('reset').onclick=resetView;
$('gray').onclick=async()=>{
  gray=!gray;const expected=gray,serial=viewSerial;
  $('gray').textContent=gray?'切换彩色':'切换灰度';const result=job.stages['2'];
  try {
    const image=await warmImage(job.id,gray?result.gray:result.image);
    if(serial!==viewSerial || gray!==expected)return;
    if($('result-image')!==image)$('result-image').replaceWith(image);transformImage();
  }catch(error){showError(`图片载入失败：${error.message}`);}
};
$('point-size').oninput=()=>{const size=Number($('point-size').value);if(pointMaterial) pointMaterial.size=size;if(denseObject) denseObject.material.size=size;};
$('point-density').onchange=()=>{if(selected===3) selectStep(3);};
$('point-mask-before').onclick=()=>{pointMasked=false;denseEnabled.delete(job?.id);updatePointMask();};
$('point-mask-after').onclick=()=>{pointMasked=true;updatePointMask();};
$('densify-before').onclick=()=>{denseEnabled.delete(job.id);pointMasked=true;updatePointMask();};
$('densify-after').onclick=()=>{denseEnabled.add(job.id);pointMasked=true;updatePointMask();void loadDensePoints(job.id,viewSerial,pointPreviewFull);};
for(const [id,variant] of [['mask-before','unmasked'],['mask-after','masked']]) $(id).onclick=()=>{
  if(gaussianVariant===variant) return;
  gaussianVariant=variant;if(variant==='unmasked')gaussianDense=false;void selectStep(selected);
};
for(const [id,enabled] of [['gaussian-dense-before',false],['gaussian-dense-after',true]]) $(id).onclick=()=>{
  gaussianVariant='masked';gaussianDense=enabled;void selectStep(selected);
};
$('gaussian-dense-build').onclick=async()=>{
  if(!job || job.busy || gaussianDensePending)return;
  const jid=job.id,step=selected;gaussianDensePending=true;renderState();updateGaussianDensePanel();
  disposeObject();viewCache.clear();$('view-loading').hidden=false;$('view-loading').textContent='生成期间释放预览缓存，完成后重新预加载…';
  try {
    let state=await api(`/api/jobs/${jid}/gaussian-densify/${step}`,{method:'POST'});
    while(state.status==='running') {
      $('gaussian-dense-status').textContent=state.message||'正在生成加密高斯…';
      await new Promise(resolve=>setTimeout(resolve,1200));
      state=await api(`/api/jobs/${jid}/gaussian-densify/${step}`);
    }
    if(state.status!=='ready')throw new Error(state.message||'加密生成失败');
    if(job?.id===jid)job.stages[String(step)].densification=state.result;
  }catch(error){showError(error.message);}
  finally {
    gaussianDensePending=false;renderState();
    if(job?.id===jid){await selectStep(selected);void preloadAll();}
  }
};
$('mask-build').onclick=async()=>{
  if(!job || job.busy) return;
  $('mask-build').disabled=true;showError(null);
  try {job=await api(`/api/jobs/${job.id}/gaussian-mask/${selected}`,{method:'POST'});renderState();if(!job.busy) await selectStep(selected);}
  catch(error) {showError(error.message);renderState();}
};
$('distance-retry').onclick=()=>{if(selected===3 && normalStats && !distancePending) void loadPointDistances(job.id,viewSerial,pointPreviewFull);};
$('normal-retry').onclick=()=>{if(selected===3 && object?.isPoints && !normalPending) void loadPointNormals(job.id,viewSerial,pointPreviewFull);};
for(const element of [$('image-view'),$('three-view')]) {
  element.addEventListener('contextmenu',e=>e.preventDefault());
  element.addEventListener('pointerdown',e=>{drag={element,x:e.clientX,y:e.clientY,button:e.button};element.setPointerCapture(e.pointerId);element.focus();});
  element.addEventListener('pointerup',()=>drag=null);
  element.addEventListener('pointercancel',()=>drag=null);
  element.addEventListener('pointermove',e=>{
    if(!drag||drag.element!==element||novel.busy) return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
    if(element.id==='image-view') {imgX+=dx;imgY+=dy;transformImage();}
    else if(camera) {
      if(drag.button===2) {camera.translateX(-dx*.003);camera.translateY(dy*.003);}
      else {yaw-=dx*.003;pitch=Math.max(-1.55,Math.min(1.55,pitch-dy*.003));camera.rotation.set(pitch,yaw,0);}
    }
  });
  element.addEventListener('wheel',e=>{
    e.preventDefault();
    if(novel.busy)return;
    if(element.id==='image-view') {imgScale=Math.max(1,Math.min(12,imgScale*Math.exp(-e.deltaY*.001)));transformImage();}
    else if(camera) camera.translateZ(Math.max(-.5,Math.min(.5,e.deltaY*.002)));
  },{passive:false});
}
$('three-view').addEventListener('keydown',e=>{if('wasdqeWASDQE'.includes(e.key)||e.key==='Shift') {keys.add(e.key);e.preventDefault();}});
window.addEventListener('keyup',e=>keys.delete(e.key));window.addEventListener('blur',()=>keys.clear());
$('three-view').addEventListener('blur',()=>keys.clear());
async function refreshDebug(stage=selected) {
  if(!job) return;
  try {
    debugData=await api(`/api/jobs/${job.id}/debug/${stage}`);
    debugData.stage=stage;
    $('debug-data').textContent=`STEP ${stage} · ${stepName(stage)}\n\n`+JSON.stringify({pipeline:debugData.pipeline,result:debugData.result,...(stage===3?{densification:debugData.densification}:{}),...(stage>=4?{densification:debugData.gaussian_densification}:{})},null,2)+'\n\n运行日志\n'+debugData.log+(stage===3?'\n\n加密预览日志\n'+debugData.densification_log:'');
  } catch(error) {$('debug-data').textContent=error.message;}
}
$('debug-panel').addEventListener('toggle',()=>{if($('debug-panel').open) refreshDebug();});
$('refresh-debug').onclick=()=>refreshDebug();
$('running-debug').onclick=()=>refreshDebug(job?.running_stage||selected);
$('download-debug').onclick=async()=>{
  await refreshDebug();if(!debugData) return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(debugData,null,2)],{type:'application/json'}));
  const link=document.createElement('a');link.href=url;link.download=`sharp-stage-${debugData.stage}-debug.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
};
$('direction').onchange=()=>{
  if(!camera) return;
  const preset={front:[0,0],right:[-Math.PI/2,0],back:[Math.PI,0],left:[Math.PI/2,0],up:[0,Math.PI/2],down:[0,-Math.PI/2]};
  [yaw,pitch]=preset[$('direction').value];camera.rotation.set(pitch,yaw,0);
};
$('fov').onchange=()=>{if(camera){camera.fov=Math.max(30,Math.min(120,Number($('fov').value)||75));$('fov').value=camera.fov;camera.updateProjectionMatrix();}};
$('save-pose').onclick=()=>{if(camera){savedPose={position:camera.position.clone(),yaw,pitch,fov:camera.fov};$('restore-pose').disabled=false;}};
$('restore-pose').onclick=()=>{if(camera&&savedPose){camera.position.copy(savedPose.position);yaw=savedPose.yaw;pitch=savedPose.pitch;camera.rotation.set(pitch,yaw,0);camera.fov=savedPose.fov;$('fov').value=camera.fov;camera.updateProjectionMatrix();}};
$('compare').onclick=()=>selectStep(selected===4?5:4);
// Read-only diagnostics for browser devtools and reproducible UI checks.
window.sharpDebug=()=>({step:selected,camera:camera?{position:camera.position.toArray(),yaw,pitch,fov:camera.fov}:null,
  novel:novel.debug(),
  cache:{ready:viewCache.ready,entries:viewCache.entries.size,preloading:preloadJob,lastSwitch},
  objectType:object?.type,points:object?.geometry?.attributes?.position?.count,
  densification:selected===3?{enabled:denseEnabled.has(job?.id),visible:denseShown(),loading:denseLoadingSerial===viewSerial,
    addedLoaded:denseObject?.geometry?.attributes?.position?.count??0,addedVisible:denseShown()?denseObject.geometry.attributes.position.count:0,
    pointSize:denseObject?.material?.size,state:denseRecords.get(job?.id),downstream:true}:null,
  gaussianDensification:selected>=4&&selected<=5?{enabled:gaussianDense,ready:!!job.stages[String(selected)]?.densification,
    visibleParts:activeEntry?.meshes?Object.entries(activeEntry.meshes).filter(([,m])=>m.visible).map(([k])=>k):[],
    info:job.stages[String(selected)]?.densification}:null,
  pointMask:selected===3?{variant:pointMasked?'masked':'unmasked',visible:normalStats?(pointMasked?(distanceStats?distanceCounts(distanceStats,DISTANCE_NEIGHBORS.indexOf(FIXED_DISTANCE_NEIGHBORS),FIXED_DISTANCE_CM).visible:visibleAt(normalStats,FIXED_NORMAL_ANGLE)):normalStats.total):null}:null,
  normalFilter:selected===3?{ready:!!normalStats,loading:normalPending,threshold:FIXED_NORMAL_ANGLE,enabled:pointMasked,
    visible:normalStats?visibleAt(normalStats,normalThreshold.value):null,
    hidden:normalStats?normalStats.total-visibleAt(normalStats,normalThreshold.value):null,
    unreliable:normalStats?.unreliable,downstreamMask:true}:null,
  distanceFilter:selected===3?{ready:!!distanceStats,loading:distancePending,
    neighbors:FIXED_DISTANCE_NEIGHBORS,centimeters:FIXED_DISTANCE_CM,
    enabled:distanceUniforms.enabled.value===1,iterations:1,downstreamMask:true,
    ...(distanceStats?(pointMasked?distanceCounts(distanceStats,DISTANCE_NEIGHBORS.indexOf(FIXED_DISTANCE_NEIGHBORS),FIXED_DISTANCE_CM):{visible:distanceStats.total,hidden:0,unknownRemoved:0}):{})}:null,
  gaussianMask:selected>=4&&selected<=5?{variant:job.stages[String(selected)]?.variants?gaussianVariant:'unmasked',
    download:$('download').getAttribute('href'),details:job.stages[String(selected)]?.mask}:null,
  splats:activeEntry?.meshes?Object.values(activeEntry.meshes).filter(m=>m.visible).reduce((n,m)=>n+m.splats.getNumSplats(),0):null,render:renderer?.info?.render,
  loading:!$('view-loading').hidden,error:$('error').hidden?null:$('error').textContent});
let qualityResuming=false,qualityResumeFailed=null;
async function resumeQualityCapture(){
  const run=job?.quality_run;
  if(!run||run.status!=='awaiting_capture'||qualityResuming||qualityResumeFailed===run.pending_id)return;
  qualityResuming=true;const jid=job.id,attempt=run.pending_id;
  try{
    const captureState=automatic?await prepareAutomaticCapture(run.round_start):null;
    if(!automatic)await selectStep(run.round_start);
    if(job?.id!==jid||job.quality_run?.pending_id!==attempt)return;
    if((captureState||novel.state)?.source?.sha256!==run.source_sha256||(captureState||novel.state)?.highlighted!==run.pending_camera)throw new Error('待采集相机或场景已改变');
    if(!automatic)await novel.ensureRecommended();
    job=await api(`/api/jobs/${jid}/quality/resume`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({attempt_id:attempt})});
    if(automatic)viewCache.retain(activeEntry);
    else {disposeObject();viewCache.clear();await selectStep(run.next_stage);}
    renderState();
  }catch(error){qualityResumeFailed=attempt;if(automatic)automaticFailed=jid;showError('自动采集暂停：'+error.message+(automatic?'；请切换调试模式查看。':'；可点击继续重试采集。'));}
  finally{qualityResuming=false;renderState();}
}
setInterval(async()=>{
  if(initializing||polling||qualityResuming||advancing||loadingUpload)return;
  if(!job?.busy&&job?.quality_run?.status!=='awaiting_capture'){await advanceAutomatically();return;}
  polling=true;
  const jid=job.id, previous=Object.keys(job.stages).length;
  const previousSelected=JSON.stringify(job.stages[String(selected)]);
  try {
    const updated=await api(`/api/jobs/${jid}`);
    if(job.id!==jid) return;
    job=updated;renderState();showError(job.error);
    if(job.quality_run?.status==='awaiting_capture'&&(!automatic||job.execution_mode==='auto')){await resumeQualityCapture();return;}
    const completed=Object.keys(job.stages).length;
    if(automatic&&!job.busy&&(job.error||['exhausted','cancelled','error'].includes(job.quality_run?.status)))await selectStep(latestVisibleStep(job,mode));
    else if(completed>previous&&visibleStep(completed,mode)) await selectStep(completed);
    else if(previousSelected!==JSON.stringify(job.stages[String(selected)])) await selectStep(selected);
  } catch(error) {showError(`获取进度失败：${error.message}`);}
  finally {polling=false;}
  await advanceAutomatically();
},1200);
try {
  const state=await api('/api/state');token=state.token;job=state.job;
  const requestedJob=new URLSearchParams(location.search).get('job');
  if(requestedJob){if(!/^[a-f0-9]{32}$/.test(requestedJob))throw new Error('无效流程编号');job=await api(`/api/jobs/${requestedJob}`);}
  $('health').textContent=state.health.depthmaster&&state.health.sharp?'本机模型已就绪 · 仅本地访问':'模型或环境缺失，请检查安装';
  renderState();if(job) {
    const requested=Number(new URLSearchParams(location.search).get('step'));
    await selectStep(job.stages[String(requested)]||job.running_stage===requested||(requested===7&&job.mask_update_pending)?requested:Object.keys(job.stages).length);
  }
  await advanceAutomatically();
} catch(error) {showError(`连接本机服务失败：${error.message}`);}
finally {initializing=false;}
