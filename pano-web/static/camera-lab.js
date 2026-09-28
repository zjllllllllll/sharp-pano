import * as THREE from 'three';
import {maskFraction,filterState,summarize,MASK_THRESHOLD,SELECTION_VERSION} from './camera-lab-filter.mjs';

const $=id=>document.getElementById(id);
const elements={scene:$('lab-scene'),camera:$('lab-camera'),map:$('lab-map'),pano:$('lab-panorama'),
  mapEmpty:$('lab-map-empty'),previewEmpty:$('lab-preview-empty'),loading:$('lab-loading'),error:$('lab-error')};
let catalog=null,scene=null,camera=null,mapMode='camera_map',mode='flat',yaw=0,pitch=0,requestId=0;
let renderVariant='masked';
let renderer=null,material=null,threeScene=null,drawCamera=null,texture=null;
const textures=new Map();
const url=relative=>'/camera-lab-artifacts/'+relative.split('/').map(encodeURIComponent).join('/')+'?v='+encodeURIComponent((catalog?.revision||catalog?.run||'legacy')+'-'+(scene?.sample?.render_fingerprint?.implementation||''));
const sceneStamp=s=>s?.render_complete+':'+(s?.cameras||[]).map(c=>c.id+Boolean(c.panorama)).join(',')+'@'+(s?.sample?.render_fingerprint?.implementation||'')+'@'+Boolean(s?.camera_map_base);
const query=new URLSearchParams(location.search);
let wantedScene=query.get('scene'),wantedCamera=query.get('camera');
const allCameras=()=>scene?.sample?.all_cameras||scene?.cameras||[];
const activeCameras=()=>summarize(allCameras()).kept;
const cameraState=c=>filterState(c,allCameras());
const stateLabels={origin:'原相机',retained:'候选保留',excluded:'Mask 超过 15%，已排除',disconnected:'与原点不连通，已排除',pending:'待渲染'};
const mapColors={origin:'#55bfff',retained:'#b9f66f',excluded:'#ff7777',disconnected:'#ffb55a',pending:'#a4a9aa'};
const svgNS='http://www.w3.org/2000/svg';
function svgElement(name,attrs={}){const node=document.createElementNS(svgNS,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);return node;}
function saveBlob(blob,name){const href=URL.createObjectURL(blob),link=document.createElement('a');link.href=href;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(href),10000);}

function drawCandidateMap(){
  const svg=$('lab-map-interactive'),layout=scene?.sample?.camera_map_projection;
  svg.replaceChildren();if(!layout||!scene.camera_map_base)return;
  const [w,h]=layout.image_size,[left,top,width,height]=layout.axes_px;
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`);
  svg.append(svgElement('image',{href:url(scene.camera_map_base),width:w,height:h}));
  for(const item of allCameras()){
    const [x,,z]=item.position_world_m;
    const px=left+(x-layout.xlim[0])/(layout.xlim[1]-layout.xlim[0])*width;
    const py=top+(layout.zlim[1]-z)/(layout.zlim[1]-layout.zlim[0])*height;
    const g=svgElement('g',{'data-map-camera':item.id,transform:`translate(${px},${py})`,tabindex:0,role:'button'});
    const radius=Math.max(6,w/380);
    g.append(svgElement('circle',{r:radius+8,fill:'transparent',class:'map-hit'}));
    g.append(svgElement('circle',{r:radius+5,fill:'none','stroke-width':3,class:'map-selected'}));
    g.append(svgElement('circle',{r:radius,class:'map-dot',stroke:'#101716','stroke-width':1.5}));
    const cross=svgElement('path',{d:`M${-radius},${-radius} L${radius},${radius} M${-radius},${radius} L${radius},${-radius}`,'stroke-width':3,class:'map-cross'});g.append(cross);
    const label=svgElement('text',{x:radius+5,y:-radius-3,'font-size':Math.max(18,w/115),'paint-order':'stroke',stroke:'#101716','stroke-width':3});label.textContent=item.id;g.append(label);
    g.append(svgElement('title'));
    g.onclick=()=>chooseCamera(item.id);
    g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseCamera(item.id);}};
    svg.append(g);
  }
  updateMapPoints();
}
function updateMapPoints(){
  const byID=new Map(allCameras().map(c=>[c.id,c]));
  for(const g of $('lab-map-interactive').querySelectorAll('[data-map-camera]')){
    const item=byID.get(g.dataset.mapCamera),state=cameraState(item),color=mapColors[state],removed=['excluded','disconnected'].includes(state);
    g.dataset.filterState=state;g.querySelector('.map-dot').setAttribute('fill',color);
    g.querySelector('.map-dot').style.display=removed?'none':'';
    g.querySelector('.map-cross').setAttribute('stroke',color);g.querySelector('.map-cross').style.display=removed?'':'none';
    g.querySelector('text').setAttribute('fill',color);
    const ring=g.querySelector('.map-selected');ring.setAttribute('stroke','#ffffff');ring.style.display=item.id===camera?.id?'':'none';
    const fraction=maskFraction(item),label=`${item.id} · ${stateLabels[state]} · Mask ${fraction===null?'待计算':(fraction*100).toFixed(2)+'%'}`;
    g.setAttribute('aria-label',label);g.querySelector('title').textContent=label;
  }
}
function renderCameraChoices(){
  const cams=activeCameras();elements.camera.replaceChildren();$('lab-camera-list').replaceChildren();
  elements.camera.disabled=!cams.length;
  for(const item of cams){
    elements.camera.add(new Option(item.id+(item.panorama?'':' · 待渲染'),item.id));
    const button=document.createElement('button');button.textContent=item.id;button.dataset.camera=item.id;
    button.setAttribute('aria-pressed','false');button.onclick=()=>chooseCamera(item.id);$('lab-camera-list').append(button);
  }
}
function updateFilter({restore=null}={}){
  if(!scene)return;
  const result=summarize(allCameras()),sample=scene.sample;
  $('lab-filter-summary').textContent=`几何点 ${allCameras().length} · 最终候选 ${result.kept.length}（含原相机） · Mask 超限 ${result.excluded.length} · 不连通 ${result.disconnected.length} · 待计算 ${result.pending.length}`;
  $('lab-filter-summary').dataset.retained=result.kept.length;$('lab-filter-summary').dataset.excluded=result.excluded.length;
  $('lab-sampling-info').textContent=sample?`0.5 m 点阵 / 原相机同高 · offset ${sample.requested_offset_m??sample.inward_offset_m} → ${sample.inward_offset_m} m · 原相机八邻域：几何候选 ${sample.neighbor_count??'—'}/8，固定筛选后 ${result.neighbors}/8。${sample.sampling_warning||''}`:'等待采样结果。';
  const error=sample?.sampling_error||sample?.selection_error||(result.onlyOrigin?'候选采样失败：Mask ≤15% 及四邻接筛选后仅剩原相机 C000。请检查深度、Mask 或轮廓；固定规则不会自动放宽。':'');
  $('lab-sampling-error').hidden=!error;$('lab-sampling-error').textContent=error;
  const oldID=restore||camera?.id;
  renderCameraChoices();updateMapPoints();
  if(!result.kept.length){clearPreview('没有已完成渲染并通过阈值的相机。');return;}
  const next=allCameras().find(c=>c.id===oldID);
  if(restore&&next){chooseCamera(next.id);return;}
  if(!next||!['origin','retained'].includes(cameraState(next)))chooseCamera(result.kept[0].id);
  else{elements.camera.value=next.id;updateControls();updateMapPoints();}
}

async function exportMap(){
  const clone=$('lab-map-interactive').cloneNode(true),background=clone.querySelector('image');
  if(!background)return;
  const response=await fetch(background.getAttribute('href'));if(!response.ok)throw new Error('俯视底图读取失败');
  const data=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;response.blob().then(blob=>reader.readAsDataURL(blob));});
  background.setAttribute('href',data);clone.removeAttribute('hidden');clone.setAttribute('xmlns',svgNS);
  clone.setAttribute('width',scene.sample.camera_map_projection.image_size[0]);clone.setAttribute('height',scene.sample.camera_map_projection.image_size[1]);
  saveBlob(new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'}),`${scene.id}-cameras-mask15-connected.svg`);
}

function showError(error){elements.error.textContent=String(error?.message||error);elements.error.hidden=false;}
function updateSceneControls(){
  const index=catalog?.scenes.findIndex(s=>s.id===scene?.id)??-1;
  $('lab-scene-prev').disabled=index<=0;
  $('lab-scene-next').disabled=index<0||index>=catalog.scenes.length-1;
}
function stepScene(delta){
  const index=catalog?.scenes.findIndex(s=>s.id===scene?.id)??-1;
  const next=catalog?.scenes[index+delta];if(index>=0&&next)chooseScene(next.id);
}
function updateURL(){const params=new URLSearchParams();if(scene)params.set('scene',scene.id);if(camera)params.set('camera',camera.id);history.replaceState(null,'','/camera-lab?'+params);}
function renderMap(){
  const source=scene?.[mapMode],interactive=mapMode==='camera_map'&&scene?.camera_map_base&&scene?.sample?.camera_map_projection;
  elements.map.hidden=!source||!!interactive;elements.mapEmpty.hidden=!!source;
  // SVGElement does not reflect the HTML .hidden property into its attribute.
  $('lab-map-interactive').toggleAttribute('hidden',!interactive);
  $('lab-map-download').textContent=interactive?'导出当前点位图 ↓':'打开原图 ↗';
  $('lab-map-download').onclick=interactive?e=>{e.preventDefault();exportMap().catch(showError);}:null;
  if(interactive)drawCandidateMap();
  if(source){elements.map.src=url(source);$('lab-map-download').href=url(source);}
  else elements.mapEmpty.textContent=scene?.error||'本张图的结果尚未生成。';
  $('lab-map-download').hidden=!source;
  const diagnostics=scene?.diagnostics;
  $('lab-outline-info').textContent=diagnostics?`拟合包络 ${diagnostics.area_m2.toFixed(1)} m² · 弱支撑/推断边界 ${(diagnostics.inferred_perimeter_fraction*100).toFixed(1)}%。绿色实线：较强支撑；橙色虚线：推断；蓝色：内缩区域。轮廓不等于真实墙线，请结合点云检查。`:'';
  document.querySelectorAll('[data-map]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.map===mapMode)));
}
function clearPreview(message){
  requestId++;camera=null;texture=null;elements.pano.hidden=true;$('lab-look-canvas').hidden=true;
  $('lab-preview-title').textContent=`${scene?.id||''} · 低清全景预览`;
  elements.error.hidden=true;
  elements.previewEmpty.hidden=false;elements.previewEmpty.textContent=message;elements.loading.hidden=true;
  $('lab-pano-download').hidden=true;$('lab-prev').disabled=true;$('lab-next').disabled=true;
  $('lab-render-info').textContent='没有可预览的采样相机。';
  $('lab-mask-views').hidden=true;$('lab-mask-warning').hidden=true;$('lab-mask-legend').hidden=true;
}
function chooseScene(id,restore=null){
  scene=catalog.scenes.find(s=>s.id===id)||catalog.scenes[0];updateSceneControls();if(!scene)return;
  elements.scene.value=scene.id;$('lab-title').textContent=scene.name+' · 俯视轮廓与采样点';renderMap();
  camera=null;
  const cams=scene.cameras||[];elements.camera.replaceChildren();$('lab-camera-list').replaceChildren();
  elements.camera.disabled=!cams.length;
  if(!cams.length){
    elements.camera.add(new Option(scene.sample?'内缩后没有点阵位置':'等待采样结果',''));
    clearPreview(scene.sample?`该轮廓向内 offset ${scene.sample.inward_offset_m} m 后没有符合规则的点阵位置。未放宽条件。`:'该样本正在处理，稍后自动刷新。');
    $('lab-camera-info').textContent=scene.sample?`采样相机：0\n原相机同高 / ${scene.sample.spacing_m} m 点阵\n内缩 ${scene.sample.inward_offset_m} m（严格执行）`:'等待本张图的反投影与轮廓拟合。';
    updateURL();return;
  }
  updateFilter({restore:cams.some(c=>c.id===restore)?restore:cams[0].id});
}
function updateControls(){
  const cams=activeCameras(),index=cams.findIndex(c=>c.id===camera?.id);
  $('lab-prev').disabled=index<=0;$('lab-next').disabled=index<0||index>=cams.length-1;
  document.querySelectorAll('[data-camera]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.camera===camera?.id)));
}
function previewSources(item,variant=renderVariant){
  const masked=variant==='masked'&&item.views?.raw&&item.views?.mask;
  return [url(scene.id+'/'+(masked?item.views.raw:item.views?.[variant]||item.panorama)),masked?url(scene.id+'/'+item.views.mask):null];
}
async function fetchTexture(source,maskSource=null){
  const cacheKey=source+'|'+(maskSource||'');
  if(textures.has(cacheKey)){const existing=textures.get(cacheKey);textures.delete(cacheKey);textures.set(cacheKey,existing);return existing;}
  const pending=(async()=>{
    let t=await new THREE.TextureLoader().loadAsync(source);
    if(maskSource){
      const m=await new THREE.TextureLoader().loadAsync(maskSource),canvas=document.createElement('canvas');
      canvas.width=t.image.width;canvas.height=t.image.height;
      const ctx=canvas.getContext('2d');ctx.drawImage(m.image,0,0);
      const mask=ctx.getImageData(0,0,canvas.width,canvas.height).data;ctx.drawImage(t.image,0,0);
      const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
      for(let i=0;i<mask.length;i+=4)if(mask[i]){pixels.data[i]=0;pixels.data[i+1]=255;pixels.data[i+2]=0;}
      ctx.putImageData(pixels,0,0);t.dispose();m.dispose();t=new THREE.CanvasTexture(canvas);t.userData.previewUrl=canvas.toDataURL('image/png');
    }else t.userData.previewUrl=source;
    t.flipY=false;t.colorSpace=THREE.SRGBColorSpace;t.needsUpdate=true;return t;
  })();
  textures.set(cacheKey,pending);
  try{const result=await pending;while(textures.size>8){const [key,value]=textures.entries().next().value;textures.delete(key);value.then(t=>{if(t!==texture)t.dispose();});}return result;}
  catch(error){textures.delete(cacheKey);throw error;}
}
async function chooseCamera(id){
  camera=allCameras().find(c=>c.id===id);if(!camera)return;
  elements.camera.querySelectorAll('[data-inspection]').forEach(option=>option.remove());
  if(!activeCameras().some(c=>c.id===id)){const option=new Option(id+' · 已排除，仅检查',id);option.dataset.inspection='true';elements.camera.add(option);}
  elements.camera.value=camera.id;updateControls();updateURL();
  updateMapPoints();
  const [x,y,z]=camera.position_world_m;
  $('lab-camera-info').textContent=`${scene.name} / ${camera.id}\n位置 X=${x.toFixed(2)}  Y=${y.toFixed(2)}  Z=${z.toFixed(2)} m\n距轮廓 ${camera.boundary_distance_m.toFixed(2)} m\n${camera.coincides_with_original?'原相机 · 0 号参考位':'与原相机同高'+(camera.square_ring!=null?' · 第 '+camera.square_ring+' 圈':'')}${camera.reference_only?'\n仅作原视点参考，不属于内缩区新增采样点':''}`;
  const fraction=maskFraction(camera);$('lab-camera-info').textContent+=`\nMask ${fraction===null?'待计算':(fraction*100).toFixed(2)+'%'} · ${stateLabels[cameraState(camera)]}${['excluded','disconnected'].includes(cameraState(camera))?'；仅供检查，不参与候选导出':''}`;
  $('lab-preview-title').textContent=`${scene.id} / ${camera.id} · 低清全景预览`;
  const ticket=++requestId;texture=null;elements.pano.hidden=true;$('lab-look-canvas').hidden=true;
  $('lab-render-info').textContent='等待载入该相机的全景结果。';
  $('lab-mask-views').hidden=!camera.views;$('lab-mask-legend').hidden=!camera.views;
  document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===renderVariant)));
  const stats=camera.mask_stats;
  $('lab-mask-warning').hidden=!stats||stats.camera_state==='known_free';
  $('lab-mask-warning').textContent=stats&&stats.camera_state!=='known_free'?'该相机位于未知或未确认空域，整张图保守标为 Mask。非原相机将因超过 15% 被排除；这里仅保留诊断预览，不代表确认相机在实体内部。':'';
  $('lab-pano-download').hidden=true;elements.error.hidden=true;elements.loading.hidden=false;elements.previewEmpty.hidden=true;
  if(!camera.panorama){elements.loading.hidden=true;elements.previewEmpty.hidden=false;elements.previewEmpty.textContent='该相机全景图尚未渲染完成。';return;}
  const sources=previewSources(camera);
  try{
    const next=await fetchTexture(...sources);if(ticket!==requestId)return;
    texture=next;elements.pano.src=next.userData.previewUrl;$('lab-pano-download').href=next.userData.previewUrl;$('lab-pano-download').hidden=false;
    const size=scene.sample?.render_resolution||[1024,512];
    $('lab-render-info').textContent=stats?`${size.join(' × ')} · 原始 RGB-D 可靠表面 / 未知体积首次遮挡（非最终高斯） · Mask ${(stats.mask_fraction*100).toFixed(1)}% · 保留 ${(camera.coverage*100).toFixed(1)}% · 相机${stats.camera_state==='known_free'?'在已知空域':'空域未确认'}。纯绿区域待修补，尚未 AI 补图。展开与自由转动显示同一版本。`:`1024 × 512 · 旧版 SHARP 第 3 步双层点云预览 · 覆盖率 ${(camera.coverage*100).toFixed(1)}%。`;
    elements.loading.hidden=true;setMode(mode);
    const cams=activeCameras(),index=cams.indexOf(camera);const neighbors=[cams[index+1],cams[index-1]];
    for(const item of neighbors)if(item?.panorama)fetchTexture(...previewSources(item)).catch(()=>{});
    for(const variant of ['masked','raw','mask','overlay'])if(camera.views?.[variant])fetchTexture(...previewSources(camera,variant)).catch(()=>{});
  }catch(error){if(ticket===requestId){elements.loading.hidden=true;elements.previewEmpty.hidden=false;elements.previewEmpty.textContent='全景图载入失败。';showError(error);}}
}
function initRenderer(){
  if(renderer)return;
  const target=$('lab-look-canvas');
  renderer=new THREE.WebGLRenderer({antialias:false,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  renderer.outputColorSpace=THREE.SRGBColorSpace;target.append(renderer.domElement);
  material=new THREE.ShaderMaterial({uniforms:{pano:{value:null},yaw:{value:0},pitch:{value:0},fov:{value:75},aspect:{value:2}},
    vertexShader:'varying vec2 uvScreen; void main(){uvScreen=uv; gl_Position=vec4(position.xy,0.0,1.0);}',
    fragmentShader:`uniform sampler2D pano; uniform float yaw,pitch,fov,aspect; varying vec2 uvScreen;
      void main(){float h=tan(radians(fov)*.5);vec2 q=uvScreen*2.0-1.0;
        vec3 r=normalize(vec3(q.x*h,q.y*h/aspect,1.0));
        r=vec3(r.x,cos(pitch)*r.y+sin(pitch)*r.z,-sin(pitch)*r.y+cos(pitch)*r.z);
        r=vec3(cos(yaw)*r.x+sin(yaw)*r.z,r.y,-sin(yaw)*r.x+cos(yaw)*r.z);
        vec2 uv=vec2(fract(atan(r.x,r.z)/6.28318530718+.5),.5-asin(clamp(r.y,-1.0,1.0))/3.14159265359);
        gl_FragColor=texture2D(pano,uv);
        #include <colorspace_fragment>
      }`});
  threeScene=new THREE.Scene();threeScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2),material));drawCamera=new THREE.Camera();
  new ResizeObserver(draw).observe(target);
  let drag=null;
  target.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY,yaw,pitch];target.setPointerCapture(e.pointerId);$('lab-preview').focus();});
  target.addEventListener('pointermove',e=>{if(!drag)return;yaw=drag[2]-(e.clientX-drag[0])*.004;pitch=Math.max(-1.56,Math.min(1.56,drag[3]+(e.clientY-drag[1])*.004));draw();});
  target.addEventListener('pointerup',()=>{drag=null;});target.addEventListener('pointercancel',()=>{drag=null;});
  target.addEventListener('wheel',e=>{e.preventDefault();$('lab-fov').value=Math.max(30,Math.min(110,Number($('lab-fov').value)+Math.sign(e.deltaY)*5));draw();},{passive:false});
}
function draw(){
  if(!renderer||!texture||mode!=='look')return;
  const target=$('lab-look-canvas'),w=target.clientWidth,h=target.clientHeight;if(!w||!h)return;
  renderer.setSize(w,h,false);material.uniforms.pano.value=texture;material.uniforms.yaw.value=yaw;material.uniforms.pitch.value=pitch;
  const fov=Math.max(30,Math.min(110,Number($('lab-fov').value)||75));
  material.uniforms.fov.value=fov;material.uniforms.aspect.value=w/h;renderer.render(threeScene,drawCamera);
  target.dataset.yaw=yaw.toFixed(4);target.dataset.pitch=pitch.toFixed(4);target.dataset.rendered='true';
}
function setMode(next){
  mode=next;$('lab-flat').setAttribute('aria-pressed',String(mode==='flat'));$('lab-look').setAttribute('aria-pressed',String(mode==='look'));
  elements.pano.hidden=!texture||mode!=='flat';$('lab-look-canvas').hidden=!texture||mode!=='look';
  if(texture&&mode==='look'){try{initRenderer();draw();}catch(error){showError('自由视角初始化失败：'+error.message);}}
}
async function refresh(){
  try{
    const response=await fetch('/api/camera-lab',{cache:'no-store'});if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const next=await response.json();const previousScene=scene?.id||wantedScene,previousCamera=camera?.id||wantedCamera;
    const revisionChanged=catalog&&(catalog.revision||catalog.run)!==(next.revision||next.run);
    if(revisionChanged){requestId++;texture=null;for(const pending of textures.values())pending.then(t=>t.dispose()).catch(()=>{});textures.clear();}
    const previousStamp=sceneStamp(scene);
    catalog=next;$('lab-progress').textContent=`${next.total} 张测试图 · 轮廓 ${next.outline_ready}/${next.total}\n全景完成 ${next.render_complete}/${next.total} · 最终候选 ${next.camera_count}（含原相机）`;
    $('lab-rules').textContent='固定规则：原相机同高 / 0.5 m 点阵。offset 从 0.5 m 每次减 0.05 m，直到原点八邻域至少 4 点或 offset < 0.1 m。随后保留 Mask ≤15% 且与原点四邻接连通的点；仅剩 C000 报错。C000 始终作参考保留。';
    elements.scene.replaceChildren();for(const item of next.scenes)elements.scene.add(new Option(item.name+(item.render_complete?'':' · 处理中'),item.id));
    elements.scene.disabled=!next.scenes.length;
    const updated=next.scenes.find(s=>s.id===previousScene);
    const nextStamp=sceneStamp(updated);
    if(!scene||!updated||revisionChanged||previousStamp!==nextStamp)chooseScene(previousScene,previousCamera);
    else{scene=updated;elements.scene.value=scene.id;camera=allCameras().find(c=>c.id===previousCamera)||null;updateSceneControls();}
    wantedScene=wantedCamera=null;
  }catch(error){showError(error);}
}
elements.scene.onchange=()=>chooseScene(elements.scene.value);
$('lab-scene-prev').onclick=()=>stepScene(-1);
$('lab-scene-next').onclick=()=>stepScene(1);
elements.camera.onchange=()=>chooseCamera(elements.camera.value);
$('lab-prev').onclick=()=>{const cams=activeCameras(),i=cams.indexOf(camera);if(i>0)chooseCamera(cams[i-1].id);};
$('lab-next').onclick=()=>{const cams=activeCameras(),i=cams.indexOf(camera);if(i>=0&&i+1<cams.length)chooseCamera(cams[i+1].id);};
$('lab-export-cameras').onclick=()=>{if(!scene)return;const result=summarize(allCameras());saveBlob(new Blob([JSON.stringify({scene:scene.id,run:catalog.run,selection_version:SELECTION_VERSION,spacing_m:.5,inward_offset_m:scene.sample.inward_offset_m,mask_threshold:MASK_THRESHOLD,connectivity:4,anchor:'C000',comparison:'mask_fraction > 0.15 excluded; C000 exempt; then origin four-connected component',count:result.kept.length,cameras:result.kept,excluded_camera_ids:result.excluded.map(c=>c.id),disconnected_camera_ids:result.disconnected.map(c=>c.id),pending_camera_ids:result.pending.map(c=>c.id),sampling_error:scene.sample.sampling_error||null,selection_error:scene.sample.selection_error||(result.onlyOrigin?'Only original camera remains':null)},null,2)],{type:'application/json'}),`${scene.id}-selected-cameras.json`);};
document.querySelectorAll('[data-map]').forEach(b=>b.onclick=()=>{mapMode=b.dataset.map;renderMap();});
$('lab-flat').onclick=()=>setMode('flat');$('lab-look').onclick=()=>setMode('look');$('lab-fov').onchange=draw;
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{renderVariant=b.dataset.view;if(camera)chooseCamera(camera.id);});
$('lab-reset').onclick=()=>{yaw=pitch=0;$('lab-fov').value=75;draw();};$('lab-refresh').onclick=refresh;
$('lab-preview').onkeydown=e=>{const step=.07;if(mode!=='look')return;const actions={ArrowLeft:()=>yaw-=step,ArrowRight:()=>yaw+=step,ArrowUp:()=>pitch=Math.min(1.56,pitch+step),ArrowDown:()=>pitch=Math.max(-1.56,pitch-step)};if(actions[e.key]){e.preventDefault();actions[e.key]();draw();}};
await refresh();setInterval(()=>{if(catalog&&catalog.render_complete<catalog.total)refresh();},15000);
