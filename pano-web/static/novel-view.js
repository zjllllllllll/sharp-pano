import * as THREE from 'three';
// Pixel-aligned two/three-way comparison: clip full-size layers, never resize strips.
export function moveDivider(stops,index,value){
  const next=[...stops];
  if(index<0||index>=next.length||!Number.isFinite(value))return next;
  next[index]=Math.max(index?next[index-1]:0,Math.min(index+1<next.length?next[index+1]:100,value));
  return next;
}

function node(tag,text,className){const n=document.createElement(tag);if(text)n.textContent=text;if(className)n.className=className;return n;}

export function imageWipe({title,layers,aspect=2,stops,expand=true,colorLegend=[]}){
  if(![2,3].includes(layers.length))throw new Error('Comparison needs two or three layers');
  const defaults=layers.length===2?[50]:[33.333,66.667];let positions=stops?[...stops]:defaults;
  const figure=node('figure',null,'image-wipe');figure.dataset.title=title;figure.style.setProperty('--wipe-aspect',String(aspect));
  const caption=node('figcaption',null,'wipe-heading'),heading=node('strong',title),actions=node('span');
  const reset=node('button','复位');reset.type='button';reset.setAttribute('aria-label',`${title} · 复位分隔线`);
  actions.append(reset);caption.append(heading,actions);figure.append(caption);
  const stage=node('div',null,'wipe-stage');stage.style.aspectRatio=String(aspect);figure.append(stage);
  const sheets=layers.map((layer,index)=>{
    const sheet=node('div',null,'wipe-sheet');sheet.dataset.layer=String(index);
    const image=node('img');image.alt=`${title} · ${layer.label}`;image.draggable=false;image.decoding='async';
    const status=node('span','正在加载…','wipe-image-status');sheet.append(image,status);
    image.onload=()=>status.remove();image.onerror=()=>{status.textContent='图片加载失败，请刷新';sheet.dataset.error='true';};
    image.src=layer.url;stage.append(sheet);return sheet;
  });
  const handles=positions.map((_,index)=>{
    const handle=node('div',null,'wipe-divider');handle.tabIndex=0;handle.setAttribute('role','slider');
    handle.setAttribute('aria-label',`${title} · ${layers[index].label} / ${layers[index+1].label} 分隔线`);
    handle.setAttribute('aria-orientation','horizontal');handle.dataset.divider=String(index);
    const grip=node('span','↔','wipe-grip');grip.setAttribute('aria-hidden','true');handle.append(grip);stage.append(handle);
    handle.onkeydown=event=>{
      const delta=event.shiftKey?10:1;let value=positions[index];
      if(event.key==='ArrowLeft')value-=delta;else if(event.key==='ArrowRight')value+=delta;
      else if(event.key==='Home')value=index?positions[index-1]:0;
      else if(event.key==='End')value=index+1<positions.length?positions[index+1]:100;else return;
      event.preventDefault();positions=moveDivider(positions,index,value);paint();
    };
    return handle;
  });
  const legend=node('div',null,'wipe-legend');
  layers.forEach((layer,index)=>{const label=node('span',`${index+1} · ${layer.label}`);label.title=layer.note||layer.label;legend.append(label);});
  figure.append(legend);
  if(colorLegend.length){
    const colors=node('div',null,'mask-source-legend');colors.setAttribute('aria-label','Mask 来源颜色图例');
    for(const {label,color} of colorLegend){
      const item=node('span'),swatch=node('i');swatch.style.backgroundColor=color;swatch.setAttribute('aria-hidden','true');
      item.append(swatch,document.createTextNode(label));colors.append(item);
    }
    figure.append(colors);
  }
  function paint(){
    const edges=[0,...positions,100];
    sheets.forEach((sheet,i)=>sheet.style.clipPath=`inset(0 ${100-edges[i+1]}% 0 ${edges[i]}%)`);
    handles.forEach((handle,i)=>{
      handle.style.left=`${positions[i]}%`;handle.setAttribute('aria-valuemin',String(i?positions[i-1]:0));
      handle.setAttribute('aria-valuemax',String(i+1<positions.length?positions[i+1]:100));
      handle.setAttribute('aria-valuenow',positions[i].toFixed(1));handle.setAttribute('aria-valuetext',`${positions[i].toFixed(1)}%`);
    });
    stage.dataset.stops=JSON.stringify(positions);
  }
  let drag=null;
  const move=event=>{
    if(drag===null)return;
    const bounds=stage.getBoundingClientRect();
    positions=moveDivider(positions,drag,(event.clientX-bounds.left)/bounds.width*100);paint();
  };
  stage.onpointerdown=event=>{
    if(event.button!==0||!event.isPrimary)return;
    const bounds=stage.getBoundingClientRect(),x=(event.clientX-bounds.left)/bounds.width*100;
    const selected=event.target.closest('[data-divider]');
    drag=selected?Number(selected.dataset.divider):positions.reduce((best,p,i)=>Math.abs(p-x)<Math.abs(positions[best]-x)?i:best,0);
    stage.setPointerCapture(event.pointerId);handles[drag].focus({preventScroll:true});move(event);event.preventDefault();
  };
  stage.onpointermove=move;
  stage.onpointerup=stage.onpointercancel=event=>{drag=null;if(stage.hasPointerCapture(event.pointerId))stage.releasePointerCapture(event.pointerId);};
  stage.onlostpointercapture=()=>{drag=null;};
  reset.onclick=()=>{positions=[...defaults];paint();};
  if(expand){
    const enlarge=node('button','放大对比');enlarge.type='button';enlarge.setAttribute('aria-label',`${title} · 放大对比`);actions.append(enlarge);
    enlarge.onclick=()=>{
      const dialog=node('dialog',null,'wipe-modal'),close=node('button','关闭放大','wipe-close');close.type='button';
      close.onclick=()=>dialog.close();dialog.append(close,imageWipe({title,layers,aspect,stops:positions,expand:false,colorLegend}));
      dialog.onclose=()=>dialog.remove();document.body.append(dialog);dialog.showModal();close.focus();
    };
  }
  paint();return figure;
}

const $=id=>document.getElementById(id);
// 256-entry Turbo palette, shared by all three numerical depth layers.
const DEPTH_PALETTE=Uint8Array.from(atob('MBI7MhVDMxhKNBtRNR5YNiFfNyRmOCdtOSpzOi15Oy+APDKGPTWLPjiRPzuXPz6cQECiQUOnQUasQkmxQku1Q066RFG/RFTDRFbHRVnLRVzPRV7TRmHWRmTaRmbdRmngRmvjRmvjR27mR3HpR3PrR3buR3jwR3vyRn30RoD2RoL4RoX6Rof7RYr8RYz9RI/+Q5H+QpT/QZb/QJn/Ppv+PZ7+O6D9OqP8OKX7N6j6Nav4M633Ma/1L7L0LrTyLLfwKrnuKLzrJ77pJcDnI8PkIsXiIMffH8ndHsvaHM3YG9DVGtLSGtTQGdXNGNfKGNnIGNvFGN3CGN7AGOC9GeK7GeO5GuS2HOa0HeeyH+mvIOqsIuuqJeynJ+6kKu+hLPCeL/GbMvKYNfOUOPSRPPWOP/aKQ/eHRviESviATvl9Uvp6Vfp2WftzXfxvYfxsZf1paf1mbf5icf5fdf5cef5Zff9WgP9ThP9RiP9Oi/9Lj/9Jkv9Hlv5Emf5CnP5An/0/of09pPw8p/w6qfs5rPs4r/o3sfk2tPg2t/c1ufY1vPU0vvQ0wfM0w/E0xvA0yO80y+00zew00Oo00uk11Oc11+U12eQ22+I23eA339834d0349s45dk459c56dU569M57NE67s8678068cs68sk69Mc69cU69sM698E6+L45+bw5+ro5+7g4+7Y3/LM2/LE2/a41/aw0/qkz/qcy/qQx/qEw/p4v/pst/pks/pYr/pMq/pAp/Y0n/Yom/Icl/IQj+4Ei+34h+nsf+Xge+XUd+HIc928a9mwZ9WkY9GYX82MV8mAU8V0T8FsS71gR7VUQ7FMP61AO6k4N6EsM50kM5UcL5EUK4kMK4UEJ3z8I3T0I3DsH2jkH2DcG1jUG1DMF0jEF0C8Fzi0EzCsEyioEyCgDxSYDwyUDwSMCviECvCACuR4Ctx0CtBsBshoBrxgBrBcBqRYBpxQBpBMBoRIBnhABmw8BmA4BlQ0BkgsBjgoBiwkCiAgChQcCgQYCfgUCegQD'),c=>c.charCodeAt(0));

export function readDepthNpy(buffer){
  const bytes=new Uint8Array(buffer),view=new DataView(buffer);
  if(bytes.length<12||bytes[0]!==147||new TextDecoder().decode(bytes.subarray(1,6))!=='NUMPY')throw new Error('不是有效的 NPY 深度文件');
  const version=bytes[6];if(![1,2].includes(version))throw new Error('不支持的 NPY 版本');
  const start=version===1?10:12,length=version===1?view.getUint16(8,true):view.getUint32(8,true),offset=start+length;
  const header=new TextDecoder().decode(bytes.subarray(start,offset));
  if(!/['"]descr['"]\s*:\s*['"]<f4['"]/.test(header)||! /['"]fortran_order['"]\s*:\s*False/.test(header))throw new Error('需要 float32 行优先深度');
  const shape=header.match(/['"]shape['"]\s*:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,?\s*\)/);
  if(!shape)throw new Error('深度必须为二维数组');
  const height=Number(shape[1]),width=Number(shape[2]);
  if(!height||!width||width*height>4096*4096||offset%4||offset+width*height*4!==buffer.byteLength)throw new Error('深度数据长度不匹配');
  return {width,height,data:new Float32Array(buffer,offset,width*height)};
}

export function depthPixels(values,lo,hi){
  const pixels=new Uint8ClampedArray(values.length*4);
  for(let i=0;i<values.length;i++){
    const d=values[i];pixels[i*4+3]=255;
    if(!Number.isFinite(d)||d<=1e-3){pixels[i*4+1]=255;continue;}
    const index=Math.round(Math.max(0,Math.min(1,(d-lo)/Math.max(hi-lo,1e-6)))*255)*3;
    pixels.set(DEPTH_PALETTE.subarray(index,index+3),i*4);
  }
  return pixels;
}
const FACES=['front','right','back','left','up','down'];
const LABELS=['前','右','后','左','上','下'];
// The same OpenCV forward/down axes as sharp-sixface/geometry.py.
const AXES=[[[0,0,1],[0,1,0]],[[1,0,0],[0,1,0]],[[0,0,-1],[0,1,0]],
  [[-1,0,0],[0,1,0]],[[0,-1,0],[0,0,1]],[[0,1,0],[0,0,-1]]];
const display=p=>new THREE.Vector3(p[0],-p[1],-p[2]);
function element(tag,text,className){const e=document.createElement(tag);if(text)e.textContent=text;if(className)e.className=className;return e;}

const MASK_SOURCE_COLOR=[255,153,51];
const CAPTURE_VERSION='geometry-only-open3-smooth1-green-v4';

export function maskSourcePixels(raw,mask,{sourceMode=true}={}){
  if(raw.length%4||raw.length!==mask.length)throw new Error('Mask 与原图尺寸不一致');
  const pixels=new Uint8ClampedArray(raw);let geometry=0;
  for(let i=0;i<pixels.length;i+=4){
    if(mask[i]>0){pixels.set(sourceMode?MASK_SOURCE_COLOR:[0,255,0],i);geometry++;}
  }
  return {pixels,counts:{geometry},total:raw.length/4};
}

export class NovelView {
  constructor({api,context,error,onBusy}) {
    Object.assign(this,{api,context,error,onBusy});this.busy=false;this.active=false;this.labels=[];this.generation=0;this.depthComparisons=new Map();this.maskSourcePreviews=new Map();
    this.layer=element('div',null,'camera-labels');$('viewport').append(this.layer);
    this.popup=element('div',null,'camera-popup');this.popup.hidden=true;$('viewport').append(this.popup);
  }
  url(path){return `/novel-artifacts/${this.jid}/${this.state.revision}/${path}`;}
  hide(){this.generation++;this.active=false;this.layer.hidden=true;this.popup.hidden=true;$('novel-panel').hidden=true;$('repair-panel').hidden=true;if(this.group)this.group.visible=false;document.querySelectorAll('dialog.wipe-modal').forEach(d=>d.close());}
  compare(title,layers,aspect=2){return imageWipe({title,aspect,layers:layers.map(([label,path])=>({label,url:this.url(path)}))});}
  async maskSourcePreview(preview,sourceMode=true){
    const base=this.url(preview.path);
    if(!this.maskSourcePreviews.has(base)){
      const task=Promise.all(['raw.png','mask.png'].map(async name=>{
        const img=new Image();img.src=`${base}/${name}`;await img.decode();
        if(img.naturalWidth!==1024||img.naturalHeight!==512)throw new Error('Mask 来源图尺寸不一致');
        const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=512;
        const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);
        return ctx.getImageData(0,0,1024,512).data;
      })).then(([raw,mask])=>({raw,mask})).catch(error=>{this.maskSourcePreviews.delete(base);throw error;});
      this.maskSourcePreviews.set(base,task);
      if(this.maskSourcePreviews.size>4)this.maskSourcePreviews.delete(this.maskSourcePreviews.keys().next().value);
    }
    const {raw,mask}=await this.maskSourcePreviews.get(base);
    const {pixels,counts,total}=maskSourcePixels(raw,mask,{sourceMode});
    const canvas=document.createElement('canvas');canvas.width=1024;canvas.height=512;
    canvas.getContext('2d').putImageData(new ImageData(pixels,1024,512),0,0);
    const colorLegend=sourceMode?[{label:`几何 Mask（橙） ${(counts.geometry/total*100).toFixed(2)}%`,color:`rgb(${MASK_SOURCE_COLOR.join(',')})`}]:[];
    return {url:canvas.toDataURL('image/png'),colorLegend,counts,total};
  }
  repairPanorama(preview,result,generation){
    const container=element('div'),view=element('div'),details=element('p',null,'wipe-help');details.hidden=true;
    const right={label:result?.image?'最终合成':'二值 Mask',url:this.url(result?.image||`${preview.path}/mask.png`)},original=this.url(`${preview.path}/masked.png`);
    const picker=element('select');picker.setAttribute('aria-label','全景 RGB 左侧显示');
    for(const [value,label] of [['green','原图（绿幕 Mask）'],['sources','几何 Mask（橙）']]){
      const option=element('option',label);option.value=value;picker.append(option);
    }
    const current=preview.capture_version===CAPTURE_VERSION;
    if(current){
      const toolbar=element('div',null,'novel-actions'),label=element('label','全景左侧：');label.append(picker);
      toolbar.append(label,element('span',`右侧：${right.label}`));container.append(toolbar);
    }
    details.setAttribute('role','status');container.append(details,view);let serial=0;
    const show=async()=>{
      const ticket=++serial,sourceMode=picker.value==='sources';
      const stops=JSON.parse(view.querySelector('.wipe-stage')?.dataset.stops||'[50]');
      const title=`全景 RGB · ${sourceMode?'几何 Mask':'原图（带 Mask）'} / ${right.label}`;
      try{
        let left={label:'原图（带 Mask）',url:original},colorLegend=[];
        if(current){
          if(!view.querySelector('.wipe-stage'))view.replaceChildren(imageWipe({title,layers:[left,right],stops}));
          const data=await this.maskSourcePreview(preview,sourceMode);
          if(ticket!==serial||generation!==this.generation)return;
          left={label:sourceMode?'几何 Mask（橙）':'绿幕 Mask',url:data.url};colorLegend=data.colorLegend;
          const pixels=data.counts.geometry;
          details.textContent=`几何 Mask ${(pixels/data.total*100).toFixed(2)}%（${pixels.toLocaleString()} / ${data.total.toLocaleString()} 像素）。`;
          details.hidden=false;
        }
        if(ticket!==serial||generation!==this.generation)return;
        view.replaceChildren(imageWipe({title,layers:[left,right],stops,colorLegend}));
      }catch(error){
        if(ticket===serial&&generation===this.generation){
          details.textContent=`Mask 预览加载失败：${error.message}。当前显示已保存原图，切换显示选项可重试。`;details.hidden=false;
          view.replaceChildren(imageWipe({title:`全景 RGB · 已保存原图 / ${right.label}`,layers:[{label:'原图（带 Mask）',url:original},right],stops}));
        }
      }
    };
    picker.onchange=()=>{if(generation===this.generation)void show();};void show();return container;
  }
  async depthComparison(result){
    const key=`${this.jid}:${result.revision}:${result.depth_revision}:masked-input`;
    if(!this.depthComparisons.has(key)){
      const base=this.url(result.path),records=new Map(result.face_results.map(row=>[row.face,row]));
      const task=Promise.all(FACES.map(async face=>{
        const skipped=records.get(face).skipped,views={};
        for(const [name,suffix] of [['original','input-z'],['model',skipped?'input-z':'lingbot-z'],['composite','completed-z']]){
          if(name==='model'&&skipped){views.model=views.original;continue;}
          const filename=`${face}-${suffix}.npy`,response=await fetch(`${base}/${filename}`);
          if(!response.ok)throw new Error(`${face}: 模型深度文件读取失败`);
          const buffer=await response.arrayBuffer(),sha=[...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))].map(b=>b.toString(16).padStart(2,'0')).join('');
          if(sha!==result.files[filename])throw new Error(`${face}: 深度源文件校验失败`);
          const depth=readDepthNpy(buffer);if(depth.width!==512||depth.height!==512)throw new Error(`${face}: 深度画幅不匹配`);
          const canvas=document.createElement('canvas');canvas.width=depth.width;canvas.height=depth.height;
          canvas.getContext('2d').putImageData(new ImageData(depthPixels(depth.data,...result.preview_range_m),depth.width,depth.height),0,0);
          views[name]=canvas.toDataURL('image/png');
        }
        return {face,skipped,views};
      })).then(faces=>({depth_revision:result.depth_revision,faces})).catch(error=>{this.depthComparisons.delete(key);throw error;});
      this.depthComparisons.set(key,task);
      if(this.depthComparisons.size>2)this.depthComparisons.delete(this.depthComparisons.keys().next().value);
    }
    return this.depthComparisons.get(key);
  }
  async show(jid,state) {
    this.hide();this.jid=jid;this.state=state;this.active=true;this.layer.hidden=false;
    const {scene}=this.context();
    if(this.group){scene.remove(this.group);this.group.traverse(n=>{n.geometry?.dispose();n.material?.dispose();});}
    this.group=new THREE.Group();scene.add(this.group);this.labels=[];this.layer.replaceChildren();
    this.leaders=document.createElementNS('http://www.w3.org/2000/svg','svg');this.leaders.classList.add('camera-leaders');this.layer.append(this.leaders);
    const selected=new Set(state.selected_ids),frontier=new Set(state.frontier);
    const picker=$('novel-picker');picker.replaceChildren();
    for(const c of state.cameras){
      const position=display(c.position_world_m),highlight=c.id===state.highlighted;
      const color=highlight?0xffcf50:selected.has(c.id)?0x55bfff:frontier.has(c.id)?0xb8f66b:0x789077;
      const marker=new THREE.Mesh(new THREE.SphereGeometry(highlight?.065:.045,12,8),new THREE.MeshBasicMaterial({color,depthTest:false,depthWrite:false}));
      marker.position.copy(position);marker.renderOrder=100;this.group.add(marker);
      const button=element('button',c.id,`camera-label${highlight?' recommended':''}${selected.has(c.id)?' visited':''}`);
      button.title=`${c.id} · Mask ${(c.mask_stats.mask_fraction*100).toFixed(2)}% · 点击渲染高斯`;
      button.onclick=()=>void this.preview(c.id).catch(this.error);this.layer.append(button);
      const line=document.createElementNS('http://www.w3.org/2000/svg','line');this.leaders.append(line);
      this.labels.push({button,position,id:c.id,line,highlight});
      const option=element('option',`${c.id}${highlight?(state.current?' ★ 本轮已确认':' ★ 推荐'):''} · Mask ${(c.mask_stats.mask_fraction*100).toFixed(2)}%`);option.value=c.id;picker.append(option);
    }
    picker.value=state.highlighted||'C000';
    $('novel-render').onclick=()=>void this.preview(picker.value).catch(this.error);
    $('novel-highlight').onclick=()=>{
      const c=state.cameras.find(c=>c.id===picker.value);if(!c)return;
      const {camera}=this.context();camera.position.copy(display(c.position_world_m));
      camera.updateMatrixWorld(true);this.popup.hidden=true;this.tick();
    };
    const globalMask=state.selection_policy==='max-mask';
    const panel=$('novel-panel');panel.querySelector('strong').textContent=globalMask?'候选相机 · 最大 Mask':'候选相机 · 全局四邻域';
    panel.querySelectorAll('p')[1].textContent=`黄色：本轮推荐；蓝色：已选；绿色：${globalMask?'全部未选候选':'四邻域'}。点击编号会现场渲染高斯；点击不改变已选集合。`;
    panel.querySelector('.novel-privacy').textContent=`进入下一步会确认推荐相机，六面最多 3 个请求并行补洞：每次向${state.rgb_repair_model==='gpt-image-2'?'公司接口 GPT Image 2':'火山方舟 Seedream'}发送一张 1024×1024 绿幕待补图。`;
    const selectionText=state.current?`本轮已确认 ${state.highlighted}；后续补洞沿用此视角`:
      (globalMask?`全部未选候选中 Mask 最大 · 推荐 ${state.highlighted||'无'}`:`本轮四邻域 ${state.frontier.join('、')} · 推荐 ${state.highlighted||'无'}`);
    $('novel-status').textContent=`${state.count} 个候选 · 已选 ${state.selected_ids.join('、')} · ${selectionText}。场景：${state.source.variant}，${state.source.count.toLocaleString()} 个高斯。${state.iteration_index>1?` 第 ${state.iteration_index} 轮：累计 ${state.accepted_observations.length} 台已接受相机的可信空间。`:''}`;
    $('novel-panel').hidden=false;this.tick();
  }
  tick(){
    if(!this.active||!this.state)return;
    const {camera}=this.context(),rect=$('viewport').getBoundingClientRect();camera.updateMatrixWorld();
    const placed=[];
    // Same-height cameras often project onto the same scanline. Keep every
    // visible ID clickable, with a leader back to its actual spatial point.
    const ordered=[...this.labels].sort((a,b)=>Number(b.highlight)-Number(a.highlight)||a.position.distanceToSquared(camera.position)-b.position.distanceToSquared(camera.position));
    for(const row of ordered){
      const p=row.position.clone().project(camera);const inFront=row.position.clone().applyMatrix4(camera.matrixWorldInverse).z<-.01;
      row.button.hidden=!inFront||p.z<-1||p.z>1||Math.abs(p.x)>1||Math.abs(p.y)>1;
      row.line.style.display=row.button.hidden?'none':'';
      if(row.button.hidden)continue;
      const anchorX=(p.x*.5+.5)*rect.width,anchorY=(-p.y*.5+.5)*rect.height;
      const x=Math.max(26,Math.min(rect.width-26,anchorX));let y=anchorY-24;
      for(let attempt=0;attempt<30;attempt++){
        const offset=attempt===0?0:Math.ceil(attempt/2)*26*(attempt%2?-1:1);
        const candidate=Math.max(15,Math.min(rect.height-18,anchorY-24+offset));
        if(!placed.some(q=>Math.abs(q.x-x)<49&&Math.abs(q.y-candidate)<24)){y=candidate;break;}
      }
      placed.push({x,y});row.button.style.left=`${x}px`;row.button.style.top=`${y}px`;
      row.line.setAttribute('x1',anchorX);row.line.setAttribute('y1',anchorY);row.line.setAttribute('x2',x);row.line.setAttribute('y2',y);
    }
    if(!this.popup.hidden){
      const row=this.labels.find(r=>r.id===this.popupCamera),p=row?.position.clone().project(camera);
      const visible=row&&!row.button.hidden;
      const x=visible?(p.x*.5+.5)*rect.width:rect.width-185,y=visible?(-p.y*.5+.5)*rect.height:32;
      this.popup.style.left=`${Math.max(8,Math.min(rect.width-350,x+16))}px`;
      this.popup.style.top=`${Math.max(8,Math.min(rect.height-this.popup.offsetHeight-8,y+16))}px`;
    }
  }
  async renderFaces(c) {
    const {renderer,scene,spark,camera}=this.context();
    if(!spark)throw new Error('高斯尚未载入');
    const target=new THREE.WebGLRenderTarget(512,512,{type:THREE.UnsignedByteType,format:THREE.RGBAFormat,colorSpace:THREE.SRGBColorSpace});
    const previous=renderer.getRenderTarget(),auto=spark.autoUpdate,groupVisible=this.group.visible;
    const clearColor=renderer.getClearColor(new THREE.Color()),clearAlpha=renderer.getClearAlpha(),background=scene.background;
    const view=new THREE.PerspectiveCamera(100,1,.01,1000);view.position.copy(display(c.position_world_m));
    const canvas=document.createElement('canvas');canvas.width=canvas.height=512;const ctx=canvas.getContext('2d');
    const pixels=new Uint8Array(512*512*4),image=ctx.createImageData(512,512),alphaImage=ctx.createImageData(512,512),faces=[],alphaFaces=[];
    this.group.visible=false;spark.autoUpdate=false;
    try{
      renderer.setClearColor(0x000000,0);scene.background=null;
      scene.updateMatrixWorld(true);
      for(const [forward,down] of AXES){
        view.up.copy(display(down).negate());view.lookAt(view.position.clone().add(display(forward)));view.updateMatrixWorld(true);
        await spark.update({scene,camera:view});renderer.setRenderTarget(target);spark.render(scene,view);
        await renderer.readRenderTargetPixelsAsync(target,0,0,512,512,pixels);
        for(let y=0;y<512;y++)image.data.set(pixels.subarray((511-y)*2048,(512-y)*2048),y*2048);
        // Keep render alpha for quality evaluation; repair masks use geometry only.
        // Separate grayscale PNG avoids canvas premultiply/unpremultiply loss.
        for(let i=0;i<image.data.length;i+=4){
          alphaImage.data[i]=alphaImage.data[i+1]=alphaImage.data[i+2]=image.data[i+3];
          alphaImage.data[i+3]=255;image.data[i+3]=255;
        }
        ctx.putImageData(image,0,0);faces.push(canvas.toDataURL('image/png'));
        ctx.putImageData(alphaImage,0,0);alphaFaces.push(canvas.toDataURL('image/png'));
      }
    }finally{renderer.setRenderTarget(previous);renderer.setClearColor(clearColor,clearAlpha);scene.background=background;target.dispose();this.group.visible=groupVisible;spark.autoUpdate=auto;await spark.update({scene,camera});}
    return {faces,alpha_faces:alphaFaces,capture_version:CAPTURE_VERSION};
  }
  async preview(id) {
    if(this.busy)throw new Error('正在渲染，请等待当前相机完成');
    if(!this.active)throw new Error('请先进入候选相机步骤');
    const c=this.state.cameras.find(c=>c.id===id);if(!c)throw new Error('无效相机');
    this.busy=true;this.onBusy(true);this.popupCamera=id;this.popup.hidden=false;this.popup.replaceChildren(element('strong',`${id} · 正在渲染高斯全景…`));this.tick();
    try{
      const capture=await this.renderFaces(c);
      const result=await this.api(`/api/jobs/${this.jid}/novel/preview/${id}`,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({revision:this.state.revision,source_sha256:this.state.source.sha256,...capture})});
      this.state.previews[id]=result;this.showPopup(result);return result;
    }catch(error){this.popup.replaceChildren(element('p',error.message));throw error;}
    finally{this.busy=false;this.onBusy(false);}
  }
  showPopup(result){
    const heading=element('div',null,'camera-popup-heading');heading.append(element('strong',`${result.camera_id} · 高斯全景`));
    const close=element('button','×');close.setAttribute('aria-label','关闭相机预览');close.onclick=()=>this.popup.hidden=true;heading.append(close);
    const img=element('img');img.alt=`${result.camera_id} 实际高斯全景`;img.src=this.url(`${result.path}/masked.png`);
    const link=element('a');link.href=img.src;link.target='_blank';link.rel='noopener';link.append(img);
    const controls=element('div',null,'camera-popup-controls');
    for(const [label,name] of [['含 Mask','masked.png'],['原始渲染','raw.png'],['Mask','mask.png']]){
      const b=element('button',label);b.onclick=()=>{img.src=this.url(`${result.path}/${name}`);link.href=img.src;};controls.append(b);
    }
    this.popup.replaceChildren(heading,link,controls,element('small',`1024 × 512 · Mask ${(result.mask_fraction*100).toFixed(2)}% · 点击图片放大。预览不会选中此相机。`));
    this.tick();
  }
  async ensureRecommended(){
    if(!this.active)throw new Error('请先打开本轮候选相机步骤');
    const id=this.state.highlighted;
    if(!id)throw new Error('没有可选的下一台相机');
    const cached=this.state.previews[id];
    return cached?.capture_version===CAPTURE_VERSION?cached:await this.preview(id);
  }
  async qualityContext(jid,result){
    if(result?.quality_faces){
      let depth=result.kind==='depth-completion'?result:null;
      if(result.kind==='repair'){
        const state=await this.api(`/api/jobs/${jid}`),candidate=state.stages['8'];
        if(candidate?.accepted_bundle?.rgb_sha256===result.image_sha256&&candidate.camera_id===result.camera_id)depth=candidate;
      }
      return {live:true,rows:result.quality_faces,jointDepth:depth};
    }
    try{return await this.api(`/api/jobs/${jid}/face-quality`);}
    catch{return {status:'not_evaluated',rows:[]};}
  }
  qualityCard(name,kind,result,context){
    const box=element('div',null,'face-quality');
    const historical=!context.live;
    const audit=historical?context.rows?.find(r=>r.camera===result?.camera_id&&r.face===name&&r.kind===kind&&
      (!r.source_revision||r.source_revision===result.revision)&&(!r.depth_revision||kind!=='depth'||r.depth_revision===result.depth_revision)&&
      (!r.capture_path||kind!=='rgb'||r.capture_path===result.capture.path)):null;
    const q=historical?audit?.assessment:context.rows.find(r=>r.face===name);
    if(!q){box.append(element('p','质量检查：未评估'));return box;}
    const pair=context.jointDepth?.accepted_bundle?.faces.find(r=>r.face===name);
    const decision=historical?(audit.joint_accepted?'联合通过':audit.disposition):pair?(pair.accepted?'联合通过':q.status==='passed'?'连带弃用':'整面弃用'):'RGB 检查后结果，等待深度联合确认';
    const labels={passed:'通过',rejected:'拒收',insufficient_evidence:'证据不足',not_needed:'无需修补'};
    box.dataset.status=q.status;
    box.append(element('strong',`${labels[q.status]||q.status} · 质量分 ${q.score===null?'—':q.score.toFixed(1)} · ${decision}`));
    if(q.reasons?.length)box.append(element('p',q.reasons.join('；')));
    if(q.warnings?.length&&q.status==='passed')box.append(element('p','超过提示线（不弃用）：'+q.warnings.join('；'),'wipe-help'));
    const details=element('details');details.append(element('summary','指标、评估区域与差异热图'));
    const pre=element('pre',JSON.stringify({metrics:q.metrics,rejection_limits:q.limits,warning_limits:q.warning_limits,evidence:q.evidence,policy:q.policy_version},null,2));details.append(pre);
    const maps=element('div',null,'quality-maps');
    const labelsMap={anchors:'评估区域',metric_support:'指标有效区域',difference:'差异热图',bad:'异常区域'};
    for(const [key,path] of Object.entries(historical?audit.maps:q.maps||{})){
      const img=element('img');img.loading='lazy';img.alt=labelsMap[key]||key;
      img.src=historical?path:this.url(`${kind==='rgb'?result.capture.path+'/repair':result.path}/${path}`);
      const figure=element('figure');figure.append(img,element('figcaption',img.alt));maps.append(figure);
    }
    details.append(maps);box.append(details);return box;
  }
  qualityNotice(panel,context){
    if(context.live){panel.append(element('p','质量分检查 Mask 外一致性；轻微偏差仅提示，明显失效整面弃用。','wipe-help'));return;}
    if(context.status==='ready'){
      const p=element('p','离线诊断尚未应用到已生成场景；现有全景、深度和 PLY 保持原结果。','novel-privacy');
      const a=element('a','查看全部 24 张诊断报告 ↗');a.href=context.report_url;a.target='_blank';a.rel='noopener';p.append(' ',a);panel.append(p);
    }
  }
  async showRepair(jid,result,variant='gpt') {
    const rootResult=result,variants=result?.comparison_variants||[];
    this.hide();const generation=this.generation;this.jid=jid;const state=await this.api(`/api/jobs/${jid}/novel`);
    if(generation!==this.generation)return;this.state=state;
    const choice=variants.find(v=>v.id===variant);
    if(choice&&choice.id!=='gpt')result=await this.api(this.url(choice.path));
    if(generation!==this.generation)return;this.repairResult=result;
    const preview=result?.capture||this.state.current?.preview;
    if(!preview)throw new Error('尚未确认补洞相机');
    const panel=$('repair-panel');panel.replaceChildren();panel.hidden=false;panel.setAttribute('aria-label','六面补洞与全景融合');
    const quality=await this.qualityContext(jid,result);if(generation!==this.generation)return;this.qualityNotice(panel,quality);
    if(variants.length){
      const controls=element('div',null,'novel-actions'),label=element('label','RGB 测试版本：'),picker=element('select');picker.setAttribute('aria-label','RGB 修复对比版本');
      for(const v of variants){const option=element('option',v.label);option.value=v.id;picker.append(option);}
      picker.value=choice?.id||'gpt';picker.onchange=()=>void this.showRepair(jid,rootResult,picker.value).catch(this.error);
      label.append(picker);controls.append(label);panel.append(controls);
      panel.append(element('p',rootResult.comparison_note||'相同单张 1024×1024 输入和提示词。GPT 请求 2048×2048，Seedream 原返回 1920×1920。本轮仅 RGB，没有运行新深度；切换只用于对比，不改变后续输入。','wipe-help'));
    }
    const outer=!result||result.composite?.direction==='outward';
    if(preview.capture_version===CAPTURE_VERSION)panel.append(element('p',`纯绿为几何未知区域。${preview.mask_cleanup?'已做 3×3 开运算并删除小于 16 像素的连通块。':''}${preview.mask_cleanup?.smoothing?'边缘已高斯羽化并以 50% 阈值恢复为二值 Mask。':''}真实黑色保留。`,'wipe-help'));
    panel.append(element('p',`相机 ${preview.camera_id} · ${result?.request?.model||(state.rgb_repair_model==='gpt-image-2'?'GPT Image 2':'Seedream')} · ${outer?'Mask 内完整使用新图，向外 4 像素羽化；过渡带外保持原图。':'旧版洞内羽化；Mask 外保持原图。'}`));
    panel.append(element('p','拖动竖线左右对比；六面用两条竖线分为三段，也可放大查看。所有“原图”均为带 Mask 的待修补图；API 返回是未经本地合成的模型输出。','wipe-help'));
    const accepted=quality.jointDepth?.accepted_bundle;
    const displayed=accepted?{...result,image:accepted.rgb}:result;
    this.repairResult=displayed;
    panel.append(this.repairPanorama(preview,displayed,generation));
    if(accepted)panel.append(element('p','上方默认显示 RGB 与深度联合通过的全景；各面原始返回仍保留供检查。','wipe-help'));
    if(accepted?.unresolved_mask||result?.unresolved_mask)panel.append(this.compare('剩余未修补区域',[['请求修补 Mask',`${preview.path}/mask.png`],['尚未修补',accepted?.unresolved_mask||result.unresolved_mask]]));
    const inputs=element('details');inputs.append(element('summary','查看待补图与二值 Mask'),this.compare('全景待补图 / 二值 Mask',[
      ['原图（带 Mask）',`${preview.path}/masked.png`],['二值 Mask',`${preview.path}/mask.png`]]));panel.append(inputs);
    panel.append(element('h2',result?.image?'六面 RGB · 原图（带 Mask） / API 返回 / 最终合成':'六面 RGB · 原图（带 Mask） / 二值 Mask'));
    const grid=element('div',null,'repair-faces');
    for(let i=0;i<6;i++){
      const name=FACES[i],original=`${preview.path}/${name}-masked.png`,record=result?.faces.find(r=>r.face===name);
      const pair=accepted?.faces.find(r=>r.face===name);
      const layers=result?.image?[
        ['原图（带 Mask）',original],[record?.skipped?'未调用 · 原图':'API 返回',record?.skipped?original:`${preview.path}/repair/${name}-model.png`],
        [pair&&!pair.accepted?'此面弃用 · 保留缺口':'RGB 合成',pair&&!pair.accepted?original:`${preview.path}/repair/${name}-repaired.png`]]:[['原图（带 Mask）',original],['二值 Mask',`${preview.path}/${name}-mask.png`]];
      const card=element('section');card.append(this.compare(`${LABELS[i]} · RGB · FOV 100°`,layers,1),this.qualityCard(name,'rgb',result,quality));grid.append(card);
    }
    panel.append(grid);
    if(result?.image){
      const checks=outer?`洞内未完整采用新图：${result.inside_generated_mismatch_pixels} 像素；外羽化带之外变化：${result.outside_feather_changed_pixels} 像素；外侧过渡带实际变化：${result.outside_mask_changed_pixels} 像素。`:`最终全景 Mask 外变化像素：${result.outside_mask_changed_pixels}。`;
      panel.append(element('p',`${checks}无洞的面没有调用 API，中间一段按原图显示并单独标注。`));
      $('description').textContent=result.description;
      $('download').hidden=false;$('download').href=this.url(displayed.image)+'?download=1';$('download').textContent='下载当前补洞全景 PNG ↗';
      const seedVariant=variants.find(v=>v.id==='seedream-outer');
      if(seedVariant&&!rootResult.comparison_note)panel.append(this.compare('同一外羽化 · Seedream / GPT',[
        ['Seedream',seedVariant.image],['GPT Image 2',rootResult.image]]));
    }else{
      panel.append(element('p',state.repair_running?`正在以最多 3 个请求并行调用 ${state.rgb_repair_model==='gpt-image-2'?'GPT Image 2':'Seedream'}；已成功的面会保存，运行日志在下方。`:'新版 Mask 和六面待补图已就绪，尚未重新生成 RGB、深度及高斯补丁。','repair-progress'));
      $('description').textContent=`待补 Mask 仅来自几何未知区域，已完成开运算和小连通块清理。${preview.mask_cleanup?.smoothing?'边缘平滑采用高斯 σ=1 px 羽化后取 50% 边缘。':''}`;
      $('download').hidden=false;$('download').href=this.url(`${preview.path}/masked.png`)+'?download=1';$('download').textContent='下载当前待补全景 PNG ↗';
    }
  }
  showGaussianRepair(jid,result,depth){
    this.hide();this.jid=jid;this.state={revision:result?.revision||depth.revision};
    const panel=$('repair-panel');panel.replaceChildren();panel.hidden=false;panel.setAttribute('aria-label','本轮高斯补洞');
    if(!result){panel.append(element('p','正在投影 RGB-D、运行 SHARP 深度锚定并校验六面补丁，完成后自动加载。','repair-progress'));return;}
    panel.append(element('p',`相机 ${result.camera_id} · 本轮新增 ${result.added_count.toLocaleString()} 个高斯 · 本轮输入 ${result.base_count.toLocaleString()} 个高斯逐字节保留。补丁来自 Mask 与六环过渡区域。上方“原始高斯”始终显示首次生成、尚未补洞的场景。`));
    const pointFilter=result.acceptance.point_filter;
    if(pointFilter)panel.append(element('p',`已应用首次生成的点剔除规则：法向 82° → 64 邻居平均距离 15 cm。补丁筛选前 ${pointFilter.patch_before.toLocaleString()} 个，剔除 ${pointFilter.patch_removed.toLocaleString()} 个，保留 ${pointFilter.patch_kept.toLocaleString()} 个；过渡带 6 像素。`));
    if(result.warnings?.length)panel.append(element('p',`需复核：${result.warnings.join('；')}。原验收阈值保持不变，此版本保留严格点剔除后的结果。`,'novel-privacy'));
    panel.append(this.compare('同一相机实际渲染 · 本轮补洞前 / 补洞后',[['本轮补洞前',result.before_image],['本轮合并后',result.image]]));
    const filled=result.acceptance.face_results.reduce((sum,row)=>sum+(row.projection?.extrapolated_pixels||0),0);
    panel.append(element('p',`RGB 来自本轮融合全景；深度来自本轮 LingBot 直接全景。六面输入中 ${filled.toLocaleString()} 个无效采样按参考方法外推，已记录位置；有效 LingBot 深度保持不变。`));
    const links=element('div',null,'novel-actions');
    for(const [text,path] of [['下载新增补丁 PLY',result.patch],['下载合并后 PLY',result.download],['查看验收报告',`${result.path}/acceptance.json`]]){
      const a=element('a',text,'secondary');a.href=this.url(path)+(path.endsWith('.ply')?'?download=1':'');links.append(a);
    }
    panel.append(links);
  }
  async showDepth(jid,result,repair){
    this.hide();const generation=this.generation;this.jid=jid;this.state={revision:result?.revision||repair.revision};
    const panel=$('repair-panel');panel.replaceChildren();panel.hidden=false;panel.setAttribute('aria-label','本轮深度补全');
    panel.append(element('p',`相机 ${result?.camera_id||repair.camera_id} · 同一高斯场景 · 本轮补好的 RGB 引导 LingBot。透视面为相机 Z 深度，全景为径向距离，单位均为米。`));
    const input=element('details');input.append(element('summary','对比 RGB 引导图'),this.compare('RGB 引导全景',[
      ['原图（带 Mask）',`${repair.capture.path}/masked.png`],['补好的 RGB',repair.image]]));panel.append(input);
    if(!result){panel.append(element('p','正在本地渲染六面深度并进行 LingBot 补全；进度与日志见下方。前序结果保留不变。','repair-progress'));return;}
    const quality=await this.qualityContext(jid,result);if(generation!==this.generation)return;this.qualityNotice(panel,quality);
    if(result.accepted_bundle){
      panel.append(element('p','LingBot 实际输入为各自通过检查的面 RGB；全景只融合联合通过的面。'));
      panel.append(this.compare('RGB 联合接受结果',[['RGB 检查后',repair.image],['RGB 与深度联合通过',result.accepted_bundle.rgb]]));
      panel.append(this.compare('剩余未修补区域',[['请求修补 Mask',`${repair.capture.path}/mask.png`],['尚未修补',result.accepted_bundle.unresolved_mask]]));
    }
    const direct=result.panorama_source==='lingbot_raw_faces',panoramaLabel=direct?'LingBot 直接全景':'最终合成深度';
    panel.append(element('p',direct?'全景直接投影融合 LingBot 六面原始返回，不按 Mask 拼回原图。模型无有效深度的位置保留为零，并以纯绿显示。下方六面的局部合成保留作对照，不参与全景输出。':'全景拖动一条竖线对比；六面拖动两条竖线对比带 Mask 的原始输入、LingBot 原始返回与最终合成。输入中的待补深度数值仍为零，六面预览以纯绿标记；模型返回未做尺度校准、外推或边缘融合。','wipe-help'));
    const toolbar=element('div',null,'novel-actions'),label=element('label','全景左侧：'),select=element('select');select.setAttribute('aria-label','全景深度左侧版本');
    const labels={input:'原图（带 Mask）',hole:'深度待补 Mask'};
    for(const [value,text] of Object.entries(labels)){const option=element('option',text);option.value=value;select.append(option);}
    label.append(select);toolbar.append(label,element('span',`右侧：${panoramaLabel}`));
    const view=element('div');const show=()=>view.replaceChildren(this.compare('全景径向深度',[[labels[select.value],result.views[select.value]],[panoramaLabel,result.views.completed]]));select.onchange=show;show();
    const downloads=element('div',null,'novel-actions');
    for(const [key,label] of [['rendered','原始深度 NPY'],['input','带洞深度 NPY'],['completed',direct?'LingBot 全景 NPY':'最终深度 NPY']]){
      const a=element('a',label,'secondary');a.href=this.url(result.arrays[key])+'?download=1';downloads.append(a);
    }
    panel.append(toolbar,view,downloads,element('p',`统一色标：${result.preview_range_m[0].toFixed(2)}—${result.preview_range_m[1].toFixed(2)} m（仅显示拉伸，NPY 保留原始米值）。${direct?`全景有效率 ${(result.panorama_valid_coverage*100).toFixed(2)}%；无效深度 ${result.invalid_completed_pixels} 像素`:`区域外变化像素 ${result.outside_changed_pixels}`}；补洞覆盖率 ${(result.hole_coverage*100).toFixed(2)}%；原始额外无效深度 ${result.additional_invalid_panorama_pixels} 像素。`));
    if(result.warnings?.length)panel.append(element('p',result.warnings.join('；'),'novel-privacy'));
    panel.append(element('h2',`六面相机 Z 深度 · 原图（带 Mask） / LingBot 返回 / ${direct?'局部合成（对照）':'最终合成'}`));
    const grid=element('div',null,'repair-faces');
    const loading=element('p','正在读取已保存的模型深度并生成对比预览，不会重新推理…');panel.append(loading,grid);
    try{
      const data=await this.depthComparison(result);
      if(generation!==this.generation)return;
      if(data.depth_revision!==result.depth_revision)throw new Error('深度版本已变化，请刷新页面');
      for(let i=0;i<6;i++){
        const row=data.faces.find(r=>r.face===FACES[i]),card=element('section');card.append(imageWipe({title:`${LABELS[i]} · 深度 · FOV 100°`,aspect:1,layers:[
          {label:'原图（带 Mask）',url:row.views.original},{label:row.skipped?'未调用 · 原图':result.face_results.find(r=>r.face===FACES[i])?.quality_skipped?'RGB 拒收 · 未调用':'LingBot 返回',url:row.views.model},{label:direct?'局部合成（对照）':'最终合成',url:row.views.composite}]}),this.qualityCard(FACES[i],'depth',result,quality));grid.append(card);
      }
      loading.remove();
    }catch(error){if(generation===this.generation){loading.textContent='深度对比加载失败：'+error.message;throw error;}}
  }
  debug(){return {active:this.active,busy:this.busy,source:this.state?.source,recommended:this.state?.recommended,highlighted:this.state?.highlighted,
    frontier:this.state?.frontier,selected:this.state?.selected_ids,markers:this.labels.length,preview:this.popupCamera};}
}
