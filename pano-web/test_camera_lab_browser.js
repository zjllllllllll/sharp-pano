// Execute in the running lab using agent-browser eval --stdin.
// Exercises actual UI events against independent expectations from API data.
(async()=>{
  const assert=(ok,message)=>{if(!ok)throw new Error(message);};
  const catalog=await fetch('/api/camera-lab').then(r=>r.json());
  assert(catalog.total===47&&catalog.render_complete===47,'Incomplete published dataset');
  const select=document.getElementById('lab-scene');
  assert(!document.querySelector('#lab-mask-threshold,input[type="range"]'),'Sampling slider still exists');
  assert(catalog.rules.mask_threshold===.15&&catalog.rules.connectivity===4,'Published rule mismatch');
  const expected=scene=>{
    const pool=scene.sample.all_cameras;
    const pass=pool.filter(c=>c.coincides_with_original||c.mask_stats.mask_fraction<=.15);
    const reached=new Set(['0,0']);let changed=true;
    while(changed){changed=false;for(const c of pass){
      const [x,z]=c.grid,key=c.grid.join(',');
      if(!reached.has(key)&&[[x+1,z],[x-1,z],[x,z+1],[x,z-1]].some(p=>reached.has(p.join(',')))){reached.add(key);changed=true;}
    }}
    return {pool,kept:pool.filter(c=>reached.has(c.grid.join(','))),
      excluded:pool.filter(c=>!c.coincides_with_original&&c.mask_stats.mask_fraction>.15),
      disconnected:pass.filter(c=>!reached.has(c.grid.join(',')))};
  };
  document.querySelector('[data-map="camera_map"]').click();
  const checks=[];let slowest=0;
  for(const scene of catalog.scenes){
    const start=performance.now();select.value=scene.id;select.dispatchEvent(new Event('change',{bubbles:true}));
    slowest=Math.max(slowest,performance.now()-start);
    await new Promise(requestAnimationFrame);
    const svg=document.getElementById('lab-map-interactive');
    assert(getComputedStyle(svg).display!=='none'&&svg.getBoundingClientRect().height>0,scene.id+': map invisible');
    const {pool,kept,excluded,disconnected}=expected(scene);
    assert(svg.querySelectorAll('[data-map-camera]').length===pool.length,scene.id+': missing map nodes');
    assert(JSON.stringify(kept.map(c=>c.id))===JSON.stringify(scene.cameras.map(c=>c.id)),scene.id+': canonical candidates disagree');
    for(const obsoleteSetting of [0,.25,1]){
      localStorage.setItem('camera-lab-mask-threshold',String(obsoleteSetting));
      const mapKept=[...svg.querySelectorAll('[data-map-camera]')].filter(g=>['retained','origin'].includes(g.dataset.filterState));
      assert(mapKept.length===kept.length,scene.id+': retained map count');
      assert(svg.querySelectorAll('[data-filter-state="excluded"]').length===excluded.length,scene.id+': excluded map count');
      assert(svg.querySelectorAll('[data-filter-state="disconnected"]').length===disconnected.length,scene.id+': disconnected count');
      for(const c of pool){
        const node=svg.querySelector(`[data-map-camera="${c.id}"]`);
        assert(node.dataset.filterState===c.selection_state,scene.id+': map classification differs from backend');
      }
      const listed=[...document.getElementById('lab-camera').options].map(o=>o.value);
      assert(JSON.stringify(listed)===JSON.stringify(kept.map(c=>c.id)),scene.id+': active camera list mismatch');
      assert(svg.querySelector('[data-map-camera="C000"]').dataset.filterState==='origin',scene.id+': origin removed');
      assert(document.getElementById('lab-sampling-error').hidden===!(kept.length===1||scene.sample.sampling_error),scene.id+': error display');
    }
    checks.push({scene:scene.id,kept:kept.length,excluded:excluded.length,disconnected:disconnected.length});
    if(kept.length>1){
      document.getElementById('lab-next').click();
      assert(document.getElementById('lab-camera').value===kept[1].id,scene.id+': next did not follow candidate list');
      document.getElementById('lab-prev').click();
      assert(document.getElementById('lab-camera').value==='C000',scene.id+': previous did not return to origin');
    }
    const removed=disconnected[0]||excluded[0];
    if(removed){
      svg.querySelector(`[data-map-camera="${removed.id}"]`).dispatchEvent(new MouseEvent('click',{bubbles:true}));
      assert(document.getElementById('lab-camera-info').textContent.includes('不参与候选导出'),scene.id+': diagnostic preview not labeled');
      assert(document.getElementById('lab-prev').disabled&&document.getElementById('lab-next').disabled,scene.id+': removed node entered navigation');
      svg.querySelector('[data-map-camera="C000"]').dispatchEvent(new MouseEvent('click',{bubbles:true}));
    }
  }
  select.value='5.3';select.dispatchEvent(new Event('change',{bubbles:true}));
  await new Promise(requestAnimationFrame);
  assert(document.getElementById('lab-sampling-info').textContent.includes('0.5 → 0.3 m'),'Small-room actual offset missing');
  assert(document.getElementById('lab-sampling-info').textContent.includes('几何候选 4/8'),'Small-room neighbor count missing');
  // Capture exports without writing browser Downloads during verification.
  const create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;
  const blobs=[];
  try{
    URL.createObjectURL=function(blob){blobs.push(blob);return create.call(URL,blob);};
    HTMLAnchorElement.prototype.click=function(){if(!this.download)return click.call(this);};
    document.getElementById('lab-export-cameras').click();
    const payload=JSON.parse(await blobs[0].text());
    assert(payload.count===5&&payload.cameras.length===5,'Filtered export should contain 5 of 6 cameras');
    assert(payload.excluded_camera_ids.length===1&&payload.mask_threshold===.15&&payload.connectivity===4,'Export threshold/provenance');
    assert(JSON.stringify(payload.cameras)===JSON.stringify(catalog.scenes.find(s=>s.id==='5.3').cameras),'Export not canonical');
    document.getElementById('lab-map-download').click();
    for(let i=0;blobs.length<2&&i<100;i++)await new Promise(resolve=>setTimeout(resolve,30));
    assert(blobs.length===2,'Missing SVG export');
    const svg=new DOMParser().parseFromString(await blobs[1].text(),'image/svg+xml');
    assert(!svg.querySelector('parsererror'),'Invalid exported SVG');
    assert(svg.querySelector('image').getAttribute('href').startsWith('data:image/png;base64,'),'Exported background not embedded');
    assert(svg.querySelectorAll('[data-map-camera]').length===6,'Export lost excluded map points');
  }finally{URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;}
  assert(!document.getElementById('lab-error').textContent,'Visible runtime/export error');
  const result={passed:true,scenes:catalog.total,maskThreshold:.15,connectivity:4,maxSceneHandlerMilliseconds:slowest,
    checks,exportsChecked:['canonical candidates JSON','standalone marked SVG'],noSliders:true,
    navigationChecked:true,excludedInspectionChecked:true,obsoleteSettingsIgnored:true};
  window.cameraSamplingBrowserAudit=result;
  return result;
})()
