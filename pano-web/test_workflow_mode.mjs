import test from 'node:test';
import assert from 'node:assert/strict';
import {visibleStep, displayStep, latestVisibleStep, nextAutomaticStage} from './static/workflow-mode.mjs';
const job = (count, extra={}) => ({execution_mode:'auto',stages:Object.fromEntries(Array.from({length:count},(_,i)=>[i+1,{}])),...extra});

test('default hides depth and each round preparation; debug retains every stage',()=>{
  const all=Array.from({length:25},(_,i)=>i+1);
  assert.deepEqual(all.filter(s=>visibleStep(s)),[1,3,4,5,9,13,17,21,25]);
  assert.deepEqual(all.filter(s=>visibleStep(s,'debug')),all);
  for(const [count,visible] of [[2,1],[6,5],[8,5],[10,9],[12,9],[24,21]]) {
    assert.equal(latestVisibleStep(job(count)),visible);
    assert.equal(latestVisibleStep(job(count),'debug'),count);
  }
  assert.equal(latestVisibleStep(job(13),'auto',8),5);
});
test('automatic execution includes every hidden stage and ends at existing round limit',()=>{
  for(let done=1;done<25;done++)assert.equal(nextAutomaticStage(job(done)),done+1);
  assert.equal(nextAutomaticStage(job(25)),null);
});
test('viewing legacy or debug jobs cannot start paid repair automatically',()=>{
  assert.equal(nextAutomaticStage(job(6,{execution_mode:undefined})),null);
  assert.equal(nextAutomaticStage(job(6,{execution_mode:'debug'})),null);
  assert.equal(nextAutomaticStage(null),null);
});
test('running, failed, stopped, and quality capture states do not advance',()=>{
  for(const extra of [{busy:true},{error:'failed'},{iteration_status:'stopped'},
    ...['awaiting_capture','exhausted','cancelled','error'].map(status=>({quality_run:{status}}))]) {
    assert.equal(nextAutomaticStage(job(9,extra)),null);
  }
  assert.equal(nextAutomaticStage(job(7,{quality_run:{status:'ready'}})),8);
  assert.equal(nextAutomaticStage(job(9,{quality_run:{status:'complete'}})),10);
});

test('a stale automatic request displays the result completed by another tab',async()=>{
  const {readFileSync}=await import('node:fs');
  const {runInNewContext}=await import('node:vm');
  const source=readFileSync(new URL('./static/app.js',import.meta.url),'utf8');
  const advance=source.slice(source.indexOf('async function advanceStep('),source.indexOf("$('next').onclick=()=>advanceStep();"));
  const selected=[],requests=[];
  const context={job:job(8),advancing:false,automatic:true,automaticFailed:null,
    gaussianDense:true,gaussianVariant:'unmasked',visibleStep,mode:'auto',
    localStep:step=>step<6?step:6+(step-6)%4,
    renderState(){},showError(){},activeEntry:{id:9},viewCache:{retain(){}},
    selectStep:async step=>selected.push(step),
    api:async(url,options)=>{requests.push(JSON.parse(options.body));return job(9,{iteration_status:'stopped'});},
  };
  runInNewContext(advance+';globalThis.advance=advanceStep;',context);
  await context.advance(9);
  assert.deepEqual(requests,[{expected_stage:9,gaussian_variant:'masked'}]);
  assert.deepEqual(selected,[9]);
  assert.equal(context.advancing,false);
});

test('default display numbers are consecutive across hidden steps and repair rounds',()=>{
  const visible=[1,3,4,5,9,13,17,21,25];
  assert.deepEqual(visible.map(step=>displayStep(step)),[1,2,3,4,5,6,7,8,9]);
  for(let step=1;step<=25;step++)assert.equal(displayStep(step,'debug'),step);
});

test('starting the next automatic stage preserves the newly selected viewer',async()=>{
  const {readFileSync}=await import('node:fs');
  const {runInNewContext}=await import('node:vm');
  const source=readFileSync(new URL('./static/app.js',import.meta.url),'utf8');
  const advance=source.slice(source.indexOf('async function advanceStep('),source.indexOf("$('next').onclick=()=>advanceStep();"));
  const active={id:'visible-result'},retained=[];
  const context={job:job(4),advancing:false,automatic:true,automaticFailed:null,activeEntry:active,
    visibleStep,mode:'auto',localStep:step=>step<6?step:6+(step-6)%4,
    renderState(){},showError(message){assert.equal(message,null);},
    disposeObject(){assert.fail('Current viewer was disposed');},
    viewCache:{retain(value){retained.push(value);},clear(){assert.fail('Current cache was cleared');}},
    document:{body:{classList:{add(){assert.fail('Current viewer was hidden');}}}},
    selectStep(){assert.fail('Next stage has no result to select yet');},
    api:async()=>job(4,{busy:true,running_stage:5}),
  };
  runInNewContext(advance+';globalThis.advance=advanceStep;',context);
  await context.advance(5);
  assert.deepEqual(retained,[active]);assert.equal(context.advancing,false);
});

test('automatic advancement waits for a user-selected viewer to finish loading',async()=>{
  const {readFileSync}=await import('node:fs');
  const {runInNewContext}=await import('node:vm');
  const source=readFileSync(new URL('./static/app.js',import.meta.url),'utf8');
  const action=source.slice(source.indexOf('async function advanceAutomatically(){'),source.indexOf('async function prepareAutomaticCapture(step){'));
  const started=[];
  const context={automatic:true,viewSelections:1,advancing:false,polling:false,qualityResuming:false,
    loadingUpload:false,automaticFailed:null,job:job(4),nextAutomaticStage,
    advanceStep:async step=>started.push(step)};
  runInNewContext(action+';globalThis.advance=advanceAutomatically;',context);
  await context.advance();assert.deepEqual(started,[]);
  context.viewSelections=0;await context.advance();assert.deepEqual(started,[5]);
});
