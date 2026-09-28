import test from 'node:test';
import assert from 'node:assert/strict';
import {filterState,summarize,MASK_THRESHOLD} from './static/camera-lab-filter.mjs';
const make=(id,grid,mask)=>({id,grid,coincides_with_original:id==='C000',mask_stats:{mask_fraction:mask}});
const cameras=[make('C000',[0,0],1),make('C001',[0,1],.15),make('C002',[1,1],.151),make('C003',[2,0],null)];
test('strict threshold equality and original exemption',()=>{
  assert.equal(MASK_THRESHOLD,.15);
  assert.equal(filterState(cameras[0],cameras),'origin');assert.equal(filterState(cameras[1],cameras),'retained');
  assert.equal(filterState(cameras[2],cameras),'excluded');assert.equal(filterState(cameras[3],cameras),'pending');
});
test('fixed rule ignores obsolete threshold argument and does not mutate input',()=>{
  const before=JSON.stringify(cameras),low=summarize(cameras,.25),high=summarize(cameras,1);
  assert.deepEqual(low.kept.map(c=>c.id),['C000','C001']);assert.equal(low.neighbors,1);
  assert.equal(high.kept.length,2);assert.equal(high.pending.length,1);assert.equal(JSON.stringify(cameras),before);
});
test('only original reports error; pending renders are not false errors',()=>{
  assert.equal(summarize(cameras,0).onlyOrigin,false);
  assert.equal(summarize([cameras[0],cameras[2]]).onlyOrigin,true);
});
test('missing, invalid mask stats are never silently accepted',()=>{
  for(const value of [null,undefined,NaN,Infinity,-.1,1.1]){
    const c=make('C004',[1,0],value);assert.equal(filterState(c,[cameras[0],c]),'pending');
  }
});
test('diagonal and gap do not connect; excluded bridge cuts the branch',()=>{
  const pool=[cameras[0],make('C001',[1,1],.1),make('C002',[1,0],.2),make('C003',[2,0],.1),make('C004',[-2,0],.1)];
  const result=summarize(pool);assert.equal(result.kept.length,1);assert.equal(result.disconnected.length,3);
});
test('valid cardinal paths reach diagonals and preserve input order',()=>{
  const pool=[cameras[0],make('C001',[1,1],.1),make('C002',[1,0],.1),make('C003',[0,-1],.1),make('C004',[-1,-1],.1)];
  assert.deepEqual(summarize(pool).kept,pool);
});
