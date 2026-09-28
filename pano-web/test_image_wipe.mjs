import {test} from 'node:test';
import assert from 'node:assert/strict';
import {moveDivider,readDepthNpy,depthPixels} from './static/novel-view.js';

test('single divider reaches both edges without altering original state',()=>{
  const stops=[50];assert.deepEqual(moveDivider(stops,0,-10),[0]);assert.deepEqual(moveDivider(stops,0,120),[100]);assert.deepEqual(stops,[50]);
});
test('two dividers cannot cross and reject non-finite input',()=>{
  assert.deepEqual(moveDivider([30,70],0,90),[70,70]);assert.deepEqual(moveDivider([30,70],1,-9),[30,30]);
  assert.deepEqual(moveDivider([30,70],0,NaN),[30,70]);assert.deepEqual(moveDivider([30,70],4,40),[30,70]);
});
test('arbitrary pointer sequences preserve clipping order',()=>{
  let stops=[33,67];
  for(let i=0;i<1000;i++){stops=moveDivider(stops,i%2,Math.sin(i)*150);assert.ok(stops[0]>=0&&stops[0]<=stops[1]&&stops[1]<=100);}
});
