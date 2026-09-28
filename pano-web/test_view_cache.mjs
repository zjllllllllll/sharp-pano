import test from 'node:test';
import assert from 'node:assert/strict';
import {ViewCache} from './static/view-cache.mjs';

test('concurrent loads and instant reuse share the same resource',async()=>{
  let loads=0;const cache=new ViewCache(()=>{});
  const load=async()=>({id:++loads});
  const [a,b]=await Promise.all([cache.get('same',load),cache.get('same',load)]);
  assert.equal(a,b);assert.equal(loads,1);assert.equal(cache.peek('same'),a);
  assert.equal(await cache.get('same',load),a);
});
test('a failed load can retry; switching jobs disposes ready and late loads',async()=>{
  const disposed=[];const cache=new ViewCache(v=>disposed.push(v));
  await assert.rejects(cache.get('bad',()=>Promise.reject(new Error('fail'))));
  assert.equal(await cache.get('bad',async()=>2),2);
  let resolve;const pending=cache.get('late',()=>new Promise(r=>resolve=r));
  await Promise.resolve();cache.clear();resolve(3);
  await assert.rejects(pending,/失效/);assert.deepEqual(disposed,[2,3]);assert.equal(cache.ready,0);
});
test('new version evicts only the old resource',async()=>{
  const disposed=[];const cache=new ViewCache(v=>disposed.push(v));
  await cache.get('a',async()=>1);await cache.get('b',async()=>2);
  cache.delete('a');assert.deepEqual(disposed,[1]);assert.equal(cache.peek('b'),2);
});
