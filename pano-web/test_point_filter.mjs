import test from 'node:test';
import assert from 'node:assert/strict';
import {FIXED_NORMAL_ANGLE, angleStatistics, visibleAt, attachAngleShader,
  FIXED_DISTANCE_NEIGHBORS, FIXED_DISTANCE_CM, distanceStatistics, distanceCounts} from './static/point-filter.mjs';

test('production mask is fixed to 64 neighbors and 15 cm, strict boundary', () => {
  assert.equal(FIXED_DISTANCE_NEIGHBORS,64);
  assert.equal(FIXED_DISTANCE_CM,15);
  const stats=distanceStatistics(new Float32Array([82,83,-1,-1]),new Float32Array([.15,-1,.15,.151]),1);
  assert.equal(distanceCounts(stats,0,FIXED_DISTANCE_CM).visible,2);
});

test('strictly greater threshold, unreliable retention, reversible half-degree steps', () => {
  const angles = new Float32Array([-1, 0, 45, 80, 80.1, 80.5, 90, NaN, 100]);
  const stats = angleStatistics(angles);
  assert.equal(stats.unreliable, 3);
  assert.equal(visibleAt(stats, 90), 9);
  assert.equal(visibleAt(stats, 80), 6);
  assert.equal(visibleAt(stats, 80.5), 8);
  assert.equal(visibleAt(stats, 0), 4);
  assert.equal(visibleAt(stats, 90), 9);
  for (let t=0; t<=90; t+=.5) {
    assert.equal(visibleAt(stats, t), angles.filter(a=>a<0 || a<=t).length);
  }
});

test('GPU threshold uniform changes without modifying point arrays', () => {
  const material = {}, threshold = {value: 90};
  attachAngleShader(material, threshold);
  const shader = {uniforms: {}, vertexShader: '#include <begin_vertex>',
    fragmentShader: '#include <clipping_planes_fragment>\n#include <opaque_fragment>'};
  material.onBeforeCompile(shader);
  assert.match(shader.vertexShader, /vNormalAngle = normalAngle/);
  assert.match(shader.fragmentShader, /vNormalAngle > normalAngleThreshold\) discard/);
  assert.ok(shader.fragmentShader.indexOf('discard') < shader.fragmentShader.indexOf('opaque_fragment'));
  threshold.value = 85;
  assert.equal(shader.uniforms.normalAngleThreshold.value, 85);
});

test('fixed 82 degrees precedes distance filtering and unreliable normals can be removed', () => {
  assert.equal(FIXED_NORMAL_ANGLE,82);
  const angles=new Float32Array([0,82,82.1,-1,-1]);
  const values=new Float32Array([.01,.02, .03,.04, -1,-1, .10,.11, 0,0]);
  const stats=distanceStatistics(angles,values,2);
  assert.equal(stats.normalKept,4);
  assert.deepEqual(distanceCounts(stats,0,3),{visible:3,hidden:1,unknownRemoved:1});
  assert.deepEqual(distanceCounts(stats,1,3),{visible:2,hidden:2,unknownRemoved:1});
  assert.deepEqual(distanceCounts(stats,0,3,false),{visible:4,hidden:0,unknownRemoved:0});
  assert.equal(distanceCounts(stats,0,10).visible,4); // float32 equality must survive.
});

test('histogram agrees with float32 GPU threshold at every millimeter', () => {
  const values=new Float32Array(Array.from({length:2002},(_,i)=>i/2000));
  const angles=new Float32Array(values.length);
  const stats=distanceStatistics(angles,values,1);
  for(let mm=0;mm<=1000;mm++) {
    const expected=values.filter(d=>d<=Math.fround(mm/1000)).length;
    assert.equal(distanceCounts(stats,0,mm/10).visible,expected,`at ${mm} mm`);
  }
});

test('combined shader masks normal then distance, with reversible distance toggle', () => {
  const material={},threshold={value:82},distance={threshold:{value:.03},enabled:{value:1}};
  attachAngleShader(material,threshold,distance);
  const shader={uniforms:{},vertexShader:'#include <begin_vertex>',fragmentShader:'#include <clipping_planes_fragment>\n#include <opaque_fragment>'};
  material.onBeforeCompile(shader);
  assert.match(shader.vertexShader,/vMeanNeighborDistance = meanNeighborDistance/);
  assert.ok(shader.fragmentShader.indexOf('if (vNormalAngle')<shader.fragmentShader.indexOf('if (neighborDistanceEnabled'));
  assert.equal(shader.uniforms.normalAngleThreshold.value,82);
  distance.enabled.value=0;
  assert.equal(shader.uniforms.neighborDistanceEnabled.value,0);
});
