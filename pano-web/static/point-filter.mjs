// Preview-only: point order, geometry, original colors and downstream data stay intact.
export const FIXED_NORMAL_ANGLE = 82;
export const DISTANCE_NEIGHBORS = [4, 8, 16, 32, 64];
export const FIXED_DISTANCE_NEIGHBORS = 64;
export const FIXED_DISTANCE_CM = 15;
export function angleStatistics(angles) {
  const bins = new Uint32Array(181);
  let unreliable = 0;
  for (let i = 0; i < angles.length; i++) {
    const angle = angles[i];
    if (!Number.isFinite(angle) || angle < 0 || angle > 90) {
      angles[i] = -1; // Same keep-visible sentinel in CPU counts and the shader.
      unreliable++;
    } else bins[Math.ceil(angle * 2)]++;
  }
  const cumulative = new Uint32Array(181);
  let count = unreliable;
  for (let i = 0; i < bins.length; i++) cumulative[i] = count += bins[i];
  return {total: angles.length, unreliable, counts: cumulative};
}

export function visibleAt(statistics, threshold) {
  return statistics.counts[Math.round(Math.max(0, Math.min(90, threshold)) * 2)];
}

export function distanceStatistics(angles, distances, stride=DISTANCE_NEIGHBORS.length) {
  if (distances.length !== angles.length*stride) throw new Error('距离与点云数量不匹配');
  const counts = Array.from({length:stride}, () => new Uint32Array(1001));
  const unknownCounts = Array.from({length:stride}, () => new Uint32Array(1001));
  let normalKept=0, unreliable=0;
  for (let i=0; i<angles.length; i++) {
    if (angles[i]>FIXED_NORMAL_ANGLE) continue;
    normalKept++;
    const unknown=angles[i]<0;
    if (unknown) unreliable++;
    for (let k=0; k<stride; k++) {
      const value=distances[i*stride+k];
      if (Number.isNaN(value) || value<0) throw new Error('法向保留点缺少有效距离');
      if (value>1) continue; // Above the maximum 100 cm threshold, always hidden when enabled.
      let bin=Math.floor(value*1000);
      // Match float32 shader thresholds exactly, including equality at e.g. 10 cm.
      if (value>Math.fround(bin/1000)) bin++;
      if (bin>1000) continue;
      counts[k][bin]++;
      if (unknown) unknownCounts[k][bin]++;
    }
  }
  for (let k=0; k<stride; k++) for (let b=1; b<=1000; b++) {
    counts[k][b]+=counts[k][b-1];
    unknownCounts[k][b]+=unknownCounts[k][b-1];
  }
  return {counts,unknownCounts,normalKept,unreliable,total:angles.length};
}

export function distanceCounts(stats, kIndex, centimeters, enabled=true) {
  const bin=Math.round(Math.max(0,Math.min(100,centimeters))*10);
  const visible=enabled?stats.counts[kIndex][bin]:stats.normalKept;
  const unknownVisible=enabled?stats.unknownCounts[kIndex][bin]:stats.unreliable;
  return {visible,hidden:stats.normalKept-visible,unknownRemoved:stats.unreliable-unknownVisible};
}

export function attachAngleShader(material, threshold, distance=null) {
  material.onBeforeCompile = shader => {
    shader.uniforms.normalAngleThreshold = threshold;
    shader.vertexShader = 'attribute float normalAngle;\nvarying float vNormalAngle;\n' + shader.vertexShader;
    shader.vertexShader = shader.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\nvNormalAngle = normalAngle;');
    shader.fragmentShader = 'uniform float normalAngleThreshold;\nvarying float vNormalAngle;\n' + shader.fragmentShader;
    // Discard before color/depth output, so hidden points do not occlude others.
    shader.fragmentShader = shader.fragmentShader.replace('#include <clipping_planes_fragment>',
      '#include <clipping_planes_fragment>\nif (vNormalAngle > normalAngleThreshold) discard;');
    if (distance) {
      shader.uniforms.neighborDistanceThreshold=distance.threshold;
      shader.uniforms.neighborDistanceEnabled=distance.enabled;
      shader.vertexShader='attribute float meanNeighborDistance;\nvarying float vMeanNeighborDistance;\n'+shader.vertexShader;
      shader.vertexShader=shader.vertexShader.replace('#include <begin_vertex>',
        '#include <begin_vertex>\nvMeanNeighborDistance = meanNeighborDistance;');
      shader.fragmentShader='uniform float neighborDistanceThreshold;\nuniform float neighborDistanceEnabled;\nvarying float vMeanNeighborDistance;\n'+shader.fragmentShader;
      shader.fragmentShader=shader.fragmentShader.replace('if (vNormalAngle > normalAngleThreshold) discard;',
        'if (vNormalAngle > normalAngleThreshold) discard;\nif (neighborDistanceEnabled > 0.5 && vMeanNeighborDistance > neighborDistanceThreshold) discard;');
    }
  };
  material.customProgramCacheKey = () => `sharp-point-filter-v2-${distance?'distance':'normal'}`;
  material.needsUpdate = true;
}
