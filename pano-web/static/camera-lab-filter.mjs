// Fixed threshold followed by C000's four-connected 0.5 m grid component.
export const MASK_THRESHOLD=.15;
export const SELECTION_VERSION='mask15-origin-four-connected-v1';
export function maskFraction(camera){
  const value=camera?.mask_stats?.mask_fraction;
  return typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=1?value:null;
}
const cache=new WeakMap();
export function summarize(cameras){
  if(cache.has(cameras))return cache.get(cameras);
  const groups={origin:[],retained:[],excluded:[],disconnected:[],pending:[]},states=new Map(),grid=new Map();
  for(const c of cameras){
    const value=maskFraction(c);
    states.set(c.id,c.coincides_with_original?'origin':value===null?'pending':value>MASK_THRESHOLD?'excluded':'disconnected');
    grid.set(c.grid.join(','),c);
  }
  const queue=grid.has('0,0')?[[0,0]]:[];
  while(queue.length){
    const [x,z]=queue.pop();
    for(const [dx,dz] of [[1,0],[-1,0],[0,1],[0,-1]]){
      const c=grid.get([x+dx,z+dz].join(','));
      if(c&&states.get(c.id)==='disconnected'){states.set(c.id,'retained');queue.push(c.grid);}
    }
  }
  for(const c of cameras)groups[states.get(c.id)].push(c);
  const kept=cameras.filter(c=>['origin','retained'].includes(states.get(c.id)));
  const result={...groups,states,kept,neighbors:kept.filter(c=>Math.max(...c.grid.map(Math.abs))===1).length,
    onlyOrigin:kept.length===1&&groups.origin.length===1&&groups.pending.length===0};
  cache.set(cameras,result);return result;
}
export function filterState(camera,cameras){return summarize(cameras).states.get(camera.id);}
