// Display policy does not change the stages executed by the pipeline.
export const visibleStep = (step, mode = 'auto') => mode === 'debug' ||
  (step !== 2 && (step <= 5 || (step - 9) % 4 === 0));
export function displayStep(step, mode = 'auto') {
  if (mode === 'debug') return step;
  return Array.from({length:step}, (_,i)=>i+1).filter(s=>visibleStep(s,mode)).length;
}
export function latestVisibleStep(job, mode = 'auto', before = Infinity) {
  return Math.max(1, ...Object.keys(job?.stages || {}).map(Number)
    .filter(step => step <= before && visibleStep(step, mode)));
}
export function nextAutomaticStage(job) {
  if (!job || job.execution_mode !== 'auto' || job.busy || job.error ||
      job.iteration_status === 'stopped' ||
      ['awaiting_capture', 'exhausted', 'cancelled', 'error'].includes(job.quality_run?.status)) return null;
  const next = Object.keys(job.stages).length + 1;
  return next <= 25 ? next : null;
}
