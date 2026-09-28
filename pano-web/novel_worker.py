from pathlib import Path
import argparse
import time
from novel_pipeline import prepare,repair,write_json,read

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',type=int,choices=[6,7,8,9]);p.add_argument('job',type=Path)
    a=p.parse_args();start=time.perf_counter()
    if a.stage==9:
        from novel_gaussian import repair_gaussians
        from quality_retry import QualityNoContribution
        try:result=repair_gaussians(a.job)
        except QualityNoContribution as exc:result=dict(quality_no_contribution=True,reason=str(exc))
    elif a.stage==8:
        from novel_depth import complete_depth
        result=complete_depth(a.job)
    elif a.stage==7 and read(a.job/'job.json').get('rgb_repair_model')=='gpt-image-2':
        from gpt_green_repair import repair_fresh
        result=repair_fresh(a.job)
    elif a.stage==6 and read(a.job/'job.json').get('iteration_parent'):
        from novel_iteration import prepare_iteration
        result=prepare_iteration(a.job)
    else:result=(prepare if a.stage==6 else repair)(a.job)
    result['seconds']=time.perf_counter()-start
    write_json(a.job/f'stage-{a.stage}.json',result)
    print('本步骤完成',flush=True)
