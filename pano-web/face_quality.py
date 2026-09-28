"""Deterministic, CPU-only consistency gates for RAW model returns.

Scores are engineering consistency scores, not calibrated probabilities.
Never register, fit, composite, or extrapolate predictions before evaluation.
"""
import hashlib
import json
import numpy as np
from scipy import ndimage as ndi

VERSION = 'face-quality-v2-obvious-failure'
PARAMETERS = dict(version=VERSION, size=512, anchor_alpha=.90, margin=8,
    min_coverage=.20, grid=4, min_cells=4, cell_coverage=.25, min_rows_cols=2,
    ssim_sigma=1.5, ssim_radius=5, edge_sigma=1., edge_radius=3,
    edge_threshold=.03, edge_tolerance=2, min_edges=128, min_edge_fraction=.005,
    component_pixels=16, score_pass=80, depth_epsilon=.001,
    collapse_min_range_m=.20, collapse_min_relative_range=.10, collapse_ratio=.20,
    rgb_limits=dict(ssim=(.70,.98), color_error=(.12,.02), edge_f1=(.50,.95),
                    bad_fraction=(.30,.02), largest_bad_fraction=(.15,.005)),
    depth_limits=dict(valid_fraction=(.90,1.), median_relative_error=(.15,.01),
                      p90_relative_error=(.50,.03), bad_fraction=(.40,.01),
                      largest_bad_fraction=(.30,.005)),
    warning_limits=dict(rgb=dict(ssim=.90,color_error=.06,edge_f1=.80,bad_fraction=.10,largest_bad_fraction=.03),
        depth=dict(valid_fraction=.98,median_relative_error=.03,p90_relative_error=.10,bad_fraction=.05,largest_bad_fraction=.02)))
POLICY_HASH = hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest()
HIGHER = {'ssim','edge_f1','valid_fraction'}
REASONS = dict(insufficient_evidence='有效参照面积或空间分布不足',
    invalid_input='输入尺寸、数值或来源不匹配', missing_input='缺少原始输入文件',
    ssim='原有结构变化', color_error='原有颜色变化', edge_f1='原物体边缘变化',
    bad_fraction='异常区域占比超限', largest_bad_fraction='连续异常区域过大',
    valid_fraction='有效深度比例不足', median_relative_error='深度中位误差超限',
    p90_relative_error='深度 P90 误差超限', depth_collapse='深度层次塌缩',
    upstream_rejected='同面的 RGB 未通过', paired_rejected='同面的另一阶段未通过')


def record(kind, status, reasons=(), **extra):
    return dict(kind=kind, policy_version=VERSION, policy_hash=POLICY_HASH,
        status=status, score=0. if any(r in ('invalid_input','missing_input') for r in reasons) else None, reason_codes=list(reasons),
        reasons=[REASONS.get(r,r) for r in reasons], metrics={}, **extra)


def evidence(mask):
    mask=np.asarray(mask,bool); cells=np.array([[p.mean()>=.25 for p in np.array_split(row,4,axis=1)]
                                              for row in np.array_split(mask,4,axis=0)])
    result=dict(pixels=int(mask.sum()), coverage=float(mask.mean()), cells=int(cells.sum()),
        rows=int(cells.any(1).sum()), columns=int(cells.any(0).sum()))
    result['sufficient']=(result['coverage']>=.20 and result['cells']>=4 and
                          result['rows']>=2 and result['columns']>=2)
    return result


def anchors(mask, geometry, alpha, source_z=None):
    mask=np.asarray(mask,bool);geometry=np.asarray(geometry,bool);alpha=np.asarray(alpha)
    if mask.shape!=(512,512) or geometry.shape!=mask.shape or alpha.shape!=mask.shape:
        raise ValueError('Expected registered 512x512 inputs')
    if alpha.dtype==np.uint8:alpha=alpha.astype(np.float64)/255.
    excluded=mask|geometry
    distance=ndi.distance_transform_edt(~excluded) if excluded.any() else np.full(mask.shape,np.inf)
    a=(distance>8)&np.isfinite(alpha)&(alpha>=.90)
    a[:8]=False;a[-8:]=False;a[:,:8]=False;a[:,-8:]=False
    if source_z is not None:
        source_z=np.asarray(source_z)
        if source_z.shape!=mask.shape:raise ValueError('Unregistered source depth')
        a &= np.isfinite(source_z)&(source_z>.001)
    return a


def supported(mask, radius):
    return ndi.minimum_filter(np.asarray(mask,np.uint8),size=2*radius+1,mode='constant',cval=0)>0


def components(bad, minimum=16):
    labels,_=ndi.label(bad,structure=np.ones((3,3),bool))
    counts=np.bincount(labels.ravel());keep=counts>=minimum;keep[0]=False
    cleaned=keep[labels]
    return cleaned,int(counts[keep].max()) if keep.any() else 0


def metric_score(name,value,limit,good):
    # The same signed interpolation handles both larger/smaller-is-better.
    return float(np.clip(80+20*(value-limit)/(good-limit),0,100))


def finish(kind, metrics, areas, hard=(), maps=None):
    limits=PARAMETERS[kind+'_limits'];scores={};failed=list(hard)
    for name,value in metrics.items():
        if name not in limits or value is None:continue
        limit,good=limits[name];scores[name]=metric_score(name,value,limit,good)
        if (value<limit if name in HIGHER else value>limit):failed.append(name)
    sufficient=all(a['sufficient'] for a in areas.values())
    if not sufficient:failed.insert(0,'insufficient_evidence')
    status='insufficient_evidence' if not sufficient else 'rejected' if failed else 'passed'
    r=record(kind,status,failed);r.update(metrics=metrics,evidence=areas,limits=limits,
        component_scores=scores,score=None if not sufficient else 0. if hard else min(scores.values(),default=100.))
    warnings=[name for name,limit in PARAMETERS['warning_limits'][kind].items() if metrics.get(name) is not None
              and (metrics[name]<limit if name in HIGHER else metrics[name]>limit)]
    r.update(warning_codes=warnings,warnings=[REASONS[n] for n in warnings],warning_limits=PARAMETERS['warning_limits'][kind])
    return r,maps or {}


def ssim_map(a,b):
    filt=lambda x:ndi.gaussian_filter(x,1.5,radius=5,mode='reflect')
    ma,mb=filt(a),filt(b)
    va=np.maximum(filt(a*a)-ma*ma,0);vb=np.maximum(filt(b*b)-mb*mb,0)
    cov=filt(a*b)-ma*mb
    return np.clip(((2*ma*mb+.01**2)*(2*cov+.03**2))/
                   ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2)),-1,1)


def evaluate_rgb(source,prediction,mask,geometry,alpha):
    try:
        a=anchors(mask,geometry,alpha)
        x=np.asarray(source,np.float64)/255.;y=np.asarray(prediction,np.float64)/255.
        if x.shape!=(512,512,3) or y.shape!=x.shape or not np.isfinite([x,y]).all():
            raise ValueError('Unregistered RGB')
    except (ValueError,TypeError):return record('rgb','rejected',['invalid_input']),{}
    gray=lambda v:v@np.array([.2126,.7152,.0722])
    gx,gy=gray(x),gray(y);s=ssim_map(gx,gy);v=supported(a,5)
    small=lambda z:z.reshape(256,2,256,2).mean((1,3))
    v2=supported(a.reshape(256,2,256,2).all((1,3)),5)
    s2=ssim_map(small(gx),small(gy))
    mean=lambda values,where:float(values[where].mean()) if where.any() else 0.
    color=np.abs(x-y).mean(-1)
    bad,largest=components(v&(((s<.75)&(color>.05))|(color>.15)))
    edge_support=supported(a,4)
    def edge(z):
        z=ndi.gaussian_filter(z,1.,radius=3,mode='reflect')
        return (np.hypot(ndi.sobel(z,0)/8,ndi.sobel(z,1)/8)>.03)&edge_support
    e0,e1=edge(gx),edge(gy);en0,en1=int(e0.sum()),int(e1.sum())
    required=max(128,.005*edge_support.sum());f1=None;edge_note='纹理不足，不适用'
    if (e0|e1).sum()>=required:
        if min(en0,en1)<required<=max(en0,en1):f1=0.
        else:
            near0=ndi.distance_transform_edt(~e0)<=2 if en0 else np.zeros_like(e0)
            near1=ndi.distance_transform_edt(~e1)<=2 if en1 else np.zeros_like(e1)
            precision=float((e1&near0).sum())/max(1,en1);recall=float((e0&near1).sum())/max(1,en0)
            f1=2*precision*recall/max(precision+recall,1e-15)
        edge_note='双向 2 px 匹配'
    metrics=dict(ssim=min(mean(s,v),mean(s2,v2)),ssim_512=mean(s,v),ssim_256=mean(s2,v2),
        color_error=mean(color,a),edge_f1=f1,edge_note=edge_note,source_edges=en0,predicted_edges=en1,
        bad_fraction=float(bad.sum()/max(1,v.sum())),largest_bad_fraction=float(largest/max(1,v.sum())))
    return finish('rgb',metrics,dict(anchor=evidence(a),ssim_512=evidence(v),ssim_256=evidence(v2),edge=evidence(edge_support)),
        maps=dict(anchors=a,metric_support=v,difference=np.where(a,color,0),bad=bad))


def evaluate_depth(source,prediction,model_valid,mask,geometry,alpha):
    try:
        x=np.asarray(source,np.float64);y=np.asarray(prediction,np.float64)
        a=anchors(mask,geometry,alpha,x);valid=np.asarray(model_valid,bool).copy()
        if y.shape!=x.shape or valid.shape!=x.shape:raise ValueError('Unregistered depth')
    except (ValueError,TypeError):return record('depth','rejected',['invalid_input']),{}
    valid &= np.isfinite(y)&(y>.001);good=a&valid
    error=np.zeros_like(x);error[good]=np.abs(y[good]-x[good])/np.maximum(x[good],.001)
    bad,largest=components(a&(~valid|(error>.10)),minimum=1)
    med,p90=np.quantile(error[good],[.5,.9]) if good.any() else (1e6,1e6)
    source_range=float(np.ptp(np.quantile(x[good],[.1,.9]))) if good.any() else 0.
    prediction_range=float(np.ptp(np.quantile(y[good],[.1,.9]))) if good.any() else 0.
    collapsed=(good.any() and source_range>=max(.20,.10*float(np.median(x[good]))) and
               prediction_range<.20*source_range)
    metrics=dict(valid_fraction=float(good.sum()/max(1,a.sum())),median_relative_error=float(med),
        p90_relative_error=float(p90),bad_fraction=float(bad.sum()/max(1,a.sum())),
        largest_bad_fraction=float(largest/max(1,a.sum())),source_range_m=source_range,
        predicted_range_m=prediction_range,collapse=bool(collapsed))
    return finish('depth',metrics,dict(anchor=evidence(a)),hard=['depth_collapse'] if collapsed else (),
        maps=dict(anchors=a,metric_support=a,difference=np.where(a,np.where(valid,error,1),0),bad=bad))


def joint(rgb,depth):
    allowed={'passed','not_needed'}
    accepted=rgb['status'] in allowed and depth['status'] in allowed
    scores=[r['score'] for r in (rgb,depth) if r.get('score') is not None]
    return dict(accepted=accepted,status='accepted' if accepted else 'rejected',
        score=min(scores) if scores and len(scores)==sum(r['status']!='not_needed' for r in (rgb,depth)) else None,
        rgb_status=rgb['status'],depth_status=depth['status'],
        reasons=[] if accepted else [kind for kind,r in [('rgb',rgb),('depth',depth)] if r['status'] not in allowed])
