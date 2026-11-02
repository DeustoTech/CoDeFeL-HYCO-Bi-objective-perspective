"""Reproducible elliptic Bi-HYCO campaign E0-E8.

Reuses exact geometry/FEM/architectures of the supplied
BiHYCO_XPINN_Elliptic_3Seeds.ipynb, and the previously generated coefficient
misspecification references. Everything stays local; no external downloads.

Scientific guardrails:
 - Truth parameters and fine-grid state NEVER enter training, validation selection
   or hyperparameter tuning; used only for post-selection evaluation.
 - All model comparisons share data_id, reference and mesh.
 - No model receives the unknown Gaussian bump in its forward/residual model.
 - Training caches are keyed by method, optimizer, split, data hash and parameters.
 - E8 changes the parameterization; it is an extension, not a repetition of E1-E7.
"""
from __future__ import annotations
import copy, csv, hashlib, io, json, math, os, platform, shutil, time, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm
import scipy
from scipy.linalg import eigh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import brentq
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pandas as pd
import original_elliptic_core as orig

ROOT=Path(__file__).resolve().parent
REF=ROOT/'references'; CACHE=ROOT/'cache'; RUNS=ROOT/'runs'; REPORTS=ROOT/'reports'; FIGURES=ROOT/'figures'
ALGORITHM_VERSION='v2.0-reduced-2026-10-27'
BASE_SEED=1240; BASE_TRIAL_SEEDS=(1251,1252,1253,1254,1255)
BG=np.array([2.0,1.0]); INIT=np.array([0.5,0.5])
LEVELS=(0.0,0.01,0.03); REGIONS=('Omega1','Omega2')
DEVICE=orig.DEVICE
for p in (REF,CACHE,RUNS,REPORTS,FIGURES): p.mkdir(parents=True,exist_ok=True)

# Explicit FULL/PILOT distinction prevents publishing short-test results by accident.
MODE='FULL'; VAL_EVERY=500; XP_VAL_EVERY=1000
FULL_ROUNDS=20000; FULL_XP_EPOCHS=30000
PILOT_ROUNDS=8; PILOT_XP_EPOCHS=4

def setup(mode='FULL', rounds=None, xp_epochs=None):
    global MODE, VAL_EVERY, XP_VAL_EVERY, FULL_ROUNDS, FULL_XP_EPOCHS
    mode=str(mode).upper()
    if mode not in ('FULL','PILOT'): raise ValueError('MODE must be FULL or PILOT')
    MODE=mode
    if rounds is not None:
        if mode=='FULL': FULL_ROUNDS=int(rounds)
        else:
            global PILOT_ROUNDS
            PILOT_ROUNDS=int(rounds)
    if xp_epochs is not None:
        if mode=='FULL': FULL_XP_EPOCHS=int(xp_epochs)
        else:
            global PILOT_XP_EPOCHS
            PILOT_XP_EPOCHS=int(xp_epochs)
    VAL_EVERY=500 if mode=='FULL' else 4
    XP_VAL_EVERY=1000 if mode=='FULL' else 2
    print('Campaign mode:',MODE, '| Bi-HYCO rounds:',training_rounds(),'| XPINN epochs:',xp_epochs_count())

def training_rounds(): return FULL_ROUNDS if MODE=='FULL' else PILOT_ROUNDS

def xp_epochs_count(): return FULL_XP_EPOCHS if MODE=='FULL' else PILOT_XP_EPOCHS

def sha_bytes(*chunks):
    h=hashlib.sha256()
    for o in chunks:
        if isinstance(o,str): h.update(o.encode())
        elif isinstance(o,bytes): h.update(o)
        else:
            a=np.ascontiguousarray(np.asarray(o)); h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()[:20]

def key_json(obj): return json.dumps(obj,sort_keys=True,separators=(',',':'),default=str)

def safe_id(s): return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in str(s))

def levstr(x): return f'm{int(round(float(x)*100)):02d}'

def noise_str(x): return f'eta{int(round(float(x)*100)):02d}'

def ref_path(region,mismatch):
    if region not in REGIONS or float(mismatch) not in LEVELS: raise ValueError((region,mismatch))
    return REF/f'uref-elliptic-coeff-{region}-{levstr(mismatch)}.npz'

def load_reference(region,mismatch):
    path=ref_path(region,mismatch)
    if not path.is_file(): raise FileNotFoundError(f'Required bundled reference missing: {path}. Unzip the FULL package.')
    with np.load(path,allow_pickle=False) as z: r={k:np.array(z[k]) for k in z.files}
    assert str(r['perturbation_type'])=='diffusion_coefficient'
    assert str(r['bump_region'])==region
    assert abs(float(r['target_model_mismatch'])-float(mismatch))<1e-12
    assert np.allclose(r['a_true'],BG)
    assert len(r['y_phy_clean'])==len(r['y_syn_clean'])==800
    if mismatch>0: assert float(r['epsilon'])>0
    r['reference_id']=sha_bytes(r['points_fine'],r['u_reference_fine'],r['true_coefficient_fine_elements'])
    return r

_FEMS={}
def get_fems():
    if 'pair' not in _FEMS:
        coarse=orig.FixedFEM(*orig.build_unstructured_mesh(900,180,160,BASE_SEED))
        fine=orig.FixedFEM(*orig.build_unstructured_mesh(1800,360,320,BASE_SEED+999))
        _FEMS['pair']=(coarse,fine)
    return _FEMS['pair']

def check_ref_mesh(r,coarse=None,fine=None):
    coarse,fine=(coarse,fine) if coarse is not None else get_fems()
    assert np.allclose(r['points'],coarse.points,atol=1e-12,rtol=0), 'coarse FEM mismatch'
    assert np.array_equal(r['triangles'],coarse.triangles), 'coarse connectivity mismatch'
    assert np.allclose(r['points_fine'],fine.points,atol=1e-12,rtol=0), 'fine FEM mismatch'
    assert np.array_equal(r['triangles_fine'],fine.triangles), 'fine connectivity mismatch'
    assert np.allclose(r['lumped_mass'],coarse.mass,atol=1e-12)

def _val_test_splits():
    f=CACHE/'splits_v1.npz'
    if f.exists():
        with np.load(f,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}
    d={'x1_val':orig.sample_uniform_in_inner(200,seed=BASE_SEED+3001),
       'x2_val':orig.sample_uniform_in_outer_shell(200,seed=BASE_SEED+3002),
       'x1_test':orig.sample_uniform_in_inner(200,seed=BASE_SEED+4001),
       'x2_test':orig.sample_uniform_in_outer_shell(200,seed=BASE_SEED+4002)}
    assert all(np.all(orig.inside_outer_np(x)) for x in d.values())
    np.savez_compressed(f,**d)
    return d

def _case_noise_seed(seed,region,mismatch,noise,kind):
    # Fixed, independent RNG streams for observation noise; independent of training seed.
    # At eta=0 the data are IDENTICAL for all training seeds.
    q=f'{region}/{mismatch:.4f}/{noise:.4f}/{kind}/{seed if noise else "identical"}'
    return int(hashlib.sha256(q.encode()).hexdigest()[:8],16)

def ensure_data(region,mismatch,noise,seed):
    """Immutable train/validation/test split with local noisy observations."""
    noise=float(noise); seed=int(seed)
    r=load_reference(region,mismatch)
    f=CACHE/'data'/f'{region}_{levstr(mismatch)}_{noise_str(noise)}_seed{seed}.npz'
    f.parent.mkdir(exist_ok=True,parents=True)
    if f.exists():
        with np.load(f,allow_pickle=False) as z: d={k:np.array(z[k]) for k in z.files}
        if str(d['reference_id'])!=r['reference_id']: raise ValueError('Old cached observations do not match reference.')
        return d
    split=_val_test_splits()
    interp=orig.make_interpolator(r['points_fine'],r['u_reference_fine'])
    d={'x1_train':r['X_phy'],'x2_train':r['X_syn'],
       'x1_val':split['x1_val'],'x2_val':split['x2_val'],
       'x1_test':split['x1_test'],'x2_test':split['x2_test']}
    # Note: original train observations are the same ones as in the supplied notebook.
    for q in ('train','val','test'):
        for k in (1,2):
            clean=np.asarray(r['y_phy_clean'] if k==1 else r['y_syn_clean']) if q=='train' else np.asarray(interp(d[f'x{k}_{q}']))
            sd=float(np.std(clean))
            rng=np.random.default_rng(_case_noise_seed(seed,region,mismatch,noise,f'{q}/{k}'))
            d[f'y{k}_{q}']=clean+noise*sd*rng.normal(size=len(clean))
            d[f'y{k}_{q}_clean']=clean
    d['data_id']=np.array(sha_bytes(r['reference_id'],*(d[k] for k in ('x1_train','y1_train','x2_train','y2_train','x1_val','y1_val','x2_val','y2_val','x1_test','y1_test','x2_test','y2_test'))))
    d['reference_id']=np.array(r['reference_id'])
    d['region']=np.array(region);d['mismatch']=np.array(mismatch);d['eta']=np.array(noise)
    np.savez_compressed(f,**d)
    return d

def case(region,mismatch,noise,seed):
    r=load_reference(region,mismatch)
    d=ensure_data(region,mismatch,noise,seed)
    check_ref_mesh(r)
    fem,_=get_fems()
    return dict(region=region,mismatch=float(mismatch),eta=float(noise),seed=int(seed),
                ref=r,data=d,fem=fem,data_id=str(d['data_id']),reference_id=r['reference_id'])

def mesh_cache():
    """Return a checked FEM spectral cache, tolerating platform-specific roundoff.

    Earlier releases compared a bitwise SHA including the FEM right-hand side.
    Tiny differences in floating-point assembly on another operating system
    can change that SHA even when the stored decomposition is correct. We now
    *numerically* verify the cached physical solver against the current sparse
    FEM for three independent coefficient vectors. An incompatible cache is
    safely moved aside and rebuilt, without touching reference or run files.
    """
    f = CACHE / 'spectral_2parameter_v1.npz'
    coarse, _ = get_fems()
    mesh_id = sha_bytes(coarse.points, coarse.triangles, coarse.rhs)

    if f.exists():
        reason = None
        try:
            with np.load(f, allow_pickle=False) as z:
                eig = np.asarray(z['eig'], dtype=np.float64)
                V = np.asarray(z['V'], dtype=np.float64)
                load = np.asarray(z['load'], dtype=np.float64)
                free = np.asarray(z['free'], dtype=np.int64)
                cached_id = str(z['mesh_id']) if 'mesh_id' in z.files else ''

            n = len(coarse.free)
            if (not np.array_equal(free, coarse.free) or
                eig.shape != (n,) or V.shape != (n, n) or load.shape != (n,) or
                not np.all(np.isfinite(eig)) or
                not np.all(np.isfinite(V)) or
                not np.all(np.isfinite(load))):
                reason = 'incompatible free nodes, array shapes, or nonfinite values'
            else:
                # These checks also detect stale stiffness/load assemblies.
                for a1, a2 in ((0.5, 0.5), (2.0, 1.0), (3.5, 1.1)):
                    denom = a2 + (a1 - a2) * eig
                    if not np.all(np.isfinite(denom)) or np.min(np.abs(denom)) < 1e-12:
                        reason = 'invalid spectral denominators'
                        break
                    u_cached = np.zeros(len(coarse.points), dtype=np.float64)
                    u_cached[free] = V @ (load / denom)
                    u_fem = coarse.solve_numpy((a1, a2))
                    rel_error = norm(u_cached - u_fem) / max(norm(u_fem), 1e-12)
                    if not np.isfinite(rel_error) or rel_error > 1e-7:
                        reason = f'forward-solver mismatch (relative error {rel_error:.3e})'
                        break

            if reason is None:
                if cached_id != mesh_id:
                    warnings.warn(
                        'Spectral-cache checksum differs across environments, '
                        'but the stored basis agrees with the current FEM; reusing it.',
                        RuntimeWarning, stacklevel=2)
                return f
        except (OSError, ValueError, KeyError, TypeError) as ex:
            reason = f'could not read or verify cache: {ex}'

        # A *genuinely* incompatible file must not be used for training.
        backup = CACHE / 'spectral_2parameter_v1.incompatible.npz'
        index = 1
        while backup.exists():
            backup = CACHE / f'spectral_2parameter_v1.incompatible_{index}.npz'
            index += 1
        f.rename(backup)
        warnings.warn(
            f'Spectral cache rejected ({reason}); moved to {backup.name}. '
            'Rebuilding from the current FEM.', RuntimeWarning, stacklevel=2)

    k1 = coarse.K1_free.toarray()
    k0 = (coarse.K1_free + coarse.K2_free).toarray()
    eig, V = eigh(k1, k0, check_finite=False)
    np.savez_compressed(f, eig=eig, V=V,
                        load=V.T @ coarse.rhs[coarse.free],
                        free=coarse.free, mesh_id=mesh_id)
    return f

_PHY={}
def physical_torch():
    if 'physical' not in _PHY:
        f=mesh_cache()
        with np.load(f,allow_pickle=False) as z:
            eig,V,load,free=[np.array(z[k]) for k in ('eig','V','load','free')]
        n=len(get_fems()[0].points)
        _PHY['physical']=orig.SpectralFEM(n,free,eig,V,load,DEVICE).to(DEVICE)
    return _PHY['physical']

def prepare_problem(c):
    fem=c['fem'];d=c['data'];interp=orig.NodalInterpolator(fem.points)
    return dict(fem=fem,physical=physical_torch(),interpolator=interp,
        observation_map=interp.torch_map(d['x1_train'],DEVICE),
        x1_train=orig.tensor(d['x1_train']),x2_train=orig.tensor(d['x2_train']),
        x1_val=orig.tensor(d['x1_val']),x2_val=orig.tensor(d['x2_val']),
        reference=c['ref']['u_reference'],case=c)

def state_errors(pred,ref,fem): return orig.state_errors(np.asarray(pred),np.asarray(ref),fem)

def test_and_val(pred_fn,case_):
    d=case_['data']
    output={}
    for q in ('val','test'):
        vv=[]
        for k in (1,2):
            pred=np.asarray(pred_fn(d[f'x{k}_{q}']))
            y=np.asarray(d[f'y{k}_{q}'])
            vv.append(float(np.mean((pred-y)**2)))
        output[f'{q}_mse']=.5*sum(vv)
        output[f'{q}_1_mse']=vv[0];output[f'{q}_2_mse']=vv[1]
    return output

def physical_validation(a,c,split='val'):
    fem=c['fem'];d=c['data'];u=fem.solve_numpy(a)
    inter=orig.NodalInterpolator(fem.points)
    return .5*sum(float(np.mean((inter.torch_map(d[f'x{k}_{split}'],DEVICE)(orig.tensor(u)).detach().cpu().numpy()-d[f'y{k}_{split}'])**2)) for k in (1,2))

def quick_interp_eval(u,xy,fem):
    idx,w=orig.NodalInterpolator(fem.points).indices_weights(xy)
    return (u[idx]*w).sum(axis=-1)

def evaluate_selected(c,a,u_direct=None,method='GN-dist'):
    """Ground truth is accessed only here, AFTER method and checkpoint selection."""
    fem=c['fem'];r=c['ref'];d=c['data'];a=np.asarray(a,dtype=float)
    u=fem.solve_numpy(a)
    def map_phys(xy): return quick_interp_eval(u,xy,fem)
    v=test_and_val(map_phys,c)
    v.update({f'e_phy_{k}':val for k,val in state_errors(u,r['u_reference'],fem).items()})
    delta=(a-BG)/BG
    v.update(a1=float(a[0]),a2=float(a[1]),e_bg=float(norm(delta)),data_id=c['data_id'],
             region=c['region'],mismatch=c['mismatch'],eta=c['eta'],seed=c['seed'],
             method=method,reference_id=c['reference_id'])
    tri=r['triangles_fine'];cent=r['points_fine'][tri].mean(axis=1)
    a_model=np.where(orig.inside_inner_np(cent),a[0],a[1]);a_true=np.asarray(r['true_coefficient_fine_elements'])
    tri_nodes=r['points_fine'][tri]
    area=np.abs(np.cross(tri_nodes[:,1]-tri_nodes[:,0],tri_nodes[:,2]-tri_nodes[:,0]))/2
    v['e_coefficient']=float(np.sqrt(np.sum(area*(a_model-a_true)**2)/(np.sum(area*a_true**2)+1e-30)))
    if u_direct is not None:
        u_direct=np.asarray(u_direct,dtype=float)
        es=state_errors(u_direct,r['u_reference'],fem)
        v.update({f'e_direct_{k}':vv for k,vv in es.items()})
        def map_direct(xy): return quick_interp_eval(u_direct,xy,fem)
        v.update({f'direct_{k}':vv for k,vv in test_and_val(map_direct,c).items()})
    return v

def _atomic_npz(path,**data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.temporary.npz')
    np.savez_compressed(tmp,**data);os.replace(tmp,path)

def _portable_torch_structure(value):
    # Keep optimizer tensors but convert NumPy scalars/arrays to safe Python types.
    # This allows torch.load(..., weights_only=True) without relaxing safety.
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {k:_portable_torch_structure(v) for k,v in value.items()}
    if isinstance(value,list):return [_portable_torch_structure(v) for v in value]
    if isinstance(value,tuple):return tuple(_portable_torch_structure(v) for v in value)
    return value

def _atomic_torch(path,blob):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.temporary.pt')
    torch.save(_portable_torch_structure(blob),tmp)
    os.replace(tmp,path)

def _atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str));os.replace(tmp,path)

def save_csv(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not records:
        pd.DataFrame().to_csv(path,index=False);return
    pd.DataFrame(records).to_csv(path,index=False)

def read_csv(path):
    p=Path(path)
    return pd.read_csv(p).to_dict('records') if p.exists() and p.stat().st_size>0 else []

def _run_folder(kind,case_,params):
    payload=dict(version=ALGORITHM_VERSION,kind=kind,data_id=case_['data_id'],seed=case_['seed'],params=params)
    digest=sha_bytes(key_json(payload))
    p=RUNS/MODE/kind/f'{case_["region"]}_{levstr(case_["mismatch"])}_{noise_str(case_["eta"])}_s{case_["seed"]}_{digest}'
    p.mkdir(exist_ok=True,parents=True)
    _atomic_json(p/'manifest.json',payload)
    return p

def _save_result(folder,row,files=None):
    _atomic_json(folder/'result.json',row)
    if files:
        _atomic_npz(folder/'states.npz',**files)

def _cached_result(folder):
    f=folder/'result.json'
    if f.exists():
        x=json.loads(f.read_text())
        if x.get('complete',False): return x
    return None

def grouped_summary(rows,by=('region','mismatch','eta','method')):
    if not rows:return pd.DataFrame()
    df=pd.DataFrame(rows)
    cols=[x for x in ('e_bg','e_coefficient','e_phy_global_error','e_direct_global_error','val_mse','test_mse','seconds','selected_step') if x in df]
    summary=df.groupby(list(by),dropna=False).agg({col:['mean','std','count'] for col in cols})
    summary.columns=['_'.join(map(str,x)) for x in summary.columns]
    summary=summary.reset_index()
    if 'data_id' in df:
        distinct=df.groupby(list(by),dropna=False)['data_id'].nunique().reset_index(name='n_distinct_datasets')
        summary=summary.merge(distinct,on=list(by),how='left',validate='one_to_one')
    return summary

def save_figure(fig,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p,dpi=600,bbox_inches='tight'); fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)

def evaluation_figure(case_,out):
    r=case_['ref'];fig,ax=plt.subplots(figsize=(7,6))
    tri=mtri.Triangulation(r['points_fine'][:,0],r['points_fine'][:,1],r['triangles_fine'])
    im=ax.tripcolor(tri,facecolors=r['true_coefficient_fine_elements'],cmap='viridis',shading='flat')
    theta=np.linspace(0,2*np.pi,300);rad=orig.interface_radius(theta)
    ax.plot(rad*np.cos(theta),rad*np.sin(theta),'w-',lw=1.5)
    rad=orig.outer_radius(theta);ax.plot(rad*np.cos(theta),rad*np.sin(theta),'k-',lw=1.3)
    ax.plot(case_['data']['x1_train'][:,0],case_['data']['x1_train'][:,1],'.',ms=.4,alpha=.22,color='black')
    ax.plot(case_['data']['x2_train'][:,0],case_['data']['x2_train'][:,1],'.',ms=.4,alpha=.22,color='white')
    ax.set_aspect('equal');ax.set_xlabel('x');ax.set_ylabel('y');ax.set_title(f'{case_["region"]}: {case_["mismatch"]:.0%} state discrepancy')
    fig.colorbar(im,ax=ax,label='True heterogeneous diffusion coefficient')
    save_figure(fig,out)


# -------------------------- Distributed physical inversion ----------------------
class LocalGN:
    """Only locally available data are accessed inside a unit."""
    def __init__(self,fem,xy,y,beta):
        self.idx,self.w=orig.NodalInterpolator(fem.points).indices_weights(xy)
        self.y=np.asarray(y,dtype=float);self.scale=np.sqrt(float(beta)/max(len(y),1))
    def pred(self,u): return np.sum(u[self.idx]*self.w,axis=1)
    def summary(self,u,du):
        r=self.scale*(self.pred(u)-self.y)
        J=self.scale*np.stack([self.pred(du[:,j]) for j in range(du.shape[1])],axis=1)
        return J.T@J,J.T@r,.5*float(r@r)
    def loss(self,u):
        r=self.scale*(self.pred(u)-self.y)
        return .5*float(r@r)

def forward_sensitivities(log_a,fem):
    a=np.exp(np.asarray(log_a));L=splu((a[0]*fem.K1_free+a[1]*fem.K2_free).tocsc())
    u=np.zeros(len(fem.points));u[fem.free]=L.solve(fem.rhs[fem.free])
    du=np.zeros((len(fem.points),2));uf=u[fem.free]
    for j,Kj in enumerate((fem.K1_free,fem.K2_free)):
        du[fem.free,j]=-L.solve(a[j]*(Kj@uf))
    return u,du

def run_gn(c,method='GN-dist',betas=(.5,.5),prior=0.0,max_iterations=70):
    # GN is deterministic: at eta=0 all training seeds have identical data.
    # Reuse its unique physical solve instead of rerunning an identical inversion.
    if c['eta']==0 and c['seed']!=BASE_TRIAL_SEEDS[0]:
        original=case(c['region'],c['mismatch'],c['eta'],BASE_TRIAL_SEEDS[0])
        if original['data_id']!=c['data_id']:
            raise AssertionError('Noise-free data unexpectedly differ across training seeds')
        canonical=run_gn(original,method=method,betas=betas,prior=prior,max_iterations=max_iterations)
        result=copy.deepcopy(canonical)
        result['seed']=c['seed']
        result['reused_deterministic_GN']=True
        result['n_distinct_observations']=1
        return result
    params=dict(method=method,betas=list(betas),prior=prior,max_iterations=max_iterations)
    folder=_run_folder('GN',c,params);cached=_cached_result(folder)
    if cached is not None:return cached
    d=c['data'];fem=c['fem']
    local=[LocalGN(fem,d[f'x{k}_train'],d[f'y{k}_train'],betas[k-1]) for k in (1,2)]
    lower=np.log(np.array([.05,.05]));upper=np.log(np.array([20.,20.]))
    log_a=np.log(INIT.copy());mu=1.e-3;n_forward=n_sens=0;history=[]
    start=time.perf_counter()
    for iteration in range(max_iterations):
        u,du=forward_sensitivities(log_a,fem);n_forward+=1;n_sens+=2
        sums=[z.summary(u,du) for z in local]
        H=sum((v[0] for v in sums),np.zeros((2,2)));g=sum((v[1] for v in sums),np.zeros(2))
        loss=sum(v[2] for v in sums)
        if prior>0:
            dx=log_a-np.log(INIT);H+=prior*np.eye(2);g+=prior*dx;loss+=prior/2*float(dx@dx)
        gnorm=float(norm(g,np.inf));hist=dict(iter=iteration,gnorm=gnorm,loss=loss,a1=float(np.exp(log_a[0])),a2=float(np.exp(log_a[1])),mu=mu)
        if gnorm<2e-11:
            history.append({**hist,'accepted':True});break
        accepted=False
        scale=max(np.trace(H)/2,1e-14)
        for trial in range(16):
            delta=np.linalg.solve(H+mu*scale*np.eye(2),-g)
            newx=np.clip(log_a+delta,lower,upper)
            if norm(newx-log_a)<1e-10:break
            unew=fem.solve_numpy(np.exp(newx));n_forward+=1
            newloss=sum(z.loss(unew) for z in local)
            if prior>0:newloss+=prior/2*float(norm(newx-np.log(INIT))**2)
            if np.isfinite(newloss) and newloss<loss:
                log_a=newx;mu=max(mu/3,1e-12);accepted=True;break
            mu=min(mu*6,1e12)
        history.append({**hist,'accepted':accepted})
        if not accepted:break
    a=np.exp(log_a);u=fem.solve_numpy(a);n_forward+=1
    seconds=time.perf_counter()-start
    row=evaluate_selected(c,a,method=method)
    row.update(complete=True,seconds=seconds,selection='converged GN; validation used only for GN-weighted weight choice',
         selected_step=len(history),n_forward_solves=n_forward,n_sensitivity_solves=n_sens,
         n_neural_steps=0,n_communications=len(history),
         transmitted_bytes=int(len(history)*(2*2*8+2*8+2*8)),
         beta_1=float(betas[0]),beta_2=float(betas[1]),prior=float(prior),
         output_folder=str(folder.relative_to(ROOT)),n_distinct_observations=1)
    _save_result(folder,row,dict(coefficients=a,physical_state=u))
    save_csv(folder/'history.csv',history)
    return row

def run_weighted_gn(c,beta_grid=(.10,.25,.50,.75,.90)):
    """Choose weights ONLY by held-out PHYSICAL validation loss."""
    trials=[run_gn(c,method=f'GN-weighted-b{beta:.2f}',betas=(beta,1-beta)) for beta in beta_grid]
    chosen=min(trials,key=lambda d:d['val_mse'])
    # Selection is based on observations alone; NEVER on e_bg.
    result=copy.deepcopy(chosen); result['method']='GN-weighted'
    result['gn_weight_grid']=list(beta_grid);result['selection']='min validation physical MSE'
    p=_run_folder('GN_weighted_selected',c,dict(grid=list(beta_grid)))
    _save_result(p,result)
    return result


# ------------------------------- Bi-HYCO --------------------------------------
def _validation_from_model(model,c):
    a=model.coefficients().detach().cpu().numpy().astype(float)
    if not np.isfinite(a).all(): raise FloatingPointError('Nonfinite Bi-HYCO coefficients')
    fem=c['fem'];u=fem.solve_numpy(a);d=c['data']
    val_phy=.5*sum(np.mean((quick_interp_eval(u,d[f'x{k}_val'],fem)-d[f'y{k}_val'])**2) for k in (1,2))
    with torch.no_grad():
        val_syn=.5*sum(np.mean((model.net(orig.tensor(d[f'x{k}_val'])).cpu().numpy().ravel()-d[f'y{k}_val'])**2) for k in (1,2))
    val_coop=.5*(np.mean((quick_interp_eval(u,d['x1_val'],fem)-d['y1_val'])**2)+
               np.mean((model.net(orig.tensor(d['x2_val'])).detach().cpu().numpy().ravel()-d['y2_val'])**2))
    return dict(val_phy=float(val_phy),val_syn=float(val_syn),val_coop=float(val_coop),a1=float(a[0]),a2=float(a[1]))

def run_bihyco(c,seed=None,interaction=1.,local_steps=2,algorithm='federated',epochs=None):
    """Faithful original two-copy Adam; parameter average only, persistent local moments.

    A few evaluations of true error may be stored in history for AFTER-THE-FACT
    diagnostics, but never influence stopping, selection or hyperparameters.
    """
    seed=int(c['seed'] if seed is None else seed)
    epochs=training_rounds() if epochs is None else int(epochs)
    if algorithm not in ('federated','central') :raise ValueError(algorithm)
    params=dict(interaction=float(interaction),local_steps=int(local_steps),algorithm=algorithm,
       hyco_width=orig.HYCO_WIDTH,hyco_depth=orig.HYCO_DEPTH,
       alpha_phy=orig.ALPHA_PHY,alpha_syn=orig.ALPHA_SYN,
       omega_phy=orig.OMEGA_PHY,omega_syn=orig.OMEGA_SYN,
       init=INIT.tolist(),clip=orig.HYCO_CLIP_NORM,lr_decay=orig.LR_DECAY)
    folder=_run_folder('BiHYCO',c,params)
    row=_cached_result(folder)
    if row is not None and int(row.get('completed_rounds',0))>=epochs:return row
    problem=prepare_problem(c)
    torch.manual_seed(seed)
    if DEVICE.type=='cuda':torch.cuda.manual_seed_all(seed)
    local=[orig.BiHYCO().to(DEVICE)]
    if algorithm=='federated':local.append(copy.deepcopy(local[0]))
    optimizers=[torch.optim.Adam([{'params':[mod.raw_coefficients],'lr':orig.HYCO_COEFFICIENT_LR},
                                  {'params':mod.net.parameters(),'lr':orig.HYCO_NETWORK_LR}]) for mod in local]
    rng=np.random.default_rng(seed+22001)
    chk=folder/'training_checkpoint.pt';completed=0;elapsed_past=0;hist=[];best_val=float('inf');best_step=0
    if chk.exists():
        saved=torch.load(chk,map_location=DEVICE,weights_only=True)
        if saved['params']!=params or saved['data_id']!=c['data_id']:
            raise ValueError('Bi-HYCO checkpoint mismatch: moved or modified data/configuration.')
        completed=int(saved['completed']);elapsed_past=float(saved['seconds'])
        hist=list(saved['history']);best_val=float(saved['best_val']);best_step=int(saved['best_step'])
        for j in range(len(local)):
            local[j].load_state_dict(saved['states'][j]);optimizers[j].load_state_dict(saved['optimizer_states'][j])
        rng.bit_generator.state=saved['numpy_rng_state']
        torch.set_rng_state(saved['torch_rng_cpu'].cpu())
        # GPU RNG state is intentionally restored if available.
        if DEVICE.type=='cuda' and saved.get('torch_rng_cuda') is not None:
            torch.cuda.set_rng_state_all([z.cpu() for z in saved['torch_rng_cuda']])
        print('Continuing existing Bi-HYCO checkpoint:',folder.name,'from round',completed)
    y1,y2=[orig.tensor(c['data'][f'y{k}_train']) for k in (1,2)]
    start=time.perf_counter()
    w1,w2=orig.UNIT_WEIGHTS
    for round_number in range(completed+1,epochs+1):
        ghost=orig.interaction_points(rng);z=orig.tensor(ghost)
        iz=problem['interpolator'].torch_map(ghost,DEVICE) if interaction>0 else None
        for idx,mod in enumerate(local):
            optim=optimizers[idx]
            for local_step in range(int(local_steps)):
                optim.zero_grad(set_to_none=True)
                a=mod.coefficients();u=problem['physical'](a)
                if algorithm=='federated':
                    if idx==0:
                        loss=orig.OMEGA_PHY*orig.ALPHA_PHY*((problem['observation_map'](u)-y1).square().mean()+orig.BETA_PHY*a.square().sum())
                    else:
                        loss=orig.OMEGA_SYN*orig.ALPHA_SYN*((mod.net(problem['x2_train']).flatten()-y2).square().mean()+.5*orig.BETA_SYN*sum(p.square().sum() for p in mod.net.parameters()))
                else:
                    loss=w1*orig.OMEGA_PHY*orig.ALPHA_PHY*((problem['observation_map'](u)-y1).square().mean()+orig.BETA_PHY*a.square().sum())
                    loss+=w2*orig.OMEGA_SYN*orig.ALPHA_SYN*((mod.net(problem['x2_train']).flatten()-y2).square().mean()+.5*orig.BETA_SYN*sum(p.square().sum() for p in mod.net.parameters()))
                if interaction>0:loss=loss+float(interaction)*(iz(u)-mod.net(z).flatten()).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError(f'Bi-HYCO loss became NaN at round {round_number}')
                loss.backward();torch.nn.utils.clip_grad_norm_(mod.net.parameters(),orig.HYCO_CLIP_NORM)
                optim.step()
        if len(local)==2:
            with torch.no_grad():
                a_avg=w1*local[0].coefficients()+w2*local[1].coefficients()
                if not torch.isfinite(a_avg).all():raise FloatingPointError('Non-finite aggregated parameters')
                pos=a_avg-orig.COEFFICIENT_FLOOR
                raw=pos+torch.log(-torch.expm1(-pos))
                for mod in local:mod.raw_coefficients.copy_(raw)
                for p1,p2 in zip(local[0].net.parameters(),local[1].net.parameters()):
                    avg=w1*p1+w2*p2;p1.copy_(avg);p2.copy_(avg)
        if round_number%orig.LR_DECAY_EVERY==0:
            for optim in optimizers:
                for group in optim.param_groups:group['lr']*=orig.LR_DECAY
        # Train checkpoint+validation only every VAL_EVERY steps, not using ground truth.
        if round_number%VAL_EVERY==0 or round_number==epochs:
            mdl=local[0];mdl.eval()
            vals=_validation_from_model(mdl,c)
            elapsed=elapsed_past+time.perf_counter()-start
            vals.update(step=round_number,seconds=elapsed)
            # The truth error below is DIAGNOSTIC ONLY, not used to select a model.
            vals['diagnostic_e_bg']=float(norm((np.array([vals['a1'],vals['a2']])-BG)/BG))
            hist.append(vals)
            if vals['val_phy']<best_val:
                best_val=float(vals['val_phy']);best_step=round_number
                _atomic_torch(folder/'model_best.pt',mdl.state_dict())
            # Always preserve final/resumable LOCAL optimizer states and RNG streams.
            blob=dict(params=params,data_id=c['data_id'],completed=round_number,seconds=elapsed,
                history=hist,best_val=best_val,best_step=best_step,
                states=[m.state_dict() for m in local],optimizer_states=[o.state_dict() for o in optimizers],
                numpy_rng_state=rng.bit_generator.state,torch_rng_cpu=torch.get_rng_state(),
                torch_rng_cuda=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None)
            _atomic_torch(chk,blob)
            save_csv(folder/'training_history.csv',hist)
    elapsed=elapsed_past+time.perf_counter()-start
    selected=orig.BiHYCO().to(DEVICE)
    selected.load_state_dict(torch.load(folder/'model_best.pt',map_location=DEVICE,weights_only=True))
    selected.eval()
    a=selected.coefficients().detach().cpu().numpy();u=c['fem'].solve_numpy(a)
    with torch.no_grad():u_syn=selected.net(orig.tensor(c['fem'].points)).flatten().cpu().numpy().astype(float)
    row=evaluate_selected(c,a,u_direct=u_syn,method=('Bi-HYCO' if interaction else 'Bi-HYCO-0') if algorithm=='federated' else 'Bi-HYCO-central')
    npar=sum(p.numel() for p in selected.parameters())
    row.update(complete=True,seconds=float(elapsed),selection='minimum held-out physical validation MSE',
       selected_step=int(best_step),completed_rounds=int(epochs),last_step=int(epochs),lambda_int=float(interaction),
       local_steps=int(local_steps),n_neural_steps=epochs*local_steps*len(local),
       n_forward_solves_approx=int(epochs*local_steps*len(local)+len(hist)),
       n_communications=epochs if len(local)==2 else 0,
       transmitted_bytes=int(epochs*len(local)*npar*4) if len(local)==2 else 0,
       output_folder=str(folder.relative_to(ROOT)),val_phy_selected=float(best_val),
       n_parameters=npar,implementation='original two-copy Adam + coefficient/parameter averaging' if len(local)==2 else 'centralized full objective Adam')
    _save_result(folder,row,dict(a_selected=a,physical_state=u,synthetic_state=u_syn))
    return row


# --------------------------------- XPINN ---------------------------------------
def run_xpinn(c):
    """Faithful two-network XPINN from supplied code, with validation checkpoints.

    For publication, the FULL setting uses its original 30,000 Adam epochs and
    its original loss and sampling schedule. Unknown bump is NEVER used in residual.
    """
    params=dict(width=orig.XPINN_WIDTH,depth=orig.XPINN_DEPTH,lr=orig.XPINN_LR,
       interface_flux_weight=2,interface_value_weight=10,collocation_batch=orig.BATCH_INTERIOR)
    folder=_run_folder('XPINN',c,params)
    epochs=xp_epochs_count()
    row=_cached_result(folder)
    if row is not None and int(row.get('completed_epochs',0))>=epochs:return row
    d=c['data'];seed=c['seed']
    torch.manual_seed(seed)
    if DEVICE.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=orig.XPINN().to(DEVICE)
    pool1=orig.tensor(orig.sample_uniform_in_inner(orig.POOL_INTERIOR,seed=seed+11001))
    pool2=orig.tensor(orig.sample_uniform_in_outer_shell(orig.POOL_INTERIOR,seed=seed+11002))
    outer_bnd=orig.tensor(orig.outer_boundary_points(orig.POOL_BOUNDARY))
    iface,normal=orig.interface_points_and_normals_np(orig.POOL_INTERFACE)
    iface_t,normal_t=orig.tensor(iface),orig.tensor(normal)
    unit1=orig.XPINNLocalUnit(model.inner,orig.tensor(d['x1_train']),d['y1_train'],pool1)
    unit2=orig.XPINNLocalUnit(model.outer,orig.tensor(d['x2_train']),d['y2_train'],pool2,outer_bnd)
    optim=torch.optim.Adam(model.parameters(),lr=orig.XPINN_LR)
    chk=folder/'training_checkpoint.pt';completed=0;elapsed_past=0;hist=[];best_val=float('inf');best_step=0
    if chk.exists():
        prev=torch.load(chk,map_location=DEVICE,weights_only=True)
        if prev['params']!=params or prev['data_id']!=c['data_id']:raise ValueError('XPINN checkpoint mismatch')
        model.load_state_dict(prev['model']);optim.load_state_dict(prev['optimizer'])
        completed=int(prev['completed']);elapsed_past=float(prev['seconds']);hist=list(prev['history'])
        best_val=float(prev['best_val']);best_step=int(prev['best_step'])
        torch.set_rng_state(prev['torch_rng_cpu'].cpu())
        if DEVICE.type=='cuda' and prev.get('torch_rng_cuda') is not None:
            torch.cuda.set_rng_state_all([z.cpu() for z in prev['torch_rng_cuda']])
        print('Continuing XPINN:',folder.name,'from epoch',completed)
    start=time.perf_counter()
    for ep in range(completed+1,epochs+1):
        optim.zero_grad(set_to_none=True)
        i1=torch.randint(len(pool1),(orig.BATCH_INTERIOR,),device=DEVICE)
        i2=torch.randint(len(pool2),(orig.BATCH_INTERIOR,),device=DEVICE)
        ib=torch.randint(len(outer_bnd),(orig.BATCH_BOUNDARY,),device=DEVICE)
        ii=torch.randint(len(iface_t),(orig.BATCH_INTERFACE,),device=DEVICE)
        local1=unit1.backward_local(model.coefficients()[0],i1)
        local2=unit2.backward_local(model.coefficients()[1],i2,ib)
        xf,nf=iface_t[ii],normal_t[ii]
        u1,dn1=orig.normal_gradient(model.inner,xf,nf)
        u2,dn2=orig.normal_gradient(model.outer,xf,nf)
        a1,a2=model.coefficients().unbind()
        coupling=10.*(u1-u2).square().mean()+2.*(a1*dn1-a2*dn2).square().mean()
        coupling.backward()
        optim.step()
        if ep%XP_VAL_EVERY==0 or ep==epochs:
            model.eval()
            a=model.coefficients().detach().cpu().numpy().astype(float)
            if not np.isfinite(a).all():raise FloatingPointError(f'XPINN parameters not finite at epoch {ep}')
            val_phy=physical_validation(a,c)
            direct1=orig.predict_xpinn(model,d['x1_val']);direct2=orig.predict_xpinn(model,d['x2_val'])
            val_nn=.5*(np.mean((direct1-d['y1_val'])**2)+np.mean((direct2-d['y2_val'])**2))
            elapsed=elapsed_past+time.perf_counter()-start
            snapshot=dict(step=ep,seconds=elapsed,val_phy=val_phy,val_direct=float(val_nn),
                          a1=float(a[0]),a2=float(a[1]),
                          diagnostic_e_bg=float(norm((a-BG)/BG)))
            hist.append(snapshot)
            if val_phy<best_val:
                best_val=float(val_phy);best_step=ep
                _atomic_torch(folder/'model_best.pt',model.state_dict())
            blob=dict(params=params,data_id=c['data_id'],completed=ep,seconds=elapsed,
               model=model.state_dict(),optimizer=optim.state_dict(),history=hist,
               best_val=best_val,best_step=best_step,torch_rng_cpu=torch.get_rng_state(),
               torch_rng_cuda=torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else None)
            _atomic_torch(chk,blob);save_csv(folder/'training_history.csv',hist)
    elapsed=elapsed_past+time.perf_counter()-start
    best=orig.XPINN().to(DEVICE)
    best.load_state_dict(torch.load(folder/'model_best.pt',map_location=DEVICE,weights_only=True));best.eval()
    a=best.coefficients().detach().cpu().numpy().astype(float)
    state=orig.predict_xpinn(best,c['fem'].points)
    row=evaluate_selected(c,a,u_direct=state,method='XPINN')
    row.update(complete=True,seconds=float(elapsed),selected_step=int(best_step),
        completed_epochs=int(epochs),selection='minimum held-out physical validation MSE',
        n_neural_steps=int(epochs),n_forward_solves_approx=int(len(hist)),
        n_communications=epochs,transmitted_bytes=0,
        n_parameters=sum(x.numel() for x in best.parameters()),
        output_folder=str(folder.relative_to(ROOT)))
    _save_result(folder,row,dict(a_selected=a,physical_state=c['fem'].solve_numpy(a),xpinn_state=state))
    return row

# --------------------------------- reporting ---------------------------------
def rows_to_summary(rows,file_prefix):
    if not rows:return pd.DataFrame()
    df=pd.DataFrame(rows).drop_duplicates(subset=['method','data_id','seed','lambda_int','local_steps'] if all(k in rows[0] for k in ('lambda_int','local_steps')) else ['method','data_id','seed'],keep='last')
    df.to_csv(REPORTS/f'{file_prefix}_runs.csv',index=False)
    summary=grouped_summary(df.to_dict('records'))
    summary.to_csv(REPORTS/f'{file_prefix}_summary.csv',index=False)
    return df

def paired_difference(df,m1='Bi-HYCO',m2='GN-dist',key='e_bg'):
    d1=df[df['method']==m1];d2=df[df['method']==m2]
    on=['region','mismatch','eta','seed','data_id']
    if d1.duplicated(on).any() or d2.duplicated(on).any():
        raise ValueError('Paired comparison requires one row per method, configuration and seed')
    z=d1.merge(d2,on=on,suffixes=('_A','_B'),validate='one_to_one')
    if not len(z):return pd.DataFrame()
    z[f'delta_{key}']=z[f'{key}_A']-z[f'{key}_B']
    return z

def basic_parameter_plot(df,prefix):
    if len(df)==0:return
    df=df.copy().dropna(subset=['e_bg'])
    cases=df[['region','mismatch','eta']].drop_duplicates().to_dict('records')
    cases=cases[:6]
    fig,axes=plt.subplots(1,len(cases),figsize=(max(6,3.7*len(cases)),4.1),squeeze=False)
    for ax,c in zip(axes[0],cases):
        subset=df[(df.region==c['region'])&(df.mismatch==c['mismatch'])&(df.eta==c['eta'])]
        methods=sorted(subset.method.unique())
        for j,m in enumerate(methods):
            values=subset[subset.method==m]['e_bg'].to_numpy()
            ax.scatter(j+np.linspace(-.11,.11,len(values)),values*100,s=26,alpha=.75)
        ax.set_xticks(range(len(methods)));ax.set_xticklabels(methods,rotation=32,ha='right',fontsize=8)
        ax.set_title(f"{c['region']}, {c['mismatch']:.0%} mismatch, eta={c['eta']:.2f}")
        ax.set_ylabel('Background-parameter error (%)');ax.grid(axis='y',alpha=.25)
    fig.suptitle('Individual training/data seeds (not only averages)',fontsize=11)
    save_figure(fig,FIGURES/f'{prefix}_parameter_errors.png')

def collected_results(full_only=True):
    rows=[]
    path=RUNS/('FULL' if full_only else MODE)
    if not path.exists():return pd.DataFrame()
    for p in path.glob('**/result.json'):
        try:
            r=json.loads(p.read_text())
            if r.get('complete',False): rows.append(r)
        except Exception as ex:warnings.warn(f'Could not load {p}: {ex}')
    return pd.DataFrame(rows)

def diagnose_case(co,cf,method_labels=('Bi-HYCO','GN-dist','XPINN')):
    # Generate selected-state snapshots only if they were saved; no retraining.
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    tri=mtri.Triangulation(co['fem'].points[:,0],co['fem'].points[:,1],co['fem'].triangles)
    arr=[('Reference',co['ref']['u_reference'])]
    for r in cf:
        if r.get('method') in method_labels and r.get('data_id')==co['data_id']:
            p=ROOT/r['output_folder']/'states.npz'
            if p.exists():
                with np.load(p,allow_pickle=False) as z:
                    if 'physical_state' in z: arr.append((r['method'],z['physical_state'].copy()))
    for ax,(name,u) in zip(axes,arr[:4]):
        im=ax.tricontourf(tri,u,levels=24,cmap='viridis');ax.set_aspect('equal');ax.set_title(name)
    save_figure(fig,FIGURES/f'{co["region"]}_{levstr(co["mismatch"])}_fields.png')


# --------------------------------- campaign E0-E7 ------------------------------
def E0_prepare(seeds=BASE_TRIAL_SEEDS,noises=(0.,0.10),show=True):
    """Validate six bundled reference solutions; cache ONLY the reduced campaign's data.

    The experiment library can still request additional independent data via case().
    No fine-resolution PDE is recomputed here.
    """
    coarse,fine=get_fems()
    for region in REGIONS:
        for mismatch in LEVELS:
            rr=load_reference(region,mismatch);check_ref_mesh(rr,coarse,fine)
            if mismatch==0: assert float(rr['epsilon'])==0
            else: assert float(rr['epsilon'])>0
            if show and mismatch==.03:
                evaluation_figure(case(region,mismatch,0,seeds[0]),FIGURES/f'E0_geometry_{region}.png')
    # E1 (one control), E2/E3 (three seeds, BOTH regions), E4 (five, Omega1 only).
    requests=[('Omega1',0.,0.,[seeds[0]])]
    requests += [(r,.03,0.,list(seeds[:3])) for r in REGIONS]
    requests += [('Omega1',.03,0.,list(seeds)),
                 ('Omega1',.01,0.,list(seeds)),
                 ('Omega1',.03,.10,list(seeds))]
    prepared=set()
    for region,level,noise,seed_list in requests:
        for seed in seed_list:
            key=(region,level,noise,int(seed))
            if key not in prepared:
                ensure_data(*key);prepared.add(key)
    basis=mesh_cache()
    signature=dict(version=ALGORITHM_VERSION,reference_count=6,
       splits=dict(train_per_unit=800,val_per_unit=200,test_per_unit=200),
       study='reduced: E2/E3 3 seeds in both regions, E4 5 seeds Omega1 only',
       robustness_scenarios=[dict(region='Omega1',mismatch=level,eta=eta) for level,eta in ((.03,0.),(.01,0.),(.03,.10))],
       n_prepared_data=len(prepared),seeds=list(seeds),
       original_geometry='Rout=1+0.12 cos(3theta)+0.08 sin(5theta); Rint=0.42(1+0.22 cos(5theta))',
       coarse_mesh=[900,180,160],fine_mesh=[1800,360,320],true_background=[2,1],
       checkpoint_selection='minimum physical MSE on held-out observations; no ground-truth parameter access',
       device=str(DEVICE),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
       spectral_cache=basis.name)
    _atomic_json(CACHE/'campaign_manifest.json',signature)
    print(f'E0: verified six reference files and prepared {len(prepared)} unique observation cases; reused FEM cache.')
    return signature


def E1_control(seeds=BASE_TRIAL_SEEDS[:1],include_xpinn=True):
    """One unperturbed-control run; previously published multi-seed control exists in manuscript."""
    rows=[]
    for seed in seeds:
        c=case('Omega1',0,0,seed)
        rows.extend([run_gn(c),run_bihyco(c,interaction=1.)])
        if include_xpinn:rows.append(run_xpinn(c))
    df=rows_to_summary(rows,'E1_control');basic_parameter_plot(df,'E1')
    return df


def E2_main(seeds=BASE_TRIAL_SEEDS[:3],include_xpinn=True):
    """Both locations, 3% coefficient mismatch, eta=0, three paired seeds."""
    rows=[]
    for region in REGIONS:
        for seed in seeds:
            c=case(region,.03,0.,seed)
            rows.extend([run_gn(c),run_gn(c,method='GN-unit1',betas=(1.,0.)),
                         run_gn(c,method='GN-unit2',betas=(0.,1.)),run_weighted_gn(c),
                         run_bihyco(c,interaction=1.)])
            if include_xpinn:rows.append(run_xpinn(c))
            if seed==seeds[0]:diagnose_case(c,rows)
    df=rows_to_summary(rows,'E2_main');basic_parameter_plot(df,'E2')
    p=paired_difference(df)
    p.to_csv(REPORTS/'E2_paired_BiHYCO_minus_GN.csv',index=False)
    if len(p):
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for ax,region in zip(axes,REGIONS):
            sub=p[p['region']==region].sort_values('seed')
            for _,r in sub.iterrows():
                ax.plot([0,1],[100*r['e_bg_A'],100*r['e_bg_B']],marker='o',alpha=.6)
            ax.set_xticks([0,1],['Bi-HYCO','GN-dist']);ax.set_title(region)
            ax.set_ylabel('Background-parameter error (%)');ax.grid(axis='y',alpha=.2)
        fig.suptitle('Paired observation/training seeds; no statistical significance implied')
        save_figure(fig,FIGURES/'E2_paired_background_error.png')
    return df


def E3_ablation(seeds=BASE_TRIAL_SEEDS[:3]):
    """Both locations; λ=0 and 1, same 3 seeds and physical-validation checkpoint rule."""
    rows=[]
    for region in REGIONS:
        for seed in seeds:
            c=case(region,.03,0.,seed)
            rows.extend([run_bihyco(c,interaction=1.),run_bihyco(c,interaction=0.)])
    df=rows_to_summary(rows,'E3_ablation')
    pp=paired_difference(df,'Bi-HYCO','Bi-HYCO-0')
    pp.to_csv(REPORTS/'E3_paired_interaction.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,region in zip(axes,REGIONS):
        for method in ('Bi-HYCO','Bi-HYCO-0'):
            for row in [r for r in rows if r['region']==region and r['method']==method]:
                history=read_csv(ROOT/row['output_folder']/'training_history.csv')
                if history:
                    ax.plot([r['step'] for r in history],[r['val_phy'] for r in history],alpha=.65,
                            label=method if row['seed']==seeds[0] else None)
                    ax.axvline(row['selected_step'],color='gray',lw=.5,alpha=.2)
        ax.set_title(region);ax.set_xlabel('Coordinator rounds');ax.set_ylabel('Held-out physical MSE')
        ax.set_yscale('log');ax.legend(fontsize=8)
    save_figure(fig,FIGURES/'E3_validation_trajectories.png')
    # Selected and final parameter errors: oracle diagnostic, NEVER checkpoint selection.
    data=[]
    for r in rows:
        h=read_csv(ROOT/r['output_folder']/'training_history.csv')
        if h:
            last=h[-1]
            data.append(dict(region=r['region'],seed=r['seed'],method=r['method'],
                selected_step=r['selected_step'],final_step=last['step'],
                e_bg_selected=r['e_bg'],e_bg_final=last['diagnostic_e_bg'],
                validation_selected=r['val_mse'],validation_final=last['val_phy']))
    save_csv(REPORTS/'E3_selected_vs_final.csv',data)
    return df


def E4_robustness(seeds=BASE_TRIAL_SEEDS,
                  scenarios=((.03,0.),(.01,0.),(.03,.10)),region='Omega1',
                  include_weighted_gn=True):
    """Targeted five-seed robustness ONLY in Omega1, λ=1, with E2 run reuse.

    Δ=3%, η=0 shares 3 of its 5 Bi-HYCO runs with E2;
    Δ=1%, η=0 and Δ=3%, η=.10 each add 5 runs.
    XPINN and Bi-HYCO λ=0 are deliberately excluded from the robustness extension.
    """
    if region!='Omega1':
        raise ValueError('Reduced E4 is pre-registered for Omega1 only; use E2/E3 for Omega2.')
    if len(seeds)!=5 or len(set(seeds))!=5:
        raise ValueError('E4 requires exactly five DISTINCT seed identifiers')
    rows=[]
    for mismatch,eta in scenarios:
        for seed in seeds:
            c=case(region,mismatch,eta,seed)
            rows.extend([run_gn(c),run_bihyco(c,interaction=1.)])
            if include_weighted_gn:rows.append(run_weighted_gn(c))
    df=rows_to_summary(rows,'E4_robustness')
    paired=paired_difference(df)
    paired.to_csv(REPORTS/'E4_paired_background_error.csv',index=False)
    if len(paired):
        fig,axes=plt.subplots(1,2,figsize=(11,4.1))
        order=list(scenarios)
        for j,(mismatch,eta) in enumerate(order):
            sub=paired[(paired['mismatch']==mismatch)&(paired['eta']==eta)].sort_values('seed')
            axes[0].scatter(np.full(len(sub),j)-.10,sub['e_bg_A']*100,s=26,label='Bi-HYCO' if j==0 else None)
            axes[0].scatter(np.full(len(sub),j)+.10,sub['e_bg_B']*100,s=26,label='GN-dist' if j==0 else None)
            for _,r in sub.iterrows():
                axes[1].scatter(j,100*r['delta_e_bg'],alpha=.60,color='#666666',s=19)
            if len(sub):
                axes[1].errorbar(j,100*sub['delta_e_bg'].mean(),yerr=100*sub['delta_e_bg'].std(ddof=1),
                                 marker='D',capsize=3,color='black')
        labs=[f'δ={v:.0%}\nη={n:.2f}' for v,n in order]
        for ax in axes:
            ax.set_xticks(range(len(order)),labs);ax.grid(axis='y',alpha=.25)
            ax.set_xlabel('Omega1 perturbation / observation noise')
        axes[0].set_title('Individual seed errors');axes[0].set_ylabel('Background-parameter error (%)')
        axes[1].set_title('Paired Bi-HYCO minus GN-dist');axes[1].set_ylabel('Difference (percentage points)')
        axes[1].axhline(0,color='gray',linestyle='--',lw=1);axes[0].legend(fontsize=8)
        save_figure(fig,FIGURES/'E4_Omega1_reduced_five_seed_robustness.png')
    return df


def E5_interaction_sweep(seeds=BASE_TRIAL_SEEDS[:3],lambdas=(0.,.1,1.,10.),region='Omega1'):
    """One region / 3 seeds; only λ=.1 and 10 are new compared to E2/E3."""
    if region!='Omega1':raise ValueError('Reduced E5 uses Omega1, 3% mismatch only')
    rows=[]
    for seed in seeds:
        c=case(region,.03,0,seed)
        for lam in lambdas:rows.append(run_bihyco(c,interaction=lam))
    df=pd.DataFrame(rows)
    df.to_csv(REPORTS/'E5_interaction_runs.csv',index=False)
    summary=df.groupby(['region','lambda_int']).agg(e_bg_mean=('e_bg','mean'),e_bg_std=('e_bg','std'),
        e_phy_mean=('e_phy_global_error','mean'),e_phy_std=('e_phy_global_error','std'),n=('seed','count')).reset_index()
    summary.to_csv(REPORTS/'E5_interaction_summary.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9.8,3.9))
    x=summary.sort_values('lambda_int')
    xx=np.arange(len(x))
    axes[0].errorbar(xx,x.e_bg_mean*100,yerr=x.e_bg_std.fillna(0)*100,fmt='o-',capsize=3)
    axes[0].set_ylabel('Background-parameter error (%)')
    axes[1].errorbar(xx,x.e_phy_mean*100,yerr=x.e_phy_std.fillna(0)*100,fmt='o-',capsize=3)
    axes[1].set_ylabel('Physical-state error (%)')
    for ax in axes:
        ax.set_xticks(xx,[str(z) for z in x.lambda_int]);ax.set_xlabel('Interaction weight λ');ax.grid(alpha=.25)
    fig.suptitle('Omega1, δ=3%, η=0; three matched initialization seeds')
    save_figure(fig,FIGURES/'E5_Omega1_interaction_sweep.png')
    return df

def exact_coordinator_diagnostic(c, rounds=120, step_scale=.05, local_steps=2):
    """Deterministic, fixed ghost points, full gradient (no Adam) demonstration.

    Not a proof of boundedness or KL. Numerically checks the algebraic E=1 identity,
    E=2 local-averaging perturbation, and traces aggregate gradient norm.
    """
    params=dict(rounds=int(rounds),step_scale=step_scale,local_steps=int(local_steps),seed=c['seed'])
    folder=_run_folder('exact_coordinator',c,params)
    if (folder/'history.csv').exists():return pd.read_csv(folder/'history.csv')
    prob=prepare_problem(c);d=c['data'];seed=c['seed'];torch.manual_seed(seed)
    model=orig.BiHYCO().to(DEVICE)
    fixed_ghost=orig.tensor(orig.interaction_points(np.random.default_rng(seed+22001)))
    fixed_map=prob['interpolator'].torch_map(fixed_ghost.detach().cpu().numpy(),DEVICE)
    y1,y2=orig.tensor(d['y1_train']),orig.tensor(d['y2_train'])
    weights=orig.UNIT_WEIGHTS
    def obj1(m):
        a=m.coefficients();u=prob['physical'](a)
        return orig.OMEGA_PHY*orig.ALPHA_PHY*((prob['observation_map'](u)-y1).square().mean()+orig.BETA_PHY*a.square().sum())+(
             fixed_map(u)-m.net(fixed_ghost).flatten()).square().mean()
    def obj2(m):
        a=m.coefficients();u=prob['physical'](a)
        return orig.OMEGA_SYN*orig.ALPHA_SYN*((m.net(prob['x2_train']).flatten()-y2).square().mean()+.5*orig.BETA_SYN*sum(p.square().sum() for p in m.net.parameters()))+(
             fixed_map(u)-m.net(fixed_ghost).flatten()).square().mean()
    def flat_pars(m):return torch.cat([p.reshape(-1) for p in m.parameters()])
    def local_step(m,objective,eta):
        loss=objective(m);pars=list(m.parameters())
        grads=torch.autograd.grad(loss,pars)
        with torch.no_grad():
            for p,g in zip(pars,grads):p.sub_(eta*g)
    hist=[];started=time.perf_counter()
    for i in range(1,int(rounds)+1):
        eta=step_scale/(5+i)
        m1,m2=copy.deepcopy(model),copy.deepcopy(model)
        # Gradient at current point for the aggregate objective, weighted 1/2.
        loss=.5*obj1(model)+.5*obj2(model)
        grads=torch.autograd.grad(loss,list(model.parameters()))
        flatg=torch.cat([g.reshape(-1) for g in grads]);normg=float(norm(flatg.detach().cpu().numpy()))
        v0=flat_pars(model).detach().clone()
        for _ in range(int(local_steps)):
            local_step(m1,obj1,eta);local_step(m2,obj2,eta)
        # All vectors are averaged in RAW parameter coordinates in the theorem.
        with torch.no_grad():
            for p,p1,p2 in zip(model.parameters(),m1.parameters(),m2.parameters()):
                p.copy_(weights[0]*p1+weights[1]*p2)
        vm=flat_pars(model).detach()
        perturb=float(norm((vm-(v0-int(local_steps)*eta*flatg)).cpu().numpy()))
        hist.append(dict(round=i,eta=eta,grad_norm=normg,
             coordinator_perturb=perturb,perturb_over_eta_squared=perturb/(eta**2),
             seconds=time.perf_counter()-started))
    save_csv(folder/'history.csv',hist)
    return pd.DataFrame(hist)

def E6_coordinator(seeds=BASE_TRIAL_SEEDS[:1],run_exact=True):
    """One illustrative Omega1 seed: reused E2 E=2, new E=1 and centralized check."""
    rows=[]
    for seed in seeds:
        c=case('Omega1',.03,0,seed)
        rows.extend([run_bihyco(c,interaction=1.,local_steps=1),
                     run_bihyco(c,interaction=1.,local_steps=2),
                     run_bihyco(c,interaction=1.,local_steps=1,algorithm='central')])
    df=rows_to_summary(rows,'E6_coordinator')
    if run_exact:
        example=case('Omega1',.03,0,seeds[0])
        one=exact_coordinator_diagnostic(example,rounds=20 if MODE=='PILOT' else 120,local_steps=1)
        two=exact_coordinator_diagnostic(example,rounds=20 if MODE=='PILOT' else 120,local_steps=2)
        fig,axes=plt.subplots(1,2,figsize=(9,3.7))
        axes[0].semilogy(two['round'],two['grad_norm']);axes[0].set_ylabel('||∇J||')
        axes[1].plot(one['round'],one['coordinator_perturb'],label='E=1')
        axes[1].plot(two['round'],two['coordinator_perturb'],label='E=2')
        axes[1].set_yscale('symlog',linthresh=1e-10);axes[1].set_ylabel('||T(x) - x + Eη∇J(x)||')
        for ax in axes:ax.set_xlabel('Coordinator round');ax.grid(alpha=.2)
        axes[1].legend();save_figure(fig,FIGURES/'E6_exact_gradient_coordinator.png')
    return df


def E7_sensitivity_cost(seeds=BASE_TRIAL_SEEDS[:1]):
    rows=[]
    for region in REGIONS:
        for mismatch in (0,.01,.03):
            c=case(region,mismatch,0,seeds[0]);u,du=forward_sensitivities(np.log(BG),c['fem'])
            d=c['data']
            for tag,ks in [('unit1',[1]),('unit2',[2]),('combined',[1,2])]:
                J=np.vstack([np.column_stack([quick_interp_eval(du[:,j],d[f'x{k}_train'],c['fem']) for j in range(2)]) for k in ks])
                sv=np.linalg.svd(J,compute_uv=False)
                rows.append(dict(region=region,mismatch=mismatch,unit=tag,
                                 singular_max=float(sv[0]),singular_min=float(sv[-1]),
                                 condition=float(sv[0]/max(sv[-1],1e-16)),rank=int(np.linalg.matrix_rank(J))))
    save_csv(REPORTS/'E7_sensitivities.csv',rows)
    df=collected_results(full_only=(MODE=='FULL'))
    if len(df):
        df.to_csv(REPORTS/'E7_all_cached_runs.csv',index=False)
        useful=df[(df['region'].isin(REGIONS)) & (df['mismatch']==.03) & (df['eta']==0)]
        if len(useful):
            fig,ax=plt.subplots(figsize=(7.5,4.5))
            for m,sub in useful.groupby('method'):
                sub=sub[(sub['seconds']>0)&(sub['e_bg']>0)]
                if len(sub):ax.scatter(sub['seconds'],100*sub['e_bg'],label=m,alpha=.7,s=30)
            ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Reported wall-clock time (s)')
            ax.set_ylabel('Background parameter error (%)');ax.set_title('Accuracy–cost tradeoff (do not infer parallel speedup)')
            ax.legend(fontsize=7);ax.grid(alpha=.2)
            save_figure(fig,FIGURES/'E7_accuracy_cost.png')
    return pd.DataFrame(rows)


# -------- E8: genuinely higher-dimensional, smooth-in-each-region diffusivity ----
# Parameterization: a(x;g)=a_base(x)*exp(sum_j g_j * phi_j(x)).
# The TRUE coefficient adds an UNMODELED positive Gaussian bump, as in E2.
# This is a separate inverse problem, NOT a claim of reproducing the original 2-param one.
def _e8_basis_spec(p):
    if p not in (8,16):raise ValueError('E8 supports 8 and 16 spatial unknowns')
    half=p//2;centers=[];sigmas=[];region_ids=[]
    for sub in (1,2):
        for j in range(half):
            theta=2*np.pi*j/half + (0 if sub==1 else .2)
            radius=(.21 if sub==1 else .7) + (.07 if (half==8 and j%2) else 0)
            centers.append([radius*np.cos(theta),radius*np.sin(theta)])
            sigmas.append(.21 if sub==1 else .28)
            region_ids.append(sub)
    return np.array(centers),np.array(sigmas),np.array(region_ids)

def e8_features(xy,p,with_grad=False):
    xy=np.atleast_2d(np.asarray(xy,dtype=float))
    center,sigma,regions=_e8_basis_spec(p)
    diff=xy[:,None,:]-center[None,:,:]
    features=np.exp(-.5*np.sum(diff**2,axis=2)/sigma[None,:]**2)
    inner=orig.inside_inner_np(xy)
    features*=np.where(regions[None,:]==1,inner[:,None],(~inner)[:,None])
    if not with_grad:return features
    grad=-diff/(sigma[None,:,None]**2)*features[:,:,None]
    return features,grad

def e8_true_gamma(p):
    x=np.arange(1,p+1,dtype=float)
    return .24*np.sin(1.6*x)+.07*np.cos(.6*x)

class SpatialEllipticFEM:
    def __init__(self,original_fem,p):
        self.fem=original_fem;self.p=int(p);self.points=original_fem.points
        self.tri=original_fem.triangles;self.free=original_fem.free
        t=self.points[self.tri];self.cent=t.mean(axis=1)
        x1,y1=t[:,0,0],t[:,0,1];x2,y2=t[:,1,0],t[:,1,1];x3,y3=t[:,2,0],t[:,2,1]
        self.areas=.5*np.abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
        b=np.column_stack([y2-y3,y3-y1,y1-y2])/(2*self.areas[:,None])
        cc=np.column_stack([x3-x2,x1-x3,x2-x1])/(2*self.areas[:,None])
        self.Kloc=self.areas[:,None,None]*(b[:,:,None]*b[:,None,:]+cc[:,:,None]*cc[:,None,:])
        self.phi=e8_features(self.cent,p)
        self.base=np.where(orig.inside_inner_np(self.cent),2.,1.)
        self.rows=np.repeat(self.tri,3,axis=1).ravel();self.cols=np.tile(self.tri,(1,3)).ravel()
        self.n=len(self.points)
    def a_elements(self,gamma):
        return self.base*np.exp(np.clip(self.phi@np.asarray(gamma,dtype=float),-20,20))
    def matrix(self,coefficient):
        values=(self.Kloc*np.asarray(coefficient)[:,None,None]).ravel()
        return coo_matrix((values,(self.rows,self.cols)),shape=(self.n,self.n)).tocsr()[self.free][:,self.free]
    def solve(self,gamma,extra_elements=None,return_lu=False):
        coeff=self.a_elements(gamma)
        if extra_elements is not None:coeff=coeff+np.asarray(extra_elements)
        if coeff.min()<=0:raise ValueError('Non-positive true diffusion in E8')
        lu=splu(self.matrix(coeff).tocsc())
        u=np.zeros(self.n);u[self.free]=lu.solve(self.fem.rhs[self.free])
        return (u,lu,coeff) if return_lu else u
    def jac_solutions(self,gamma):
        u,lu,a=self.solve(gamma,return_lu=True)
        derivative=np.zeros((self.n,self.p))
        # Each sensitivity solves K du_j = -(dK/dg_j)u, same sparse LU.
        for j in range(self.p):
            dk=self.matrix(a*self.phi[:,j]);tmp=-(dk@u[self.free])
            derivative[self.free,j]=lu.solve(tmp)
        return u,derivative

class SparseStateSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx,gamma,spatial_solver):
        g=gamma.detach().cpu().numpy().astype(float)
        u,lu,coeff=spatial_solver.solve(g,return_lu=True)
        ctx.solver=spatial_solver;ctx.lu=lu;ctx.u=u;ctx.coeff=coeff
        return gamma.new_tensor(u)
    @staticmethod
    def backward(ctx,grad_u):
        solver=ctx.solver
        v=np.zeros(solver.n)
        v[solver.free]=ctx.lu.solve(grad_u.detach().cpu().numpy()[solver.free].astype(float),trans='T')
        ue=ctx.u[solver.tri];ve=v[solver.tri]
        energy=np.einsum('ei,eij,ej->e',ve,solver.Kloc,ue,optimize=True)
        grad=-solver.phi.T@(ctx.coeff*energy)
        return grad_u.new_tensor(grad),None

class SpatialHybrid(nn.Module):
    def __init__(self,p):
        super().__init__();self.g=nn.Parameter(torch.zeros(p,dtype=orig.DTYPE))
        self.net=orig.MLP(orig.HYCO_WIDTH,orig.HYCO_DEPTH)
    def coefficients(self):return self.g


def e8_reference(p,region='Omega1',mismatch=.03):
    p=int(p);filename=CACHE/f'E8_reference_p{p}_{region}_{levstr(mismatch)}.npz'
    if filename.exists():
        with np.load(filename,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}
    if region not in REGIONS:raise ValueError(region)
    coarse,fine=get_fems();fine_model=SpatialEllipticFEM(fine,p)
    gt=e8_true_gamma(p);baseline=fine_model.solve(gt)
    center=np.array([-.22,-.22] if region=='Omega1' else [.75,.15])
    sigma=.16 if region=='Omega1' else .075
    bump=np.exp(-np.sum((fine_model.cent-center)**2,axis=1)/(2*sigma**2))
    bump*=orig.inside_inner_np(fine_model.cent) if region=='Omega1' else ~orig.inside_inner_np(fine_model.cent)
    if mismatch==0:epsilon=0.;u=baseline.copy()
    else:
        def loss(e):
            uu=fine_model.solve(gt,extra_elements=e*bump)
            return float(np.sqrt(np.sum(fine.mass*(uu-baseline)**2)/(np.sum(fine.mass*uu**2)+1e-20)))
        hi=1.
        while loss(hi)<mismatch and hi<300:hi*=2
        if loss(hi)<mismatch:raise RuntimeError('Could not reach mismatch; modify spatial basis/bump geometry')
        epsilon=brentq(lambda e:loss(e)-mismatch,0,hi,xtol=1e-10)
        u=fine_model.solve(gt,extra_elements=epsilon*bump)
    interp=orig.make_interpolator(fine.points,u)
    x1=orig.sample_uniform_in_inner(800,seed=BASE_SEED+101)
    x2=orig.sample_uniform_in_outer_shell(800,seed=BASE_SEED+102)
    data=dict(p=p,region=region,mismatch=mismatch,
        points_fine=fine.points,triangles_fine=fine.triangles,u_reference_fine=u,
        u_reference=interp(coarse.points),mass_coarse=coarse.mass,
        x1_train=x1,x2_train=x2,y1_clean=interp(x1),y2_clean=interp(x2),
        true_gamma=gt,epsilon=epsilon,bump_center=center,bump_sigma=sigma,
        coefficient_true=fine_model.a_elements(gt)+epsilon*bump,
        coefficient_bg=fine_model.a_elements(gt))
    _atomic_npz(filename,**data)
    return data

def e8_case(p,region='Omega1',mismatch=.03,noise=0,seed=1251,n_train=120):
    r=e8_reference(p,region,mismatch);coarse,_=get_fems()
    d={};sp=_val_test_splits();inter=orig.make_interpolator(r['points_fine'],r['u_reference_fine'])
    for k in (1,2):
        train_x=r[f'x{k}_train'][:n_train];val_x=sp[f'x{k}_val'];test_x=sp[f'x{k}_test']
        for label,x in [('train',train_x),('val',val_x),('test',test_x)]:
            yy=r[f'y{k}_clean'][:n_train] if label=='train' else inter(x)
            rng=np.random.default_rng(_case_noise_seed(seed,region,mismatch,noise,f'E8_p{p}/{label}/{k}'))
            d[f'x{k}_{label}']=x;d[f'y{k}_{label}']=yy+float(noise)*np.std(yy)*rng.normal(size=len(yy))
    data_id=sha_bytes(f'E8|{p}|{region}|{mismatch}',*[d[k] for k in sorted(d)])
    return dict(p=p,region=region,mismatch=mismatch,eta=noise,seed=int(seed),
                data_id=data_id,reference_id=sha_bytes(r['u_reference_fine'],r['coefficient_true']),
                data=d,ref=r,fem=coarse,solver=SpatialEllipticFEM(coarse,p),true_gamma=r['true_gamma'])

def _e8_error_row(c,gamma,physical_state,synthetic_state=None,method='E8'):
    g=np.asarray(gamma,dtype=float);gt=np.asarray(c['true_gamma']);r=c['ref']
    d=c['data'];fem=c['fem'];err=state_errors(physical_state,r['u_reference'],fem)
    row=dict(method=method,p=c['p'],region=c['region'],mismatch=c['mismatch'],eta=c['eta'],
        seed=c['seed'],data_id=c['data_id'],reference_id=c['reference_id'],
        e_gamma=float(norm(g-gt)/(norm(gt)+1e-15)),
        e_phy_global_error=float(err['global_error']),
        e_phy_inner_error=float(err['inner_error']),e_phy_outer_error=float(err['outer_error']))
    true_bg=np.asarray(r['coefficient_bg'])
    m=SpatialEllipticFEM(get_fems()[1],c['p']);est=m.a_elements(g)
    area=m.areas
    row['e_bg_field']=float(np.sqrt(np.sum(area*(est-true_bg)**2)/(np.sum(area*true_bg**2)+1e-30)))
    row['e_true_coefficient']=float(np.sqrt(np.sum(area*(est-r['coefficient_true'])**2)/(np.sum(area*r['coefficient_true']**2)+1e-30)))
    for split in ('val','test'):
        row[split+'_mse']=.5*sum(np.mean((quick_interp_eval(physical_state,d[f'x{k}_{split}'],fem)-d[f'y{k}_{split}'])**2) for k in (1,2))
    if synthetic_state is not None:
        row['e_syn_global_error']=state_errors(synthetic_state,r['u_reference'],fem)['global_error']
        row['test_syn_mse']=.5*sum(np.mean((quick_interp_eval(synthetic_state,d[f'x{k}_test'],fem)-d[f'y{k}_test'])**2) for k in (1,2))
    return row

def run_e8_gn(c,penalties=(1e-5,1e-4,1e-3),max_iter=35):
    params=dict(p=c['p'],penalties=list(penalties),max_iter=max_iter,scheme='distributed damped GN; ridge gamma prior')
    folder=_run_folder('E8_GN',c,params);previous=_cached_result(folder)
    if previous:return previous
    solver=c['solver'];d=c['data'];p=c['p']
    idxmaps=[orig.NodalInterpolator(c['fem'].points).indices_weights(d[f'x{k}_train']) for k in (1,2)]
    def sample(u,iw): idx,wt=iw;return np.sum(u[idx]*wt,axis=1)
    best=None;history=[];start=time.perf_counter()
    for prior in penalties:
        x=np.zeros(p);mu=1e-3;counter=0
        for it in range(max_iter):
            u,du=solver.jac_solutions(x);counter+=p+1
            H=prior*np.eye(p);g=prior*x;loss=.5*prior*float(x@x)
            for k in (1,2):
                w=1/np.sqrt(2*len(d[f'y{k}_train']))
                r=w*(sample(u,idxmaps[k-1])-d[f'y{k}_train'])
                J=w*np.column_stack([sample(du[:,j],idxmaps[k-1]) for j in range(p)])
                H+=J.T@J;g+=J.T@r;loss+=.5*float(r@r)
            if norm(g,np.inf)<1e-9:break
            accepted=False
            for j in range(14):
                step=np.linalg.solve(H+mu*max(np.trace(H)/p,1e-12)*np.eye(p),-g)
                cand=np.clip(x+step,-2,2);ut=solver.solve(cand);counter+=1
                newloss=.5*prior*float(cand@cand)
                for k in (1,2):
                    pred=sample(ut,idxmaps[k-1]);r=(pred-d[f'y{k}_train'])/np.sqrt(2*len(pred));newloss+=.5*float(r@r)
                if newloss<loss: x=cand;mu=max(mu/3,1e-10);accepted=True;break
                mu=min(mu*6,1e10)
            if not accepted:break
        u=solver.solve(x);counter+=1
        val=.5*sum(np.mean((quick_interp_eval(u,d[f'x{k}_val'],c['fem'])-d[f'y{k}_val'])**2) for k in (1,2))
        history.append(dict(prior=prior,iteration=it,val_mse=val,forward_sensitivity_equiv=counter))
        if best is None or val<best['val']:
            best=dict(gamma=x.copy(),u=u.copy(),val=val,prior=prior,it=it,evals=counter)
    row=_e8_error_row(c,best['gamma'],best['u'],method='E8-GN-dist')
    row.update(complete=True,seconds=time.perf_counter()-start,selected_step=int(best['it']),
        selection='minimum held-out validation loss over predeclared ridge grid',
        ridge_selected=float(best['prior']),n_forward_solves_approx=int(best['evals']),
        n_communications=int(best['it']),n_neural_steps=0,
        output_folder=str(folder.relative_to(ROOT)))
    _save_result(folder,row,dict(gamma=best['gamma'],physical_state=best['u']))
    save_csv(folder/'ridge_history.csv',history)
    return row


def run_e8_bihyco(c,interaction=1.,rounds=None):
    p=c['p'];rounds=(1000 if MODE=='FULL' else 4) if rounds is None else int(rounds)
    params=dict(p=p,interaction=float(interaction),rounds_per_local=2,
                width=orig.HYCO_WIDTH,depth=orig.HYCO_DEPTH,
                lr_g=3e-3,lr_net=8e-4,physical_reg=1e-4)
    folder=_run_folder('E8_BiHYCO',c,params)
    cached=_cached_result(folder)
    if cached and cached.get('completed_rounds',0)>=rounds:return cached
    torch.manual_seed(c['seed']);local=[SpatialHybrid(p).to(DEVICE)];local.append(copy.deepcopy(local[0]))
    optimizers=[torch.optim.Adam([{'params':[m.g],'lr':3e-3},
        {'params':m.net.parameters(),'lr':8e-4}]) for m in local]
    interp=orig.NodalInterpolator(c['fem'].points)
    d=c['data'];map1=interp.torch_map(d['x1_train'],DEVICE)
    y1,y2=orig.tensor(d['y1_train']),orig.tensor(d['y2_train']);x2=orig.tensor(d['x2_train'])
    rng=np.random.default_rng(c['seed']+22001)
    history=[];best_val=float('inf');best_step=0;elapsed_past=0;completed=0
    chk=folder/'training_checkpoint.pt'
    if chk.exists():
        old=torch.load(chk,map_location=DEVICE,weights_only=True)
        if old['params']!=params or old['data_id']!=c['data_id']:raise ValueError('E8 checkpoint signature mismatch')
        for k in range(2):local[k].load_state_dict(old['models'][k]);optimizers[k].load_state_dict(old['optimizers'][k])
        completed=int(old['completed']);best_step=int(old['best_step']);best_val=float(old['best_val']);elapsed_past=float(old['seconds'])
        history=list(old['history']);rng.bit_generator.state=old['rng'];torch.set_rng_state(old['torch_rng_cpu'].cpu())
    checkpoint_every=100 if MODE=='FULL' else 2
    started=time.perf_counter()
    for it in range(completed+1,rounds+1):
        xyz=orig.interaction_points(rng);mapghost=interp.torch_map(xyz,DEVICE);ghost=orig.tensor(xyz)
        for unit_idx,(m,optim) in enumerate(zip(local,optimizers)):
            for _ in range(2):
                optim.zero_grad(set_to_none=True)
                u=SparseStateSolve.apply(m.g,c['solver'])
                if unit_idx==0:
                    loss=orig.OMEGA_PHY*orig.ALPHA_PHY*((map1(u)-y1).square().mean()+1.e-4*(m.g**2).sum())
                else:
                    loss=orig.OMEGA_SYN*orig.ALPHA_SYN*((m.net(x2).flatten()-y2).square().mean()+.5*orig.BETA_SYN*sum(q.square().sum() for q in m.net.parameters()))
                if interaction>0:loss=loss+float(interaction)*(mapghost(u)-m.net(ghost).flatten()).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError(f'E8 Bi-HYCO nan at iteration {it}')
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.0)
                optim.step()
        with torch.no_grad():
            for vals in zip(local[0].parameters(),local[1].parameters()):
                avg=.5*(vals[0]+vals[1]);vals[0].copy_(avg);vals[1].copy_(avg)
        if it%checkpoint_every==0 or it==rounds:
            g=local[0].g.detach().cpu().numpy().copy();u=c['solver'].solve(g)
            v=.5*sum(np.mean((quick_interp_eval(u,d[f'x{k}_val'],c['fem'])-d[f'y{k}_val'])**2) for k in (1,2))
            seconds=elapsed_past+time.perf_counter()-started
            history.append(dict(round=it,val_phy=v,diagnostic_e_gamma=float(norm(g-c['true_gamma'])/norm(c['true_gamma'])),seconds=seconds))
            if v<best_val:
                best_val=v;best_step=it
                _atomic_torch(folder/'model_best.pt',local[0].state_dict())
            _atomic_torch(chk,dict(params=params,data_id=c['data_id'],models=[z.state_dict() for z in local],
                optimizers=[o.state_dict() for o in optimizers],completed=it,best_val=best_val,best_step=best_step,
                history=history,seconds=seconds,rng=rng.bit_generator.state,torch_rng_cpu=torch.get_rng_state()))
            save_csv(folder/'training_history.csv',history)
    model=SpatialHybrid(p).to(DEVICE)
    model.load_state_dict(torch.load(folder/'model_best.pt',map_location=DEVICE,weights_only=True));model.eval()
    g=model.g.detach().cpu().numpy().copy();u=c['solver'].solve(g)
    with torch.no_grad():us=model.net(orig.tensor(c['fem'].points)).cpu().numpy().flatten()
    row=_e8_error_row(c,g,u,us,method='E8-Bi-HYCO' if interaction else 'E8-Bi-HYCO-0')
    row.update(complete=True,seconds=elapsed_past+time.perf_counter()-started,
        selection='minimum held-out PHYSICAL validation MSE',selected_step=best_step,completed_rounds=rounds,
        n_neural_steps=rounds*4,n_forward_solves_approx=rounds*4+len(history),
        n_communications=rounds,n_parameters=sum(q.numel() for q in model.parameters()),
        lambda_int=float(interaction),output_folder=str(folder.relative_to(ROOT)))
    _save_result(folder,row,dict(gamma_selected=g,physical_state=u,synthetic_state=us))
    return row

class E8XPINN(nn.Module):
    def __init__(self,p):
        super().__init__();self.inner=orig.MLP(orig.XPINN_WIDTH,orig.XPINN_DEPTH)
        self.outer=orig.MLP(orig.XPINN_WIDTH,orig.XPINN_DEPTH)
        self.g=nn.Parameter(torch.zeros(p,dtype=orig.DTYPE))
        center,sigma,sub=_e8_basis_spec(p)
        self.register_buffer('center',orig.tensor(center));self.register_buffer('sigma',orig.tensor(sigma))
        self.register_buffer('sub',torch.tensor(sub,device=DEVICE))
    def a_and_gradient(self,x,region):
        diff=x[:,None,:]-self.center[None,:,:]
        phi=torch.exp(-.5*(diff**2).sum(2)/(self.sigma[None,:]**2))
        phi=phi*(self.sub[None,:]==region)
        gradphi=-diff/(self.sigma[None,:,None]**2)*phi[:,:,None]
        logfactor=phi@self.g
        a=(2. if region==1 else 1.)*torch.exp(logfactor)
        grad_a=a[:,None]*torch.einsum('npi,p->ni',gradphi,self.g)
        return a[:,None],grad_a

def e8_xpinn_residual(model,network,x,region):
    x=x.detach().requires_grad_(True)
    pred=network(x)
    grad=torch.autograd.grad(pred.sum(),x,create_graph=True)[0]
    lap=sum(torch.autograd.grad(grad[:,i].sum(),x,create_graph=True)[0][:,i:i+1] for i in range(2))
    a,agrad=model.a_and_gradient(x,region)
    return -a*lap-(agrad*grad).sum(dim=1,keepdim=True)-orig.source_torch(x)

def e8_xpinn_predict(model,xy):
    xy=np.asarray(xy);inner=orig.inside_inner_np(xy);v=np.zeros(len(xy))
    with torch.no_grad():
        if inner.any():v[inner]=model.inner(orig.tensor(xy[inner])).cpu().numpy().ravel()
        if (~inner).any():v[~inner]=model.outer(orig.tensor(xy[~inner])).cpu().numpy().ravel()
    return v

def run_e8_xpinn(c,epochs=None):
    epochs=(3000 if MODE=='FULL' else 3) if epochs is None else int(epochs)
    params=dict(p=c['p'],lr=1e-3,epochs_parameterization='shared smooth coefficient',
               epochs_training=epochs,architecture='two tanh networks, 4x80',
               residual='-a Delta u - grad a . grad u - f')
    folder=_run_folder('E8_XPINN',c,params)
    cached=_cached_result(folder)
    if cached and cached.get('completed_epochs',0)>=epochs:return cached
    seed=c['seed'];torch.manual_seed(seed)
    m=E8XPINN(c['p']).to(DEVICE);d=c['data']
    p1=orig.tensor(orig.sample_uniform_in_inner(orig.POOL_INTERIOR,seed=seed+11001))
    p2=orig.tensor(orig.sample_uniform_in_outer_shell(orig.POOL_INTERIOR,seed=seed+11002))
    bnd=orig.tensor(orig.outer_boundary_points(orig.POOL_BOUNDARY))
    interface,normals=orig.interface_points_and_normals_np(orig.POOL_INTERFACE)
    interface=orig.tensor(interface);normals=orig.tensor(normals)
    x1,x2=orig.tensor(d['x1_train']),orig.tensor(d['x2_train'])
    y1,y2=orig.tensor(d['y1_train']).reshape(-1,1),orig.tensor(d['y2_train']).reshape(-1,1)
    opt=torch.optim.Adam(m.parameters(),lr=1.e-3)
    history=[];best_val=float('inf');best_ep=0;seconds0=0;completed=0
    chk=folder/'training_checkpoint.pt'
    if chk.exists():
        blob=torch.load(chk,map_location=DEVICE,weights_only=True)
        if blob['data_id']!=c['data_id'] or blob['params']!=params:raise ValueError('E8 XPINN checkpoint mismatch')
        m.load_state_dict(blob['model']);opt.load_state_dict(blob['optimizer']);completed=int(blob['completed'])
        history=list(blob['history']);best_val=float(blob['best_val']);best_ep=int(blob['best_ep']);seconds0=float(blob['seconds'])
        torch.set_rng_state(blob['torch_rng_cpu'].cpu())
    ckpt_every=500 if MODE=='FULL' else 2;start=time.perf_counter()
    for epoch in range(completed+1,epochs+1):
        opt.zero_grad(set_to_none=True)
        i1=torch.randint(len(p1),(min(512,len(p1)),),device=DEVICE)
        i2=torch.randint(len(p2),(min(512,len(p2)),),device=DEVICE)
        ib=torch.randint(len(bnd),(min(256,len(bnd)),),device=DEVICE)
        ii=torch.randint(len(interface),(min(256,len(interface)),),device=DEVICE)
        r1=e8_xpinn_residual(m,m.inner,p1[i1],1);r2=e8_xpinn_residual(m,m.outer,p2[i2],2)
        loss=5*((m.inner(x1)-y1).square().mean()+(m.outer(x2)-y2).square().mean())
        loss+=.5*(r1.square().mean()+r2.square().mean())+10*m.outer(bnd[ib]).square().mean()+.0001*(m.g**2).sum()
        xf=interface[ii].detach().requires_grad_(True);nf=normals[ii]
        u1=m.inner(xf);u2=m.outer(xf)
        grad1=torch.autograd.grad(u1.sum(),xf,create_graph=True)[0];grad2=torch.autograd.grad(u2.sum(),xf,create_graph=True)[0]
        a1,_=m.a_and_gradient(xf,1);a2,_=m.a_and_gradient(xf,2)
        loss+=10*(u1-u2).square().mean()+2*(a1*(grad1*nf).sum(1,keepdim=True)-a2*(grad2*nf).sum(1,keepdim=True)).square().mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'E8 XPINN nonfinite loss at epoch {epoch}')
        loss.backward();opt.step()
        if epoch%ckpt_every==0 or epoch==epochs:
            g=m.g.detach().cpu().numpy().copy();u=c['solver'].solve(g)
            v=.5*sum(np.mean((quick_interp_eval(u,d[f'x{k}_val'],c['fem'])-d[f'y{k}_val'])**2) for k in (1,2))
            elapsed=seconds0+time.perf_counter()-start
            history.append(dict(epoch=epoch,val_phy=v,diagnostic_e_gamma=float(norm(g-c['true_gamma'])/norm(c['true_gamma'])),seconds=elapsed))
            if v<best_val:
                best_val=v;best_ep=epoch;_atomic_torch(folder/'model_best.pt',m.state_dict())
            _atomic_torch(chk,dict(data_id=c['data_id'],params=params,completed=epoch,history=history,
                best_val=best_val,best_ep=best_ep,seconds=elapsed,model=m.state_dict(),optimizer=opt.state_dict(),
                torch_rng_cpu=torch.get_rng_state()))
            save_csv(folder/'training_history.csv',history)
    best=E8XPINN(c['p']).to(DEVICE)
    best.load_state_dict(torch.load(folder/'model_best.pt',map_location=DEVICE,weights_only=True));best.eval()
    g=best.g.detach().cpu().numpy().copy();u=c['solver'].solve(g)
    us=e8_xpinn_predict(best,c['fem'].points)
    row=_e8_error_row(c,g,u,us,method='E8-XPINN')
    row.update(complete=True,seconds=seconds0+time.perf_counter()-start,selected_step=best_ep,
        selection='minimum held-out PHYSICAL validation MSE',completed_epochs=epochs,
        n_neural_steps=epochs,n_forward_solves_approx=len(history),n_communications=epochs,
        n_parameters=sum(t.numel() for t in best.parameters()),output_folder=str(folder.relative_to(ROOT)))
    _save_result(folder,row,dict(gamma_selected=g,physical_state=u,xpinn_state=us))
    return row

def E8_higher_dimensional(seeds=BASE_TRIAL_SEEDS[:3],dims=(8,16),region='Omega1',mismatch=.03,
                          noise=0.,n_train=120,include_xpinn=True):
    """Actual 8/16-parameter FEM inversion with noncommuting stiffness matrices.

    New fine references are generated ONCE, cached. GN, Bi-HYCO, Bi-HYCO-0,
    and revised variable-coefficient XPINN share exactly the same data.
    """
    allrows=[]
    for p in dims:
        ref=e8_reference(p,region,mismatch)
        for seed in seeds:
            c=e8_case(p,region,mismatch,noise,seed,n_train=n_train)
            allrows += [run_e8_gn(c),run_e8_bihyco(c,interaction=1.),run_e8_bihyco(c,interaction=0.)]
            if include_xpinn:allrows.append(run_e8_xpinn(c))
    df=pd.DataFrame(allrows)
    df.to_csv(REPORTS/'E8_highdimensional_runs.csv',index=False)
    summ=df.groupby(['p','method']).agg(
        e_gamma_mean=('e_gamma','mean'),e_gamma_std=('e_gamma','std'),
        e_bg_field_mean=('e_bg_field','mean'),e_phy_mean=('e_phy_global_error','mean'),
        seconds_mean=('seconds','mean'),n=('seed','count')).reset_index()
    summ.to_csv(REPORTS/'E8_highdimensional_summary.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.3))
    for method,sub in summ.groupby('method'):
        ax.plot(sub['p'],sub['e_bg_field_mean']*100,marker='o',label=method)
    ax.set_xlabel('Number of spatial physical parameters');ax.set_ylabel('Error of background diffusion field (%)')
    ax.set_xticks(list(dims));ax.legend(fontsize=8);ax.grid(alpha=.2)
    save_figure(fig,FIGURES/'E8_dimension_background_error.png')
    return df

