# -*- coding: utf-8 -*-
"""Allen--Cahn: one-seed, two-parameter comparison with directional diagnostics.

Unknowns: positive nu and constant background alpha. All FOUR source amplitudes
are KNOWN. Three localized, positive reaction heterogeneities generate data only.
GN-DIST, cooperative joint-Adam Bi-HYCO, and XPINN-C share the same observations.

This is a development benchmark, not evidence of any method's superiority.
See the notebook and README for checkpoint and sensitivity interpretation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import copy, hashlib, json, math, os, platform, random, sys, time, warnings, gc
from typing import Callable
import numpy as np
import pandas as pd
from scipy.fft import dctn, idctn
from scipy.integrate import trapezoid
from scipy.optimize import brentq
from scipy.ndimage import map_coordinates
from scipy.linalg import svd
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt

VERSION = 'localized-two-parameter-v1'
T_END = 0.42
P_BACKGROUND = np.array([0.075, 65.0])
P_INITIAL = np.array([0.055, 50.0])
P_LOW, P_HIGH = np.array([0.02, 20.0]), np.array([0.16, 160.0])
KNOWN_BETA = np.array([15.0, -12.0, 10.0, -9.0])
SOURCE_GAUSSIANS = ((0.19,0.41,0.090),(0.76,0.29,0.095),
                    (0.37,0.80,0.085),(0.77,0.71,0.090))
# Used ONLY by the reference generator and post-training diagnostics.
HIDDEN_GAUSSIANS = ((0.68,0.35,0.065,1.0),(0.33,0.66,0.065,1.0),
                    (0.63,0.31,0.055,0.8))
STAR_R, STAR_A, STAR_M, STAR_W = .25, .09, 7, .028
PARAM_NAMES = ('nu','alpha')


def json_ready(x):
    if isinstance(x, dict): return {str(k):json_ready(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [json_ready(v) for v in x]
    if isinstance(x, np.ndarray): return json_ready(x.tolist())
    if isinstance(x, np.generic): return json_ready(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def save_json(path, x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(json_ready(x),indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def atomic_savez(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp,path)


def digest(x):
    return hashlib.sha256(json.dumps(json_ready(x),sort_keys=True).encode()).hexdigest()[:16]


def atomic_save(obj,path):
    path=Path(path); tmp=path.with_name(path.name+'.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)


def load_checkpoint(path, device='cpu'):
    # Load ONLY checkpoints produced by this program or other trusted files.
    return torch.load(path,map_location=device,weights_only=False)


def clone_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    mode: str = 'pilot'                 # 'smoke' tests wiring, not accuracy
    seed: int = 7
    root: str = 'AllenCahn_TwoParameter_Results'
    device: str = 'cpu'                 # CPU recommended; CUDA is optional
    threads: int = 1
    target_discrepancy: float = .12      # State discrepancy, NOT coefficient amplitude
    reference_n: int = 192
    reference_dt: float = .000525
    reference_stride: int = 4
    inverse_n: int = 64
    inverse_dt: float = .0015
    n_train: int = 1200                 # per holder
    n_val: int = 300
    n_test: int = 300
    hyco_rounds: int = 3000             # upper limit; extend without losing Adam states
    hyco_min_rounds: int = 1500
    hyco_patience: int = 400
    hyco_val_every: int = 20
    local_steps: int = 2                # BOTH parameter blocks, in BOTH holders
    hyco_width: int = 64
    hyco_physical_lr: float = .001
    hyco_neural_lr: float = .0008
    interaction_weight: float = .18
    ramp_rounds: int = 150
    ghost_points: int = 640             # uniformly sampled; twice the old count
    ic_points: int = 96
    ic_weight: float = .1               # cooperative loss; previously tested in continuations
    pretrain_steps: int = 1000
    pretrain_ic_weight: float = 1.5     # separate from cooperative IC weight
    fixed_diagnostic_every: int = 100
    diagnostic_ghost_points: int = 512
    diagnostic_ic_points: int = 256
    xp_epochs: int = 10000
    xp_min_epochs: int = 5000
    xp_patience: int = 2000
    xp_val_every: int = 200
    xp_warmup: int = 900
    xp_ramp_epochs: int = 900
    xp_width: int = 64
    xp_nn_lr: float = .0008
    xp_phys_lr: float = .0002
    xp_collocation: int = 96            # per quadrant; 50/25/25 mixed sampling
    xp_observation_batch: int = 96
    xp_ic_points: int = 64
    xp_bc_points: int = 48
    xp_interface_points: int = 48
    xp_data_weight: float = 6.
    xp_pde_weight: float = 8.
    xp_ic_weight: float = 7.            # NOT replaced by the Bi-HYCO IC weight
    xp_bc_weight: float = .5
    xp_interface_weight: float = 2.
    xp_residual_scale: float = 20.
    gradient_clip: float = 5.
    min_relative_improvement: float = .003  # patience only; exact minima are saved
    gn_max_iterations: int = 60
    gn_fd_step: float = 1e-3
    run_local_gn_diagnostic: bool = True # supplementary, not a fourth main method
    evaluation_frames: int = 61
    plot_dpi: int = 600
    export_pdf: bool = True
    make_plots: bool = True
    refinement_check: bool = True     # additional reference N -> N+64, dt/2
    sensitivity_fd_step: float = 1e-3
    run_interaction_sensitivity: bool = True
    interaction_data_points: int = 192
    interaction_points: int = 96
    interaction_ic_points: int = 96
    sensitivity_svd_rtol: tuple = (1e-6,1e-8,1e-10)
    force_continue: bool = False       # only to override a recorded early stop

    def effective(self):
        if self.mode not in ('smoke','pilot'):
            raise ValueError("mode must be 'smoke' or 'pilot'")
        if not (0 <= self.target_discrepancy <= .30):
            raise ValueError('Expected a prescribed state discrepancy in [0, .30].')
        c=copy.deepcopy(self)
        if self.mode=='smoke':
            vals=dict(reference_n=40,reference_dt=.007,reference_stride=1,
                inverse_n=20,inverse_dt=.014,n_train=48,n_val=16,n_test=16,
                hyco_rounds=min(c.hyco_rounds,6),hyco_min_rounds=6,hyco_patience=4,
                hyco_val_every=1,hyco_width=16,pretrain_steps=8,ramp_rounds=3,
                ghost_points=40,ic_points=16,fixed_diagnostic_every=2,
                diagnostic_ghost_points=48,diagnostic_ic_points=24,
                xp_epochs=min(c.xp_epochs,24),xp_min_epochs=24,xp_patience=12,
                xp_val_every=4,xp_warmup=4,xp_ramp_epochs=4,xp_width=16,
                xp_collocation=16,xp_observation_batch=12,xp_ic_points=12,
                xp_bc_points=12,xp_interface_points=12,gn_max_iterations=12,
                evaluation_frames=13,interaction_data_points=24,
                interaction_points=16,interaction_ic_points=16,refinement_check=False)
            c=replace(c,**vals)
        if c.reference_n<=c.inverse_n or c.reference_dt>=c.inverse_dt:
            raise ValueError('Reference must be finer in both space and time.')
        if c.plot_dpi!=600:
            warnings.warn('The requested publication setting is plot_dpi=600.')
        return c


# %% Reference equation, forward solvers and independent observation splits

def star_initial(x,y):
    x=np.asarray(x);y=np.asarray(y)
    r=np.hypot(x-.5,y-.5);th=np.arctan2(y-.5,x-.5)
    return np.tanh((STAR_R+STAR_A*np.cos(STAR_M*th)-r)/STAR_W)


def gaussian(x,y,cx,cy,sigma):
    return np.exp(-.5*(((x-cx)/sigma)**2+((y-cy)/sigma)**2))


def hidden_shape(x,y):
    return sum(a*gaussian(x,y,cx,cy,s) for cx,cy,s,a in HIDDEN_GAUSSIANS)


def envelopes(t):
    tau=np.asarray(t)/T_END
    return np.stack([np.sin(np.pi*tau)**2,.3+.7*np.cos(np.pi*tau)**2,
        np.exp(-((tau-.37)/.24)**2),.3+.7*np.sin(2*np.pi*tau)**2],axis=-1)


@dataclass
class Trajectory:
    states: np.ndarray
    times: np.ndarray
    x: np.ndarray

    def sample(self,points):
        p=np.asarray(points,dtype=float)
        if len(p)==0:return np.empty(0)
        n=len(self.x);h=self.x[1]-self.x[0]
        xf=np.clip((p[:,0]-self.x[0])/h,0,n-1-1e-10)
        yf=np.clip((p[:,1]-self.x[0])/h,0,n-1-1e-10)
        i=xf.astype(int);j=yf.astype(int);wx=xf-i;wy=yf-j
        k=np.clip(np.searchsorted(self.times,p[:,2],side='right')-1,0,len(self.times)-2)
        wt=np.clip((p[:,2]-self.times[k])/(self.times[k+1]-self.times[k]),0,1)
        a=self.states
        def bil(q):
            return ((1-wx)*(1-wy)*a[q,i,j]+wx*(1-wy)*a[q,i+1,j]
                    +(1-wx)*wy*a[q,i,j+1]+wx*wy*a[q,i+1,j+1])
        return (1-wt)*bil(k)+wt*bil(k+1)


def save_trajectory(path,tr,**extra):
    atomic_savez(path,states=tr.states.astype(np.float32),times=tr.times,
                        x=tr.x,y=tr.x,**extra)


def load_trajectory(path):
    with np.load(path,allow_pickle=False) as a:
        return Trajectory(a['states'],a['times'],a['x'])


class ForwardSolver:
    """Cell-centred Neumann finite differences; exact subflows, Strang composition.

    Source half-step, reaction half-step, diffusion step, reaction half-step,
    source half-step. Midpoint forcing. Computation and in-memory derivatives use
    float64. Saved fields use float32 only to reduce disk space.
    """
    def __init__(self,n,dt,stride=1):
        self.n=int(n);self.steps=int(np.ceil(T_END/dt));self.dt=T_END/self.steps
        self.stride=stride;self.x=(np.arange(n)+.5)/n
        X,Y=np.meshgrid(self.x,self.x,indexing='ij')
        self.u0=star_initial(X,Y)
        lam=-4*n*n*np.sin(np.pi*np.arange(n)/(2*n))**2
        self.L=lam[:,None]+lam[None,:]
        self.hidden=hidden_shape(X,Y)
        self.src=np.stack([gaussian(X,Y,*s) for s in SOURCE_GAUSSIANS])
        self.q=envelopes((np.arange(self.steps)+.5)*self.dt)*KNOWN_BETA
        # Known forcing is precomputed (no unknown source amplitudes).
        self.forcing=(np.einsum('kj,jxy->kxy',self.q,self.src) if n<=96 else None)

    def solve(self,p,heterogeneity_scale=0.):
        p=np.asarray(p,dtype=float)
        if p.shape!=(2,) or not np.all(p>0):raise ValueError('p must contain positive nu, alpha')
        diff=np.exp(p[0]*self.dt*self.L)
        decay=np.exp(-p[1]*np.exp(heterogeneity_scale*self.hidden)*self.dt)
        u=self.u0.copy();states=[u.copy()];times=[0.]
        for k in range(self.steps):
            f=self.forcing[k] if self.forcing is not None else np.einsum('j,jxy->xy',self.q[k],self.src)
            u=u+.5*self.dt*f
            u=u/np.sqrt(decay+(1-decay)*u*u)
            u=idctn(dctn(u,type=2,norm='ortho',workers=1)*diff,
                    type=2,norm='ortho',workers=1)
            u=u/np.sqrt(decay+(1-decay)*u*u)
            u=u+.5*self.dt*f
            if (k+1)%self.stride==0 or k==self.steps-1:
                states.append(u.copy());times.append((k+1)*self.dt)
        a=np.stack(states)
        if not np.all(np.isfinite(a)):raise FloatingPointError('Nonfinite PDE solution')
        return Trajectory(a,np.asarray(times),self.x)


def st_norm(tr):
    return float(np.sqrt(trapezoid(np.mean(tr.states.astype(float)**2,axis=(1,2)),tr.times)))


def uniform_points(rng,n):
    a=rng.random((n,3));a[:,2]*=T_END;return a


def ic_points(rng,n):
    a=rng.random((n,3));a[:,2]=0.;return a,star_initial(a[:,0],a[:,1])


def sample_holder(rng,holder,n):
    # Same 40%-initial-front design as the earlier experiment. No hidden centers.
    p=np.empty((n,3))
    for i in range(n):
        x=y=np.nan
        if rng.random()<.4:
            for _ in range(300):
                th=rng.uniform(-np.pi,np.pi)
                r=STAR_R+STAR_A*np.cos(STAR_M*th)+rng.normal(0,.055)
                a=.5+r*np.cos(th);b=.5+r*np.sin(th)
                if 0<a<1 and 0<b<1 and (((a<.5)==(b<.5))==(holder==0)):
                    x,y=a,b;break
        if not(np.isfinite(x) and np.isfinite(y)):
            while True:
                x,y=rng.random(2)
                if (((x<.5)==(y<.5))==(holder==0)):break
        p[i]=x,y,T_END*((i%6)+rng.random())/6
    return p[rng.permutation(n)]


def reference_signature(c):
    return dict(version=VERSION,T=T_END,background=P_BACKGROUND,known_beta=KNOWN_BETA,
        sources=SOURCE_GAUSSIANS,hidden=HIDDEN_GAUSSIANS,star=[STAR_R,STAR_A,STAR_M,STAR_W],
        target=c.target_discrepancy,n=c.reference_n,dt=c.reference_dt,stride=c.reference_stride)


def prepare_data(c,root):
    shared=root/'reference_cache'/digest(reference_signature(c));shared.mkdir(parents=True,exist_ok=True)
    rf=shared/'reference_heterogeneous.npz';hf=shared/'reference_homogeneous.npz'
    calfile=shared/'calibration.json'
    if not all(p.exists() for p in (rf,hf,calfile)):
        print('Calibrating the heterogeneous reference (independent of training)...',flush=True)
        start=time.perf_counter();solver=ForwardSolver(c.reference_n,c.reference_dt,c.reference_stride)
        homogeneous=solver.solve(P_BACKGROUND)
        def score(s):
            tr=solver.solve(P_BACKGROUND,float(s))
            return st_norm(Trajectory(tr.states-homogeneous.states,tr.times,tr.x))/st_norm(tr)
        if c.target_discrepancy==0:scale=0.
        else:
            upper=.5
            while score(upper)<c.target_discrepancy and upper<2.5:upper+=.5
            if score(upper)<c.target_discrepancy:
                raise RuntimeError('Target discrepancy not bracketed; do not silently increase it.')
            scale=brentq(lambda s:score(s)-c.target_discrepancy,0,upper,xtol=1e-5)
        reference=solver.solve(P_BACKGROUND,scale)
        achieved=st_norm(Trajectory(reference.states-homogeneous.states,reference.times,reference.x))/st_norm(reference)
        alpha=P_BACKGROUND[1]*np.exp(scale*solver.hidden)
        cal=dict(target=c.target_discrepancy,achieved=achieved,heterogeneity_scale=scale,
            alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),
            calibration_seconds=time.perf_counter()-start,reference_signature=reference_signature(c))
        save_trajectory(hf,homogeneous);save_trajectory(rf,reference,alpha_true=alpha)
        save_json(calfile,cal)
    cal=json.loads(calfile.read_text());reference=load_trajectory(rf);homogeneous=load_trajectory(hf)
    if abs(cal['achieved']-c.target_discrepancy)>5e-4:raise RuntimeError('Cached discrepancy mismatch')
    obs_key=digest(dict(ref=reference_signature(c),seed=c.seed,train=c.n_train,
                        val=c.n_val,test=c.n_test,sampler='alternating-quadrants-40percent-front-v1',noise=0.))
    datafolder=shared/'observations'/obs_key;datafolder.mkdir(parents=True,exist_ok=True)
    file=datafolder/'observations.npz'
    if not file.exists():
        rng=np.random.default_rng(c.seed+131);obs={}
        for holder in (0,1):
            for split,n in (('train',c.n_train),('val',c.n_val),('test',c.n_test)):
                X=sample_holder(rng,holder,n)
                obs[f'X_{split}_{holder}']=X;obs[f'Y_{split}_{holder}']=reference.sample(X)
        atomic_savez(file,**obs)
    with np.load(file,allow_pickle=False) as a:obs={k:a[k] for k in a.files}
    for holder in (0,1):
        for split,n in (('train',c.n_train),('val',c.n_val),('test',c.n_test)):
            X=obs[f'X_{split}_{holder}'];assert len(X)==n
            assert np.all((((X[:,0]<.5)==(X[:,1]<.5))==(holder==0)))
    XX=np.concatenate([obs[k] for k in sorted(obs) if k.startswith('X_')])
    if len(np.unique(XX,axis=0))!=len(XX):raise RuntimeError('Duplicate observation coordinates across splits')
    dataid=hashlib.sha256(b''.join(k.encode()+obs[k].tobytes() for k in sorted(obs))).hexdigest()[:16]
    save_json(datafolder/'metadata.json',dict(data_id=dataid,seed=c.seed,noise=0.,
        known_source_amplitudes=KNOWN_BETA,unknowns=PARAM_NAMES,reference_path=rf))
    print(f'Data ID: {dataid}; calibrated state mismatch: {cal["achieved"]:.4%}',flush=True)
    return obs,reference,homogeneous,cal,dataid,datafolder


# %% Differentiable forward solver and shared neural parametrization

class TorchDCT2:
    """Orthonormal DCT-II and its inverse, exactly matching scipy.fft.dctn.

    Implemented using even-extension FFTs, with a differentiable computational
    graph and O(n^2 log n) spatial complexity. No external torch-dct package.
    """
    def __init__(self,n,device,dtype):
        self.n=n
        k=torch.arange(n,dtype=dtype,device=device)
        self.phase_fwd=torch.polar(torch.ones_like(k),-torch.pi*k/(2*n))
        self.phase_inv=torch.conj(self.phase_fwd)
        self.scale=math.sqrt(2/n)
        self.inv_sqrt2=2**-.5

    def fwd1(self,x,dim):
        n=self.n
        v=torch.cat([x,torch.flip(x,dims=(dim,))],dim=dim)
        f=torch.fft.rfft(v,dim=dim).narrow(dim,0,n)
        shape=[1]*x.ndim;shape[dim]=n
        a=.5*self.scale*(f*self.phase_fwd.view(shape)).real
        return torch.cat([a.narrow(dim,0,1)*self.inv_sqrt2,
                          a.narrow(dim,1,n-1)],dim=dim)

    def inv1(self,x,dim):
        n=self.n
        shape=[1]*x.ndim;shape[dim]=n
        a=torch.cat([x.narrow(dim,0,1)/self.inv_sqrt2,
                     x.narrow(dim,1,n-1)],dim=dim)
        f=(2/self.scale)*a*self.phase_inv.view(shape)
        f=torch.cat([f,torch.zeros_like(f.narrow(dim,0,1))],dim=dim)
        return torch.fft.irfft(f,n=2*n,dim=dim).narrow(dim,0,n)

    def fwd2(self,x):
        return self.fwd1(self.fwd1(x,-1),-2)

    def inv2(self,x):
        return self.inv1(self.inv1(x,-2),-1)


class ObservationSampler:
    """Fixed differentiable trilinear sampler on cell-centered x/y and uniform t."""
    def __init__(self,points,n,dt,nsteps,device,dtype):
        p=np.asarray(points,dtype=float)
        i_float=np.clip(p[:,0]*n-.5,0,n-1-1e-10)
        j_float=np.clip(p[:,1]*n-.5,0,n-1-1e-10)
        it=np.floor(i_float).astype(np.int64)
        jt=np.floor(j_float).astype(np.int64)
        t_float=np.clip(p[:,2]/dt,0,nsteps)
        kt=np.minimum(np.floor(t_float).astype(np.int64),nsteps-1)
        self.i=torch.tensor(it,device=device,dtype=torch.long)
        self.j=torch.tensor(jt,device=device,dtype=torch.long)
        self.k=torch.tensor(kt,device=device,dtype=torch.long)
        self.wx=torch.tensor(i_float-it,dtype=dtype,device=device)
        self.wy=torch.tensor(j_float-jt,dtype=dtype,device=device)
        self.wt=torch.tensor(np.clip(t_float-kt,0,1),dtype=dtype,device=device)

    def __call__(self,states):
        i,j,k=self.i,self.j,self.k
        wx,wy,wt=self.wx,self.wy,self.wt
        def bil(kk):
            return ((1-wx)*(1-wy)*states[kk,i,j]
                    +wx*(1-wy)*states[kk,i+1,j]
                    +(1-wx)*wy*states[kk,i,j+1]
                    +wx*wy*states[kk,i+1,j+1])
        return (1-wt)*bil(k)+wt*bil(k+1)


class TorchForward:
    """Same discrete physical map as ForwardSolver; differentiate w.r.t. log nu, log alpha."""
    def __init__(self,solver,device='cpu'):
        self.n=solver.n;self.steps=solver.steps;self.dt=solver.dt
        self.device=torch.device(device);self.dtype=torch.float64
        self.u0=torch.tensor(solver.u0,device=device,dtype=self.dtype)
        self.L=torch.tensor(solver.L,device=device,dtype=self.dtype)
        forcing=solver.forcing if solver.forcing is not None else np.einsum('kj,jxy->kxy',solver.q,solver.src)
        self.forcing=torch.tensor(forcing,device=device,dtype=self.dtype)
        self.dct=TorchDCT2(solver.n,self.device,self.dtype)

    def sampler(self,X):
        return ObservationSampler(X,self.n,self.dt,self.steps,self.device,self.dtype)

    def solve(self,z):
        p=torch.exp(z);diff=torch.exp(p[0]*self.dt*self.L);decay=torch.exp(-p[1]*self.dt)
        u=self.u0;states=[u]
        for k in range(self.steps):
            u=u+.5*self.dt*self.forcing[k]
            u=u/torch.sqrt(decay+(1-decay)*u*u)
            u=self.dct.inv2(self.dct.fwd2(u)*diff)
            u=u/torch.sqrt(decay+(1-decay)*u*u)
            u=u+.5*self.dt*self.forcing[k]
            states.append(u)
        return torch.stack(states)


class StateNet(nn.Module):
    def __init__(self,width=64):
        super().__init__();layers=[];d=3
        for _ in range(3):layers.extend([nn.Linear(d,width),nn.Tanh()]);d=width
        layers.append(nn.Linear(d,1));self.layers=nn.Sequential(*layers)
    def forward(self,x):
        v=torch.stack((2*x[:,0]-1,2*x[:,1]-1,2*x[:,2]/T_END-1),dim=1)
        return self.layers(v).squeeze(-1)


def tensor(a,device='cpu',grad=False,dtype=torch.float32):
    return torch.tensor(np.asarray(a),device=device,dtype=dtype,requires_grad=grad)


def predict_network(net,X,batch=8192):
    dtype=next(net.parameters()).dtype;device=next(net.parameters()).device
    with torch.no_grad():
        return np.concatenate([net(tensor(X[k:k+batch],device,dtype=dtype)).cpu().numpy()
                               for k in range(0,len(X),batch)])


def physical_validation(solver,p,obs,split='val'):
    tr=solver.solve(p)
    local=[float(np.mean((tr.sample(obs[f'X_{split}_{i}'])-obs[f'Y_{split}_{i}'])**2)) for i in (0,1)]
    return .5*sum(local)


def neural_validation(net,obs,split='val'):
    return .5*sum(float(np.mean((predict_network(net,obs[f'X_{split}_{i}'])-
                    obs[f'Y_{split}_{i}'])**2)) for i in (0,1))


def check_solver(torch_solver,solver):
    p=np.array([.063,57.]);z=np.log(p);rng=np.random.default_rng(246)
    X=uniform_points(rng,32);sample=torch_solver.sampler(X)
    ref=solver.solve(p)
    target=tensor(ref.sample(X)+.02*np.sin(np.arange(len(X))),torch_solver.device,dtype=torch.float64)
    zz=tensor(z,torch_solver.device,True,torch.float64)
    state=torch_solver.solve(zz);objective=(sample(state)-target).square().mean()
    grad=torch.autograd.grad(objective,zz)[0].detach().cpu().numpy()
    fd=[]
    for j in range(2):
        a=z.copy();b=z.copy();a[j]+=2e-4;b[j]-=2e-4
        aa=solver.solve(np.exp(a)).sample(X);bb=solver.solve(np.exp(b)).sample(X)
        fd.append((np.mean((aa-target.cpu().numpy())**2)-np.mean((bb-target.cpu().numpy())**2))/4e-4)
    result=dict(max_field_difference=float(np.max(abs(state.detach().cpu().numpy()-ref.states))),
        max_sample_difference=float(np.max(abs(sample(state).detach().cpu().numpy()-ref.sample(X)))),
        relative_gradient_error=float(np.linalg.norm(grad-fd)/max(np.linalg.norm(fd),1e-14)),
        autodiff=grad,finite_difference=fd)
    if result['max_field_difference']>2e-8 or result['relative_gradient_error']>2e-3:
        raise RuntimeError(f'Physical solver / gradient check failed: {result}')
    return result


def output_folder(root,dataid,c,method):
    d=asdict(c)
    relevant={k:v for k,v in d.items() if k.startswith('hyco_')} if method=='bihyco' else {}
    if method=='bihyco':
        for k in ('local_steps','interaction_weight','ramp_rounds','ghost_points','ic_points',
                  'ic_weight','pretrain_steps','pretrain_ic_weight','gradient_clip'):
            relevant[k]=d[k]
    elif method=='xpinn_c':relevant={k:v for k,v in d.items() if k.startswith('xp_')}
    else:relevant={k:v for k,v in d.items() if k.startswith('gn_')}
    # Budget, validation frequency and stopping settings may be extended without
    # changing weights/RNG/Adam states. Optimizer/architecture changes create a new run.
    for k in tuple(relevant):
        if any(x in k for x in ('rounds','epochs','patience','val_every','max_iterations')) and k not in ('ramp_rounds','xp_ramp_epochs'):
            relevant.pop(k)
    relevant.update(version=VERSION,method=method,seed=c.seed,data_id=dataid,
                    inverse_n=c.inverse_n,inverse_dt=c.inverse_dt,device=c.device)
    path=root/'runs'/f'seed{c.seed}_{dataid}'/f'{method}_{digest(relevant)}'
    path.mkdir(parents=True,exist_ok=True)
    save_json(path/'configuration.json',dict(config=asdict(c),signature=relevant,
        unknowns=PARAM_NAMES,known_source_amplitudes=KNOWN_BETA))
    return path


def selected_payload(net,p,step,score):
    return dict(step=int(step),params=np.asarray(p).copy(),score=float(score),model_state=clone_state(net))


class Selection:
    """Exact best-checkpoint selection; patience updated only by significant gains."""
    def __init__(self,rel=.003):
        self.rel=rel;self.best_ph=None;self.best_nn=None
        self.pat_ph=np.inf;self.pat_nn=np.inf;self.last_ph=0;self.last_nn=0
    def update(self,net,p,step,vp,vn,out):
        if np.isfinite(vp):
            if self.best_ph is None or vp<self.best_ph['score']:
                self.best_ph=selected_payload(net,p,step,vp);atomic_save(self.best_ph,out/'best_physical_checkpoint.pt')
            if vp<self.pat_ph*(1-self.rel):self.pat_ph=vp;self.last_ph=step
        if np.isfinite(vn):
            if self.best_nn is None or vn<self.best_nn['score']:
                self.best_nn=selected_payload(net,p,step,vn);atomic_save(self.best_nn,out/'best_neural_checkpoint.pt')
            if vn<self.pat_nn*(1-self.rel):self.pat_nn=vn;self.last_nn=step
    def stop(self,step,minimum,patience):
        return step>=minimum and step-max(self.last_ph,self.last_nn)>=patience
    def pack(self):return copy.deepcopy(self.__dict__)
    def unpack(self,x):self.__dict__.update(x)


# %% GN-DIST: aggregate local normal-equation statistics, not raw observations

def train_gn(c,solver,obs,out,weights=(.5,.5),name='gn_dist'):
    path=out/'gn_checkpoint.pt'
    if path.exists():
        cp=load_checkpoint(path)
        if cp['finished'] and cp['max_iterations']>=c.gn_max_iterations:
            print(f'Reusing {name}: {path}',flush=True);return cp
    start=time.perf_counter();z=np.log(P_INITIAL);damping=.01;hist=[];solves=0
    lower,upper=np.log(P_LOW),np.log(P_HIGH)
    def solve(z):
        nonlocal solves
        solves+=1;return solver.solve(np.exp(z))
    def holder_statistics(i,base,plus,minus,h):
        X=obs[f'X_train_{i}'];y=obs[f'Y_train_{i}'];r=base.sample(X)-y
        J=np.column_stack([(p.sample(X)-m.sample(X))/(2*h) for p,m in zip(plus,minus)])
        return J.T@J/len(r),J.T@r/len(r),float(r@r/len(r))
    def scalar_cost(tr):
        return sum(w*float(np.mean((tr.sample(obs[f'X_train_{i}'])-obs[f'Y_train_{i}'])**2))
                   for i,w in enumerate(weights) if w)
    reason='iteration_limit';projected=np.inf
    for it in range(1,c.gn_max_iterations+1):
        tr=solve(z);h=c.gn_fd_step;plus=[];minus=[]
        for j in range(2):
            zp=z.copy();zm=z.copy();zp[j]+=h;zm[j]-=h
            plus.append(solve(zp));minus.append(solve(zm))
        local=[holder_statistics(i,tr,plus,minus,h) for i in (0,1)]
        A=sum(w*s[0] for w,s in zip(weights,local));b=sum(w*s[1] for w,s in zip(weights,local))
        cost=sum(w*s[2] for w,s in zip(weights,local))
        projected=float(np.linalg.norm(z-np.clip(z-b,lower,upper),ord=np.inf))
        accepted=False;delta=np.zeros(2);trialcost=cost
        if projected<1e-9:
            reason='projected_gradient_tolerance'
        else:
            diagonal=np.maximum(np.diag(A),1e-8)
            for _ in range(8):
                dz=-np.linalg.solve(A+damping*np.diag(diagonal)+1e-12*np.eye(2),b)
                dz=np.clip(dz,-.7,.7)
                for factor in (1.,.5,.25,.125,.0625):
                    zz=np.clip(z+factor*dz,lower,upper);tt=solve(zz);cc=scalar_cost(tt)
                    if cc<cost-max(1e-14,1e-9*cost):
                        delta=zz-z;z=zz;trialcost=cc;accepted=True;damping=max(1e-10,damping/2);break
                if accepted:break
                damping=min(1e8,damping*5)
        hist.append(dict(iteration=it,train_cost_before=cost,train_cost_after=trialcost,
            projected_gradient_before=projected,damping=damping,accepted=accepted,
            nu_hat=np.exp(z[0]),alpha_hat=np.exp(z[1]),step_norm=np.linalg.norm(delta),
            val_physical_mse=physical_validation(solver,np.exp(z),obs),solves=solves))
        pd.DataFrame(hist).to_csv(out/'gn_history.csv',index=False)
        if reason=='projected_gradient_tolerance':break
        if not accepted:reason='no_accepted_descent_step';break
        if np.linalg.norm(delta)<1e-7:reason='step_tolerance';break
    cp=dict(params=np.exp(z),step=it,history=hist,time_seconds=time.perf_counter()-start,
        solver_calls=solves,termination=reason,projected_gradient_before_last_step=projected,
        finished=True,max_iterations=c.gn_max_iterations,method=name,
        communication='two local 2x2 normal matrices, two 2-vectors and scalar costs',weights=weights)
    atomic_save(cp,path);save_json(out/'gn_diagnostics.json',cp)
    print(f'{name}: params={np.exp(z)}, stop={reason}, solves={solves}',flush=True)
    return cp
# %% Bi-HYCO: two complete copies, simultaneous local Adam, parameter coordinator

def fixed_hyco_diagnostic(net,p,physics,obs,c,points):
    """Diagnostic quadrature is fixed. Does not update models or choose checkpoints."""
    device=physics.device;z=tensor(np.log(p),device,True,torch.float64)
    tr=physics.solve(z)
    g,ic=points['ghost'],points['ic']
    yp=tensor(obs['Y_train_0'],device,dtype=torch.float64)
    ys=tensor(obs['Y_train_1'],device)
    lp=(physics.sampler(obs['X_train_0'])(tr)-yp).square().mean()
    ls=(net(tensor(obs['X_train_1'],device))-ys).square().mean()
    li=(physics.sampler(g)(tr)-net(tensor(g,device)).double()).square().mean()
    lic=(net(tensor(ic,device))-tensor(star_initial(ic[:,0],ic[:,1]),device)).square().mean()
    losses=[lp,ls,li,lic];weights=[.5,.5,c.interaction_weight,c.ic_weight]
    neural=list(net.parameters());params=[z]+neural
    details={};totals=[torch.zeros_like(z),None]
    for key,weight,loss in zip(('physical','synthetic','interaction','initial'),weights,losses):
        grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        gz=grad[0] if grad[0] is not None else torch.zeros_like(z)
        gn=torch.cat([(v if v is not None else torch.zeros_like(q)).reshape(-1)
                      for v,q in zip(grad[1:],neural)])
        totals[0]=totals[0]+weight*gz
        totals[1]=weight*gn if totals[1] is None else totals[1]+weight*gn
        details.update({f'L_{key}':float(loss.detach()),f'weighted_{key}':weight*float(loss.detach()),
            f'grad_{key}_lognu':float(gz[0]),f'grad_{key}_logalpha':float(gz[1]),
            f'grad_{key}_neural_norm':float(gn.norm())})
    details.update(J=sum(w*float(l.detach()) for w,l in zip(weights,losses)),
        grad_J_lognu=float(totals[0][0]),grad_J_logalpha=float(totals[0][1]),
        grad_J_physical_norm=float(totals[0].norm()),grad_J_neural_norm=float(totals[1].norm()))
    return details


def train_bihyco(c,solver,obs,out):
    device=torch.device(c.device);set_seed(c.seed+197)
    physics=TorchForward(solver,device)
    parity=out/'solver_parity.json'
    if not parity.exists():save_json(parity,check_solver(physics,solver))
    rng=np.random.default_rng(c.seed+197)
    rng_diag=np.random.default_rng(c.seed+80197)
    di=uniform_points(rng_diag,c.diagnostic_ghost_points)
    xi,_=ic_points(rng_diag,c.diagnostic_ic_points)
    points=dict(ghost=di,ic=xi)
    atomic_savez(out/'fixed_diagnostic_points.npz',**points)
    global_net=StateNet(c.hyco_width).to(device);global_p=P_INITIAL.copy()
    prefile=out/'pretrained_neural.pt'
    Xsyn=tensor(obs['X_train_1'],device);ysyn=tensor(obs['Y_train_1'],device)
    pretime=0.
    if prefile.exists():
        pp=load_checkpoint(prefile,device);global_net.load_state_dict(pp['model_state'])
        rng.bit_generator.state=pp['rng'];torch.set_rng_state(pp['torch_rng'].cpu());pretime=pp['seconds']
    else:
        preopt=torch.optim.Adam(global_net.parameters(),lr=c.hyco_neural_lr)
        phist=[];prestart=time.perf_counter()
        for step in range(1,c.pretrain_steps+1):
            X0,y0=ic_points(rng,256 if c.mode=='pilot' else 16)
            preopt.zero_grad(set_to_none=True)
            data=(global_net(Xsyn)-ysyn).square().mean()
            ic=(global_net(tensor(X0,device))-tensor(y0,device)).square().mean()
            loss=data+c.pretrain_ic_weight*ic;loss.backward()
            torch.nn.utils.clip_grad_norm_(global_net.parameters(),c.gradient_clip);preopt.step()
            if step%50==0 or step==1 or step==c.pretrain_steps:
                phist.append(dict(step=step,data=float(data.detach()),initial=float(ic.detach()),loss=float(loss.detach())))
        pretime=time.perf_counter()-prestart
        pd.DataFrame(phist).to_csv(out/'pretraining_history.csv',index=False)
        atomic_save(dict(model_state=clone_state(global_net),rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),seconds=pretime),prefile)
    nets=[StateNet(c.hyco_width).to(device) for _ in (0,1)]
    zunits=[nn.Parameter(tensor(np.log(global_p),device,dtype=torch.float64)) for _ in (0,1)]
    opts=[torch.optim.Adam([{'params':[zunits[i]],'lr':c.hyco_physical_lr},
                           {'params':nets[i].parameters(),'lr':c.hyco_neural_lr}]) for i in (0,1)]
    for net in nets:net.load_state_dict(global_net.state_dict())
    sel=Selection(c.min_relative_improvement);hist=[];diagnostics=[]
    last=out/'last_training_checkpoint.pt';prior=pretime;diag_prior=0.;start_round=1
    if last.exists():
        cp=load_checkpoint(last,device)
        if cp.get('stopped_early',False) and not c.force_continue:
            print('Bi-HYCO previously early-stopped. Set force_continue=True to override.',flush=True);return cp
        if cp['step']>=c.hyco_rounds:return cp
        global_p=np.asarray(cp['global_p']);global_net.load_state_dict(cp['global_net'])
        for i in (0,1):
            nets[i].load_state_dict(cp['local_nets'][i])
            with torch.no_grad():zunits[i].copy_(cp['local_z'][i])
            opts[i].load_state_dict(cp['optimizers'][i])
        rng.bit_generator.state=cp['rng_state'];torch.set_rng_state(cp['torch_rng'].cpu())
        if device.type=='cuda' and cp.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(cp['cuda_rng'])
        hist=cp['history'];diagnostics=cp['fixed_diagnostics'];sel.unpack(cp['selection'])
        prior=cp['wall_seconds'];diag_prior=cp['diagnostic_seconds'];start_round=cp['step']+1
        print('Resuming Bi-HYCO at round',start_round,flush=True)
    phy_sampler=physics.sampler(obs['X_train_0']);yp=tensor(obs['Y_train_0'],device,dtype=torch.float64)
    zlo=tensor(np.log(P_LOW),device,dtype=torch.float64);zhi=tensor(np.log(P_HIGH),device,dtype=torch.float64)
    timer=time.perf_counter();diagnostic_seconds=diag_prior
    for rd in range(start_round,c.hyco_rounds+1):
        lam=c.interaction_weight*min(1.,rd/max(c.ramp_rounds,1))
        ghost=uniform_points(rng,c.ghost_points);gs=physics.sampler(ghost);Xg=tensor(ghost,device)
        X0,y0=ic_points(rng,c.ic_points);X0=tensor(X0,device);y0=tensor(y0,device)
        localp=[];localsd=[];gnorm=[];parts=[]
        for holder in (0,1):
            nets[holder].load_state_dict(global_net.state_dict())
            with torch.no_grad():zunits[holder].copy_(tensor(np.log(global_p),device,dtype=torch.float64))
            for st in range(c.local_steps):
                opt=opts[holder];net=nets[holder];z=zunits[holder]
                opt.zero_grad(set_to_none=True)
                states=physics.solve(z)
                interaction=(gs(states)-net(Xg).double()).square().mean()
                data=(phy_sampler(states)-yp).square().mean() if holder==0 else (net(Xsyn)-ysyn).square().mean()
                ic=(net(X0)-y0).square().mean()
                loss=data+lam*interaction+c.ic_weight*ic
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite Bi-HYCO loss at round {rd}')
                # Both gradients evaluated BEFORE either block is changed.
                loss.backward()
                pg=float(z.grad.norm());ng=float(torch.sqrt(sum(q.grad.square().sum() for q in net.parameters())))
                torch.nn.utils.clip_grad_norm_([z],c.gradient_clip)
                torch.nn.utils.clip_grad_norm_(net.parameters(),c.gradient_clip)
                opt.step()
                with torch.no_grad():z.clamp_(min=zlo,max=zhi)
            localp.append(np.exp(z.detach().cpu().numpy()));localsd.append(clone_state(net))
            gnorm.append((pg,ng));parts.append([float(data.detach()),float(interaction.detach()),float(ic.detach())])
        oldp=global_p.copy();global_p=.5*(localp[0]+localp[1])
        global_net.load_state_dict({k:.5*(localsd[0][k]+localsd[1][k]) for k in localsd[0]})
        if rd%c.hyco_val_every==0 or rd==1 or rd==c.hyco_rounds:
            vp=physical_validation(solver,global_p,obs);vn=neural_validation(global_net,obs)
            sel.update(global_net,global_p,rd,vp,vn,out)
            disagreement=np.linalg.norm(np.log(localp[0])-np.log(localp[1]))
            row=dict(round=rd,lambda_effective=lam,val_physical_mse=vp,val_neural_mse=vn,
                nu_hat=global_p[0],alpha_hat=global_p[1],physical_scaled_disagreement=disagreement,
                step_log_norm=np.linalg.norm(np.log(global_p)-np.log(oldp)))
            for i in (0,1):
                row.update({f'unit{i+1}_nu':localp[i][0],f'unit{i+1}_alpha':localp[i][1],
                    f'unit{i+1}_physical_gradient_norm':gnorm[i][0],f'unit{i+1}_neural_gradient_norm':gnorm[i][1],
                    f'unit{i+1}_data_loss':parts[i][0],f'unit{i+1}_interaction_loss':parts[i][1],
                    f'unit{i+1}_initial_loss':parts[i][2]})
            if rd%c.fixed_diagnostic_every==0 or rd==1 or rd==c.hyco_rounds:
                dstart=time.perf_counter()
                diagnostic=dict(round=rd,**fixed_hyco_diagnostic(global_net,global_p,physics,obs,c,points))
                diagnostics.append(diagnostic);diagnostic_seconds+=time.perf_counter()-dstart
                pd.DataFrame(diagnostics).to_csv(out/'fixed_loss_and_gradient_diagnostics.csv',index=False)
            wall=prior+time.perf_counter()-timer
            row.update(wall_seconds=wall,optimization_validation_seconds=wall-diagnostic_seconds)
            hist.append(row)
            stopped=sel.stop(rd,c.hyco_min_rounds,c.hyco_patience)
            cp=dict(step=rd,global_p=global_p.copy(),global_net=clone_state(global_net),
                local_z=[z.detach().cpu().clone() for z in zunits],local_nets=localsd,
                optimizers=[copy.deepcopy(o.state_dict()) for o in opts],selection=sel.pack(),
                rng_state=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                history=hist,fixed_diagnostics=diagnostics,wall_seconds=wall,
                diagnostic_seconds=diagnostic_seconds,time_seconds=wall-diagnostic_seconds,
                stopped_early=stopped,method='bihyco')
            atomic_save(cp,last);pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
            print(f'Bi-HYCO {rd:5d}: Vphy={vp:.4e}; Vsyn={vn:.4e}; p={np.round(global_p,6)}',flush=True)
            if stopped:print('Bi-HYCO validation-based early stop.',flush=True);break
    return cp


# %% XPINN-C: mixed known-geometry sampling, warm-up and joint parameter inference

def quadrant_np(x):return (x[:,0]>=.5).astype(int)+2*(x[:,1]>=.5).astype(int)


class XPINN(nn.Module):
    def __init__(self,width):
        super().__init__();self.parts=nn.ModuleList([StateNet(width) for _ in range(4)])
    def forward(self,x):
        labels=(x[:,0]>=.5).long()+2*(x[:,1]>=.5).long()
        out=torch.zeros(len(x),dtype=x.dtype,device=x.device)
        for k in range(4):
            ix=torch.where(labels==k)[0]
            if len(ix):out=out.index_copy(0,ix,self.parts[k](x[ix]))
        return out


def sector_sample(rng,k,n):
    lo_x=0 if k in (0,2) else .5;lo_y=0 if k in (0,1) else .5
    return np.column_stack((rng.uniform(lo_x,lo_x+.5,n),rng.uniform(lo_y,lo_y+.5,n),rng.uniform(0,T_END,n)))


def mixed_collocation(rng,k,n):
    nu=n//2;nf=n//4;ns=n-nu-nf
    a=sector_sample(rng,k,nu)
    bounds=[(np.pi,1.5*np.pi),(1.5*np.pi,2*np.pi),(.5*np.pi,np.pi),(0,.5*np.pi)]
    th=rng.uniform(*bounds[k],nf);r=np.clip(STAR_R+STAR_A*np.cos(STAR_M*th)+rng.normal(0,.05,nf),.015,.43)
    b=np.column_stack((.5+r*np.cos(th),.5+r*np.sin(th),T_END*rng.beta(1.4,2.,nf)))
    cx,cy,sigma=SOURCE_GAUSSIANS[k]
    lx=0 if k in (0,2) else .5;ly=0 if k in (0,1) else .5
    # Match previous C sampler: clipped known-source neighbourhoods, not hidden heterogeneities.
    cc=np.column_stack((np.clip(rng.normal(cx,1.2*sigma,ns),lx+1e-6,lx+.5-1e-6),
        np.clip(rng.normal(cy,1.2*sigma,ns),ly+1e-6,ly+.5-1e-6),rng.uniform(0,T_END,ns)))
    allx=np.concatenate((a,b,cc));return allx[rng.permutation(n)]


def known_source_torch(x):
    tau=x[:,2]/T_END
    q=[torch.sin(np.pi*tau)**2,.3+.7*torch.cos(np.pi*tau)**2,
       torch.exp(-((tau-.37)/.24)**2),.3+.7*torch.sin(2*np.pi*tau)**2]
    total=torch.zeros(len(x),dtype=x.dtype,device=x.device)
    for beta,(cx,cy,sigma),env in zip(KNOWN_BETA,SOURCE_GAUSSIANS,q):
        total=total+float(beta)*torch.exp(-.5*(((x[:,0]-cx)/sigma)**2+((x[:,1]-cy)/sigma)**2))*env
    return total


def derivatives(net,x):
    u=net(x);d=torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    xx=torch.autograd.grad(d[:,0].sum(),x,create_graph=True)[0][:,0]
    yy=torch.autograd.grad(d[:,1].sum(),x,create_graph=True)[0][:,1]
    return u,d[:,2],xx+yy


def xpinn_loss(model,z,c,batches,rng,ep):
    device=z.device;nu,alpha=torch.exp(z);ramp=max(0.,min(1.,(ep-c.xp_warmup)/max(1,c.xp_ramp_epochs)))
    zero=sum(q.sum()*0 for q in model.parameters())
    losses={k:zero for k in ('data','pde','initial','boundary','interface')}
    for k in range(4):
        net=model.parts[k]
        if ramp>0:
            x=tensor(mixed_collocation(rng,k,c.xp_collocation),device,True)
            u,ut,lap=derivatives(net,x)
            res=ut-nu*lap+alpha*(u**3-u)-known_source_torch(x)
            losses['pde']=losses['pde']+(res/c.xp_residual_scale).square().mean()/4
        X,y=batches[k];ix=rng.integers(0,len(X),size=min(c.xp_observation_batch,len(X)))
        losses['data']=losses['data']+(net(X[ix])-y[ix]).square().mean()/4
        pp=sector_sample(rng,k,c.xp_ic_points);pp[:,2]=0.
        losses['initial']=losses['initial']+(net(tensor(pp,device))-tensor(star_initial(pp[:,0],pp[:,1]),device)).square().mean()/4
        b=sector_sample(rng,k,c.xp_bc_points);half=len(b)//2
        b[:half,0]=0. if k in (0,2) else 1.;b[half:,1]=0. if k in (0,1) else 1.
        xb=tensor(b,device,True);ub=net(xb)
        g=torch.autograd.grad(ub.sum(),xb,create_graph=True)[0]
        losses['boundary']=losses['boundary']+(.5*g[:half,0].square().mean()+.5*g[half:,1].square().mean())/4
    for ka,kb,axis in ((0,1,0),(2,3,0),(0,2,1),(1,3,1)):
        pp=sector_sample(rng,ka,c.xp_interface_points);pp[:,axis]=.5;x=tensor(pp,device,True)
        a=model.parts[ka](x);b=model.parts[kb](x)
        ga=torch.autograd.grad(a.sum(),x,create_graph=True)[0][:,axis]
        gb=torch.autograd.grad(b.sum(),x,create_graph=True)[0][:,axis]
        losses['interface']=losses['interface']+((a-b).square().mean()+.05*(ga-gb).square().mean())/4
    total=(c.xp_data_weight*losses['data']+ramp*c.xp_pde_weight*losses['pde']+
        c.xp_ic_weight*losses['initial']+c.xp_bc_weight*losses['boundary']+
        c.xp_interface_weight*losses['interface'])
    return total,losses,ramp


def train_xpinn(c,solver,obs,out):
    device=torch.device(c.device);set_seed(c.seed+1093);rng=np.random.default_rng(c.seed+1093)
    model=XPINN(c.xp_width).to(device);z=nn.Parameter(tensor(np.log(P_INITIAL),device))
    optimizer=torch.optim.Adam([{'params':model.parameters(),'lr':c.xp_nn_lr},
                                {'params':[z],'lr':c.xp_phys_lr}])
    sel=Selection(c.min_relative_improvement);hist=[];losses=[];prior=0.;start_epoch=1
    last=out/'last_training_checkpoint.pt'
    if last.exists():
        cp=load_checkpoint(last,device)
        if cp.get('stopped_early',False) and not c.force_continue:
            print('XPINN-C previously early-stopped; force_continue=True overrides.',flush=True);return cp
        if cp['step']>=c.xp_epochs:return cp
        model.load_state_dict(cp['model_state'])
        with torch.no_grad():z.copy_(cp['z'])
        optimizer.load_state_dict(cp['optimizer']);sel.unpack(cp['selection'])
        rng.bit_generator.state=cp['rng_state'];torch.set_rng_state(cp['torch_rng'].cpu())
        if device.type=='cuda' and cp.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(cp['cuda_rng'])
        hist=cp['history'];losses=cp['losses'];prior=cp['wall_seconds'];start_epoch=cp['step']+1
        print('Resuming XPINN-C at epoch',start_epoch,flush=True)
    batches=[]
    for k in range(4):
        holder=0 if k in (0,3) else 1
        X=obs[f'X_train_{holder}'];Y=obs[f'Y_train_{holder}'];ix=quadrant_np(X)==k
        if not np.any(ix):raise ValueError('No training observations in a quadrant')
        batches.append((tensor(X[ix],device),tensor(Y[ix],device)))
    low=tensor(np.log(P_LOW),device);high=tensor(np.log(P_HIGH),device)
    timer=time.perf_counter()
    for ep in range(start_epoch,c.xp_epochs+1):
        z.requires_grad_(ep>c.xp_warmup)
        optimizer.zero_grad(set_to_none=True)
        total,terms,ramp=xpinn_loss(model,z,c,batches,rng,ep)
        if not torch.isfinite(total):raise FloatingPointError(f'Nonfinite XPINN-C loss at {ep}')
        total.backward()
        pg=float(z.grad.norm()) if z.grad is not None else 0.
        ng=float(torch.sqrt(sum(v.grad.square().sum() for v in model.parameters() if v.grad is not None)))
        torch.nn.utils.clip_grad_norm_(model.parameters(),c.gradient_clip)
        if z.grad is not None:torch.nn.utils.clip_grad_norm_([z],c.gradient_clip)
        optimizer.step()
        with torch.no_grad():z.clamp_(min=low,max=high)
        if ep%max(1,c.xp_val_every//4)==0 or ep==1 or ep==c.xp_epochs:
            losses.append(dict(epoch=ep,total=float(total.detach()),pde_ramp=ramp,
                **{k:float(v.detach()) for k,v in terms.items()},
                physical_gradient_norm=pg,neural_gradient_norm=ng))
        if ep%c.xp_val_every==0 or ep==1 or ep==c.xp_epochs:
            p=np.exp(z.detach().cpu().numpy().astype(float))
            vp=physical_validation(solver,p,obs);vn=neural_validation(model,obs)
            sel.update(model,p,ep,vp,vn,out)
            wall=prior+time.perf_counter()-timer
            hist.append(dict(epoch=ep,val_physical_mse=vp,val_neural_mse=vn,
                nu_hat=p[0],alpha_hat=p[1],pde_ramp=ramp,physical_gradient_norm=pg,
                neural_gradient_norm=ng,wall_seconds=wall))
            stopped=sel.stop(ep,c.xp_min_epochs,c.xp_patience)
            cp=dict(step=ep,params=p,model_state=clone_state(model),z=z.detach().cpu().clone(),
                optimizer=copy.deepcopy(optimizer.state_dict()),selection=sel.pack(),
                rng_state=copy.deepcopy(rng.bit_generator.state),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
                history=hist,losses=losses,wall_seconds=wall,time_seconds=wall,
                stopped_early=stopped,method='xpinn_c')
            atomic_save(cp,last);pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
            pd.DataFrame(losses).to_csv(out/'loss_components.csv',index=False)
            print(f'XPINN-C {ep:6d}: Vphy={vp:.4e}; Vnn={vn:.4e}; p={np.round(p,6)}',flush=True)
            if stopped:print('XPINN-C validation-based early stop.',flush=True);break
    return cp
# %% Observation sensitivities and discrepancy directions (post-processing only)

def interpolate_frames(tr,times,xgrid):
    h=tr.x[1]-tr.x[0]
    XX,YY=np.meshgrid((xgrid-tr.x[0])/h,(xgrid-tr.x[0])/h,indexing='ij')
    coords=np.array([XX.ravel(),YY.ravel()]);out=[]
    for t in times:
        k=int(np.clip(np.searchsorted(tr.times,t,side='right')-1,0,len(tr.times)-2))
        a=float(np.clip((t-tr.times[k])/(tr.times[k+1]-tr.times[k]),0,1))
        field=(1-a)*tr.states[k]+a*tr.states[k+1]
        out.append(map_coordinates(field,coords,order=1,mode='nearest').reshape(len(xgrid),len(xgrid)))
    return np.asarray(out)


def observation_jacobians(solver,p,Xlist,h):
    """Derivatives in log coordinates: columns correspond to relative parameter changes."""
    z=np.log(p);parts=[];plus=[];minus=[]
    for j in range(2):
        a=z.copy();b=z.copy();a[j]+=h;b[j]-=h
        plus.append(solver.solve(np.exp(a)));minus.append(solver.solve(np.exp(b)))
    for X in Xlist:
        parts.append(np.column_stack([(a.sample(X)-b.sample(X))/(2*h) for a,b in zip(plus,minus)]))
    return parts,plus,minus


def matrix_sensitivity(A):
    U,s,Vt=svd(A,full_matrices=False,check_finite=True)
    for j in range(len(s)):
        ix=int(np.argmax(abs(Vt[j])))
        if Vt[j,ix]<0:Vt[j]*=-1;U[:,j]*=-1
    tol=max(A.shape)*np.finfo(float).eps*(s[0] if len(s) else 0)
    rank=int(np.sum(s>tol));norms=np.linalg.norm(A,axis=0)
    cosine=float(A[:,0]@A[:,1]/max(norms[0]*norms[1],1e-300))
    # Direction for one parameter after the other parameter is allowed to compensate.
    residual0=A[:,0]-A[:,1]*(A[:,1]@A[:,0])/max(norms[1]**2,1e-300)
    residual1=A[:,1]-A[:,0]*(A[:,0]@A[:,1])/max(norms[0]**2,1e-300)
    result=dict(sigma_max=s[0],sigma_min=s[-1],condition=s[0]/s[-1] if rank==2 else np.inf,
        numerical_rank=rank,column_norm_lognu=norms[0],column_norm_logalpha=norms[1],
        column_cosine=cosine,profiled_lognu_norm=np.linalg.norm(residual0),
        profiled_logalpha_norm=np.linalg.norm(residual1),
        weakest_lognu=Vt[-1,0],weakest_logalpha=Vt[-1,1],
        strongest_lognu=Vt[0,0],strongest_logalpha=Vt[0,1])
    return result,(U,s,Vt)


def sensitivity_analysis(c,solver,obs,reference,homogeneous,p,label,out):
    out=out/label;out.mkdir(parents=True,exist_ok=True)
    X=[obs['X_train_0'],obs['X_train_1']]
    deriv,plus,minus=observation_jacobians(solver,p,X,c.sensitivity_fd_step)
    small,_,_=observation_jacobians(solver,p,X,c.sensitivity_fd_step/2)
    fdrelative=[float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-14)) for a,b in zip(deriv,small)]
    matrices={'unit1_raw':deriv[0],'unit2_raw':deriv[1],'combined_raw':np.vstack(deriv),
              'unit1_rms':deriv[0]/np.sqrt(len(X[0])),
              'unit2_rms':deriv[1]/np.sqrt(len(X[1])),
              'balanced_rms':np.vstack([np.sqrt(.5/len(xx))*a for xx,a in zip(X,deriv)])}
    base=solver.solve(p);rows=[];projection_rows=[];directions=[]
    # r_total = structural discrepancy + fine/coarse numerical gap + parameter shift.
    structural=[reference.sample(x)-homogeneous.sample(x) for x in X]
    coarse_background=solver.solve(P_BACKGROUND)
    numerical=[homogeneous.sample(x)-coarse_background.sample(x) for x in X]
    param_shift=[coarse_background.sample(x)-base.sample(x) for x in X]
    total=[obs[f'Y_train_{i}']-base.sample(x) for i,x in enumerate(X)]
    for name,A in matrices.items():
        row,decomp=matrix_sensitivity(A);U,s,Vt=decomp
        rows.append(dict(point=label,observation_map=name,nu=p[0],alpha=p[1],**row))
        if name=='unit1_rms':ids=[0];weights=[1.]
        elif name=='unit2_rms':ids=[1];weights=[1.]
        elif name=='balanced_rms':ids=[0,1];weights=[.5,.5]
        else:continue
        for what,source in [('structural',structural),('discretization',numerical),('parameter_shift',param_shift),('total',total)]:
            r=np.concatenate([np.sqrt(w/len(X[i]))*source[i] for w,i in zip(weights,ids)])
            ur=U.T@r;ranktol=1e-10*s[0]
            coefficients=np.divide(ur,s,out=np.zeros_like(ur),where=s>ranktol)
            dz=Vt.T@coefficients
            orth=r-U@ur
            projection_rows.append(dict(point=label,observation_map=name,component=what,
                residual_norm=np.linalg.norm(r),tangent_projection_norm=np.linalg.norm(ur),
                orthogonal_norm=np.linalg.norm(orth),
                linear_shift_lognu=dz[0],linear_shift_logalpha=dz[1],
                residual_along_strong_left=ur[0],residual_along_weak_left=ur[-1],
                coefficient_strong=coefficients[0],coefficient_weak=coefficients[-1]))
        for strength,j in [('strong',0),('weak',-1)]:
            direction=Vt[j]
            for eps in (-.05,-.025,.025,.05):
                pp=np.asarray(p)*np.exp(eps*direction)
                trial=solver.solve(pp)
                change=np.concatenate([np.sqrt(w/len(X[i]))*(trial.sample(X[i])-base.sample(X[i]))
                                       for w,i in zip(weights,ids)])
                linear=A@(eps*direction)
                directions.append(dict(point=label,observation_map=name,direction=strength,
                    epsilon=eps,dlognu=direction[0],dlogalpha=direction[1],
                    actual_change_norm=np.linalg.norm(change),linear_change_norm=np.linalg.norm(linear),
                    relative_linearization_error=np.linalg.norm(change-linear)/max(np.linalg.norm(change),1e-14)))
    pd.DataFrame(rows).to_csv(out/'sensitivity_table.csv',index=False)
    pd.DataFrame(projection_rows).to_csv(out/'discrepancy_projections.csv',index=False)
    pd.DataFrame(directions).to_csv(out/'finite_direction_checks.csv',index=False)
    atomic_savez(out/'observation_jacobians.npz',**matrices,parameters=p,
                        coordinates=np.array(['log_nu','log_alpha']))
    save_json(out/'diagnostic_metadata.json',dict(linearization_parameters=p,fd_step=c.sensitivity_fd_step,
        relative_fd_change_h_to_halfh=fdrelative,linearization_point=label,
        uses_background_truth_for_posthoc_analysis_only=label=='background',
        interpretation='Local derivatives of the homogeneous forward map; rank is NOT global identifiability. '
        'Discrepancy projections are linear predictions, NOT guaranteed estimator errors. '
        'Raw matrices match Section 4.7; RMS-normalized matrices match empirical loss normalization.'))
    if any(e>1e-2 for e in fdrelative):warnings.warn(f'FD sensitivity check above 1% at {label}: {fdrelative}')
    if label=='background':
        times=np.linspace(0,T_END,13)
        fields=np.stack([(interpolate_frames(a,times,solver.x)-interpolate_frames(b,times,solver.x))
                          /(2*c.sensitivity_fd_step) for a,b in zip(plus,minus)])
        atomic_savez(out/'sensitivity_fields.npz',derivatives=fields.astype('float32'),
            times=times,x=solver.x,parameters=p,coordinates=np.array(['log_nu','log_alpha']))
    return pd.DataFrame(rows)


def neural_parameter_jacobian(net,X,chunk=32):
    """All trainable neural parameters, not just final-layer features; float64 CPU."""
    model=copy.deepcopy(net).cpu().double().eval()
    params=dict(model.named_parameters())
    try:
        from torch.func import functional_call, jacrev, vmap
        def scalar(par,x):return functional_call(model,par,(x.unsqueeze(0),)).squeeze()
        J=[]
        for k in range(0,len(X),chunk):
            jac=vmap(jacrev(scalar),in_dims=(None,0))(params,tensor(X[k:k+chunk],dtype=torch.float64))
            J.append(torch.cat([jac[name].reshape(len(X[k:k+chunk]),-1) for name in params],dim=1).detach().numpy())
        return np.vstack(J)
    except (ImportError,RuntimeError) as exc:
        warnings.warn(f'Using row-wise neural Jacobian fallback: {exc}')
        rows=[];pp=list(model.parameters())
        for x in X:
            y=model(tensor(x[None,:],dtype=torch.float64))
            grad=torch.autograd.grad(y.sum(),pp)
            rows.append(torch.cat([a.reshape(-1) for a in grad]).detach().numpy())
        return np.vstack(rows)


def interaction_sensitivity(c,solver,obs,p,net,out):
    """Finite-sample version of Proposition 3.8 with unrestricted neural increments.

    K = cp*A.T*A + F.T*(I-E*E^+)*F, cp=cs=1/2.
    Also evaluate the ACTUAL objective with omega*||E_ic w||^2.
    SVD truncation levels are all reported. No ridge is silently inserted.
    More neural parameters than constraints can make the interaction gain zero.
    """
    out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(c.seed+42198)
    Xp=obs['X_train_0']
    ids=rng.choice(len(obs['X_train_1']),min(c.interaction_data_points,len(obs['X_train_1'])),replace=False)
    Xs=obs['X_train_1'][ids];Z=uniform_points(rng,c.interaction_points)
    Xi,_=ic_points(rng,c.interaction_ic_points)
    jac,_,_=observation_jacobians(solver,p,[Xp,Z],c.sensitivity_fd_step)
    A=jac[0]/np.sqrt(len(Xp));C=jac[1]/np.sqrt(len(Z))
    B=neural_parameter_jacobian(net,Xs)/np.sqrt(len(Xs))
    D=neural_parameter_jacobian(net,Z)/np.sqrt(len(Z))
    Eic=neural_parameter_jacobian(net,Xi)/np.sqrt(len(Xi))
    atomic_savez(out/'linearized_maps.npz',A=A,B=B,C=C,D=D,E_ic=Eic,
        physical_points=Xp,synthetic_points=Xs,ghost_points=Z,initial_points=Xi,parameters=p)
    K0=.5*A.T@A;rows=[];matrices={}
    for include_ic in (False,True):
        for lam in (0.,.06,c.interaction_weight):
            blocks=[np.sqrt(.5)*B,np.sqrt(lam)*D]
            rhs=[np.zeros((len(B),2)),np.sqrt(lam)*C]
            if include_ic:
                blocks.append(np.sqrt(c.ic_weight)*Eic);rhs.append(np.zeros((len(Eic),2)))
            E=np.vstack(blocks);F=np.vstack(rhs)
            U,s,_=svd(E,full_matrices=False,check_finite=True,lapack_driver='gesdd')
            frozen=K0+lam*C.T@C
            for tol in c.sensitivity_svd_rtol:
                rank=int(np.sum(s>tol*s[0])) if len(s) and s[0]>0 else 0
                R=F-U[:,:rank]@(U[:,:rank].T@F)
                K=K0+R.T@R;evals,V=np.linalg.eigh(K)
                v=V[:,0];v=v*(1 if v[np.argmax(abs(v))]>=0 else -1)
                row=dict(include_initial_penalty=include_ic,lambda_value=lam,svd_rtol=tol,
                    neural_parameter_count=E.shape[1],constraint_count=E.shape[0],
                    numerical_neural_rank=rank,smallest_neural_singular=s[-1],
                    kappa_physical=np.sqrt(max(0,np.linalg.eigvalsh(K0)[0])),
                    kappa_effective=np.sqrt(max(0,evals[0])),
                    max_effective_sensitivity=np.sqrt(max(0,evals[-1])),
                    gain_trace=np.trace(K-K0),gain_norm=np.linalg.norm(K-K0),
                    weakest_lognu=v[0],weakest_logalpha=v[1],
                    physical_lower_bound_residual=np.linalg.eigvalsh(K-K0)[0],
                    frozen_upper_bound_residual=np.linalg.eigvalsh(frozen-K)[0],
                    K00=K[0,0],K01=K[0,1],K11=K[1,1])
                rows.append(row);matrices[f'K_ic{int(include_ic)}_lambda{lam}_tol{tol}']=K
    df=pd.DataFrame(rows);df.to_csv(out/'interaction_profiled_sensitivity.csv',index=False)
    atomic_savez(out/'profiled_matrices.npz',K0=K0,**matrices)
    save_json(out/'interpretation.json',dict(physical_parameters=p,synthetic_observation_subset=len(Xs),
        interaction_points=len(Z),initial_points=len(Xi),cp=.5,cs=.5,
        important='This is a finite-sample local diagnostic at one trained checkpoint, NOT a recovery guarantee. '
        'The unrestricted tangent space includes all neural parameters. If it fits all constraints, the '
        'profiled interaction gain can vanish. Rank and kappa depend on the declared SVD tolerance. '
        'The IC-free version corresponds to Proposition 3.8; the IC version adds the actual auxiliary penalty. '
        'Neither the frozen-network matrix nor a ridge-regularized matrix is substituted for the profiled one.'))
    return df
# %% Final evaluation, plots, tabular outputs and reusable trajectories

def field_error(pred,true,times):
    den=trapezoid(np.mean(np.asarray(true,dtype=float)**2,axis=(1,2)),times)
    num=trapezoid(np.mean((np.asarray(pred,dtype=float)-true)**2,axis=(1,2)),times)
    return float(np.sqrt(num/max(den,1e-30)))


def parameter_errors(p,cal,reference):
    rel=(np.asarray(p)-P_BACKGROUND)/P_BACKGROUND
    X,Y=np.meshgrid(reference.x,reference.x,indexing='ij')
    alpha=P_BACKGROUND[1]*np.exp(cal['heterogeneity_scale']*hidden_shape(X,Y))
    return dict(nu_hat=p[0],alpha_hat=p[1],e_nu=abs(rel[0]),e_alpha_background=abs(rel[1]),
        signed_nu=rel[0],signed_alpha=rel[1],e_gamma_joint=np.linalg.norm(rel),
        e_alpha_full_field=np.linalg.norm(p[1]-alpha)/np.linalg.norm(alpha))


def network_frames(model,times,x):
    X,Y=np.meshgrid(x,x,indexing='ij');xy=np.column_stack([X.ravel(),Y.ravel()]);fields=[]
    for t in times:
        pts=np.column_stack((xy,np.full(len(xy),t)))
        fields.append(predict_network(model,pts).reshape(len(x),len(x)))
    return np.asarray(fields,dtype=np.float32)


def evaluate_method(c,solver,obs,ref,cal,dataid,name,cp,out):
    start=time.perf_counter();times=np.linspace(0,T_END,c.evaluation_frames)
    true=interpolate_frames(ref,times,ref.x).astype(np.float32)
    has_nn=name in ('bihyco','xpinn_c')
    if has_nn:
        selected=cp['selection'];ph=selected['best_ph'];ns=selected['best_nn'];p=np.asarray(ph['params'])
    else:p=np.asarray(cp['params']);ph=dict(step=cp['step']);ns=None
    metrics=dict(method=name,seed=c.seed,data_id=dataid,**parameter_errors(p,cal,ref),
        physical_checkpoint=int(ph['step']),time_seconds=cp['time_seconds'],
        wall_training_seconds=cp.get('wall_seconds',cp['time_seconds']),last_iteration=cp['step'])
    physical=solver.solve(p)
    save_trajectory(out/'physical_full_inverse_trajectory.npz',physical,parameters=p,
        interpretation='Auxiliary forward solve only for XPINN-C; not its neural reconstruction')
    # XPINN physical output is deliberately excluded from the main state table.
    if name!='xpinn_c':
        states=interpolate_frames(physical,times,ref.x).astype('float32')
        save_trajectory(out/'physical_states.npz',Trajectory(states,times,ref.x),parameters=p,checkpoint=ph['step'])
        metrics.update(physical_state_error_st=field_error(states,true,times),
            physical_state_error_final=np.linalg.norm(states[-1]-true[-1])/np.linalg.norm(true[-1]),
            physical_test_mse=physical_validation(solver,p,obs,'test'))
    metrics['physical_validation_mse']=physical_validation(solver,p,obs,'val')
    if has_nn:
        factory=lambda: StateNet(c.hyco_width) if name=='bihyco' else XPINN(c.xp_width)
        net=factory().to(c.device);net.load_state_dict(ns['model_state']);net.eval()
        fields=network_frames(net,times,ref.x)
        save_trajectory(out/'neural_states.npz',Trajectory(fields,times,ref.x),
            checkpoint=ns['step'],parameters_at_neural_checkpoint=ns['params'])
        metrics.update(neural_checkpoint=int(ns['step']),neural_checkpoint_params=ns['params'],
            neural_state_error_st=field_error(fields,true,times),
            neural_state_error_final=np.linalg.norm(fields[-1]-true[-1])/np.linalg.norm(true[-1]),
            neural_test_mse=neural_validation(net,obs,'test'),neural_validation_mse=ns['score'])
        net.load_state_dict(ph['model_state'])
        paired=network_frames(net,times,ref.x)
        save_trajectory(out/'neural_paired_states.npz',Trajectory(paired,times,ref.x),checkpoint=ph['step'])
        metrics.update(neural_paired_state_error_st=field_error(paired,true,times),
                       neural_paired_test_mse=neural_validation(net,obs,'test'))
    metrics['postprocessing_seconds']=time.perf_counter()-start
    save_json(out/'metrics.json',metrics)
    pd.DataFrame([metrics]).to_csv(out/'metrics.csv',index=False)
    return metrics


def finish_figure(fig,path,c):
    """Every exported figure: 600 dpi; all axes/ticks/text 16 pt; no titles."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ax in fig.axes:
        ax.set_title('');ax.tick_params(labelsize=16)
        ax.xaxis.label.set_size(16);ax.yaxis.label.set_size(16)
        ax.xaxis.get_offset_text().set_size(16);ax.yaxis.get_offset_text().set_size(16)
        for text in ax.texts:text.set_fontsize(16)
        legend=ax.get_legend()
        if legend:
            for t in legend.get_texts():t.set_fontsize(16)
    fig.savefig(path.with_suffix('.png'),dpi=c.plot_dpi,bbox_inches='tight')
    if c.export_pdf:fig.savefig(path.with_suffix('.pdf'),dpi=c.plot_dpi,bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print('Saved figure:', path.name, flush=True)


def snapshot_figure(tr,path,c):
    times=np.linspace(0,T_END,12);a=interpolate_frames(tr,times,tr.x)
    fig,axes=plt.subplots(3,4,figsize=(12,9),layout='constrained')
    lo=min(-1.,float(a.min()));hi=max(1.,float(a.max()))
    for ax,field,t in zip(axes.flat,a,times):
        im=ax.imshow(field.T,origin='lower',extent=[0,1,0,1],interpolation='none',vmin=lo,vmax=hi)
        ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$')
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
        ax.text(.04,.96,f'$t={t:.3f}$',transform=ax.transAxes,fontsize=16,va='top',
                bbox=dict(facecolor=plt.rcParams['figure.facecolor'],alpha=.8,edgecolor='none',pad=1.))
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label='$u$')
    finish_figure(fig,path,c)
    atomic_savez(Path(path).with_suffix('.npz'),states=a.astype('float32'),times=times,x=tr.x,y=tr.x)


def figures_and_tables(exp,results):
    c=exp['config'];report=exp['report'];report.mkdir(parents=True,exist_ok=True)
    metrics=[r['metrics'] for r in results.values()]
    df=pd.DataFrame(metrics);df.to_csv(report/'summary.csv',index=False)
    pc=['method','seed','nu_hat','alpha_hat','e_nu','e_alpha_background','e_gamma_joint','e_alpha_full_field','physical_checkpoint']
    pc=[x for x in pc if x in df]
    df[pc].to_csv(report/'parameter_errors.csv',index=False)
    sc=['method','physical_state_error_st','neural_state_error_st','neural_paired_state_error_st',
        'physical_test_mse','neural_test_mse','time_seconds','physical_checkpoint','neural_checkpoint']
    df[[x for x in sc if x in df]].to_csv(report/'state_errors_and_cost.csv',index=False)
    names={'gn_dist':'GN-DIST','bihyco':'Bi-HYCO','xpinn_c':'XPINN-C'}
    # English LaTeX tables; relative errors in percent, not an invented std for one seed.
    rows=['\\begin{tabular}{lrrrr}\\toprule',
          r'Method & $100e_\nu$ & $100e_\alpha$ & $100e_\gamma$ & Time (s)\\\midrule']
    for m in metrics:
        rows.append(f"{names[m['method']]} & {100*m['e_nu']:.3f} & {100*m['e_alpha_background']:.3f} & {100*m['e_gamma_joint']:.3f} & {m['time_seconds']:.1f}\\\\")
    rows.extend([r'\bottomrule\end{tabular}',r'% Requires \usepackage{booktabs}. Parameter checkpoint: physical validation.'])
    (report/'parameters.tex').write_text('\n'.join(rows),encoding='utf-8')
    rows=[r'\begin{tabular}{lrrrr}\toprule',r'Output & $100e_u(Q)$ & $V_{\mathrm{test}}$ & Checkpoint & Time (s)\\\midrule']
    for m in metrics:
        if m['method']!='xpinn_c':
            label=names[m['method']]+('-PHY' if m['method']=='bihyco' else '')
            rows.append(f"{label} & {100*m['physical_state_error_st']:.3f} & {m['physical_test_mse']:.3e} & {m['physical_checkpoint']} & {m['time_seconds']:.1f}\\\\")
        if m['method'] in ('bihyco','xpinn_c'):
            label='Bi-HYCO-SYN' if m['method']=='bihyco' else 'XPINN-C'
            timing='---' if m['method']=='bihyco' else f"{m['time_seconds']:.1f}"
            rows.append(f"{label} & {100*m['neural_state_error_st']:.3f} & {m['neural_test_mse']:.3e} & {m['neural_checkpoint']} & {timing}\\\\")
    rows.extend([r'\bottomrule\end{tabular}',r'% Neural output selected by neural validation; parameters by physical validation.',
                 '% Bi-HYCO has one cost, not separate costs for its outputs. No XPINN auxiliary physical error is shown.'])
    (report/'state_and_cost.tex').write_text('\n'.join(rows),encoding='utf-8')
    if not c.make_plots:return df
    plt.rcParams.update({'font.size':16,'axes.labelsize':16,'xtick.labelsize':16,'ytick.labelsize':16,'legend.fontsize':16})
    figdir=report/'figures';figdir.mkdir(exist_ok=True)
    snapshot_figure(exp['reference'],figdir/'reference_12_snapshots',c)
    panel=[('Reference',exp['reference'])]
    for name,r in results.items():
        out=r['output']
        if (out/'physical_states.npz').exists() and name!='xpinn_c':
            tr=load_trajectory(out/'physical_states.npz');panel.append((names[name]+' physical',tr))
            snapshot_figure(tr,figdir/f'{name}_physical_12_snapshots',c)
        if (out/'neural_states.npz').exists():
            tr=load_trajectory(out/'neural_states.npz');panel.append((names[name]+' neural',tr))
            snapshot_figure(tr,figdir/f'{name}_neural_12_snapshots',c)
        hp=out/'training_history.csv'
        if hp.exists():
            h=pd.read_csv(hp);x='round' if 'round' in h else 'epoch'
            fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
            ax.semilogy(h[x],h.val_physical_mse,label='Physical validation')
            ax.semilogy(h[x],h.val_neural_mse,label='Neural validation')
            ax.set_xlabel('Round' if x=='round' else 'Epoch');ax.set_ylabel('Mean-square error');ax.legend()
            finish_figure(fig,figdir/f'{name}_validation',c)
            for par,target in zip(PARAM_NAMES,P_BACKGROUND):
                fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
                ax.plot(h[x],h[par+'_hat'],label='Estimate');ax.axhline(target,linestyle='--',label='Prescribed background')
                ax.set_xlabel('Round' if x=='round' else 'Epoch');ax.set_ylabel('$\\nu$' if par=='nu' else '$\\alpha$');ax.legend()
                finish_figure(fig,figdir/f'{name}_{par}_trajectory',c)
    t=T_END/2;ref=interpolate_frames(exp['reference'],[t],exp['reference'].x)[0]
    fields=[interpolate_frames(tr,[t],exp['reference'].x)[0] for _,tr in panel]
    lo=min(-1.,min(float(a.min()) for a in fields));hi=max(1.,max(float(a.max()) for a in fields))
    errors=[abs(a-ref) for a in fields];mx=max(float(a.max()) for a in errors)
    fig,axes=plt.subplots(2,len(panel),figsize=(3.5*len(panel),6.5),layout='constrained',squeeze=False)
    for k,field in enumerate(fields):
        im=axes[0,k].imshow(field.T,origin='lower',extent=[0,1,0,1],interpolation='none',vmin=lo,vmax=hi)
        er=axes[1,k].imshow(errors[k].T,origin='lower',extent=[0,1,0,1],interpolation='none',vmin=0,vmax=mx)
        for ax in axes[:,k]:ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    fig.colorbar(im,ax=axes[0].tolist(),shrink=.75,label='$u$')
    fig.colorbar(er,ax=axes[1].tolist(),shrink=.75,label='$|u-u^\\dagger|$')
    finish_figure(fig,figdir/'state_comparison_and_absolute_errors',c)
    (figdir/'state_comparison_panel_order.txt').write_text(
        f'Time = {t}. Columns: '+ ' | '.join(n for n,_ in panel)+'\nTop: reconstructed state. Bottom: absolute error.\n'
        'Physical and neural fields can come from different validation checkpoints; see summary.csv.\n',encoding='utf-8')
    return df


def plot_sensitivity(c,sensdir):
    path=sensdir/'background'/'observation_jacobians.npz'
    if not c.make_plots or not path.exists():return
    with np.load(path) as a:
        matrices={k:a[k] for k in ('unit1_rms','unit2_rms','balanced_rms')}
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    ang=np.linspace(0,2*np.pi,361);v=np.stack([np.cos(ang),np.sin(ang)])
    for key,A in matrices.items():
        response=np.linalg.norm(A@v,axis=0)
        ax.plot(ang,response,label={'unit1_rms':'Unit 1','unit2_rms':'Unit 2','balanced_rms':'Combined (balanced)'}[key])
    ax.set_xlabel('$\\varphi$ (rad)');ax.set_ylabel('$\\|A v(\\varphi)\\|_2$');ax.legend()
    finish_figure(fig,sensdir/'directional_sensitivity',c)
    df=pd.read_csv(sensdir/'background'/'sensitivity_table.csv')
    df=df[df.observation_map.str.endswith('_rms')]
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for _,r in df.iterrows():ax.semilogy([1,2],[r.sigma_max,r.sigma_min],marker='o',label=r.observation_map.replace('_rms',''))
    ax.set_xticks([1,2]);ax.set_xlabel('Singular-value index');ax.set_ylabel('Singular value');ax.legend()
    finish_figure(fig,sensdir/'singular_values',c)


def prepare_experiment(config=None):
    c=(config or Config()).effective()
    if c.device not in ('cpu','cuda'):raise ValueError('Use cpu or cuda; this solver needs float64 and FFT autodiff.')
    if c.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(c.threads)
    root=Path(c.root).expanduser().resolve()/c.mode;root.mkdir(parents=True,exist_ok=True)
    obs,ref,hom,cal,dataid,datadir=prepare_data(c,root)
    solver=ForwardSolver(c.inverse_n,c.inverse_dt)
    report=root/'reports'/f'seed{c.seed}_{dataid}'
    # Training hyperparameters are recorded for each method separately; report is a view.
    report.mkdir(parents=True,exist_ok=True)
    save_json(report/'experiment_configuration.json',dict(config=asdict(c),version=VERSION,
        calibration=cal,data_id=dataid,observations=str(datadir),
        environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform())))
    exp=dict(config=c,root=root,obs=obs,reference=ref,homogeneous=hom,calibration=cal,
             data_id=dataid,data_folder=datadir,solver=solver,report=report,results={})
    if c.refinement_check:
        check=datadir.parent.parent/'reference_refinement.json'
        if not check.exists():
            fine=ForwardSolver(c.reference_n+64,c.reference_dt/2,c.reference_stride*2)
            finer=fine.solve(P_BACKGROUND,cal['heterogeneity_scale'])
            aligned=interpolate_frames(finer,ref.times,ref.x)
            err=field_error(ref.states,aligned,ref.times)
            save_json(check,dict(reference_relative_difference=err,refined_n=c.reference_n+64,
                refined_dt=c.reference_dt/2,warning='One refinement comparison, not a proof of grid convergence.'))
    return exp


def run_method(exp,method):
    c=exp['config'];solver=exp['solver'];obs=exp['obs']
    if method not in ('gn_dist','bihyco','xpinn_c'):raise ValueError(method)
    out=output_folder(exp['root'],exp['data_id'],c,method)
    train={'gn_dist':train_gn,'bihyco':train_bihyco,'xpinn_c':train_xpinn}[method]
    cp=train(c,solver,obs,out)
    metrics=evaluate_method(c,solver,obs,exp['reference'],exp['calibration'],exp['data_id'],method,cp,out)
    result=dict(metrics=metrics,output=out,checkpoint=cp);exp['results'][method]=result
    print(pd.DataFrame([metrics])[[x for x in ('method','e_nu','e_alpha_background','e_gamma_joint',
        'physical_state_error_st','neural_state_error_st','time_seconds') if x in metrics]].to_string(index=False))
    return result


def run_sensitivity(exp,include_fitted=True):
    c=exp['config'];start=time.perf_counter();out=exp['report']/'sensitivity';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    points=[('background',P_BACKGROUND),('initial',P_INITIAL)]
    if include_fitted:
        points += [(k,np.array([r['metrics']['nu_hat'],r['metrics']['alpha_hat']])) for k,r in exp['results'].items()]
    for label,p in points:
        print('Observation sensitivity at:',label,flush=True)
        rows.append(sensitivity_analysis(c,exp['solver'],exp['obs'],exp['reference'],exp['homogeneous'],p,label,out))
    table=pd.concat(rows,ignore_index=True);table.to_csv(out/'all_linearization_points.csv',index=False)
    if c.run_interaction_sensitivity and 'bihyco' in exp['results'] and include_fitted:
        cp=exp['results']['bihyco']['checkpoint']['selection']['best_ph']
        net=StateNet(c.hyco_width);net.load_state_dict(cp['model_state'])
        print('Profiling neural increments for the interaction diagnostic...',flush=True)
        interaction_sensitivity(c,exp['solver'],exp['obs'],np.asarray(cp['params']),net,out/'profiled_interaction')
    plot_sensitivity(c,out)
    save_json(out/'timing.json',dict(sensitivity_seconds=time.perf_counter()-start,
        excluded_from_method_training_times=True))
    return table


def run_local_physical_diagnostic(exp):
    """Control for a simple regional weighting explanation; NOT used to tune Bi-HYCO."""
    if not exp['config'].run_local_gn_diagnostic:return None
    out=exp['report']/'supplementary_gn_unit1';out.mkdir(parents=True,exist_ok=True)
    cp=train_gn(exp['config'],exp['solver'],exp['obs'],out,weights=(1.,0.),name='gn_unit1_diagnostic')
    result=dict(**parameter_errors(cp['params'],exp['calibration'],exp['reference']),
        validation_balanced_mse=physical_validation(exp['solver'],cp['params'],exp['obs']),
        time_seconds=cp['time_seconds'],warning='Local-data control, not GN-DIST with both holders.')
    save_json(out/'metrics.json',result);return result


def finalize_experiment(exp,analysis=True):
    if not exp['results']:raise RuntimeError('No completed methods; run at least one method first.')
    if analysis:
        run_local_physical_diagnostic(exp)
        run_sensitivity(exp)
    df=figures_and_tables(exp,exp['results'])
    save_json(exp['report']/'manifest.json',dict(mode=exp['config'].mode,seed=exp['config'].seed,
        completed=list(exp['results']),outputs={k:str(v['output']) for k,v in exp['results'].items()},
        table_and_plot_path=exp['report'],noiseless=True,unknowns=PARAM_NAMES,
        only_neural_state_reported_for_xpinn=True,
        note='One seed, no standard deviations. Separate parameter and neural checkpoints are explicit. '
        'These data are not the earlier six-unknown experiment.'))
    print('Report:',exp['report'],flush=True)
    return df


def run_experiment(config=None,methods=('gn_dist','bihyco','xpinn_c'),analysis=True):
    exp=prepare_experiment(config)
    for method in methods:run_method(exp,method)
    exp['summary']=finalize_experiment(exp,analysis=analysis)
    return exp


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('smoke','pilot'),default='smoke')
    ap.add_argument('--root',default='AllenCahn_TwoParameter_Results')
    ap.add_argument('--methods',default='gn_dist,bihyco,xpinn_c')
    a=ap.parse_args()
    run_experiment(Config(mode=a.mode,root=a.root),methods=tuple(a.methods.split(',')))
