"""Partition-averaged conditional-score MR. All weights depend on exposure data.

The exact engine enumerates up to 12 candidates per block. The adaptive engine
supports larger blocks. The exposure distribution determines the prior weights.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from functools import lru_cache
import itertools
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import norm

@lru_cache(maxsize=24)
def spins(m):
    if m<1 or m>12: raise ValueError('Exact enumeration requires 1 <= block size <= 12')
    return np.array(list(itertools.product([-1.,1.],repeat=m)))

def posterior(z,R,mu=2.,sigma=1.):
    """Z=G+E, E~N(0,R), G|C~N(mu*C,sigma^2 I), iid equal sign weights.
    Returns batch posterior means, total covariance, weights and component means.
    """
    z=np.atleast_2d(np.asarray(z,float)); R=np.asarray(R,float)
    n,m=z.shape; S=spins(m); I=np.eye(m)
    K=R+sigma**2*I
    inv=np.linalg.inv(K); B=sigma**2*inv
    Q=B@R
    Q=(Q+Q.T)/2
    component_shift=mu*S@(R@inv).T
    logw=mu*z@inv@S.T-.5*mu**2*np.einsum('ki,ij,kj->k',S,inv,S)[None,:]
    weights=np.exp(logw-logsumexp(logw,axis=1)[:,None])
    means=z@B.T+weights@component_shift
    delta=component_shift[None,:,:]-(weights@component_shift)[:,None,:]
    covariance=Q[None,:,:]+np.einsum('nk,nki,nkj->nij',weights,delta,delta,optimize=True)
    cm=z[:,None,:]@B.T+component_shift[None,:,:]
    return means,covariance,weights,cm,Q

def fixed_sign_posterior(z,R,mu=2.,sigma=1.):
    z=np.atleast_2d(z); m=z.shape[1]; inv=np.linalg.inv(R+sigma**2*np.eye(m))
    B=sigma**2*inv; Q=B@R; Q=(Q+Q.T)/2
    M=z@B.T+mu*np.where(z>=0,1.,-1.)@(R@inv).T
    return M,np.broadcast_to(Q,(len(z),m,m)).copy()

def gaussian_posterior(z,R,tau2):
    z=np.atleast_2d(z); I=np.eye(z.shape[1]); B=tau2*np.linalg.inv(R+tau2*I)
    Q=B@R; Q=(Q+Q.T)/2
    return z@B.T,np.broadcast_to(Q,(len(z),*Q.shape)).copy()

def selection(z,R,threshold=2.5,clump_r2=.1):
    z=np.atleast_2d(z); selected=np.zeros(z.shape,bool)
    for n,row in enumerate(z):
        banned=np.zeros(len(row),bool)
        for j in np.argsort(-abs(row),kind='stable'):
            if abs(row[j])<threshold: break
            if banned[j]: continue
            selected[n,j]=True; banned |= R[j]**2>clump_r2; banned[j]=True
    return selected

def score_terms(z,y,M,V,R,selected,sx=None,sy=None,psi=0.):
    z=np.atleast_2d(z); y=np.atleast_2d(y); n,m=z.shape
    sx=np.ones(m) if sx is None else np.asarray(sx); sy=np.ones(m) if sy is None else np.asarray(sy)
    t=z*sx/(sy**2)*selected
    tx=t*sx; ty=t*sy
    a=np.sum(t*(y-psi*sy*(z-M)),axis=-1)
    b=np.sum(tx*M,axis=-1)
    v2=np.einsum('ni,nij,nj->n',tx,V,tx)
    v1=psi*np.einsum('ni,nij,nj->n',tx,V,ty)
    v0=psi**2*np.einsum('ni,nij,nj->n',ty,V,ty)+(1-psi**2)*np.einsum('ni,ij,nj->n',ty,R,ty)
    return np.column_stack([a,b,v0,v1,v2])

def score_p(terms,beta):
    a,b,v0,v1,v2=np.asarray(terms,dtype=float).T
    scale=np.maximum.reduce([abs(a),abs(b),np.sqrt(abs(v0)),np.sqrt(abs(v1)),np.sqrt(abs(v2))])
    scale=np.where(scale>0,scale,1.)
    a=a/scale;b=b/scale;v0=(v0/scale)/scale;v1=(v1/scale)/scale;v2=(v2/scale)/scale
    var=v0-2*beta*v1+beta**2*v2
    eps=8*np.finfo(float).eps
    vtol=eps*(abs(v0)+abs(2*beta*v1)+abs(beta**2*v2))
    if np.any(var < -vtol):raise ValueError('Negative conditional score variance')
    var=np.where(abs(var)<=vtol,0.,var)
    residual=abs(a-beta*b)
    rzero=residual<=eps*(abs(a)+abs(beta*b))
    ratio=np.divide(residual,np.sqrt(var),out=np.zeros(np.broadcast(a,beta).shape),where=var>0)
    return np.where(var==0,np.where(rzero,1.,0.),2*norm.sf(ratio))

def confidence_set(terms,alpha=.05):
    a,b,v0,v1,v2=np.asarray(terms,float)
    q=norm.ppf(1-alpha/2)**2
    # Scale the score and causal parameter separately: polynomial coefficients
    # have different physical units and must not share an absolute tolerance.
    amplitude=max(abs(a),np.sqrt(abs(v0)))
    slope=max(abs(b),np.sqrt(abs(v2)))
    if amplitude==0:
        if slope==0:return [(-np.inf,np.inf)]
        return [(-np.inf,np.inf)] if (b/slope)**2<=q*(v2/slope)/slope else [(0.,0.)]
    if slope==0:
        if v1!=0:raise ValueError('Inconsistent score covariance')
        return [(-np.inf,np.inf)] if (a/amplitude)**2<=q*(v0/amplitude)/amplitude else []
    beta_scale=amplitude/slope
    a=a/amplitude;b=b/slope;v0=(v0/amplitude)/amplitude
    v1=(v1/amplitude)/slope;v2=(v2/slope)/slope
    A=b*b-q*v2; B=-2*a*b+2*q*v1; C=a*a-q*v0
    eps=8*np.finfo(float).eps
    if abs(A)<=eps*(abs(b*b)+abs(q*v2)):A=0.
    if abs(B)<=eps*(abs(2*a*b)+abs(2*q*v1)):B=0.
    if abs(C)<=eps*(abs(a*a)+abs(q*v0)):C=0.
    if A==0:
        if B==0: return [(-np.inf,np.inf)] if C<=0 else []
        r=-C/B*beta_scale; return [(-np.inf,r)] if B>0 else [(r,np.inf)]
    disc=B*B-4*A*C
    if abs(disc)<=eps*(abs(B*B)+abs(4*A*C)):disc=0.
    if disc<0: return [] if A>0 else [(-np.inf,np.inf)]
    qr=-.5*(B+np.copysign(np.sqrt(disc),B))
    lo,hi=sorted([qr/A*beta_scale,C/qr*beta_scale]) if qr!=0 else (0.,0.)
    return [(lo,hi)] if A>0 else [(-np.inf,lo),(hi,np.inf)]

def exact_single_block_p(z,y,R,selected,mu,sigma,beta,psi=0.,sx=None,sy=None):
    z=np.asarray(z); y=np.asarray(y); sx=np.ones(len(z)) if sx is None else np.asarray(sx)
    sy=np.ones(len(z)) if sy is None else np.asarray(sy)
    M,V,w,cm,Q=posterior(z,R,mu,sigma)
    t=z*sx/(sy**2)*selected
    u=t*(beta*sx-psi*sy)
    obs=t@(y-psi*sy*z)
    mean=cm[0]@u
    var=u@Q@u+(1-psi**2)*(t*sy)@R@(t*sy)
    if var<=0: return np.nan
    F=w[0]@ndtr((obs-mean)/np.sqrt(var))
    return min(1.,2*min(F,1-F))

def sample_posterior(z,R,mu,sigma,n,rng):
    M,V,w,cm,Q=posterior(z,R,mu,sigma)
    idx=rng.choice(len(w[0]),size=n,p=w[0])
    return cm[0,idx]+rng.multivariate_normal(np.zeros(len(z)),Q,size=n)

def fourier_cdf(observed,weights,component_means,within_variance,nodes=96,tolerance=1e-10):
    """Scalar convolution of independent Gaussian mixtures by Fourier inversion.
    The truncation bound is E1(q U^2/2)/(2*pi). Node refinement checks quadrature.
    weights and component_means have shape (blocks, states); q is total variance.
    """
    from scipy.special import roots_legendre, exp1
    weights=np.asarray(weights,float); means=np.asarray(component_means,float)
    center=np.sum(weights*means)
    vbetween=np.sum(weights*(means-np.sum(weights*means,axis=1)[:,None])**2)
    scale=np.sqrt(float(within_variance)+vbetween)
    if not np.isfinite(scale) or scale<=0: raise ValueError('Positive scalar variance required')
    q=float(within_variance)/(scale*scale)
    if q<=0:raise ValueError('Positive within-component variance required for tail bound')
    # Conservative exponential tail, tighter reported bound uses E1.
    cutoff=np.sqrt(2*np.log(1/tolerance)/q)
    cm=(means-np.sum(weights*means,axis=1)[:,None])/scale
    # Bound the largest oscillation frequency, including the observed tail.
    # A global quadrature rule can alias widely separated narrow components.
    frequency=abs((observed-center)/scale)+np.max(abs(cm),axis=1).sum()
    panels=max(1,int(np.ceil(cutoff*frequency/8)))
    per_panel=max(12,min(48,nodes//8))
    if panels*per_panel>2000000:raise ArithmeticError('Fourier oscillation budget exceeded')
    x,wbase=roots_legendre(per_panel)
    width=cutoff/panels
    u=(np.arange(panels)[:,None]*width+(x+1)[None,:]*width/2).ravel()
    wq=np.tile(wbase*width/2,panels)
    phi=np.ones(len(u),dtype=complex)
    for wb,mb in zip(weights,cm):phi*=np.exp(1j*np.outer(u,mb))@wb
    phi*=np.exp(-.5*q*u*u)
    integrand=np.imag(np.exp(-1j*u*(observed-center)/scale)*phi)/u
    F=.5-np.dot(wq,integrand)/np.pi
    tail=float(exp1(q*cutoff*cutoff/2)/(2*np.pi))
    return float(F),tail

def fourier_p(observed,weights,component_means,within_variance):
    if not np.isfinite(within_variance) or within_variance<=0:
        raise ValueError('Positive finite within-component variance required')
    weights=np.asarray(weights,float);component_means=np.asarray(component_means,float)
    if weights.shape[1]**len(weights)<=4096:
        w=np.ones(1);m=np.zeros(1)
        for wb,mb in zip(weights,component_means):
            w=(w[:,None]*wb).ravel();m=(m[:,None]+mb).ravel()
        value=w@ndtr((observed-m)/np.sqrt(within_variance))
        return float(2*min(value,1-value)),0.,0.,0
    f1,tail=fourier_cdf(observed,weights,component_means,within_variance,96)
    f2,_=fourier_cdf(observed,weights,component_means,within_variance,192)
    err=abs(f2-f1);nodes=192
    if err>1e-7:
        f3,_=fourier_cdf(observed,weights,component_means,within_variance,384)
        err=abs(f3-f2);f2=f3;nodes=384
    if err>1e-6 or not -1e-8<=f2<=1+1e-8:raise ArithmeticError('Fourier integration did not converge')
    f2=np.clip(f2,0.,1.)
    return float(2*min(f2,1-f2)),float(err),tail,nodes
