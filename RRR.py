# %% 
%matplotlib qt5
import scipy as sp
import neo
import matplotlib.pyplot as plt
import matplotlib as mpl
import quantities as pq
import elephant as ele
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 331
import pandas as pd
import seaborn as sns
from data_generator import *
from copy import copy
from plotting_helpers import *

# %%
dt = 0.02
kvec = sp.arange(-2,2,dt) * pq.s
t_stop = 5*60 * pq.s

nEvents = 4
rates = sp.ones(nEvents)*0.33*pq.Hz
Events = generate_events(nEvents, rates, t_stop)

# last event contains 
Events[-1] = Events[-1].merge(Events[-2].time_shift(1*pq.s)).time_slice(0,t_stop)

nKernels = nEvents
Kernels = generate_kernels(nKernels,kvec)

nUnits = 50
q = 10
Weights = sp.rand(nEvents,nUnits)**q - sp.rand(nEvents,nUnits)**q * 0.5 # power for sparseness

Asigs = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

Seg = neo.core.Segment()
[Seg.analogsignals.append(asig) for asig in Asigs]
[Seg.events.append(event) for event in Events]

# %% plot kernels
fig, axes = plot_kernels(Kernels)
fig.suptitle('Kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)

# %% plot the weights
fig, axes = plt.subplots(figsize=[6,4])
im = axes.matshow(Weights,cmap='PiYG',vmin=-1,vmax=1)
fig.colorbar(mappable=im,shrink=0.5,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from kernels')
axes.set_title('Weigths')
axes.xaxis.set_ticks_position('bottom')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
ysep = 3
N = 10
axes[0].set_ylabel('Events')
for i, asig in enumerate(Asigs[:N]):
    axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=1)
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_yticks(sp.arange(N)*ysep)
axes[1].set_yticklabels([])
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[1].set_xlim(0,30)

"""
 
 ##     ## ##     ## ##       ######## ####    ##       ##     ## 
 ###   ### ##     ## ##          ##     ##     ##       ###   ### 
 #### #### ##     ## ##          ##     ##     ##       #### #### 
 ## ### ## ##     ## ##          ##     ##     ##       ## ### ## 
 ##     ## ##     ## ##          ##     ##     ##       ##     ## 
 ##     ## ##     ## ##          ##     ##     ##       ##     ## 
 ##     ##  #######  ########    ##    ####    ######## ##     ## 
 
"""
# %% multi LM
from scipy import linalg
Y, X, lags = unpack_segment(Seg, num_lags=200, intercept=True)

# pred: ridge regression
lam = 0
I = sp.diag(sp.ones(X.shape[1]))
B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
Y_hat = X @ B_hat

# model error
e = Y_hat - Y
Rss = sp.trace(e.T @ e)
print("Model Rss:", Rss)

# %% plotting Y and Y_hat
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
ysep = 3
N = 10
axes[0].set_ylabel('Events')
for i, asig in enumerate(Asigs[:N]):
    axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=2)
    axes[1].plot(asig.times,Y_hat[:,i]+i*ysep,'C3',lw=1)

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_yticks(sp.arange(N)*ysep)
axes[1].set_yticklabels([])
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[1].set_xlim(0,30)

# %% plotting inferred kernels
"""
We can't because we don't know the weights
if we do: works
"""
kvec = Kernels.times
fig, axes = plt.subplots(figsize=[6,6], nrows=N,ncols=nEvents,sharex=True,sharey=True)
for i in range(N):
    Kernels_pred = copy(B_hat[1:,i].reshape(nKernels,-1).T)
    for j in range(nEvents):
        axes[i,j].plot(kvec,Kernels[:,j] * Weights[j,i],'k',lw=2,alpha=0.75)
        axes[i,j].plot(kvec,Kernels_pred[:,j] ,'C3',lw=1)

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')
fig.suptitle('est. kernels for each unit')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.1,wspace=0.1)

# %%
"""
 
 ########  ########  ########  
 ##     ## ##     ## ##     ## 
 ##     ## ##     ## ##     ## 
 ########  ########  ########  
 ##   ##   ##   ##   ##   ##   
 ##    ##  ##    ##  ##    ##  
 ##     ## ##     ## ##     ## 
 
"""

# decomposing and low rank approximation of B_hat
U, s, Vh = sp.linalg.svd(B_hat)

S = sp.zeros((U.shape[0],s.shape[0]))
for i in range(s.shape[0]):
    S[i,i] = s[i]

# rank is the number of events
r = nEvents
B_hat_lr = U[:,:r] @ S[:r,:r] @ Vh[:r,:]

# select which variant to look at
L = U[:,:r] @ S[:r,:r] 
W = Vh[:r,:]

# L = U[:,:r]
# W = S[:r,:r] @ Vh[:r,:]

# Y_hat_lr = X @ L @ W
Y_hat_lr = X @ B_hat_lr

#first check: how is the error?

# normal linear model error
# Y_hat = X @ B_hat
e = Y_hat - Y
Rss = sp.trace(e.T @ e)
print("Model Rss:", Rss)

# normal linear model error
e = Y_hat_lr - Y
Rss = sp.trace(e.T @ e)
print("RRR Rss:", Rss)

"""
if rank is equiv to events, error is identical
"""

# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
ysep = 3
N = 10
axes[0].set_ylabel('Events')
for i, asig in enumerate(Asigs[:N]):
    axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=1)
    axes[1].plot(asig.times,Y_hat[:,i]+i*ysep,'C3',lw=1)
    axes[1].plot(asig.times,Y_hat_lr[:,i]+i*ysep,'C4',lw=1)

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_yticks(sp.arange(N)*ysep)
axes[1].set_yticklabels([])
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[1].set_xlim(0,30)

# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
axes[0].set_ylabel('Events')
N = 10
for i, asig in enumerate(Asigs[:N]):
    axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=2,alpha=0.8)
    axes[1].plot(asig.times,Y_hat[:,i]+i*ysep,'C3',lw=1)
    axes[1].plot(asig.times,Y_hat_lr[:,i]+i*ysep,'C4',lw=1)

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[1].set_xlim(0,60)

"""
 
 #### ##    ##  ######  ########  ########  ######  ######## 
  ##  ###   ## ##    ## ##     ## ##       ##    ##    ##    
  ##  ####  ## ##       ##     ## ##       ##          ##    
  ##  ## ## ##  ######  ########  ######   ##          ##    
  ##  ##  ####       ## ##        ##       ##          ##    
  ##  ##   ### ##    ## ##        ##       ##    ##    ##    
 #### ##    ##  ######  ##        ########  ######     ##    
 
"""
"""
so the Q is, what do we gain from now having L and W? 
"""
# %% inspecting L
fig, axes = plt.subplots(figsize=[6,5],nrows=r,ncols=nEvents,sharex=True,sharey=True)
for ri in range(r):
    ll = l[:,ri]
    ll = ll.reshape(nEvents,-1).T
    for i in range(nEvents):
        axes[ri,i].plot(kvec, ll[:,i],color='k')

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')
fig.suptitle('latent factors')
plt.figtext(0.05,0.5,'rank',rotation='vertical')
sns.despine(fig)
fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])

# %% inspect W
fig, axes = plt.subplots(figsize=[6,4])
im = axes.matshow(W,cmap='PiYG',vmin=-1,vmax=1)
fig.colorbar(mappable=im,shrink=0.5,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from latent factors')
axes.set_title('Weigths')
axes.xaxis.set_ticks_position('bottom')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

"""
 
  ######   ######## ########    ##    ## ######## ########  ##    ## ######## ##        ######  
 ##    ##  ##          ##       ##   ##  ##       ##     ## ###   ## ##       ##       ##    ## 
 ##        ##          ##       ##  ##   ##       ##     ## ####  ## ##       ##       ##       
 ##   #### ######      ##       #####    ######   ########  ## ## ## ######   ##        ######  
 ##    ##  ##          ##       ##  ##   ##       ##   ##   ##  #### ##       ##             ## 
 ##    ##  ##          ##       ##   ##  ##       ##    ##  ##   ### ##       ##       ##    ## 
  ######   ########    ##       ##    ## ######## ##     ## ##    ## ######## ########  ######  
 
"""
# %%
"""
What can we do with this?
Get back Kernels?
"""

l = copy(L[1:])
Kernels_pred = l.sum(1).reshape(nEvents,-1).T / W.sum(1)
"""
this line: The weighted sum of the latent kernels per unit
recovers shape but not scale and sign
"""

# invert if peak is negative
for i in range(nKernels):
    if Kernels_pred[:,i].max() < sp.absolute(Kernels_pred[:,i]).max():
        Kernels_pred[:,i] *= -1

# TODO can this be avoided?
# normalize
for i in range(nKernels):
    Kernels_pred[:,i] /= Kernels_pred[:,i].max()

# %% plot the predicted kernels
fig, axes = plot_kernels(Kernels,color='k',lw=2)
fig.suptitle('predicted kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)

for i in range(nKernels):
    axes[i].plot(Kernels.times,Kernels_pred[:,i],'C3',alpha=1)
    


# %%
"""
 
  ######   ######## ########    ##      ## ######## ####  ######   ##     ## ########  ######  
 ##    ##  ##          ##       ##  ##  ## ##        ##  ##    ##  ##     ##    ##    ##    ## 
 ##        ##          ##       ##  ##  ## ##        ##  ##        ##     ##    ##    ##       
 ##   #### ######      ##       ##  ##  ## ######    ##  ##   #### #########    ##     ######  
 ##    ##  ##          ##       ##  ##  ## ##        ##  ##    ##  ##     ##    ##          ## 
 ##    ##  ##          ##       ##  ##  ## ##        ##  ##    ##  ##     ##    ##    ##    ## 
  ######   ########    ##        ###  ###  ######## ####  ######   ##     ##    ##     ######  
 
"""
"""
getting back the weights is possible easy if 
predicted kernels are of correct scale and sign
"""

tvec = sp.arange(0,t_stop.magnitude,dt) * pq.s
dt = sp.diff(tvec)[0]

nSamples = sp.rint((t_stop/dt).magnitude).astype('int32')
signals_pred = sp.zeros((nSamples,len(Events)))

for i, event in enumerate(Events):
    inds = times2inds(tvec, event.times)
    binds = sp.zeros(tvec.shape)
    binds[inds] = 1
    kernel = Kernels_pred[:,i]
    signals_pred[:,i] = sp.convolve(binds,kernel,mode='same')

"""
Y = signals_pred @ Weights

Weights = signals_pred^-1 @ Y
"""

Weights_pred = sp.linalg.pinv(signals_pred) @ Y

# %% 
g_kwargs = dict(width_ratios=(1,1,1,0.1))
fig, axes = plt.subplots(ncols=4,gridspec_kw=g_kwargs, figsize=[6,3])
kwargs = dict(cmap='PiYG',vmin=-1,vmax=1)
im = axes[0].matshow(Weights, **kwargs)
axes[0].set_title('true weights')
im = axes[1].matshow(Weights_pred, **kwargs)
axes[1].set_title('pred. weights')
im = axes[2].matshow(Weights - Weights_pred, **kwargs)
axes[2].set_title('true - pred.')

fig.colorbar(cax=axes[3], mappable=im,shrink=0.5,label='au')
for ax in axes[:-1]:
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('to units')
    ax.set_aspect('auto')
axes[0].set_ylabel('from Kernel')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])





























# %%



"""
 
  #######  ##       ########  
 ##     ## ##       ##     ## 
 ##     ## ##       ##     ## 
 ##     ## ##       ##     ## 
 ##     ## ##       ##     ## 
 ##     ## ##       ##     ## 
  #######  ######## ########  
 
"""

# %%
%matplotlib qt5
import scipy as sp
import neo
import matplotlib.pyplot as plt
import matplotlib as mpl
import quantities as pq
import elephant as ele
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 331
import pandas as pd
import seaborn as sns
from data_generator import *
from copy import copy
from plotting_helpers import *

# %% solution: RRR
"""
 
 ########  ########  ########  
 ##     ## ##     ## ##     ## 
 ##     ## ##     ## ##     ## 
 ########  ########  ########  
 ##   ##   ##   ##   ##   ##   
 ##    ##  ##    ##  ##    ##  
 ##     ## ##     ## ##     ## 
 
what does this solve?
recovering latent kernels?

is equivalent to
or: decomposition of B_hat by SVD

"""

dt = 0.02
kvec = sp.arange(-2,2,dt) * pq.s
t_stop = 5 * 60 * pq.s

nEvents = 4
rates = sp.ones(nEvents)*0.33*pq.Hz
Events = generate_events(nEvents, rates, t_stop)

# last event contains 
Events[-1] = Events[-1].merge(Events[-2].time_shift(1*pq.s)).time_slice(0,t_stop)

nKernels = nEvents
Kernels = generate_kernels(nKernels,kvec)

nUnits = 50
q = 10
Weights = sp.rand(nEvents,nUnits)**q - sp.rand(nEvents,nUnits)**q * 0.5# power for sparseness

Asigs = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

Seg = neo.core.Segment()
[Seg.analogsignals.append(asig) for asig in Asigs]
[Seg.events.append(event) for event in Events]

# %% plot the weights
fig, axes = plt.subplots(figsize=[6,4])
im = axes.matshow(Weights,cmap='PiYG',vmin=-1,vmax=1)
fig.colorbar(mappable=im,shrink=0.5,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from kernels')
axes.set_title('Weigths')
axes.xaxis.set_ticks_position('bottom')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

"""

 ##     ## ##     ## ##       ######## ####    ##       ##     ## 
 ###   ### ##     ## ##          ##     ##     ##       ###   ### 
 #### #### ##     ## ##          ##     ##     ##       #### #### 
 ## ### ## ##     ## ##          ##     ##     ##       ## ### ## 
 ##     ## ##     ## ##          ##     ##     ##       ##     ## 
 ##     ## ##     ## ##          ##     ##     ##       ##     ## 
 ##     ##  #######  ########    ##    ####    ######## ##     ## 
 
"""
# %% multi LM
from scipy import linalg
Y, X, lags = unpack_segment(Seg, num_lags=200, intercept=True)

# pred: ridge regression
lam = 0
I = sp.diag(sp.ones(X.shape[1]))
B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
Y_hat = X @ B_hat

# model error
e = Y_hat - Y
Rss = sp.trace(e.T @ e)
print("Model Rss:", Rss)

# %% 
"""
to get back to kernels: decomposition of B_hat matrix
"""
