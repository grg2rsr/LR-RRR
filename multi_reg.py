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

"""
but we don't
"""


# %%
