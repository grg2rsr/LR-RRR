# %%
%matplotlib qt5

from copy import copy
import numpy as np
import scipy as sp
from scipy import linalg
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 166

import quantities as pq
import neo
import elephant as ele
import seaborn as sns
from copy import copy

from pathlib import Path

from plotting_helpers import *
from data_generator import *
from RRRlib import *
"""
 
 ########  ########    ###    ########  
 ##     ## ##         ## ##   ##     ## 
 ##     ## ##        ##   ##  ##     ## 
 ########  ######   ##     ## ##     ## 
 ##   ##   ##       ######### ##     ## 
 ##    ##  ##       ##     ## ##     ## 
 ##     ## ######## ##     ## ########  
 
"""

data_dir = Path("/home/georg/data/rrr")
timestamps = pd.read_excel(data_dir / 'allo_egsession_timestamps.xlsx')
flags = pd.read_excel(data_dir / 'allo_egsession_flags.xlsx')
fnames = sp.sort(os.listdir(data_dir / "neurons"))

# %% get spiketrains
Sts = []
for fname in fnames:
    data = sp.loadtxt(data_dir / "neurons" / fname)
    data = data[~sp.isnan(data)]
    data = sp.sort(data)
    st = neo.core.SpikeTrain(data*pq.ms, t_stop = 5*pq.h)
    Sts.append(st)

# %% get events
colnames = timestamps.columns
Events = []
for name in colnames:
    t = timestamps[name].values
    t = t[~sp.isnan(t)]
    Event = neo.core.Event(t * pq.ms)
    Event.annotate(label=name)
    Events.append(Event)

# %% getting last timepoint of both
t_last_spike = np.max([st.times[-1] for st in Sts])
t_last_event = np.max([ev.times[-1] for ev in Events])
t_last = np.max([t_last_spike, t_last_event]) * pq.ms

for st in Sts:
    st.t_stop = t_last

# %% firing rate estimation
asigs = []
kernel = ele.kernels.GaussianKernel(sigma=100*pq.ms)

for i,st in enumerate(tqdm(Sts)):
    asig = ele.statistics.instantaneous_rate(st, 20*pq.ms, kernel=kernel)
    asig.annotate(id=i)
    asigs.append(asig)

# %% pack
Seg = neo.core.Segment()
Seg.analogsignals = asigs
Seg.spiketrains = Sts
Seg.events = Events

# subset to valid range
Seg = Seg.time_slice(4e5*pq.ms, 4e6*pq.ms)

# %%
import pickle
with open(data_dir / 'data.neo', 'wb') as fH:
    print("writing neo segment ... ")
    pickle.dump(Seg,fH)
    print("...done")

# %% inspect
Events = Seg.events
Asigs = Seg.analogsignals

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
ysep = 6
# N = 10 # up to N asig
N = 31

axes[0].set_ylabel('Events')
for i, asig in enumerate(Asigs[:N]):
    axes[1].plot(asig.times.rescale('ms').magnitude,asig.magnitude+i*ysep,'k',lw=1)

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_yticks(np.arange(N)*ysep)
axes[1].set_yticklabels([])
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# axes[1].set_xlim(0,60)

# %%
"""
 
 ##     ## ##     ##    ###    ##          ##          ###    ##     ## ########  ########     ###    
  ##   ##  ##     ##   ## ##   ##          ##         ## ##   ###   ### ##     ## ##     ##   ## ##   
   ## ##   ##     ##  ##   ##  ##          ##        ##   ##  #### #### ##     ## ##     ##  ##   ##  
    ###    ##     ## ##     ## ##          ##       ##     ## ## ### ## ########  ##     ## ##     ## 
   ## ##    ##   ##  ######### ##          ##       ######### ##     ## ##     ## ##     ## ######### 
  ##   ##    ## ##   ##     ## ##          ##       ##     ## ##     ## ##     ## ##     ## ##     ## 
 ##     ##    ###    ##     ## ########    ######## ##     ## ##     ## ########  ########  ##     ## 
 
"""

# %% unpack data
nLags = 300
Y, X, t_lags = unpack_segment(Seg, num_lags=nLags, intercept=True)

# %% xval lambda
n = 1000
k = 5
lam = xval_ridge_reg_lambda(Y[:n,:], X[:n,:], k)

# %% cheat
lam = 35

# %% LM model run
B_hat = LM(Y, X, lam=lam)
Y_hat = X @ B_hat

print("LM model error:")
print("LM: %5.3f " % Rss(Y,Y_hat))

# %%
fig, axes = plt.subplots()
ds = 1
ysep = 200
for i in range(Y.shape[1]):
    axes.plot(Y[::ds,i]+i*ysep,c='k')
    axes.plot(Y_hat[::ds,i]+i*ysep,c='r')


# %% plot data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
ysep = 10
N = 10
axes[0].set_ylabel('Events')
# tvec = asig.times.rescale('ms').magnitude
dt = sp.diff(asig.times)[0].rescale('ms')
t0 = asig.times[0].rescale('ms')
tvec = (sp.arange(X.shape[0]) * dt.magnitude + t0.magnitude) * pq.ms

for i, asig in enumerate(Asigs[:N]):
    # axes[1].plot(asig.times.rescale('ms'), asig.magnitude+i*ysep,'k',lw=1)
    axes[1].plot(tvec, Y_hat[:,i]+i*ysep,'C3',lw=2)
    # axes[1].plot(asig.times,Y_hat_lr[:,i]+i*ysep,'C4',lw=1)

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_yticks(np.arange(N)*ysep)
axes[1].set_yticklabels([])
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[1].set_xlim(0,60)

# %% plot kernels
N = 3 # for the first 10 units
kvec = t_lags
nEvents = len(Seg.events)
fig, axes = plt.subplots(figsize=[6,5], nrows=N,ncols=nEvents,sharex=True,sharey=True)
for i in range(N):
    Kernels_pred = copy(B_hat[1:,i].reshape(nEvents,-1).T)
    for j in range(nEvents):
        # axes[i,j].plot(kvec,Kernels[:,j] * Weights[j,i],'k',lw=2,alpha=0.75)
        axes[i,j].plot(kvec,Kernels_pred[:,j] ,'C3',lw=1)
        axes[i,j].axhline(0,linestyle=':',alpha=0.5,lw=0.5)
        axes[i,j].axvline(0,linestyle=':',alpha=0.5,lw=0.5)

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')

for i, ax in enumerate(axes[0,:]):
    ax.set_title(Seg.events[i].annotations['label'])

# sns.despine(fig)
fig.suptitle('est. kernels for each unit')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.1,wspace=0.1)


# %% xval rank

# idea: at which rank does Rss get worset than 1 SD
# select the last rank where model error does not increase by more than 1 SD

# getting the error distribution (Rss)
k = 5
ranks = list(range(17,25))
Rsss_lm, Rsss_rrr = xval_rank(Y, X, lam, ranks, k)

# rank select
ix = np.argmax(np.average(Rsss_rrr, axis=0) < np.average(Rsss_lm) + np.std(Rsss_lm))
r = ranks[ix]

# %% or cheat
r = 22

# %% inspect ranks - this will be an important inspection step
fig, axes = plt.subplots(figsize=[3,3])
axes.plot(ranks ,np.average(Rsss_rrr, axis=0), '.', color='k')
for i,r in enumerate(ranks):
    upper = sp.average(Rsss_rrr[:,i]) + sp.std(Rsss_rrr[:,i])
    lower = sp.average(Rsss_rrr[:,i]) - sp.std(Rsss_rrr[:,i])
    axes.plot([r,r],[lower, upper], alpha=0.5,color='k')

axes.axhline(np.average(Rsss_lm),linestyle='--',color='k')
axes.axhline(np.average(Rsss_lm) + np.std(Rsss_lm),linestyle=':',color='k')
# axes.axhline(np.average(Rsss_lm) - np.std(Rsss_lm),linestyle=':',color='k')
axes.set_xticks(ranks)
axes.set_xlabel('rank')
axes.set_ylabel('model error Rss')
axes.set_title('rank estimation')
fig.tight_layout()
# %%
r = 10
# RRR
B_hat_lr = RRR(Y, X, B_hat, r)
Y_hat_lr = X @ B_hat_lr

L, W = low_rank_approx(B_hat_lr,r)
l = copy(L[1:]) # wo intercept

# print error comparison
print("RRR model error:")
print("RRR: %5.3f" % Rss(Y,Y_hat_lr))
# %%
fig, axes = plt.subplots()
ds = 1
ysep = 200
for i in range(Y.shape[1]):
    axes.plot(Y[::ds,i]+i*ysep,c='k')
    axes.plot(Y_hat[::ds,i]+i*ysep,c='C3',lw=2)
    axes.plot(Y_hat_lr[::ds,i]+i*ysep,c='C4')

# %%
