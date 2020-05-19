# %% 
%matplotlib inline
import matplotlib.pyplot as plt
import scipy as sp
from copy import copy
import neo
import matplotlib as mpl
import quantities as pq
import elephant as ele
from tqdm import tqdm
mpl.rcParams['figure.dpi'] = 331
import pandas as pd
import seaborn as sns
from data_generator import *
from plotting_helpers import *

# %% generating fake data for ETA analysis
dt = 0.02
kvec = sp.arange(-2,2,dt) * pq.s

t_stop = 5 * 60 * pq.s
nEvents = 4
rates = sp.ones(nEvents)*0.33*pq.Hz
Events = generate_events(nEvents, rates, t_stop) 

nKernels = 4
Kernels = generate_kernels(nKernels,kvec)

Weights = sp.ones((nEvents,1))
asig, = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

# %% plot kernels
fig, axes = plot_kernels(Kernels)
fig.suptitle('Kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)

# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
axes[0].set_ylabel('Events')
axes[1].plot(asig.times,asig)
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_xlim(0,30)
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# %% event triggered average
fig, axes = plt.subplots(ncols=nEvents,figsize=[6,2.5])
t_slice = (-2,2) * pq.s
for i, event in enumerate(Events):
    asig_slices = []
    for t in event.times:
        try:
            asig_slice = asig.time_slice(t+t_slice[0],t+t_slice[1])
            tvec = asig_slice.times - t
            axes[i].plot(tvec, asig_slice, 'k', alpha=0.25,lw=1)
            asig_slices.append(asig_slice)
        except ValueError:
            pass
    
    # average:
    avg = sp.stack([asig_slice.magnitude for asig_slice in asig_slices],axis=1).mean(axis=1)
    axes[i].plot(tvec,avg,'r')
    axes[i].plot(Kernels.times,Kernels[:,i],'C%i'%i)
    axes[i].set_xlabel('time (s)')

axes[0].set_ylabel('signal (au)')

fig.suptitle('event triggered average')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])



"""
offset from zero but probably fine when zscored
"""

"""
 
 ######## ########    ###       ########  ########   #######  ########  ##       ######## ##     ##  ######  
 ##          ##      ## ##      ##     ## ##     ## ##     ## ##     ## ##       ##       ###   ### ##    ## 
 ##          ##     ##   ##     ##     ## ##     ## ##     ## ##     ## ##       ##       #### #### ##       
 ######      ##    ##     ##    ########  ########  ##     ## ########  ##       ######   ## ### ##  ######  
 ##          ##    #########    ##        ##   ##   ##     ## ##     ## ##       ##       ##     ##       ## 
 ##          ##    ##     ##    ##        ##    ##  ##     ## ##     ## ##       ##       ##     ## ##    ## 
 ########    ##    ##     ##    ##        ##     ##  #######  ########  ######## ######## ##     ##  ######  
 
"""
# %%
"""
introduce causal structure: one event leads to another but not vice versa
(rewards lead to licks but not licks to rewards)

make new event by merging two and time shifting it 
"""
dt = 0.02
kvec = sp.arange(-2,2,dt) * pq.s

t_stop = 10 * 60 * pq.s

nEvents = 4
rates = sp.ones(nEvents)*0.33*pq.Hz
Events = generate_events(nEvents, rates, t_stop)

# last event contains 
Events[-1] = Events[-1].merge(Events[-2].time_shift(1*pq.s)).time_slice(0,t_stop)

nKernels = nEvents
Kernels = generate_kernels(nKernels,kvec,spread=0)

Weights = sp.ones((nEvents,1))
asig, = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

# %% plot kernels
fig, axes = plot_kernels(Kernels)
fig.suptitle('Kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)

# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
axes[0].set_ylabel('Events')
axes[1].plot(asig.times,asig)
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

axes[1].set_xlim(0,30)

# add the thin line highlighting the causal connection between the events
for t in Events[-2].times:
    t = t.magnitude
    axes[0].plot([t,t+1],[nEvents-1,nEvents],':',color='k',zorder=-1,alpha=0.5,lw=1)

# %% event triggered average
fig, axes = plt.subplots(ncols=nEvents,figsize=[6,2.5])
t_slice = (-2,2) * pq.s
for i, event in enumerate(Events):
    asig_slices = []
    for t in event.times:
        try:
            asig_slice = asig.time_slice(t+t_slice[0],t+t_slice[1])
            tvec = asig_slice.times - t
            axes[i].plot(tvec, asig_slice, 'k', alpha=0.25,lw=1)
            asig_slices.append(asig_slice)
        except ValueError:
            pass
    
    # average:
    avg = sp.stack([asig_slice.magnitude for asig_slice in asig_slices],axis=1).mean(axis=1)
    axes[i].plot(tvec,avg,'r')
    axes[i].plot(Kernels.times,Kernels[:,i],'C%i'%i)
    axes[i].set_xlabel('time (s)')

axes[0].set_ylabel('signal (au)')

fig.suptitle('event triggered average')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])