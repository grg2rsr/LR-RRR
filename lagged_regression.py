
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

# %%
"""
introduce causal structure: one event leads to another but not vice versa
(rewards lead to licks but not licks to rewards)

make new event by merging two and time shifting it 
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
Kernels = generate_kernels(nKernels,kvec,spread=0)

Weights = sp.ones((nEvents,1))
asig, = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

Seg = neo.core.Segment()
Seg.analogsignals.append(asig)
[Seg.events.append(event) for event in Events]

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


# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
axes[0].set_ylabel('Events')
axes[1].plot(asig.times,asig, 'k', lw=2, alpha=0.95, label='data')
axes[1].plot(asig.times,Y_hat.flatten(), 'C3', lw=1, label='fit')
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].legend()
fig.suptitle('LR linear model fit')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

axes[1].set_xlim(0,30)


"""
observe: signal is very well approximated
"""
# %% plot the kernels
Kernels_pred = B_hat[1:].reshape(nKernels,-1).T

fig, axes = plot_kernels(Kernels,color='k',lw=2)
fig.suptitle('Kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)

for i in range(nKernels):
    axes[i].plot(Kernels.times,Kernels_pred[:,i],'C3',alpha=1)

fig.savefig('plots/lr_eta_recovered.png')
"""
observe: kernels are very well recovered
so far so good!
"""

"""
multi unit case:
get a set of kernels for each unit
a lot of kernels
is there structure?
does each event have a base kernels that is somehow distributed?
1 weight matrix, can be retrieved (w some assumptions)
are these assumptions not necessary w rrr?
"""



# %% plot X
fig, axes = plt.subplots(figsize=[4,6])
im = axes.matshow(X[:1500,:],cmap='gray_r',extent=(0,X.shape[1],0,30))
axes.set_aspect('auto')
fig.suptitle('regressor matrix X')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('plots/LR-X.png',dpi=300)


# %% plot B
fig, axes = plt.subplots(figsize=[4,6])
im = axes.matshow(B_hat,cmap='gray_r',extent=(0,X.shape[1],0,30))
axes.set_aspect('auto')
fig.suptitle('regressor matrix X')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
# fig.savefig('plots/LR-X.png',dpi=300)


# %%
