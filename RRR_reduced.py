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
"""
30min 1hz 50 q10 c100
11101?1?1??
30min .2hz 50 q10 c100
11?11?111??1
30min .2hz 50 q10 c10
?11
"""
# %%
dt = 0.02
kvec = sp.arange(-2,2,dt) * pq.s
t_stop = 30*60 * pq.s

nEvents = 5
rates = sp.ones(nEvents)*0.2*pq.Hz
Events = generate_events(nEvents, rates, t_stop)

# one event with multiple causes is equivalent to two simultaneous events?
# it is not, as it will cause duplicate regressors
# think about this
# duplication could also be a way to approach the problem:
# if multiple things are happening distr with diff weight, then each reg would get
# one of the variances?
Events[-2] = Events[-1]

nKernels = nEvents
Kernels = generate_kernels(nKernels, kvec, normed=True, spread=0.5)

nUnits = 100
q = 10
Weights = sp.rand(nEvents,nUnits)**q - sp.rand(nEvents,nUnits)**q * 0.5 # power for sparseness
# Weights = sp.load('W.npy')
# Weights = sp.load('Wb200.npy')

print("CC on this run: %2.2f"% sp.corrcoef(Weights[-1,:],Weights[-2,:])[0,1])

Asigs = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

Seg = neo.core.Segment()
[Seg.analogsignals.append(asig) for asig in Asigs]
[Seg.events.append(event) for event in Events[:-1]]
nEvents = nEvents -1

# %% plot kernels
# fig, axes = plot_kernels(Kernels)
# fig.suptitle('Kernels')
# fig.tight_layout()
# fig.subplots_adjust(top=0.85)

# %% plot kernels - both on last
fig, axes = plt.subplots(ncols=nKernels-1, figsize=[6,1.5])
for i in range(nKernels-1):
    axes[i].plot(Kernels.times,Kernels[:,i],color='C%i'%i)
axes[-1].plot(Kernels.times,Kernels[:,-1],color='C%i'%(nKernels-1))

for ax in axes:
    ax.axvline(0,linestyle=':',color='k',alpha=0.5)
    ax.set_xlabel('time (s)')
sns.despine(fig)
axes[0].set_ylabel('au')
fig.suptitle('Kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('plots/RRR_kernels.png')

# %% plot the weights
fig, axes = plt.subplots(figsize=[6,4])
im = axes.matshow(Weights,cmap='PiYG',vmin=-1,vmax=1)
fig.colorbar(mappable=im,shrink=0.5,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from kernels')
axes.set_title('Weigths')
axes.set_aspect('auto')
axes.xaxis.set_ticks_position('bottom')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('plots/RRR_weights.png')


# # %% plot the simulated data
# fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
# plot_events(Events,ax=axes[0])
# ysep = 4
# N = 10
# axes[0].set_ylabel('Events')
# for i, asig in enumerate(Asigs[:N]):
#     axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=1)
# axes[1].set_xlabel('time (s)')
# axes[1].set_ylabel('signal (au)')
# axes[1].set_yticks(sp.arange(N)*ysep)
# axes[1].set_yticklabels([])
# fig.suptitle('simulated data')
# sns.despine(fig)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# axes[1].set_xlim(0,30)
# fig.savefig('plots/RRR_sim_data.png')


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

# # %% plotting Y and Y_hat
# fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
# plot_events(Events,ax=axes[0])
# ysep = 3
# N = 10
# axes[0].set_ylabel('Events')
# for i, asig in enumerate(Asigs[:N]):
#     axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=2)
#     axes[1].plot(asig.times,Y_hat[:,i]+i*ysep,'C3',lw=1)

# axes[1].set_xlabel('time (s)')
# axes[1].set_ylabel('signal (au)')
# axes[1].set_yticks(sp.arange(N)*ysep)
# axes[1].set_yticklabels([])
# fig.suptitle('simulated data')
# sns.despine(fig)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# axes[1].set_xlim(0,30)

# # %% plotting inferred kernels
# N = 10
# kvec = Kernels.times
# fig, axes = plt.subplots(figsize=[6,5], nrows=N,ncols=nEvents,sharex=True,sharey=True)
# for i in range(N):
#     Kernels_pred = copy(B_hat[1:,i].reshape(nEvents,-1).T)
#     for j in range(nEvents):
#         axes[i,j].plot(kvec,Kernels[:,j] * Weights[j,i],'k',lw=2,alpha=0.75)
#         axes[i,j].plot(kvec,Kernels_pred[:,j] ,'C3',lw=1)

#     # the duplicate kernel
#     axes[i,j].plot(kvec,Kernels[:,-1] * Weights[-1,i],'gray',lw=3,alpha=0.75,zorder=-1)

# for ax in axes[-1,:]:
#     ax.set_xlabel('time (s)')
# fig.suptitle('est. kernels for each unit')
# sns.despine(fig)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.subplots_adjust(hspace=0.1,wspace=0.1)
# fig.savefig('plots/RRR_LM_pred_kernels.png')

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
r = nEvents + 1
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
# e = Y_hat - Y
# Rss = sp.trace(e.T @ e)
# print("Model Rss:", Rss)

# normal linear model error
e = Y_hat_lr - Y
Rss = sp.trace(e.T @ e)
print("RRR Rss:", Rss)

"""
if rank is equiv to events, error is identical
"""

# # %% plot the simulated data
# fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
# plot_events(Events,ax=axes[0])
# ysep = 3
# N = 10
# axes[0].set_ylabel('Events')
# for i, asig in enumerate(Asigs[:N]):
#     axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=1)
#     axes[1].plot(asig.times,Y_hat[:,i]+i*ysep,'C3',lw=1)
#     axes[1].plot(asig.times,Y_hat_lr[:,i]+i*ysep,'C4',lw=1)

# axes[1].set_xlabel('time (s)')
# axes[1].set_ylabel('signal (au)')
# axes[1].set_yticks(sp.arange(N)*ysep)
# axes[1].set_yticklabels([])
# fig.suptitle('simulated data')
# sns.despine(fig)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# axes[1].set_xlim(0,30)


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
# # %% inspecting L

# fig, axes = plt.subplots(figsize=[6,5],nrows=r,ncols=nEvents,sharex=True,sharey=True)
# for ri in range(r):
#     ll = l[:,ri]
#     ll = ll.reshape(nEvents,-1).T
#     for i in range(nEvents):
#         axes[ri,i].plot(kvec, ll[:,i],color='k')

# for ax in axes[-1,:]:
#     ax.set_xlabel('time (s)')
# fig.suptitle('latent factors')
# plt.figtext(0.05,0.5,'rank',rotation='vertical')
# sns.despine(fig)
# fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])

# # %% inspect W
# fig, axes = plt.subplots(figsize=[6,4])
# im = axes.matshow(W,cmap='PiYG',vmin=-1,vmax=1)
# fig.colorbar(mappable=im,shrink=0.5,label='au')
# axes.set_xlabel('to units')
# axes.set_ylabel('from latent factors')
# axes.set_title('Weigths')
# axes.xaxis.set_ticks_position('bottom')
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# %%
"""
 
  ######  ##       ##     ##  ######  ######## ######## ########  #### ##    ##  ######   
 ##    ## ##       ##     ## ##    ##    ##    ##       ##     ##  ##  ###   ## ##    ##  
 ##       ##       ##     ## ##          ##    ##       ##     ##  ##  ####  ## ##        
 ##       ##       ##     ##  ######     ##    ######   ########   ##  ## ## ## ##   #### 
 ##       ##       ##     ##       ##    ##    ##       ##   ##    ##  ##  #### ##    ##  
 ##    ## ##       ##     ## ##    ##    ##    ##       ##    ##   ##  ##   ### ##    ##  
  ######  ########  #######   ######     ##    ######## ##     ## #### ##    ##  ######   
 
"""

# %% clustering the weights
# and forming from them "patterns"
# -> candidate kernels
l = copy(L[1:])
nClusters = nEvents + 1 # nEvents+1
# nClusters = r

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=nClusters).fit(W.T)
# labels = kmeans.labels_

from sklearn.cluster import SpectralClustering
specclust = SpectralClustering(n_clusters=nClusters, assign_labels='kmeans').fit(W.T)
labels = specclust.labels_

# getting candidate kernels
nLags = 200
candidate_kernels = sp.zeros((nLags,nEvents,nClusters))
for j in range(nClusters):
    signal = l @ W[:,labels == j]
    true_signal = Y[:,labels == j]
    splits = sp.split(signal,nEvents,0)
    for i in range(nEvents):
        candidate_kernels[:,i,j] = sp.average(splits[i],1) # the candidate kernel

# # %% plotting the clusters
# ratios = [sp.sum(labels == i)/len(labels) for i in range(nClusters)]
# ratios.append(0.05)
# g_kwargs = dict(width_ratios=ratios)
# kwargs = dict(cmap='PiYG',vmin=-0.5,vmax=0.5)
# fig, axes = plt.subplots(figsize=[6,3],ncols=nClusters+1, gridspec_kw=g_kwargs)
# for i in range(nClusters):
#     im = axes[i].matshow(W[:,labels == i],**kwargs)
#     axes[i].set_aspect('auto')

# fig.colorbar(cax=axes[-1], mappable=im,shrink=0.5,label='au')
# for ax in axes[:-1]:
#     ax.xaxis.set_ticks_position('bottom')
#     ax.set_aspect('auto')
#     ax.set_xlabel('to units')
# axes[0].set_ylabel('from latent factor')
# for ax in axes[1:-1]:
#     ax.set_yticklabels([])
# fig.suptitle('W clusters')
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# fig.subplots_adjust(wspace=0.1)
# fig.savefig('plots/RRR_clusters.png')

# %% plotting the combinations
fig, axes = plt.subplots(nrows=nClusters,ncols=nEvents,sharey=True,sharex=True)
fig.suptitle('patterns of summations')
for i in range(nEvents):
    for j in range(nClusters):
        axes[j,i].plot(kvec, candidate_kernels[:,i,j])
        # axes[j,i].plot(sp.average(splits[i],1),lw=2)
        # axes[j,i].plot(splits[i],color='k',alpha=.5,lw=.5)

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')
sns.despine(fig)
plt.figtext(0.025,0.5,'rank',rotation='vertical')
fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.1,wspace=0.1)
fig.savefig('plots/RRR_combinations.png')

"""
 
    ###    ##       ######## ######## ########  ##    ##    ###    ######## #### ##     ## ########     ######  ########    ###    ########  ######## 
   ## ##   ##          ##    ##       ##     ## ###   ##   ## ##      ##     ##  ##     ## ##          ##    ##    ##      ## ##   ##     ##    ##    
  ##   ##  ##          ##    ##       ##     ## ####  ##  ##   ##     ##     ##  ##     ## ##          ##          ##     ##   ##  ##     ##    ##    
 ##     ## ##          ##    ######   ########  ## ## ## ##     ##    ##     ##  ##     ## ######       ######     ##    ##     ## ########     ##    
 ######### ##          ##    ##       ##   ##   ##  #### #########    ##     ##   ##   ##  ##                ##    ##    ######### ##   ##      ##    
 ##     ## ##          ##    ##       ##    ##  ##   ### ##     ##    ##     ##    ## ##   ##          ##    ##    ##    ##     ## ##    ##     ##    
 ##     ## ########    ##    ######## ##     ## ##    ## ##     ##    ##    ####    ###    ########     ######     ##    ##     ## ##     ##    ##    
 
"""
# %% 

dt = 0.02
tvec = sp.arange(0,t_stop.magnitude,dt) * pq.s
dt = sp.diff(tvec)[0]

nSamples = sp.rint((t_stop/dt).magnitude).astype('int32')
signals_pred = sp.zeros((nSamples,nEvents*nClusters))

m = 0
for i in range(nEvents):
    event = Events[i]
    inds = times2inds(tvec, event.times)
    binds = sp.zeros(tvec.shape)
    binds[inds] = 1
    for j in range(nClusters):
        kernel = candidate_kernels[:,i,j].flatten()
        signals_pred[:,m] = sp.convolve(binds,kernel,mode='same')
        m += 1


















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
generate template kernels from patterns of summations and attempt reconstructions?
how to get template kernels
test to predict data with each and check if better (than chance?)

alternative approach - try out all kernel clusters and see which one
predicts data in any way
"""

# %% normalize and peak invert candidate kernels
for i in range(nEvents):
    for j in range(nClusters):
        if candidate_kernels[:,i,j].max() < sp.absolute(candidate_kernels[:,i,j]).max():
            candidate_kernels[:,i,j] *= -1
        # candidate_kernels[:,i,j] /= candidate_kernels[:,i,j].max()

# %% get back the weight matrix

dt = 0.02
tvec = sp.arange(0,t_stop.magnitude,dt) * pq.s
dt = sp.diff(tvec)[0]

nSamples = sp.rint((t_stop/dt).magnitude).astype('int32')
signals_pred = sp.zeros((nSamples,nEvents*nClusters))

m = 0
for i in range(nEvents):
    event = Events[i]
    inds = times2inds(tvec, event.times)
    binds = sp.zeros(tvec.shape)
    binds[inds] = 1
    for j in range(nClusters):
        kernel = candidate_kernels[:,i,j].flatten()
        signals_pred[:,m] = sp.convolve(binds,kernel,mode='same')
        m += 1

chunks = 200
sig_chunks = sp.split(signals_pred,chunks)
Y_chunks = sp.split(Y,chunks)
from tqdm import tqdm
W_chunks = sp.zeros((nEvents*nClusters,nUnits,chunks))

for i in tqdm(range(chunks)):
    W_chunks[:,:,i] = sp.linalg.pinv(sig_chunks[i]) @ Y_chunks[i]

Weights_pred = sp.average(W_chunks,2)
# Weights_pred = sp.linalg.pinv(signals_pred) @ Y
plt.matshow(Weights_pred)

# %% combine top N
# sds = sp.std(Weights_pred,axis=1)
# v = sp.cumsum(sp.sort(sds)[::-1]) / sp.sum(sds)
# top_n = sp.sum(v>0.95)

# ix = []
# for i in range(nEvents):
#     for j in range(nClusters):
#         ix.append((i,j))

# ix = sp.array(ix)[sp.sort(sp.argsort(sds)[-top_n:])]

# %% select top ranking ones
sds = sp.std(Weights_pred,axis=1)
ix = sp.argsort(sds)[-r:]
ix = sp.sort(ix)

# swap
swap = False
if swap:
    ix_c = copy(ix)
    ix[-2] = ix_c[-1]
    ix[-1] = ix_c[-2]

Weights_pred_sel = Weights_pred[ix,:]

# %% select the r with the highest SD (or min - max?)
fig, axes = plt.subplots(figsize=[5,4])
axes.plot(sp.sort(sds)[::-1],'o')
axes.axvline(r-.5,linestyle=':',color='k',alpha=0.5)
axes.set_title('sorted SDs of predicted Weights')
axes.set_xlabel('row')
axes.set_ylabel('sd')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# %% 
g_kwargs = dict(width_ratios=(1,1,1,0.1))
fig, axes = plt.subplots(ncols=4,gridspec_kw=g_kwargs, figsize=[6,3])
kwargs = dict(cmap='PiYG',vmin=-1,vmax=1)
im = axes[0].matshow(Weights, **kwargs)
axes[0].set_title('true weights')
im = axes[1].matshow(Weights_pred_sel, **kwargs)
axes[1].set_title('pred. weights')
im = axes[2].matshow(Weights - Weights_pred_sel, **kwargs)
axes[2].set_title('true - pred.')

fig.colorbar(cax=axes[3], mappable=im,shrink=0.5,label='au')
for ax in axes[:-1]:
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('to units')
    ax.set_aspect('auto')
axes[0].set_ylabel('from Kernel')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('plots/RRR_weights_pred.png')

# %% get back which where the Kernels that were meaningful
ix = []
for i in range(nEvents):
    for j in range(nClusters):
        ix.append((i,j))

ix = sp.array(ix)[sp.sort(sp.argsort(sds)[-r:])]

Kernels_pred = []

if swap:
    ix_c = copy(ix)
    ix[-2] = ix_c[-1]
    ix[-1] = ix_c[-2]

for i,j in ix:
    Kernels_pred.append(candidate_kernels[:,i,j])

Kernels_pred = sp.array(Kernels_pred).T
nKernels_pred = Kernels_pred.shape[1]

fig, axes = plt.subplots(ncols=nKernels_pred,sharey=True, figsize=[6,1.5])
for i in range(nKernels_pred):
    axes[i].plot(kvec, Kernels_pred[:,i], 'r')
    axes[i].plot(kvec,Kernels[:,i],'k',lw=2)

fig.suptitle('predicted kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('plots/RRR_pred_kern.png')


# %%
