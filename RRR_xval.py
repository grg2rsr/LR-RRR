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
mpl.rcParams['figure.dpi'] = 331

import quantities as pq
import neo
import elephant as ele
import seaborn as sns

from plotting_helpers import *
from data_generator import *

# %%
dt = 0.02
kvec = np.arange(-2,2,dt) * pq.s
t_stop = 30*60 * pq.s

nEvents = 5
rates = np.ones(nEvents)*0.2*pq.Hz
Events = generate_events(nEvents, rates, t_stop)

# merge events
Events[0] = Events[0].merge(Events[1].time_shift(1*pq.s)).time_slice(0,t_stop)

# one event with multiple causes is equivalent to two simultaneous events?
# it is not, as it will cause duplicate regressors
# think about this
# duplication could also be a way to approach the problem:
# if multiple things are happening distr with diff weight, then each reg would get
# one of the variances?
Events[-2] = Events[-1]

nKernels = nEvents
Kernels = generate_kernels(nKernels, kvec, normed=True, spread=0.5)

nUnits = 50
q = 3
Weights = sp.rand(nEvents,nUnits)**q - sp.rand(nEvents,nUnits)**q * 0.5 # power for sparseness

# inject cov of level c in half of the units
c = 0.5
T = np.array([[1,c],[c,1]])
Weights[-2:,:int(nUnits/2)] = (Weights[-2:,:int(nUnits/2)].T @ T).T

print("CC on this run: %2.2f"% np.corrcoef(Weights[-1,:],Weights[-2,:])[0,1])

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
# axes[1].set_yticks(np.arange(N)*ysep)
# axes[1].set_yticklabels([])
# fig.suptitle('simulated data')
# sns.despine(fig)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# axes[1].set_xlim(0,30)
# fig.savefig('plots/RRR_sim_data.png')


"""
 
 ##       ##     ##    ########  ########  ########     ##     ## ##     ##    ###    ##       
 ##       ###   ###    ##     ## ##     ## ##     ##     ##   ##  ##     ##   ## ##   ##       
 ##       #### ####    ##     ## ##     ## ##     ##      ## ##   ##     ##  ##   ##  ##       
 ##       ## ### ##    ########  ########  ########        ###    ##     ## ##     ## ##       
 ##       ##     ##    ##   ##   ##   ##   ##   ##        ## ##    ##   ##  ######### ##       
 ##       ##     ##    ##    ##  ##    ##  ##    ##      ##   ##    ## ##   ##     ## ##       
 ######## ##     ##    ##     ## ##     ## ##     ##    ##     ##    ###    ##     ## ######## 
 
"""

# %% as a function
def LM(Y, X, lam=0):
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    Y_hat = X @ B_hat
    return B_hat
    
def Rss(Y, Y_hat):
    # model error
    e = Y_hat - Y
    Rss = np.trace(e.T @ e)
    return Rss/Y.shape[0]

def low_rank_approx(A,r, mode='left'):
    """ calculate low rank approximation of matrix A and 
    decomposition into L and W """
    # decomposing and low rank approximation of A
    U, s, Vh = linalg.svd(A)
    S = linalg.diagsvd(s,U.shape[0],s.shape[0])

    # L and W
    if mode is 'left':
        L = U[:,:r] @ S[:r,:r]
        W = Vh[:r,:]
    if mode is 'right':
        L = U[:,:r] 
        W = S[:r,:r] @ Vh[:r,:]
    
    return L, W

def RRR(Y, X, B_hat, r):
    """ reduced rank regression by low rank approx of B_hat """
    L, W = low_rank_approx(B_hat,r)
    B_hat_lr = L @ W
    Y_hat_lr = X @ B_hat_lr
    return B_hat_lr

# # %% find optimal lambda
# Y, X, lags = unpack_segment(Seg, num_lags=200, intercept=True)

# def obj_fun(lam, Y, X):
#     B_hat = LM(Y, X, lam=lam)
#     Y_hat = X @ B_hat
#     return Rss(Y, Y_hat)

# from scipy.optimize import minimize
# x0 = np.array([1])
# res = minimize(obj_fun, x0, args=(Y, X), bounds=[[0,np.inf]])
# lam = res.x


# # %% cheat
# Y, X, lags = unpack_segment(Seg, num_lags=200, intercept=True)
# lam = 0

# # %% xval for getting the Rss dist

# ix = np.arange(Y.shape[0])
# sp.random.shuffle(ix)

# K = 5
# ix_splits = np.split(ix,K)
# Rsss = np.zeros(K)

# rr = list(range(2,7))
# Rsss_rrr = np.zeros((K,len(rr)))

# for k in range(K):
#     print(k)
#     l = list(range(K))
#     l.remove(k)
#     train_ix = np.concatenate([ix_splits[i] for i in l])
#     test_ix = ix_splits[k]

#     # LM xval error
#     B_hat = LM(Y[train_ix], X[train_ix],lam=lam)
#     Y_hat = X[test_ix] @ B_hat
#     Rsss[k] = Rss(Y[test_ix], Y_hat)

#     # RRR
#     for i,r in enumerate(tqdm(rr)):

#         # B_hat = LM(Y[train_ix], X,lam=lam)
#         # Y_hat = X @ B_hat
        
#         B_hat_lr = RRR(Y[train_ix], X[train_ix], B_hat, r)
#         Y_hat_lr = X[test_ix] @ B_hat_lr
#         Rsss_rrr[k,i] = Rss(Y[test_ix], Y_hat_lr)

# # %% plotting r
# fig, axes = plt.subplots()
# axes.plot(rr,np.average(Rsss_rrr, axis=0))
# axes.axhline(np.average(Rsss),linestyle=':',color='k')
# axes.axhline(np.average(Rsss) + np.std(Rsss),linestyle=':',color='k')

# # %% r select
# ix = np.argmax(np.average(Rsss_rrr, axis=0) < np.average(Rsss) + np.std(Rsss))
# r = rr[ix]

# # %% or cheat
# r = nEvents + 1


# %% cheat
nLags = 200
Y, X, lags = unpack_segment(Seg, num_lags=nLags, intercept=True)
lam = 0
r = nEvents + 1

# %% full model final run
B_hat = LM(Y, X,lam=lam)
Y_hat = X @ B_hat

# RRR
B_hat_lr = RRR(Y, X, B_hat, r)
Y_hat_lr = X @ B_hat_lr

L, W = low_rank_approx(B_hat_lr,r)
l = copy(L[1:]) # wo intercept

# print error comparison
print("Full model error comparision:")
print("LM: %5.3f, RRR: %5.3f" % (Rss(Y,Y_hat), Rss(Y,Y_hat_lr)))


# %% plotting inferred kernels
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

# %% plot the simulated data
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
# axes[1].set_yticks(np.arange(N)*ysep)
# axes[1].set_yticklabels([])
# fig.suptitle('simulated data')
# sns.despine(fig)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# axes[1].set_xlim(0,30)

"""
 
 ##             ###    ##    ## ########     ##      ## 
 ##            ## ##   ###   ## ##     ##    ##  ##  ## 
 ##           ##   ##  ####  ## ##     ##    ##  ##  ## 
 ##          ##     ## ## ## ## ##     ##    ##  ##  ## 
 ##          ######### ##  #### ##     ##    ##  ##  ## 
 ##          ##     ## ##   ### ##     ##    ##  ##  ## 
 ########    ##     ## ##    ## ########      ###  ###  
 
"""
# %% inspect L
fig, axes = plt.subplots(figsize=[6,5],nrows=r,ncols=nEvents,sharex=True,sharey=True)

ll = np.array(np.split(l,nEvents,0))

for ri in range(r):
    for i in range(nEvents):
        axes[ri,i].plot(kvec, ll[i,:,ri],color='k')

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')
fig.suptitle('latent factors')
plt.figtext(0.05,0.5,'rank',rotation='vertical')
sns.despine(fig)
fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])

# %% inspect W
fig, axes = plt.subplots(figsize=[6,4])
im = axes.matshow(W,cmap='PiYG',vmin=-0.5,vmax=0.5)
fig.colorbar(mappable=im,shrink=0.5,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from latent factors')
axes.set_title('Weigths')
axes.xaxis.set_ticks_position('bottom')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


 
 ##     ##  ######  #### ##    ##  ######     ##       
 ##     ## ##    ##  ##  ###   ## ##    ##    ##       
 ##     ## ##        ##  ####  ## ##          ##       
 ##     ##  ######   ##  ## ## ## ##   ####   ##       
 ##     ##       ##  ##  ##  #### ##    ##    ##       
 ##     ## ##    ##  ##  ##   ### ##    ##    ##       
  #######   ######  #### ##    ##  ######     ######## 
 
# %% how many independent things are there per event?
"""
is this useful? an independent thing might involve 
"""

# %% ICA on latent factors to retrieve kernels?
l = copy(L[1:])
LatF = np.array(np.split(l,nEvents,0)).swapaxes(0,1)

from sklearn.decomposition import FastICA as ica
from sklearn.decomposition import PCA as pca

fig, axes = plt.subplots(ncols=nEvents,sharey=True)
for i in range(nEvents):
    D = LatF[:,i,:].T
    I = ica(n_components=2).fit(D)
    P = I.transform(D)
    axes[i].plot(kvec, D.T @ P)

""" 
this could be working!
idea: normalize, compare, select those that are candidate different
needs to be well understood ... 
"""
# %% testing around on this
nComps = 2
M = np.zeros((nLags,nEvents,nComps))

Q = np.array(np.split(B_hat_lr[1:,:],nEvents,0)).swapaxes(0,1)

for i in range(nEvents):
    # D = LatF[:,i,:].T
    D = Q[:,i,:].T
    I = ica(n_components=nComps).fit(D)
    P = I.transform(D)
    M[:,i,:] = D.T @ P

# %% normalize and peak invert candidate kernels
for i in range(nEvents):
    for j in range(nComps):
        if M[:,i,j].max() < np.absolute(M[:,i,j]).max():
            M[:,i,j] *= -1
        M[:,i,j] /= M[:,i,j].max()

# %%
fig, axes = plt.subplots(figsize=[6,1.5],ncols=nEvents,sharey=True)
for i in range(nEvents):
    axes[i].plot(kvec, M[:,i,:])
    axes[i].plot(Kernels.times,Kernels[:,i],color='k',lw=2,alpha=0.8,zorder=-1)
axes[-1].plot(Kernels.times,Kernels[:,-1],color='gray',lw=2,alpha=0.8,zorder=-1)


# %%
 
 ##     ##  ######  #### ##    ##  ######      ##      ## 
 ##     ## ##    ##  ##  ###   ## ##    ##     ##  ##  ## 
 ##     ## ##        ##  ####  ## ##           ##  ##  ## 
 ##     ##  ######   ##  ## ## ## ##   ####    ##  ##  ## 
 ##     ##       ##  ##  ##  #### ##    ##     ##  ##  ## 
 ##     ## ##    ##  ##  ##   ### ##    ##     ##  ##  ## 
  #######   ######  #### ##    ##  ######       ###  ###  
 

# %% clustering the weights
# and forming from them "patterns"
# -> candidate kernels
# l = copy(L[1:])
# nClusters = nEvents + 1 # nEvents+1
nClusters = r

# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=nClusters).fit(W.T)
# labels = kmeans.labels_

from sklearn.cluster import SpectralClustering
specclust = SpectralClustering(n_clusters=nClusters, assign_labels='kmeans').fit(W.T)
labels = specclust.labels_

# %% plotting the clusters
ratios = [np.sum(labels == i)/len(labels) for i in range(nClusters)]
ratios.append(0.05)
g_kwargs = dict(width_ratios=ratios)
kwargs = dict(cmap='PiYG',vmin=-0.5,vmax=0.5)
fig, axes = plt.subplots(figsize=[6,2],ncols=nClusters+1, gridspec_kw=g_kwargs)
for i in range(nClusters):
    im = axes[i].matshow(W[:,labels == i],**kwargs)
    axes[i].set_aspect('auto')

fig.colorbar(cax=axes[-1], mappable=im,shrink=0.5,label='au')
for ax in axes[:-1]:
    ax.xaxis.set_ticks_position('bottom')
    ax.set_aspect('auto')
    ax.set_xlabel('to units')
axes[0].set_ylabel('from latent factor')
for ax in axes[1:-1]:
    ax.set_yticklabels([])
fig.suptitle('W clusters')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(wspace=0.1)
fig.savefig('plots/RRR_clusters.png')

# %% per cluster analysis
""" each cluster is one of the things that are going on """
""" how to L combine on a per cluster basis? """

j = 0
fig, axes = plt.subplots(ncols=nEvents,nrows=nClusters,figsize=[6,4],sharey=True)
for j in range(nClusters):
    B_hat_cluster = L @ W[:,labels == j]
    Y_hat_c = X @ B_hat_cluster
    print(Rss(Y[:,labels == j], Y_hat_c))
    splits = sp.array(sp.split(B_hat_cluster[1:,:],nEvents,0)).swapaxes(0,1)
    for i in range(nEvents):
        axes[j,i].plot(kvec,splits[:,i,:])

# %%


# %%

# getting candidate kernels
nLags = 200
candidate_kernels = np.zeros((nLags,nEvents,nClusters))
for j in range(nClusters):
    B_hat_cluster = l @ W[:,labels == j]
    splits = np.split(B_hat_cluster,nEvents,0)
    for i in range(nEvents):
        candidate_kernels[:,i,j] = np.average(splits[i],1) # the candidate kernel

# %% error analysis of clusters?
for j in range(nClusters):
    B_hat_cluster = L @ W[:,labels == j]
    Y_hat_c = X @ B_hat_cluster
    print(Rss(Y[:,labels == j], Y_hat_c))

# %% plotting the combinations
fig, axes = plt.subplots(nrows=nClusters,ncols=nEvents,sharey=True,sharex=True)
fig.suptitle('patterns of summations')
for i in range(nEvents):
    for j in range(nClusters):
        axes[j,i].plot(kvec, candidate_kernels[:,i,j])

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')
sns.despine(fig)
plt.figtext(0.025,0.5,'rank',rotation='vertical')
fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.1,wspace=0.1)
fig.savefig('plots/RRR_combinations.png')


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
        if candidate_kernels[:,i,j].max() < np.absolute(candidate_kernels[:,i,j]).max():
            candidate_kernels[:,i,j] *= -1
        # candidate_kernels[:,i,j] /= candidate_kernels[:,i,j].max()

# %% select the r largest
ix_2d = []
for i in range(nEvents):
    for j in range(nClusters):
        ix_2d.append((i,j))


"""
left off here
2d0 now;
select hights amp ones and see how it flies

not well
"""

# %%
dt = 0.02
tvec = np.arange(0,t_stop.magnitude,dt) * pq.s
dt = np.diff(tvec)[0]

nSamples = np.rint((t_stop/dt).magnitude).astype('int32')
candidate_traces = np.zeros((nSamples,nEvents*nClusters))

m = 0
for i in range(nEvents):
    event = Events[i]
    inds = times2inds(tvec, event.times)
    binds = np.zeros(tvec.shape)
    binds[inds] = 1
    for j in range(nClusters):
        kernel = candidate_kernels[:,i,j].flatten()
        candidate_traces[:,m] = np.convolve(binds,kernel,mode='same')
        m += 1

# %%
fig, axes = plt.subplots()
axes.matshow(candidate_traces.T)
axes.set_aspect('auto')
""" selecting on this one - on magnitude? """

# %% check which candidate trace explains data?
Q = np.zeros((nUnits,nEvents*nClusters))
for u in range(nUnits):
    for m in range(nEvents*nClusters):
        Q[u,m] = np.corrcoef(Y[:,u],candidate_traces[:,m])[0,1]

fig, axes = plt.subplots()
axes.matshow(Q)
# axes.set_aspect('auto')
# %%

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=nClusters).fit(Q.T)
labels = kmeans.labels_

fig, axes = plt.subplots(ncols=nClusters)
for c in range(nClusters):
    ix = ix_2d[np.where(labels == c)]
    for i,j in ix:
        axes[c].plot(candidate_kernels[:,i,j])
















# %%
ix_2d = []
for i in range(nEvents):
    for j in range(nClusters):
        ix_2d.append((i,j))

ix_2d = np.array(ix_2d)[ix,:]

# %% predict signals

dt = 0.02
tvec = np.arange(0,t_stop.magnitude,dt) * pq.s
dt = np.diff(tvec)[0]

nSamples = np.rint((t_stop/dt).magnitude).astype('int32')
signals_pred = np.zeros((nSamples,ix_2d.shape[0]))

m = 0
for i,j in ix_2d:
    event = Events[i]
    inds = times2inds(tvec, event.times)
    binds = np.zeros(tvec.shape)
    binds[inds] = 1
    kernel = candidate_kernels[:,i,j].flatten()
    signals_pred[:,m] = np.convolve(binds,kernel,mode='same')
    m += 1


# %%
fig, axes = plt.subplots()
axes.matshow(signals_pred.T)
axes.set_aspect('auto')

# %%  get back weights
chunks = 10
sig_chunks = np.split(signals_pred,chunks)
Y_chunks = np.split(Y,chunks)
from tqdm import tqdm
W_chunks = np.zeros((ix_2d.shape[0],nUnits,chunks))

for i in tqdm(range(chunks)):
    W_chunks[:,:,i] = np.linalg.pinv(sig_chunks[i]) @ Y_chunks[i]

Weights_pred = np.average(W_chunks,2)
# Weights_pred = np.linalg.pinv(signals_pred) @ Y
plt.matshow(Weights_pred)

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
fig.savefig('plots/RRR_weights_pred.png')







# %% 
fig, axes = plt.subplots(ncols=len(ix),sharex=True)
for i,ii in enumerate(ix):
    axes[i].plot(K[ii])

# %%
chunks = 200
sig_chunks = np.split(signals_pred,chunks)
Y_chunks = np.split(Y,chunks)
from tqdm import tqdm
W_chunks = np.zeros((nEvents*nClusters,nUnits,chunks))

for i in tqdm(range(chunks)):
    W_chunks[:,:,i] = np.linalg.pinv(sig_chunks[i]) @ Y_chunks[i]

Weights_pred = np.average(W_chunks,2)
# Weights_pred = np.linalg.pinv(signals_pred) @ Y
plt.matshow(Weights_pred)

# %% combine top N
# sds = np.std(Weights_pred,axis=1)
# v = np.cumsum(np.sort(sds)[::-1]) / np.sum(sds)
# top_n = np.sum(v>0.95)

# ix = []
# for i in range(nEvents):
#     for j in range(nClusters):
#         ix.append((i,j))

# ix = np.array(ix)[np.sort(np.argsort(sds)[-top_n:])]

# %% select top ranking ones
sds = np.std(Weights_pred,axis=1)
ix = np.argsort(sds)[-r:]
ix = np.sort(ix)

ix_2d = []
for i in range(nEvents):
    for j in range(nClusters):
        ix_2d.append((i,j))

ix_2d = np.array(ix_2d)[ix,:]


# %% ALTERNATE START

dt = 0.02
tvec = np.arange(0,t_stop.magnitude,dt) * pq.s
dt = np.diff(tvec)[0]

nSamples = np.rint((t_stop/dt).magnitude).astype('int32')
signals_pred = np.zeros((nSamples,len(ix)))

m = 0
for i in range(nEvents):
    event = Events[i]
    inds = times2inds(tvec, event.times)
    binds = np.zeros(tvec.shape)
    binds[inds] = 1
    for j in range(nClusters):
        kernel = candidate_kernels[:,i,j].flatten()
        signals_pred[:,m] = np.convolve(binds,kernel,mode='same')
        m += 1

chunks = 200
sig_chunks = np.split(signals_pred,chunks)
Y_chunks = np.split(Y,chunks)
from tqdm import tqdm
W_chunks = np.zeros((nEvents*nClusters,nUnits,chunks))

for i in tqdm(range(chunks)):
    W_chunks[:,:,i] = np.linalg.pinv(sig_chunks[i]) @ Y_chunks[i]

Weights_pred = np.average(W_chunks,2)
# Weights_pred = np.linalg.pinv(signals_pred) @ Y
plt.matshow(Weights_pred)















# swap
swap = False
if swap:
    ix_c = copy(ix)
    ix[-2] = ix_c[-1]
    ix[-1] = ix_c[-2]

Weights_pred_sel = Weights_pred[ix,:]

# %% select the r with the highest SD (or min - max?)
fig, axes = plt.subplots(figsize=[5,4])
axes.plot(np.sort(sds)[::-1],'o')
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

ix = np.array(ix)[np.sort(np.argsort(sds)[-r:])]

Kernels_pred = []

if swap:
    ix_c = copy(ix)
    ix[-2] = ix_c[-1]
    ix[-1] = ix_c[-2]

for i,j in ix:
    Kernels_pred.append(candidate_kernels[:,i,j])

Kernels_pred = np.array(Kernels_pred).T
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
