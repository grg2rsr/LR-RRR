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
# mpl.rcParams['figure.dpi'] = 166
mpl.rcParams['figure.dpi'] = 331

import quantities as pq
import neo
import elephant as ele
import seaborn as sns
from copy import copy

from plotting_helpers import *
from data_generator import *
from RRRlib import *

# %%
"""
 
 ########     ###    ########    ###        ######   ######## ##    ## 
 ##     ##   ## ##      ##      ## ##      ##    ##  ##       ###   ## 
 ##     ##  ##   ##     ##     ##   ##     ##        ##       ####  ## 
 ##     ## ##     ##    ##    ##     ##    ##   #### ######   ## ## ## 
 ##     ## #########    ##    #########    ##    ##  ##       ##  #### 
 ##     ## ##     ##    ##    ##     ##    ##    ##  ##       ##   ### 
 ########  ##     ##    ##    ##     ##     ######   ######## ##    ## 
 
"""
# setting up simulated data
dt = 0.02 # time bin for firing rate estimation
t_stop = 1*60*60 * pq.s # 1h recording

nEvents = 5 # number of events 
rates = np.ones(nEvents)*0.2*pq.Hz # average rate of events
rates[1] /= 2
Events = generate_events(nEvents, rates, t_stop)

e1 = Events[-2]
e2 = Events[-1]
Events.append(e1)
Events.append(e2)

# nEvents = nEvents + 2

# first problem: (this one fails at ETA)
# A->B causality, with time lag for vis
# realized by merging first two events
Events[1] = Events[1].merge(Events[0].time_shift(0.5*pq.s)).time_slice(0,t_stop)

# further problems (not in order):
# multiple kernels behind two events, one with cov and one without
# realized by duplication and simulating an extra kernel for it

# problems to revisit - I think this should be the first problem
# responses in data without any events
# -> do they disturb the other predicted kernels?
# solved below, they don't 

# events without any responses in data
# -> do they acquire any respnses
# solved below - sort this mess

# simulating Kernels
kvec = np.arange(-2,2,dt) * pq.s
nKernels = len(Events)
# Kernels = generate_kernels(nKernels, kvec, normed=True, spread=0.5)
Kernels = generate_kernels(nKernels, kvec, normed=True, spread=0.5, mus=[0,0,0,1,1,-1,-1],sigs=sp.ones(nKernels)/2, scale=[1,1,0,1,1,1,1])

# simulating units
nUnits = 50

# Weight matrix: nEvents x nUnits
q = 5 # sparseness
Weights = sp.rand(nKernels,nUnits)**q - sp.rand(nKernels,nUnits)**q * 0.5

# injecting covariance of level c in half of the units w the duplicate kernels
c = 1.0
T = np.array([[1,c],[c,1]])
Weights[[-3,-1],:int(nUnits/2)] = (Weights[[-3,-1],:int(nUnits/2)].T @ T).T
# Weights[:,0] = 1
print("CC on this run: %2.2f"% np.corrcoef(Weights[-1,:],Weights[-2,:])[0,1])

# injecting covariance of level c in half of the units w the duplicate kernels
# c = 1.0
# T = np.array([[1,c],[c,1]])
# Weights[-2:,:int(nUnits/4)] = (Weights[-2:,:int(nUnits/4)].T @ T).T
# print("CC on this run: %2.2f"% np.corrcoef(Weights[-1,:],Weights[-2,:])[0,1])

# Weights[-1,int(nUnits*2/4):int(nUnits*3/4)] = 0
# Weights[-2,int(nUnits*3/4):int(nUnits*4/4)] = 0

# inject cov across units
# c = 0.5
# T = np.ones((nEvents,nEvents)) / 2
# T[np.diag_indices(nEvents)] = 1 
# Weights[:,:int(nUnits/3)] = (Weights[:,:int(nUnits/3)].T @ T).T

Asigs = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

# adding hidden signal to the signal (that can't be modelled)
add_hidden = True

if add_hidden:
    nEvents_other = nEvents
    nKernels_other = nKernels
    rates_other = np.ones(nEvents_other) * 1.0 * pq.Hz
    Events_other = generate_events(nEvents_other, rates_other, t_stop)
    Kernels_other = generate_kernels(nKernels_other, kvec, normed=True, spread=0.5)
    Weights_other = sp.rand(nEvents_other,nUnits)**q - sp.rand(nEvents_other,nUnits)**q * 0.5
    Asigs_other = generate_data(Kernels_other, Events_other, Weights_other, t_stop, noise=0.5)

    for i in range(len(Asigs)):
        Asigs[i] += Asigs_other[i]

Seg = neo.core.Segment()
[Seg.analogsignals.append(asig) for asig in Asigs]
[Seg.events.append(event) for event in Events[:-2]] # the last two duplicate events

# %% plot kernels - both on last
fig, axes = plt.subplots(ncols=nEvents, figsize=[6,1.5])
for i in range(nEvents):
    axes[i].plot(Kernels.times,Kernels[:,i],color='C%i'%i)

for i in [-2,-1]:
    axes[i].plot(Kernels.times,Kernels[:,i],color='C%i'%(nEvents+-i))

for ax in axes:
    ax.axvline(0,linestyle=':',color='k',alpha=0.5)
    ax.set_xlabel('time (s)')
    ax.set_ylim(-0.1,1.1)
sns.despine(fig)
axes[0].set_ylabel('au')
fig.suptitle('Kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('plots/RRR_kernels.png')

# %% plot the weights
fig, axes = plt.subplots(figsize=[6,2])
im = axes.matshow(Weights,cmap='PiYG',vmin=-1,vmax=1)
fig.colorbar(mappable=im,shrink=1.0,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from kernels')
axes.set_title('Weigths')
axes.set_aspect('auto')
axes.xaxis.set_ticks_position('bottom')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('plots/RRR_weights.png')

# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
ysep = 4
N = 10 # up to N asig

axes[0].set_ylabel('Events')
for i, asig in enumerate(Asigs[:N]):
    axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=1)
axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_yticks(np.arange(N)*ysep)
axes[1].set_yticklabels([])
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[1].set_xlim(0,60)
fig.savefig('plots/RRR_sim_data.png')

# %% maybe some classic ETA to compare
asig = Seg.analogsignals[0]
fig, axes = plt.subplots(ncols=nEvents,figsize=[6,2.5])
t_slice = (-2,2) * pq.s
for i, event in enumerate(Events[:-1]):
    asig_slices = []
    for t in event.times:
        try:
            asig_slice = asig.time_slice(t+t_slice[0],t+t_slice[1])
            tvec = asig_slice.times - t
            # axes[i].plot(tvec, asig_slice, 'k', alpha=0.25,lw=1)
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

# %%
"""
 
 ##     ## ##     ##    ###    ##       
  ##   ##  ##     ##   ## ##   ##       
   ## ##   ##     ##  ##   ##  ##       
    ###    ##     ## ##     ## ##       
   ## ##    ##   ##  ######### ##       
  ##   ##    ## ##   ##     ## ##       
 ##     ##    ###    ##     ## ######## 
 
"""

# %% unpack data
Y, X, t_lags = unpack_segment(Seg, num_lags=200, intercept=True)

# %% xval lambda
k = 5
lam = xval_ridge_reg_lambda(Y[:1000], X[:1000], k)

# %% cheat
lam = 24.83

# %% xval rank

# idea: at which rank does Rss get worset than 1 SD
# select the last rank where model error does not increase by more than 1 SD

# getting the error distribution (Rss)
k = 5
ranks = list(range(2,10))
Rsss_lm, Rsss_rrr = xval_rank(Y, X, lam, ranks, k)

#r select
ix = np.argmax(np.average(Rsss_rrr, axis=0) < np.average(Rsss_lm) + np.std(Rsss_lm))
r = ranks[ix]

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


# %% cheat
nLags = 200
Y, X, t_lags = unpack_segment(Seg, num_lags=nLags, intercept=True)
lam = 24.8
r = nEvents + 2

# %% full model final run
B_hat = LM(Y, X, lam=lam)
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
N = 5 # for the first N units
fig, axes = plt.subplots(figsize=[6,5], nrows=N,ncols=nEvents,sharex=True,sharey=True)

# plotting the known kernels
kvec = Kernels.times.rescale('ms')
for i in range(N):
    for j in range(nEvents):
        axes[i,j].plot(kvec, Kernels[:,j] * Weights[j,i], 'C%i'%j, lw=3, alpha=0.75)

    # the duplicate kernels
    for j in [-2,-1]:
        axes[i,j].plot(kvec, Kernels[:,j] * Weights[j,i], 'C%i'%(nEvents+-j), lw=3, alpha=0.75, zorder=-1)

# plotting predicted kernels for the units
b = copy(B_hat[1:])
Kernels_pred = np.array(np.split(b,nEvents,0)).swapaxes(0,1)
for i in range(N):
    for j in range(nEvents):
        axes[i,j].plot(t_lags, Kernels_pred[:,j,i], 'k', lw=1, alpha=0.75)

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')
fig.suptitle('est. kernels for each unit')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.1,wspace=0.1)
fig.savefig('plots/RRR_LM_pred_kernels.png')

# %% plot the simulated data
fig, axes = plt.subplots(nrows=2, sharex=True, figsize=[6,4], gridspec_kw=dict(height_ratios=(0.2,1)))
plot_events(Events,ax=axes[0])
ysep = 3
N = 10
axes[0].set_ylabel('Events')
for i, asig in enumerate(Asigs[:N]):
    axes[1].plot(asig.times,asig.magnitude+i*ysep,'k',lw=1)
    axes[1].plot(asig.times,Y_hat[:,i]+i*ysep,'C3',lw=2)
    axes[1].plot(asig.times,Y_hat_lr[:,i]+i*ysep,'C4',lw=1)

axes[1].set_xlabel('time (s)')
axes[1].set_ylabel('signal (au)')
axes[1].set_yticks(np.arange(N)*ysep)
axes[1].set_yticklabels([])
fig.suptitle('simulated data')
sns.despine(fig)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
axes[1].set_xlim(0,120)

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
# fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])

# %% inspect W
fig, axes = plt.subplots(figsize=[6,2])
im = axes.matshow(W,cmap='PiYG',vmin=-0.5,vmax=0.5)
fig.colorbar(mappable=im,shrink=1.0,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from latent factors\n(rank)')
axes.set_title('Weigths')
axes.xaxis.set_ticks_position('bottom')
axes.set_aspect('auto')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# %%
"""
 
 ##     ##  ######  #### ##    ##  ######      ##       
 ##     ## ##    ##  ##  ###   ## ##    ##     ##       
 ##     ## ##        ##  ####  ## ##           ##       
 ##     ##  ######   ##  ## ## ## ##   ####    ##       
 ##     ##       ##  ##  ##  #### ##    ##     ##       
 ##     ## ##    ##  ##  ##   ### ##    ##     ##       
  #######   ######  #### ##    ##  ######      ######## 
 
""" 

# %% THIS APPROACH SOLVES FOR COV=0
# more notes on this - the covariance error is exactly transferred
# is it?
# also does not solve for cov0?

l = copy(L[1:])
LatF = np.array(np.split(l,nEvents,0)).swapaxes(0,1)
# LatF.shape id of shape (nLags, nEvents, rank)

# general idea: orthogonalize several embedded signals if detected
# orthogonal signals -> candidate kernels based on ica
# orthogonalization based on ica

# problem:
# how to avoid that kernels become a combination of each other?

# rethinking this:
# instead of forcing basis to be orthogonal
# force the mixing matrix to have orthogonal rows?

candidate_kernels = [] # will be list arrays (nLags x r_est)

for i in range(nEvents):
    D = LatF[:,i,:]
    r_est = pca_rank_est(D, th=0.95)
    print(i, r_est)

    # this needs proper description
    R = D @ W
    P = low_rank_approx(R.T, r_est)[0]
    K = R @ linalg.pinv(P.T)

    # peak invert if negative
    for j in range(K.shape[1]):
        if K[:,j].max() < np.absolute(K[:,j]).max():
            K[:,j] *= -1

    # normalize all
    for j in range(K.shape[1]):
        K[:,j] /= K[:,j].max()

    candidate_kernels.append(K)

# # %% index flip kernels for plotting if necessary
# from copy import copy
# tmp = copy(candidate_kernels[-1])
# candidate_kernels[-1][:,0] = candidate_kernels[-1][:,1]
# candidate_kernels[-1][:,1] = tmp[:,0]

# %% plot candidate kernels
fig, axes = plt.subplots(ncols=nEvents, figsize=[6,1.5])

for i in range(len(candidate_kernels)):
    for j in range(candidate_kernels[i].shape[1]):
        axes[i].plot(t_lags, candidate_kernels[i][:,j], alpha=0.75, color='C%i'%(i+j),lw=2)

for ax in axes:
    ax.axvline(0,linestyle=':',color='k',alpha=0.5)
    ax.set_xlabel('time (s)')

sns.despine(fig)
axes[0].set_ylabel('au')
fig.suptitle('Kernels')
fig.subplots_adjust(top=0.85)
# fig.tight_layout()
fig.savefig('plots/RRR_kernels.png')

# %% With events and candidate kernels -> predict signals -> retrieve weights
dt = 0.02
tvec = np.arange(0,t_stop.magnitude,dt) * pq.s
dt = np.diff(tvec)[0]

nSamples = np.rint((t_stop/dt).magnitude).astype('int32')
nKernels_pred  = sum([K.shape[1] for K in candidate_kernels])
signals_pred = np.zeros((nSamples,nKernels_pred))

for i,K in enumerate(candidate_kernels):
    event = Events[i]
    inds = times2inds(tvec, event.times)
    binds = np.zeros(tvec.shape)
    binds[inds] = 1
    for j in range(K.shape[1]):
        kernel = K[:,j]
        signals_pred[:,i+j] = np.convolve(binds,kernel,mode='same')

# %%  get back weights
chunks = 10
sig_chunks = np.split(signals_pred,chunks)
Y_chunks = np.split(Y,chunks)
from tqdm import tqdm
W_chunks = np.zeros((nKernels_pred,nUnits,chunks))

for i in tqdm(range(chunks)):
    W_chunks[:,:,i] = np.linalg.pinv(sig_chunks[i]) @ Y_chunks[i]

Weights_pred = np.average(W_chunks,2)
# Weights_pred = np.linalg.pinv(signals_pred) @ Y
# plt.matshow(Weights_pred)

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
"""
 
 ##     ##  ######  #### ##    ##  ######      ##      ## 
 ##     ## ##    ##  ##  ###   ## ##    ##     ##  ##  ## 
 ##     ## ##        ##  ####  ## ##           ##  ##  ## 
 ##     ##  ######   ##  ## ## ## ##   ####    ##  ##  ## 
 ##     ##       ##  ##  ##  #### ##    ##     ##  ##  ## 
 ##     ## ##    ##  ##  ##   ### ##    ##     ##  ##  ## 
  #######   ######  #### ##    ##  ######       ###  ###  
 
"""

"""
W can be clustered in 2 dimensions
along units
along events

clustering events - which events lead to similar cognitive evets
clustering units - functional grouping of recorded neurons

"""
# %% clustering the weights - into how many clusters?
nClusters = r # rank is the number of things that are happening?

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=nClusters).fit(W.T)
labels = kmeans.labels_

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


# %% rerun the above on clusters?
ck = []
for q in range(r):
    l = copy(L[1:])
    LatF = np.array(np.split(l,nEvents,0)).swapaxes(0,1)

    candidate_kernels = [] # will be list arrays (nLags x r_est)

    for i in range(nEvents):
        D = LatF[:,i,:]
        r_est = pca_rank_est(D)

        # this needs proper description
        R = D @ W[:,labels==q]
        P = low_rank_approx(R.T, r_est)[0]
        K = R @ linalg.pinv(P.T)

        # peak invert if negative
        for j in range(K.shape[1]):
            if K[:,j].max() < np.absolute(K[:,j]).max():
                K[:,j] *= -1

        # normalize all
        for j in range(K.shape[1]):
            K[:,j] /= K[:,j].max()

        candidate_kernels.append(K)
        if K.shape[1] > 1:
            print(sp.corrcoef(K.T))

    # %% index flip kernels for plotting if necessary
    # from copy import copy
    # tmp = copy(candidate_kernels[-1])
    # candidate_kernels[-1][:,0] = candidate_kernels[-1][:,1]
    # candidate_kernels[-1][:,1] = tmp[:,0]

    # %% plot kernels - both on last
    # for i in range(nKernels-1):
        # axes[i].plot(Kernels.times,Kernels[:,i],color='C%i'%i)
        # axes[i].plot(Kernels.times,Kernels[:,i],color='k',lw=1)
    # axes[-1].plot(Kernels.times,Kernels[:,-1],color='C%i'%(nKernels-1))
    # axes[-1].plot(Kernels.times,Kernels[:,-1],color='k',lw=1)

    fig, axes = plt.subplots(ncols=nEvents, figsize=[6,1.5])

    for i in range(len(candidate_kernels)):
        for j in range(candidate_kernels[i].shape[1]):
            axes[i].plot(t_lags, candidate_kernels[i][:,j], alpha=0.75, color='C%i'%(i+j),lw=2)

    for ax in axes:
        ax.axvline(0,linestyle=':',color='k',alpha=0.5)
        ax.set_xlabel('time (s)')

    sns.despine(fig)
    axes[0].set_ylabel('au')
    fig.suptitle('Kernels from label %i'%q)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig('plots/RRR_kernels.png')

    ck.append(candidate_kernels)


# %%
# top level iterator : clusters







# %% REAL DATA
import pickle
with open('/home/georg/data/rrr/data.neo', 'rb') as fH:
    Seg = pickle.load(fH)










# %% how many independent things are there per event?
# this doesn't solve anything

l = copy(L[1:])
LatF = np.array(np.split(l,nEvents,0)).swapaxes(0,1)
# LatF.shape (nLags, nEvents, rank)

# general idea: orthogonalize several embedded signals if detected
# orthogonal signals -> candidate kernels based on ica
# orthogonalization based on ica

# problem:
# how to avoid that kernels become a combination of each other?

# rethinking this:
# instead of forcing basis to be orthogonal
# force the mixing matrix to have orthogonal rows?


candidate_kernels = [] # will be list arrays (nLags x r_est)

for i in range(nEvents):
    D = LatF[:,i,:]
    r_est = pca_rank_est(D) # well dig here deeper into it
    K = ica_orth(D, r=r_est)
    # print(r_est)
    # I = ICA(n_components=r_est).fit(D.T)
    # P = I.transform(D.T) # interpretation of P - projection
    # K = D @ P

    # peak invert if negative
    for j in range(K.shape[1]):
        if K[:,j].max() < np.absolute(K[:,j]).max():
            K[:,j] *= -1

    # normalize all
    for j in range(K.shape[1]):
        K[:,j] /= K[:,j].max()

    candidate_kernels.append(K)





