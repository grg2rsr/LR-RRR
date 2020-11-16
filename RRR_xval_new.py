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
t_stop = 60*60 * pq.s

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
# Kernels = generate_kernels(nKernels, kvec, normed=True, spread=0.5)
Kernels = generate_kernels(nKernels, kvec, normed=True, spread=0.5, mus=[0,0,0,1,-1],sigs=[.5,.5,.5,.5,.5])

nUnits = 50
q = 5
Weights = sp.rand(nEvents,nUnits)**q - sp.rand(nEvents,nUnits)**q * 0.5 # power for sparseness

# inject cov of level c in half of the units w the dup kernels
c = 1.0
T = np.array([[1,c],[c,1]])
Weights[-2:,:int(nUnits/2)] = (Weights[-2:,:int(nUnits/2)].T @ T).T
print("CC on this run: %2.2f"% np.corrcoef(Weights[-1,:],Weights[-2,:])[0,1])

# # inject cov across units
# c = 0.5
# T = np.ones((nEvents,nEvents)) / 2
# T[np.diag_indices(nEvents)] = 1 
# Weights[:,:int(nUnits/3)] = (Weights[:,:int(nUnits/3)].T @ T).T

Asigs = generate_data(Kernels, Events, Weights, t_stop, noise=0.5)

Seg = neo.core.Segment()
[Seg.analogsignals.append(asig) for asig in Asigs]
[Seg.events.append(event) for event in Events[:-1]]
nEvents = nEvents -1


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
axes[1].set_xlim(0,30)
fig.savefig('plots/RRR_sim_data.png')

# %% maybe some classic ETA to compare
asig = Seg.analogsignals[0]
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
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
    Y_hat = X @ B_hat
    return B_hat
    
def Rss(Y, Y_hat):
    """ evaluate model error """
    e = Y_hat - Y
    Rss = np.trace(e.T @ e)
    return Rss/Y.shape[0]

def low_rank_approx(A, r, mode='left'):
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

# %% find optimal lambda
# Seg.events = Seg.events[0:7]
Y, X, lags = unpack_segment(Seg, num_lags=200, intercept=True)

def chunk(A, n_chunks=10):
    """ split A into n chunks, ignore overhaning samples """
    chunk_size = sp.floor(A.shape[0] / n_chunks)
    drop = int(A.shape[0] % chunk_size)
    if drop != 0:
        A = A[:-drop]
    A_chunks = sp.split(A, n_chunks)
    return A_chunks

# %% xval lambda
from scipy.optimize import minimize
def obj_fun(lam, Y_train, X_train, Y_test, X_test):
    B_hat = LM(Y_train, X_train, lam=lam)
    Y_hat_test = X_test @ B_hat
    return Rss(Y_test, Y_hat_test)

k = 5

ix = sp.range(X.shape[0])
ix_chunks = chunk(ix, k)

lambdas = []
for i in tqdm(range(k),desc="xval lambda"):
    l = list(range(k))
    l.remove(i)

    ix_train = sp.concatenate([ix_chunks[i] for i in ix])
    ix_test = ix_chunks[i]
    
    x0 = np.array([1])
    res = minimize(obj_fun, x0, args=(Y[ix_train], X[ix_train],
                   Y[ix_test], X[ix_test[]), bounds=[[0,np.inf]])
    lam = res.x
    lambdas.append(lam)

# %% xval rank estimate




# %% cheat
Y, X, lags = unpack_segment(Seg, num_lags=200, intercept=True)
lam = 0

# %% xval for getting the Rss dist

ix = np.arange(Y.shape[0])
sp.random.shuffle(ix)

K = 5 # k-fold xval
ix_splits = chunk(ix,K)
Rsss = np.zeros(K)

rr = list(range(2,7)) # ranks to check
Rsss_rrr = np.zeros((K,len(rr)))

for k in range(K):
    print(k)
    l = list(range(K))
    l.remove(k)
    train_ix = np.concatenate([ix_splits[i] for i in l])
    test_ix = ix_splits[k]

    # LM xval error
    B_hat = LM(Y[train_ix], X[train_ix],lam=lam)
    Y_hat = X[test_ix] @ B_hat
    Rsss[k] = Rss(Y[test_ix], Y_hat)

    # RRR
    for i,r in enumerate(tqdm(rr)):

        # B_hat = LM(Y[train_ix], X,lam=lam)
        # Y_hat = X @ B_hat
        
        B_hat_lr = RRR(Y[train_ix], X[train_ix], B_hat, r)
        Y_hat_lr = X[test_ix] @ B_hat_lr
        Rsss_rrr[k,i] = Rss(Y[test_ix], Y_hat_lr)

# %% plotting r
fig, axes = plt.subplots()
axes.plot(rr,np.average(Rsss_rrr, axis=0))
axes.axhline(np.average(Rsss),linestyle=':',color='k')
axes.axhline(np.average(Rsss) + np.std(Rsss),linestyle=':',color='k')

# %% r select
ix = np.argmax(np.average(Rsss_rrr, axis=0) < np.average(Rsss) + np.std(Rsss))
r = rr[ix]

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
N = 10 # for the first 10 units
kvec = Kernels.times
fig, axes = plt.subplots(figsize=[6,5], nrows=N,ncols=nEvents,sharex=True,sharey=True)
for i in range(N):
    Kernels_pred = copy(B_hat[1:,i].reshape(nEvents,-1).T)
    for j in range(nEvents):
        axes[i,j].plot(kvec,Kernels[:,j] * Weights[j,i],'k',lw=2,alpha=0.75)
        axes[i,j].plot(kvec,Kernels_pred[:,j] ,'C3',lw=1)

    # the duplicate kernel
    axes[i,j].plot(kvec,Kernels[:,-1] * Weights[-1,i],'gray',lw=3,alpha=0.75,zorder=-1)

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
axes[1].set_xlim(0,30)

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

# %% this one will go to lib as well
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA as ICA
def estimate_rank(A,th=0.99):
    """ estimate rank by explained variance on PCA """
    pca = PCA(n_components=A.shape[1])
    pca.fit(A)
    var_exp = sp.cumsum(pca.explained_variance_ratio_) < th
    return 1 + np.sum(var_exp)


# %% how many independent things are there per event?
l = copy(L[1:])
LatF = np.array(np.split(l,nEvents,0)).swapaxes(0,1)
# LatF.shape (nLags, nEvents, rank)

# general idea: are there several orthogonal things embedded?
for i in range(nEvents):
    A = LatF[:,i,:]
    r_est = estimate_rank(A)
    print(r_est)

# answer is yes

# if yes which?
# retrieve candidate kernels based on ica

candidate_kernels = [] # will be list arrays (nLags x r_est)

for i in range(nEvents):
    D = LatF[:,i,:]
    r_est = estimate_rank(D) # well dig here deeper into it
    I = ICA(n_components=r_est).fit(D.T)
    P = I.transform(D.T) # interpretation of P - projection
    K = D @ P

    # peak invert
    for j in range(K.shape[1]):
        if K[:,j].max() < np.absolute(K[:,j]).max():
            K[:,j] *= -1

    # normalize each 
    for j in range(K.shape[1]):
        K[:,j] /= K[:,j].max()

    candidate_kernels.append(K)

# %% index flip kernels
from copy import copy
tmp = copy(candidate_kernels[-1])
candidate_kernels[-1][:,0] = candidate_kernels[-1][:,1]
candidate_kernels[-1][:,1] = tmp[:,0]

# %% plot kernels - both on last
fig, axes = plt.subplots(ncols=nKernels-1, figsize=[6,1.5])
for i in range(nKernels-1):
    axes[i].plot(Kernels.times,Kernels[:,i],color='C%i'%i)
axes[-1].plot(Kernels.times,Kernels[:,-1],color='C%i'%(nKernels-1))

for i in range(len(candidate_kernels)):
    for j in range(candidate_kernels[i].shape[1]):
        axes[i].plot(Kernels.times, candidate_kernels[i][:,j], alpha=0.5, color='C%i'%(i+j))

for ax in axes:
    ax.axvline(0,linestyle=':',color='k',alpha=0.5)
    ax.set_xlabel('time (s)')

sns.despine(fig)
axes[0].set_ylabel('au')
fig.suptitle('Kernels')
fig.tight_layout()
fig.subplots_adjust(top=0.85)
fig.savefig('plots/RRR_kernels.png')



# should be possible from this now to get to signals pred

# %% predict signals
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

# %% flip last axis in case
# from copy import copy
# Weights_temp = copy(Weights_pred)
# Weights_pred[-1] = Weights_pred[-2]
# Weights_pred[-2] = Weights_temp[-1]

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
 
 ##     ##  ######  #### ##    ##  ######      ##      ## 
 ##     ## ##    ##  ##  ###   ## ##    ##     ##  ##  ## 
 ##     ## ##        ##  ####  ## ##           ##  ##  ## 
 ##     ##  ######   ##  ## ## ## ##   ####    ##  ##  ## 
 ##     ##       ##  ##  ##  #### ##    ##     ##  ##  ## 
 ##     ## ##    ##  ##  ##   ### ##    ##     ##  ##  ## 
  #######   ######  #### ##    ##  ######       ###  ###  
 
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











# %% REAL DATA
import pickle
with open('/home/georg/data/rrr/data.neo', 'rb') as fH:
    Seg = pickle.load(fH)

















# %% per cluster analysis
""" each cluster is one of the things that are going on """
""" how to L combine on a per cluster basis? """
""" averaging clusters makes no sense? """

fig, axes = plt.subplots(ncols=nEvents,nrows=nClusters,figsize=[6,4],sharey=True)
for j in range(nClusters):
    B_hat_cluster = L @ W[:,labels == j]
    # Y_hat_c = X @ B_hat_cluster
    # print(Rss(Y[:,labels == j], Y_hat_c))
    splits = sp.array(sp.split(B_hat_cluster[1:,:],nEvents,0)).swapaxes(0,1)
    for i in range(nEvents):
        axes[j,i].plot(kvec,splits[:,i,:],lw=1,color='k',alpha=0.25)
        axes[j,i].plot(kvec,sp.average(splits[:,i,:],1),lw=2,color='C0')

for ax in axes[-1,:]:
    ax.set_xlabel('time (s)')

sns.despine(fig)
plt.figtext(0.025,0.5,'rank',rotation='vertical')
fig.tight_layout(rect=[0.05, 0.03, 1, 0.95])
fig.subplots_adjust(hspace=0.1,wspace=0.1)
fig.savefig('plots/RRR_combinations.png')


# %% getting candidate kernels
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
        candidate_kernels[:,i,j] /= candidate_kernels[:,i,j].max()

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
# W_chunks = np.zeros((ix_2d.shape[0],nUnits,chunks))
W_chunks = np.zeros((nKernels_pred,nUnits,chunks))

for i in tqdm(range(chunks)):
    W_chunks[:,:,i] = np.linalg.pinv(sig_chunks[i]) @ Y_chunks[i]

Weights_pred = np.average(W_chunks,2)
# Weights_pred = np.linalg.pinv(signals_pred) @ Y
# plt.matshow(Weights_pred)

# %% flip last axis
from copy import copy
Weights_temp = copy(Weights_pred)
Weights_pred[-1] = Weights_pred[-2]
Weights_pred[-2] = Weights_temp[-1]

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
