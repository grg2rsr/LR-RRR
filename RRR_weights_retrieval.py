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
"""
 
 ########  ########   #######  ########  ##       ######## ##     ##  ######  
 ##     ## ##     ## ##     ## ##     ## ##       ##       ###   ### ##    ## 
 ##     ## ##     ## ##     ## ##     ## ##       ##       #### #### ##       
 ########  ########  ##     ## ########  ##       ######   ## ### ##  ######  
 ##        ##   ##   ##     ## ##     ## ##       ##       ##     ##       ## 
 ##        ##    ##  ##     ## ##     ## ##       ##       ##     ## ##    ## 
 ##        ##     ##  #######  ########  ######## ######## ##     ##  ######  

 
what about multiple kernels per event?
reward events
-> reward predction kernels
-> reward responding kernel (diff cells, same cells)
disentangling possible?

Kernel for each neuron is not the product of a latent (one for each event) * weight
but instead
a weighted combination of an unknown number of base kernels
(can be greater than the number of events)

are the kernels that I retain for one event
Kernel = kw1*base1 + ...

orthonormal basis - these are the in the columns of L

"""

dt = 0.02
kvec = sp.arange(-2,2,dt) * pq.s
t_stop = 100 * pq.s

nEvents = 5
rates = sp.ones(nEvents)*0.33*pq.Hz
Events = generate_events(nEvents, rates, t_stop)

# one event with multiple causes is equivalent to two simultaneous events?
# it is not, as it will cause duplicate regressors
# think about this
# duplication could also be a way to approach the problem:
# if multiple things are happening distr with diff weight, then each reg would get
# one of the variances?
Events[-2] = Events[-1]

nKernels = nEvents
Kernels = generate_kernels(nKernels, kvec, normed=True, spread=1)

nUnits = 50
q = 12
Weights = sp.rand(nEvents,nUnits)**q - sp.rand(nEvents,nUnits)**q * 0.5 # power for sparseness

Asigs = generate_data(Kernels, Events, Weights, t_stop, noise=0.0)

Seg = neo.core.Segment()
[Seg.analogsignals.append(asig) for asig in Asigs]
[Seg.events.append(event) for event in Events[:-1]]
nEvents = nEvents -1

# %% plot kernels - both on last
fig, axes = plt.subplots(ncols=nKernels-1, figsize=[6,1.5])
for i in range(nKernels-1):
    axes[i].plot(Kernels.times,Kernels[:,i])
axes[-1].plot(Kernels.times,Kernels[:,-1])

# %% plot the weights
fig, axes = plt.subplots(figsize=[6,4])
im = axes.matshow(Weights,cmap='PiYG',vmin=-1,vmax=1)
fig.colorbar(mappable=im,shrink=0.7,label='au')
axes.set_xlabel('to units')
axes.set_ylabel('from kernels')
axes.set_title('Weigths')
axes.xaxis.set_ticks_position('bottom')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# plot weights
# fig, axes = plt.subplots(figsize=[6,1.5])
# im = axes.matshow(Weights,cmap='PiYG',vmin=-1,vmax=1)
# axes.set_aspect('auto')
# fig.colorbar(mappable=im)

# %% multi LM
from scipy import linalg
nLags = 200
Y, X, lags = unpack_segment(Seg, num_lags=nLags, intercept=True)
# Y += .5 * sp.randn(*Y.shape)

# pred: ridge regression
lam = 0
I = sp.diag(sp.ones(X.shape[1]))
B_hat = linalg.pinv(X.T @ X + lam *I) @ X.T @ Y # ridge regression
Y_hat = X @ B_hat

# model error
e = Y_hat - Y
Rss = sp.trace(e.T @ e)
print("Model Rss:", Rss)

# %% plot Y, Y_hat
nUnits_s = 10
fig, axes = plt.subplots(nrows=nUnits_s)
for i in range(nUnits_s):
    asig = Seg.analogsignals[i]
    axes[i].plot(asig.times,Y[:,i],'k',alpha=0.75,lw=2)
    axes[i].plot(asig.times,Y_hat[:,i],'r',alpha=0.75,lw=1)
    # axes[i].plot(asig.times,Y_hat_lr[:,i],'c',alpha=1,lw=1)

# %% both kernels in the last axes
from copy import copy
nUnits_s = 10 # subset
kvec = Kernels.times
fig, axes = plt.subplots(nrows=nUnits_s,ncols=nEvents,sharex=True,sharey=True)
for i in range(nUnits_s):
    Kernels_pred = copy(B_hat[1:,i].reshape(nEvents,-1).T)
    for j in range(nEvents):
        axes[i,j].plot(kvec,Kernels[:,j] * Weights[j,i],'k',lw=3,alpha=0.75)
        axes[i,j].plot(kvec,Kernels_pred[:,j] ,'r',lw=0.8)

    # the duplicate kernel
    axes[i,j].plot(kvec,Kernels[:,-1] * Weights[-1,i],'gray',lw=3,alpha=0.75,zorder=-1)

sns.despine(fig)

"""
observation: regressor combines both kernel (red = gray+black)
we want to disentangle those
"""

# %%
# decomposing and low rank approximation of B_hat
U, s, Vh = sp.linalg.svd(B_hat)
S = sp.linalg.diagsvd(s,U.shape[0],s.shape[0])

# from numpy.linalg import matrix_rank
# r = matrix_rank(B_hat)

r = nEvents + 1 # keep up to including rank - FIXME

B_hat_lr = U[:,:r] @ S[:r,:r] @ Vh[:r,:]

# select which variant to look at
L = U[:,:r] @ S[:r,:r] 
W = Vh[:r,:]

l = copy(L[1:])

# %% plot not only the summed thing but the kernels indiv and weighed 
""" inspecting the latent factors """

fig, axes = plt.subplots(nrows=r,ncols=nEvents,sharex=True,sharey=True)
for ri in range(r):
    ll = l[:,ri]
    ll = ll.reshape(nEvents,-1).T
    for i in range(nEvents):
        axes[ri,i].plot(kvec, ll[:,i],color='k')

fig.suptitle('latent factors')
"""
informative - the expected shapes are in there
"""

# %% clustering the weights
# and forming from them "patterns"
# -> candidate kernels

# from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
nClusters = nEvents + 1 # nEvents+1
# nClusters = r

# kmeans = KMeans(n_clusters=nClusters).fit(W.T)
# labels = kmeans.labels_

specclust = SpectralClustering(n_clusters=nClusters, assign_labels='kmeans').fit(W.T)
labels = specclust.labels_

fig, axes = plt.subplots(ncols=nClusters)
for i in range(nClusters):
    axes[i].matshow(W[:,labels == i])

# getting candidate kernels
candidate_kernels = sp.zeros((nLags,nEvents,nClusters))
for j in range(nClusters):
    signal = l @ W[:,labels == j]
    true_signal = Y[:,labels == j]
    splits = sp.split(signal,nEvents,0)
    for i in range(nEvents):
        candidate_kernels[:,i,j] = sp.average(splits[i],1) # the candidate kernel

# %% plotting them
fig, axes = plt.subplots(nrows=nClusters,ncols=nEvents,sharey=True,sharex=True)
fig.suptitle('patterns of summations')
for i in range(nEvents):
    for j in range(nClusters):
        axes[j,i].plot(kvec, candidate_kernels[:,i,j])
        # axes[j,i].plot(sp.average(splits[i],1),lw=2)
        # axes[j,i].plot(splits[i],color='k',alpha=.5,lw=.5)

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
        candidate_kernels[:,i,j] /= candidate_kernels[:,i,j].max()

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

Weights_pred = sp.linalg.pinv(signals_pred) @ Y
plt.matshow(Weights_pred)

# %% slicing weights_pred
# select sd_lim based on rank
sd_lim = sp.linspace(0,1,10000)
num = [sp.sum(Weights_pred.std(axis=1) > v) for v in sd_lim]
sd = sd_lim[sp.argmax(sp.array(num) == r)]

fig, axes = plt.subplots()
axes.plot(sd_lim,num)


Weights_pred_sel = Weights_pred[sp.std(Weights_pred,axis=1) > sd,:]



# %% 
fig, axes = plt.subplots(ncols=3,figsize=[6,2.5])
kwargs = dict(cmap='PiYG',vmin=-1,vmax=1)
axes[0].matshow(Weights,**kwargs)
axes[0].set_title('true weights')

axes[1].matshow(Weights_pred_sel,**kwargs)
axes[1].set_title('predicted weights')

DWeights =  Weights_pred_sel - Weights
axes[2].matshow(DWeights,**kwargs)
axes[2].set_title('difference')

for ax in axes:
    ax.xaxis.set_ticks_position('bottom')
    ax.set_aspect('auto')

"""
works!
what he have now:
gets back multiple kernels per event if we know the rank

functional subdivision:
populations clustered based on their embedded responses

Weights_pred_sel can be clustered for getting
populations based on functional subdivision
same labels as clustering on W? no, almost, why different?
clustering on W -> how to latent factors distribute
clutering on pred_weights ->


sth intrinsic and specific being modulated by opto
"""
# %% just to compare - there are diff clusters - mh!
from sklearn.cluster import SpectralClustering
nClusters = nEvents + 1 # nEvents+1
nClusters = r

specclust = SpectralClustering(n_clusters=nClusters, assign_labels='kmeans').fit(Weights_pred_sel.T)
labels = specclust.labels_

fig, axes = plt.subplots(ncols=nClusters)
for i in range(nClusters):
    axes[i].matshow(W[:,labels == i])

# %% get back which where the Kernels that were meaningful
ix = []
for i in range(nEvents):
    for j in range(nClusters):
        ix.append((i,j))

ix = sp.array(ix)[sp.std(Weights_pred,axis=1) > sd,:]

Kernels_pred = []
for i,j in ix:
    Kernels_pred.append(candidate_kernels[:,i,j])

Kernels_pred = sp.array(Kernels_pred).T
nKernels_pred = Kernels_pred.shape[1]

fig, axes = plt.subplots(ncols=nKernels_pred,sharey=True, figsize=[6,1.5])
for i in range(nKernels_pred):
    axes[i].plot(kvec, Kernels_pred[:,i], 'r')
    axes[i].plot(kvec,Kernels[:,i],'k',lw=2)
