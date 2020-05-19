import scipy as sp
import neo
import quantities as pq
import elephant as ele
from tqdm import tqdm
import pandas as pd

def times2inds(tvec,times):
    return [sp.argmin(sp.absolute(tvec - t)) for t in times] 

def generate_kernels(nKernels, kvec, spread=0.5, width=0.2, normed=True):
    """ generate some shapes """

    mus = sp.randn(nKernels) * spread
    sigs = sp.ones(nKernels) * width

    # kernels is matrix of shape k timepoints x events
    Kernels = sp.zeros((kvec.shape[0],nKernels))
    for i in range(nKernels):
        Kernels[:,i] = sp.stats.distributions.norm(mus[i],sigs[i]).pdf(kvec)
        if normed:
            Kernels[:,i] /= Kernels[:,i].max()

    Kernels = neo.core.AnalogSignal(Kernels,units=pq.dimensionless, t_start=kvec[0],t_stop=kvec[-1],sampling_period=sp.diff(kvec)[0])
    
    return Kernels

def distribute_latent_kernels(LatKernels, Weights, as_asig=True, normed=False):
    """ Weights define the mapping of how the Kernels are combined to features
    Weights thus contain nFeatures
    """
    dt = LatKernels.sampling_period
    kvec = LatKernels.times

    ObsKernels =  LatKernels.magnitude @ Weights
    if normed:
        ObsKernels = ObsKernels / ObsKernels.max(0)[sp.newaxis,:]
        
    if as_asig:
        ObsKernels = neo.core.AnalogSignal(ObsKernels,units=pq.dimensionless, t_start=kvec[0],t_stop=kvec[-1],sampling_period=dt)  
    return ObsKernels

def generate_events(nEvents, rates, t_stop):
    Events = []
    for i in range(nEvents):
        st = ele.spike_train_generation.homogeneous_poisson_process(rates[i],t_stop=t_stop)
        event = neo.core.Event(st.times)
        Events.append(event)
    return Events

def generate_data(Kernels, Events, Weights, t_stop, noise=0):
    """    FE_weights (n_features x n_events) is a matrix that distributes features on events
    result is a matrix (n_kvec_samples x n_events)
    
    convolve each event with the respective column of the matrix
    -> a matrix (t_samples x n_events)
    then

    how does each event act on each unit?
    multiply above matrix with another distributing matris
    EU_Weights (n_events x n_units)
    """

    dt = Kernels.sampling_period

    nSamples = sp.rint((t_stop/dt).magnitude).astype('int32')
    signals = sp.zeros((nSamples,len(Events)))

    tvec = sp.arange(0,t_stop.magnitude,dt.magnitude) * pq.s

    # convolution
    for i, event in enumerate(Events):
        inds = times2inds(tvec, event.times)
        binds = sp.zeros(tvec.shape)
        binds[inds] = 1
        kernel = Kernels[:,i].magnitude.flatten()
        signals[:,i] = sp.convolve(binds,kernel,mode='same')

    # distributing weights
    signals = signals @ Weights

    # adding noise
    signals = signals + sp.randn(*signals.shape) * noise

    # cast to neo asig
    Asigs = []
    for i in range(signals.shape[1]):
        asig = neo.core.AnalogSignal(signals[:,i],units=pq.dimensionless, sampling_period=dt)
        Asigs.append(asig)
    return Asigs

def unpack_segment(Seg, num_lags=100, intercept=True):
    """ """

    lags = sp.arange(-num_lags/2,num_lags/2,1,dtype='int32')

    Y = sp.stack([asig.magnitude.flatten() for asig in Seg.analogsignals],axis=1)
    t_stop = Seg.analogsignals[0].t_stop
    dt = Seg.analogsignals[0].sampling_period

    X = []
    for event in Seg.events:
        st = neo.core.SpikeTrain(event.times,t_stop=t_stop)
        bst = ele.conversion.BinnedSpikeTrain(st,binsize=dt,t_stop=t_stop)
        reg = bst.to_array().flatten()
        for lag in lags:
            X.append(sp.roll(reg,lag))

    X = sp.stack(X,axis=1)
    
    if intercept:
        X = sp.concatenate([sp.ones((X.shape[0],1)),X],1)

    lags = lags * dt.magnitude
    return Y, X, lags


