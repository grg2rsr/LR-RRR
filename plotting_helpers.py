
import scipy as sp
import neo
import matplotlib.pyplot as plt
import matplotlib as mpl
import quantities as pq
import elephant as ele
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from copy import copy

def plot_events(Events, ysep=1, ax=None):
    if ax is None:
        ax = plt.gca()
    for i, event in enumerate(Events):
        ax.plot(event.times,sp.ones(event.times.shape[0])+i*ysep,'.')
    ax.set_ylim(-0.5,len(Events)+1.5)
    return ax

def plot_kernels(Kernels, figsize=[6,1.5], color=None,**kwargs):
    nKernels = Kernels.shape[1]
    fig, axes = plt.subplots(ncols=nKernels, sharey=True, figsize=figsize)
    for i in range(nKernels):
        if color is None:
            col = 'C%i'%i
        else:
            col = color
        axes[i].plot(Kernels.times,Kernels[:,i],color=col,**kwargs)
        axes[i].axvline(0,linestyle=':',color='k',alpha=0.5)
        sns.despine(ax=axes[i])
        axes[i].set_xlabel('time (s)')
    axes[0].set_ylabel('au')
    return fig, axes

