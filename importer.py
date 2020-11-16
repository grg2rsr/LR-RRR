# %%
import pandas as pd
import scipy as sp
import os
from pathlib import Path
from tqdm import tqdm
import elephant as ele
import pickle
import neo
import numpy as np
import quantities as pq

data_dir = Path("/home/georg/data/rrr")

timestamps = pd.read_excel(data_dir / 'allo_egsession_timestamps.xlsx')
flags = pd.read_excel(data_dir / 'allo_egsession_flags.xlsx')

fnames = sp.sort(os.listdir(data_dir / "neurons"))

neurons = []
for fname in fnames:
    data = sp.loadtxt(data_dir / "neurons" / fname)
    data = data[~sp.isnan(data)]
    data = sp.sort(data)
    neurons.append(data)

# %% add spiketrains
Seg = neo.core.Segment()
t_last_spike = np.max([neuron[-1] for neuron in neurons])

for neuron in neurons:
    st = neo.core.SpikeTrain(neuron*pq.ms, t_stop = (t_last_spike+1)*pq.ms)
    Seg.spiketrains.append(st)

# %% add events
colnames = timestamps.columns
for name in colnames:
    t = timestamps[name].values
    t = t[~sp.isnan(t)]
    Event = neo.core.Event(t * pq.ms)
    Event.annotate(label=name)
    Seg.events.append(Event)

# correcting last timepoint
t_last = np.max([event.times[-1] for event in Seg.events])
if t_last > t_last_spike:
    for st in Seg.spiketrains:
        st.t_stop = t_last * pq.ms + 1 * pq.s

# %% firing rate estimation
for i,st in enumerate(tqdm(Seg.spiketrains)):
    print(i)
    asig = ele.statistics.instantaneous_rate(st, 20*pq.ms)
    asig.annotate(id=i)
    Seg.analogsignals.append(asig)

# %%
import pickle
with open(data_dir / 'data.neo', 'wb') as fH:
    print("writing neo segment ... ")
    pickle.dump(Seg,fH)
    print("...done")
# %%