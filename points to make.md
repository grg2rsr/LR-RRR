# points to make in a possible tech paper about RRR

closed loop behavior with physiology leads to deep data

measuring both behavior and neurons
hen and egg problem
task influences neural activity - neural activity influences behavior

some simple examples:
motor planning signals preceed movement
sensory representations succeed sensory stimuli

more interesting examples:
reward expectation/prediction and reward consumption

to find out if a neuron drives sth or is driven by sth
first pass analysis
align on event

embedded in the data are: influence of task events on neurons, and vice versa 
example: reward responding / reward predicting

temporal contingency between the two by calculating averages can only be inferred if there are no causal links between events
example: disentangling lick responding from reward responding is difficult, because they happen with non-random link

lagged regressor analysis can take care of this because ... not clear why this disentangles this actually, but it does

multi unit - not clear why to introduce this here now
maybe from the start?

multi unit - reduced rank regression in combination with lagged regressor gives back weights if kernels are assumed to be "unitary" meaning one events has the same kernel for all neurons (just different weights)
definitely wrong assumption (reward predicting / reward responding)

but clustering on weights, forming patterns and trying to predict data from the patterns pulls out all non-unitarty clusters

what is the use?
clustering of the population based on embedded responses (argument needed that clustering on W is different than clustering on the responses themselves)
gives meaningful populations - analysis can be restricted to those

gives individual driving forces and how they act on neurons
interesting how opto activation acts on those




# %%
"""
general intro:

Disentangle what neurons respond to (= are driven by)
and what neural responses are causing (= behaviors they are driving)

what neurons respond to:
first level analysis: align on "world" events (usually task events) and average

or the other way around: spike triggered average (usually averaging stimulus wrt spike)
"pattern triggered average?"

what this captures: average systematic variations
what this misses: anything that involves interactions, interdependence, covariations

# general things to think about
argument: aligning behavior on single neurons is very messy (is it?)
not single cells that drive things but entire populations that drive

functional subdivision of populations


are cluster averages




# very simple first case
Q: how does a neuron respond to a reward?
neuron driven by rewards or predicting rewards?
ETA -> should give kernel

...


"""