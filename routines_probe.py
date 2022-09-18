import numpy as np
import time
from numpy.random import choice
from random import random
import pickle
from tqdm.auto import tqdm

from routines_tab import allGates, Tab


def probe1pt(L = 20, p = 0, runtime_after = None):
    """
    OBC
    1-pt probe as order parameter for phase transition detection
    tracks the probe's entropy throughout evolution
    taken from the routines, but we don't neeed entanglement measurements throughout
    run for time L, entangle, and track probe while evolve for L
    entropy estimates are made after the layer of gates, not the layer of measurements
    """
    
    state = Tab(L+1)
    runtime = L
    
    # encoding step - no measurements
    for t in range(runtime):
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        
    # circuit in steady state, maximally entangle edge spin (site 0) with reference (site L)
    state.measure(0) 
    state.bellpair(0,L)
    
    refA = []
    # evolving step - measurements
    for t in range(runtime_after):
        
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
            
        refA.append(state.entEntropy(range(L)))

        for i in range(0, L):
            if random() < p: state.measure(i)

        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        
        refA.append(state.entEntropy(range(L)))
        
        for i in range(0, L):
            if random() < p: state.measure(i)
                    
    return(state, refA)



def exp_probe1pt(L = 20, p = 0, runtime_after = None, reps=10):
    """
    runs 1pt probe experiment
    """
    refAs = np.empty((0,runtime_after*2))
    for i in tqdm(range(reps)):
        _, refA = probe1pt(L=L, p=p, runtime_after=runtime_after)
        refAs = np.vstack((refAs,np.array(refA)))
    return(refAs)



def probe1pt_evol(L = 20, p = 0, runtime_after=None):
    """
    OBC
    1-pt probe as order parameter for phase transition detection
    only get entanglement at the end of evolution
    takne from the routines, but we don't neeed entanglement measurements throughout
    """
    
    state = Tab(L+1)
    runtime = L
    
    # encoding step - no measurements
    for t in range(runtime):
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        
    # circuit in steady state, maximally entangle edge spin (site 0) with reference (site L)
    state.measure(0) 
    state.bellpair(0,L)
    
    # evolving step - measurements
    for t in range(runtime_after):
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(0, L):
            if random() < p: state.measure(i)
        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(0, L):
            if random() < p: state.measure(i)
    
    return(state)



def exp_probe1pt_finalEntOnly(L = 20, ps=np.arange(0.05,0.18,0.02), runtime_after = None, reps=10):
    """
    runs 1 probe experiment, computes entanglement only at end of evolution
    """
    refAs_p = np.empty((len(ps)))
    for i in tqdm(range(len(ps))):
        refAs = []
        for j in range(reps):
            state = probe1pt_evol(L=L, p=ps[i], runtime_after=L)
            refAs.append(state.entEntropy(range(L)))
        refAs_p[i] = np.mean(refAs)
    return(refAs_p)



def probe2pt(L = 20, runtime=20, p = 0):
    """
    two-point correlation probe
    OBC
    taken from the routines, but we don't neeed entanglement measurements throughout
    run for time L, entangle, and track probe while evolve for L
    inspect the probe after the layer of gates, not the layer of measurements
    """
    
    state = Tab(L+2)
    
    # encoding step - no measurements
    for t in range(L):
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        
    # circuit in steady state, maximally entangle edge spins (sites 0, L-1) with references (sites L, L+1)
    state.measure(0) 
    state.bellpair(0,L)
    state.measure(L-1)
    state.bellpair(L-1,L+1)
    
    refA = []
    refB = []
    refAB = []
    # evolving step - measurements
    for t in range(runtime):
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)

#         refA.append(state.entEntropy(list(range(L)) + [L+1]))
#         refB.append(state.entEntropy(range(L+1)))
#         refAB.append(state.entEntropy(range(L)))
        
        for i in range(0, L):
            if random() < p: state.measure(i)

        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        
        refA.append(state.entEntropy(list(range(L)) + [L+1]))
        refB.append(state.entEntropy(range(L+1)))
        refAB.append(state.entEntropy(range(L)))
        
        for i in range(0, L):
            if random() < p: state.measure(i)
                        
    return(state, refA, refB, refAB)


def exp_probe2pt(L = 20, runtime = 20, p = 0, reps=10):
    """
    runs experiment on the 2pt probe
    """
    refAs = np.empty((0,runtime))
    refBs = np.empty((0,runtime))
    refABs = np.empty((0,runtime))
    for i in tqdm(range(reps)):
        _, refA, refB, refAB = probe2pt(L=L, runtime=runtime, p=p)
        refAs = np.vstack((refAs,np.array(refA)))
        refBs = np.vstack((refBs,np.array(refB)))
        refABs = np.vstack((refABs,np.array(refAB)))
    return(refAs, refBs, refABs)
