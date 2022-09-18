import numpy as np
import pickle
from tqdm.auto import tqdm
from numpy.random import choice
from random import random

from routines_tab import allGates, Tab


def tmi(ps, L, runtime, twice=False):
    """
    tripartite mutual information I(A:B:C) where A, B, C are contiguous regions of size L/4
    evolves a PBC chain to runtime with measurement probability p
    get TMI (to follow gate layer, not measurement layer) at each (inequivalent) cut - up to permutations of A,B,C,D
    
    twice False - TMI recorded once, after last layer of gates
    twice True - TMI recorded twice, after last and penultimate layer of gates (in the final timestep),
    to account for odd-even effects, which are particularly pronounced for some L. for L mod 4 == 0, these two
    protocols produce basically the same results. choose less computationally expensive twice = False.
    
    return array tmis has dimensions (len(ps),0) for twice False, (len(ps), 0, 2) for twice True
    """
    
    splits = np.arange(L//4) # TMI is symmetric up to permutations of A,B,C,D
    
    if twice:
        tmis = np.zeros((len(ps), len(splits), 2), dtype='float64') 
    else:
        tmis = np.zeros((len(ps), len(splits)), dtype='float64')

    for p_ind in range(len(ps)):
        p = ps[p_ind]
        
        state = Tab(L)
        
        for t in range(runtime-1):
            for i in range(0,L,2):
                state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
            for i in range(0, L):
                if random() < p: state.measure(i)
            for i in range(1, L+1, 2):
                state.twoQubitClif(allGates[choice(len(allGates))], i, (i+1)%L)
            for i in range(0, L):
                if random() < p: state.measure(i)
        for i in range(0,L,2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)      
        
        if twice:
            for j in range(len(splits)):
                ordering = np.roll(range(L), -splits[j])

                A = ordering[:L//4] # TMI for PBC
                B = ordering[L//4:L//2]
                C = ordering[L//2:3*L//4]

                # bc pure system, compute instead the equivalent but faster S(A_comp) instead of S(A), if |A| < |A_comp|
                Sa   = state.entEntropy(ordering[L//4:]) 
                Sb   = state.entEntropy(set(ordering)-set(B))
                Sc   = state.entEntropy(set(ordering)-set(C))
                Sab  = state.entEntropy(ordering[:L//2])
                Sac  = state.entEntropy(np.concatenate((A,C)))
                Sbc  = state.entEntropy(np.concatenate((B,C)))
                Sabc = state.entEntropy(ordering[:3*L//4])

                tmis[p_ind,j,0] = Sa + Sb + Sc - Sab - Sac - Sbc + Sabc
                
        for i in range(0, L):
            if random() < p: state.measure(i)              
        for i in range(1, L+1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, (i+1)%L)
            
        for j in range(len(splits)):
            ordering = np.roll(range(L), -splits[j])
            
            A = ordering[:L//4] 
            B = ordering[L//4:L//2]
            C = ordering[L//2:3*L//4]
            
            # bc pure system, compute instead the equivalent but faster S(A_comp) instead of S(A), if |A| < |A_comp|
            Sa   = state.entEntropy(ordering[L//4:]) 
            Sb   = state.entEntropy(set(ordering)-set(B))
            Sc   = state.entEntropy(set(ordering)-set(C))
            Sab  = state.entEntropy(ordering[:L//2])
            Sac  = state.entEntropy(np.concatenate((A,C)))
            Sbc  = state.entEntropy(np.concatenate((B,C)))
            Sabc = state.entEntropy(ordering[:3*L//4])
            
            if twice:
                tmis[p_ind,j,1] = Sa + Sb + Sc - Sab - Sac - Sbc + Sabc
            else:
                tmis[p_ind,j]   = Sa + Sb + Sc - Sab - Sac - Sbc + Sabc
    
    return(tmis)


import sys
directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
name_ext = sys.argv[4]
print('directory', directory, ' L', L, ' reps', reps, ' name_ext', name_ext)

ps = np.arange(0.13,0.19,0.0025)
# ps = np.arange(0,0.25,0.01)

# tmis_tot = np.zeros((len(ps), 0, 2)) # twice True
tmis_tot = np.zeros((len(ps),0)) # twice False

for i in tqdm(range(reps)):
    tmis = tmi(ps=ps, L=L, runtime=L, twice=False)
    tmis_tot = np.hstack((tmis_tot, tmis))

outfile = open(directory + '/tmi_L' + str(L) + '_' + name_ext, 'wb')
pickle.dump(tmis_tot, outfile)
outfile.close()
