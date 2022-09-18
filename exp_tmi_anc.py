import numpy as np
import time
import pickle
from tqdm.auto import tqdm
import sys
from numpy.random import choice
from random import random

from routines_anc import allGates, TabAnc, simulate_anc_pbc
                
def tmi_anc_pbc(L, ps, reps):
    """
    gets the TMI an ancilla system PBC
    """
    
    runtime = L 
    splits = np.arange(L//4) # TMI is symmetric up to permutations of A,B,C,D
    
    tmis = np.empty((len(ps),reps,len(splits)), dtype='float64')
    
    for p_ind in range(len(ps)):
        p = ps[p_ind]

        Sa = np.empty((reps,))
        Sb = np.empty((reps,))
        Sc = np.empty((reps,))
        Sab = np.empty((reps,))
        Sac = np.empty((reps,))
        Sbc = np.empty((reps,))
        Sabc = np.empty((reps,))

        for rep in tqdm(range(reps)):

            state = simulate_anc_pbc(runtime=runtime-1, L=L, p=p) 
            for i in range(0, L, 2):
                state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
            for i in range(L):
                if random() < p: state.couple(i,runtime)
            for i in range(1, L+1, 2):
                state.twoQubitClif(allGates[choice(len(allGates))], i, (i+1)%L)

            state.measureAnc()

            for j in range(len(splits)):
                
                A = np.arange(j,L//4+j)
                B = np.arange(L//4+j,L//2+j)
                C = np.arange(L//2+j,3*L//4+j)
                
                # bc pure system, compute instead the equivalent but faster S(A_comp) instead of S(A), if |A| < |A_comp|
                Sa   = state.entEntropyAnc(list(set(np.arange(L))-set(A)))
                Sb   = state.entEntropyAnc(list(set(np.arange(L))-set(B)))
                Sc   = state.entEntropyAnc(list(set(np.arange(L))-set(C)))
                Sab  = state.entEntropyAnc(np.concatenate((A,B)))
                Sac  = state.entEntropyAnc(np.concatenate((A,C)))
                Sbc  = state.entEntropyAnc(np.concatenate((B,C)))
                Sabc = state.entEntropyAnc(np.concatenate((A,B,C)))

                tmis[p_ind,rep,j] = Sa + Sb + Sc - Sab - Sac - Sbc + Sabc
    
    return tmis 

                
directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
name_ext = sys.argv[4]

ps = [0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18, 0.185]

start = time.time()
tmis = tmi_anc_pbc(L=L, ps=ps, reps=reps)
print("runtime total: {}".format(time.time()-start))

outfile = open(directory + '/L' + str(L) + '_' + name_ext, 'wb')
pickle.dump(tmis, outfile)
outfile.close()


