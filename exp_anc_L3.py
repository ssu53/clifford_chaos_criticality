import numpy as np
import pickle
from tqdm.auto import tqdm
import sys
from numpy.random import choice
from random import random

from routines_tab import allGates
from routines_anc import TabAnc, simulate_anc_obc


def exp_anc_L3(p, L, runtime, reps, timetraj=False):
    """
    mutual information between L/3 size subregions A and B with L/3 separation
    if timetraj: at every timestep, I(A:B) pre- and post- ancilla projection
    else:        at end of time evolution, I(A:B) pre- and post- ancilla projection
    """
    
    if timetraj:
        Sa_pre = np.empty((reps,runtime))
        Sb_pre = np.empty((reps,runtime))
        Sab_pre = np.empty((reps,runtime))
        Sa_post = np.empty((reps,runtime))
        Sb_post = np.empty((reps,runtime))
        Sab_post = np.empty((reps,runtime))   
    else:
        Sa_pre = np.empty((reps,))
        Sb_pre = np.empty((reps,))
        Sab_pre = np.empty((reps,))
        Sa_post = np.empty((reps,))
        Sb_post = np.empty((reps,))
        Sab_post = np.empty((reps,))
    
    A = range(L//3, L)
    B = range(0, 2*L//3)
    AB = list(range(L//3, 2*L//3))
    
    if timetraj:
        
        for rep in tqdm(range(reps)):

            state = TabAnc(L)

            for t in range(runtime):

                for i in range(0, L-1, 2):
                    state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)

                for i in range(L):
                    if random() < p: state.couple(i,t)

                for i in range(1, L-1, 2):
                    state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)

                for i in range(L):
                    if random() < p: state.couple(i,t)

                Sa_pre[rep,t] = state.entEntropyAnc(A)
                Sb_pre[rep,t] = state.entEntropyAnc(B)
                Sab_pre[rep,t] = state.entEntropyAnc(AB)

                tab_premeasure = np.copy(state.tab)
                state.measureAnc()

                Sa_post[rep,t] = state.entEntropyAnc(A)
                Sb_post[rep,t] = state.entEntropyAnc(B)
                Sab_post[rep,t] = state.entEntropyAnc(AB)

                state.tab = tab_premeasure
    else:
        
        for rep in tqdm(range(reps)):
        
            state = simulate_anc_obc(runtime=runtime, L=L, p=p)

            Sa_pre[rep] = state.entEntropyAnc(A)
            Sb_pre[rep] = state.entEntropyAnc(B)
            Sab_pre[rep] = state.entEntropyAnc(AB)

            state.measureAnc()

            Sa_post[rep] = state.entEntropyAnc(A)
            Sb_post[rep] = state.entEntropy(B)
            Sab_post[rep] = state.entEntropy(AB)
    
    return Sa_pre, Sb_pre, Sab_pre, Sa_post, Sb_post, Sab_post


    
def exp_anc_L3_fs(p, L, runtime, fs, reps):
    """
    mutual information between L/3 size subregions A and B with L/3 separation
    at end of time evolution, project a fraction f (given by array fs) of ancilla, record I(A:B)
    """
    
    Sa  = np.empty((reps,len(fs)))
    Sb  = np.empty((reps,len(fs)))
    Sab = np.empty((reps,len(fs)))
    
    A = range(L//3, L)
    B = range(0, 2*L//3)
    AB = list(range(0, L//3)) + list(range(2*L//3,L))
    
    for rep in tqdm(range(reps)):
        
        state = simulate_anc_obc(runtime=runtime, L=L, p=p)
        
        for k in range(len(fs)):
            
            tab_premeasure = np.copy(state.tab)
            state.measureAnc(f=fs[k])
                        
            Sa[rep,k]  = state.entEntropyAnc(A)
            Sb[rep,k]  = state.entEntropyAnc(B)
            Sab[rep,k] = state.entEntropyAnc(AB)

            state.tab = tab_premeasure       
    
    return Sa, Sb, Sab



directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
name_ext = sys.argv[4]

runtime = L

ps = np.arange(0,21,1)/100
fs = np.arange(0,11,1)/10

print(ps)
print(fs)

for p in ps:
    print('p =', p)
    ents = exp_anc_L3_fs(p, L, runtime, fs, reps)
    outfile = open(directory + '/L' + str(L) + '_p' + str(p) + '_' + name_ext, 'wb')
    pickle.dump(ents, outfile)
    outfile.close()
    