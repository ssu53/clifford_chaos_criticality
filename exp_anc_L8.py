import numpy as np
import time
import pickle
from tqdm.auto import tqdm
import sys

from routines_anc import simulate_anc_obc, simulate_anc_pbc


def exp_anc_L8_obc(p, L, reps):
    """
    L/8 size subregions A and B with 3L/4 separation
    """
    
    Sa_pre = np.empty((reps,))
    Sb_pre = np.empty((reps,))
    Sab_pre = np.empty((reps,))
    Sa_post = np.empty((reps,))
    Sb_post = np.empty((reps,))
    Sab_post = np.empty((reps,))
    
    A = np.arange(L//8,L)
    B = np.arange(0,7*L//8)
    AB = np.arange(L//8, 7*L//8)
        
    for i in tqdm(range(reps)):
        
        state = simulate_anc_obc(runtime=L, L=L, p=p)

        Sa_pre[i] = state.entEntropyAnc(A)
        Sb_pre[i] = state.entEntropyAnc(B)
        Sab_pre[i] = state.entEntropyAnc(AB)

        state.measureAnc()

        Sa_post[i] = state.entEntropyAnc(A)
        Sb_post[i] = state.entEntropyAnc(B)
        Sab_post[i] = state.entEntropyAnc(AB)
    
    return np.stack((Sa_pre, Sb_pre, Sab_pre, Sa_post, Sb_post, Sab_post))

    
def exp_anc_L8_pbc(p, L, reps):
    """
    L/8 size subregions A and B on a PBC chain (antipodal) with 3L/8 separation
    """
    
    splits = np.arange(3*L//8)
    #splits = np.arange(L//2)
    
    Sa_pre = np.empty((reps,len(splits)))
    Sb_pre = np.empty((reps,len(splits)))
    Sab_pre = np.empty((reps,len(splits)))
    Sa_post = np.empty((reps,len(splits)))
    Sb_post = np.empty((reps,len(splits)))
    Sab_post = np.empty((reps,len(splits)))
    
    for rep in tqdm(range(reps)):
        
        state = simulate_anc_pbc(runtime=L, L=L, p=p)
        
        for j in splits:
            A = set(np.arange(L))-set(np.arange(j,L//8+j))
            B = set(np.arange(L))-set(np.arange(L//2+j,5*L//8+j))
            AB = set(np.arange(L))-set(np.arange(j,L//8+j))-set(np.arange(L//2+j,5*L//8+j))
            Sa_pre[rep,j] = state.entEntropyAnc(list(A))
            Sb_pre[rep,j] = state.entEntropyAnc(list(B))
            Sab_pre[rep,j] = state.entEntropyAnc(list(AB))
        
        state.measureAnc()
        
        for j in splits:
            A = set(np.arange(L))-set(np.arange(j,L//8+j))
            B = set(np.arange(L))-set(np.arange(L//2+j,5*L//8+j))
            AB = set(np.arange(L))-set(np.arange(j,L//8+j))-set(np.arange(L//2+j,5*L//8+j))
            Sa_post[rep,j] = state.entEntropyAnc(list(A))
            Sb_post[rep,j] = state.entEntropyAnc(list(B))
            Sab_post[rep,j] = state.entEntropyAnc(list(AB))
    
    return np.stack((Sa_pre, Sb_pre, Sab_pre, Sa_post, Sb_post, Sab_post))

    
directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
name_ext = sys.argv[4]

ps = np.arange(0,21,2)/100

for p in ps:
    ents = exp_anc_L8_pbc(p=p, L=L, reps=reps)
    outfile = open(directory + '/L' + str(L) + '_p' + str(p) + '_' + name_ext, 'wb')
    pickle.dump(ents, outfile)
    outfile.close()
    
