import numpy as np
import pickle
from tqdm.auto import tqdm
import sys

from routines_tab import Tab, Tab_clipped
from routines_anc import simulate_anc_obc

def entEntropyclipped_middle(l_ends, r_ends, site_l, site_r):
    if (site_l < -1) or (site_l >= len(l_ends)) or (site_r < -1) or (site_r >= len(l_ends)):
        raise Exception("invalid site" + ' ' + str(site_l) + ' ' + str(site_r))
    return((np.sum((l_ends <= site_l) * (r_ends > site_l) * (r_ends <= site_r)) +
           np.sum((l_ends <= site_r) * (l_ends > site_l) * (r_ends > site_r)))//2)


def getMIs(l_ends,r_ends,L,qubit_pos,rs):
    Sa = np.empty(len(rs))
    Sb = np.empty(len(rs))
    Sab = np.empty(len(rs))
    N = len(l_ends)
    
    for i in range(len(rs)):
        size = (L-rs[i])//2
        
        site_l = qubit_pos[size]
        site_r = qubit_pos[L-size]
        Sa[i] = entEntropyclipped_middle(l_ends, r_ends, -1, site_l-1)
        Sb[i] = entEntropyclipped_middle(l_ends, r_ends, site_r-1, N-1)
        Sab[i] = entEntropyclipped_middle(l_ends, r_ends, site_l-1, site_r-1)
        # AB includes qubit_pos[i+1], excludes qubit_pos[L-i]

    return Sa, Sb, Sab


def exp_anc_corlen(p, L, runtime, rs, reps):
    """
    varying separation, pre- ancilla projeciton
    """
    
    Sa = np.empty((reps,len(rs)))
    Sb = np.empty((reps,len(rs)))
    Sab = np.empty((reps,len(rs)))                                                                                                            
    for rep in tqdm(range(reps)):
        
        state = simulate_anc_obc(runtime=runtime, L=L, p=p)
        
        qubit_pos = [np.sum(np.array(state.coupled_anc_q) < i) + i for i in range(L)] 
        state_clipped = Tab_clipped(state.xzAdjacent())
        Sa[rep,:], Sb[rep,:], Sab[rep,:] = getMIs(state_clipped.l_ends, state_clipped.r_ends, L, qubit_pos, rs)
    
    return np.stack((Sa, Sb, Sab))
            
    
directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
p = float(sys.argv[4])
name_ext = sys.argv[5]

runtime = L
rs = np.arange(0,L//2+3,2)

ents = exp_anc_corlen(p=p, L=L, runtime=runtime, rs=rs, reps=reps)
outfile = open(directory + '/L' + str(L) + '_p' + str(p) + '_' + name_ext, 'wb')
pickle.dump(ents, outfile)
outfile.close()
