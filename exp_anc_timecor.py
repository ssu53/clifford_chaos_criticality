import numpy as np
import pickle
from tqdm.auto import tqdm
import sys

from routines_anc import TabAnc, simulate_anc_pbc


def exp_anc_timeL3(Ls, ps, reps, directory):
    """
    PBC
    pre-measurement state L3 mutual information in time direction between ancillae
    """
    for L in Ls:
        for p in ps:
            print("L = ", L, " p = ", p)
            
            runtime = L
            Sa = np.empty((reps,))
            Sb = np.empty((reps,))
            Sab = np.empty((reps,))
            
            for i in tqdm(range(reps)):
                
                state = simulate_anc_pbc(runtime=runtime, L=L, p=p)
                
                qubit_time = np.array(np.concatenate((np.repeat(-1, L), state.coupled_anc_t)))
                A = np.where((qubit_time >= 0) & (qubit_time < runtime//3))[0]
                B = np.where(qubit_time >= runtime*2//3)[0]
                AB = np.concatenate((A,B))
                
                Sa[i] = state.entEntropy(A)
                Sb[i] = state.entEntropy(B)
                Sab[i] = state.entEntropy(AB)
            
            outfile = open(directory + '/L' + str(L) + '_p' + str(p), 'wb')
            pickle.dump(np.vstack((Sa, Sb, Sab)), outfile)
            outfile.close()

            
def exp_anc_timecor(L, p, rs, reps):
    """
    pbc
    pre-measurement correlation length in time between ancillae
    """
            
    runtime = L
    Sa = np.empty((reps,len(rs)))
    Sb = np.empty((reps,len(rs)))
    Sab = np.empty((reps,len(rs)))

    for i in tqdm(range(reps)):

        state = simulate_anc_pbc(runtime=runtime, L=L, p=p)

        qubit_time = np.array(np.concatenate((np.repeat(-1, L), state.coupled_anc_t)))

        for j in range(len(rs)):
            A = np.where((qubit_time >= 0) & (qubit_time < (runtime-rs[j])//2))[0]
            B = np.where(qubit_time >= runtime-(runtime-rs[j])//2)[0]
            AB = np.concatenate((A,B))

            Sa[i,j] = state.entEntropy(A)
            Sb[i,j] = state.entEntropy(B)
            Sab[i,j] = state.entEntropy(AB)

    return np.stack((Sa, Sb, Sab))

    
directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
name_ext = sys.argv[4]

rs = np.arange(0, L//2+3,2)
ps = np.arange(0,21,2)/100
#ps = np.arange(0,15,2)/100
# ps = np.arange(16,21,2)/100

for p in ps:
    print('p', p)
    ents = exp_anc_timecor(L, p, rs, reps)
    outfile = open(directory + '/L' + str(L) + '_p' + str(p) + '_' + name_ext, 'wb')
    pickle.dump(ents, outfile)
    outfile.close()
    
