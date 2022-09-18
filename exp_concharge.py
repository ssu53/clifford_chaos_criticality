import numpy as np
import pickle
from tqdm import tqdm
import sys


swap = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
iswap = np.matrix([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]])
swaps = [swap, iswap]

iphase = np.matrix([[1,0],[0,1j]])
identity = np.matrix([[1,0],[0,1]])
phases = [identity, iphase]


def exp_concharge(L, reps, max_t, directory):
    sff_brute = np.empty((reps, max_t), dtype='float') 

    for rep in tqdm(range(reps)):
        
        floquet_array = np.random.choice([0,1],L)
        phases_array = np.random.choice([0,1],2*L)

        U1 = swaps[floquet_array[0]]
        for ind in range(1,len(floquet_array)//2):
            U1 = np.kron(U1, swaps[floquet_array[ind]])

        P1 = phases[phases_array[0]]
        for ind in range(1,len(phases_array)//2):
            P1 = np.kron(P1, phases[phases_array[ind]])

        U1 = P1 @ U1

        U2 = swaps[floquet_array[len(floquet_array)//2]]
        for ind in range(len(floquet_array)//2+1,len(floquet_array)):
            U2 = np.kron(U2, swaps[floquet_array[ind]])

        P2 = phases[phases_array[len(phases_array)//2]]
        for ind in range(len(phases_array)//2+1,len(phases_array)):
            P2 = np.kron(P2, phases[phases_array[ind]])

        U2 = P2 @ U2

        permute = np.zeros((2**L,2**L),dtype='int8')
        for i in range(len(permute)-1): permute[2*i%(2**L-1),i] = 1
        permute[2**L-1,2**L-1] = 1

        U2 = permute.T @ U2 @ permute

        U = U2 @ U1
        
        for timesteps in range(max_t):
            Ut = np.linalg.matrix_power(U, timesteps)
            sff_brute[rep,timesteps] = np.real(np.round(np.trace(Ut.H) * np.trace(Ut),3))
            
    outfile = open(directory + '/L' + str(L) + '_evol' + str(max_t//L), 'wb')
    pickle.dump(sff_brute, outfile)
    outfile.close()
    

directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
max_t = int(sys.argv[4]) * L
# name_ext = sys.argv[5]

exp_concharge(L=L, reps=reps, max_t=max_t, directory=directory)

