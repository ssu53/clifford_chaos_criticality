import numpy as np
from numpy.random import choice
import pickle
from tqdm import tqdm
import sys
import time

from routines_tab import allGates, gf2RowRed

def lst_gen(lst):
    for item in lst:
        yield item
    
    
def twoQubitClif(tab, gate, a, b):
    """
    PHASELESS gate evolution
    this varies from the usual twoQubitClif for tableaus, here each COLUMN is a stabiliser (Pauli string)
    """
    n = np.shape(tab)[1] // 2
    matches = tab[n+b,:] + 2*tab[n+a,:] + 4*tab[b,:] + 8*tab[a,:]
    tab[a,:] = gate[matches,0] # xa
    tab[b,:] = gate[matches,1] # xb
    tab[n+a,:] = gate[matches,2] # za
    tab[n+b,:] = gate[matches,3] # zb
    
def twoQubitClif_ps(ps, gate, a, b):
    """
    PHASEFUL gate evolution
    acts on 1D array pauli string ps, where the last position indexes the phsae 
    """
    L = len(ps) // 2
    match = ps[L+b] + 2*ps[L+a] + 4*ps[b] + 8*ps[a]
    ps[a] = gate[match,0] # xa
    ps[b] = gate[match,1] # xb
    ps[L+a] = gate[match,2] # za
    ps[L+b] = gate[match,3] # zb
    ps[2*L] = (ps[2*L] + gate[match,4]) % 2 # phase
    

def exp_sff(L, reps, max_t, directory, name_ext):
    circuits = choice(np.arange(576,len(allGates)), (L,reps))
    n_evecs_t = np.zeros((max_t,reps))
    allphase1 = np.full((max_t,reps), True)

    for rep in tqdm(range(reps)):
        
        pauli_strings = np.identity(2*L, dtype='int8')
        floquet = lst_gen(circuits[:,rep])
        
        for i in range(0, L ,2):
            twoQubitClif(pauli_strings, allGates[next(floquet)], i, i+1)
        for i in range(1, L+1, 2):
            twoQubitClif(pauli_strings, allGates[next(floquet)], i, (i+1)%L)
        
        for timesteps in range(max_t):

            gate = np.linalg.matrix_power(pauli_strings, timesteps) % 2
            
            eig1 = (gate - np.eye(2*L, dtype='int8')) % 2
            system = np.vstack((eig1, np.identity(2*L, dtype='int8')))
            system_solved = gf2RowRed(system.T.copy()).T
            

            n_evecs = 0
            eigvecs = []
            for col in range(2*L):
                if np.sum(system_solved[:2*L,col]) == 0:
                    n_evecs += 1
                    eigvec = system_solved[2*L:,col]
                    eigvecs.append(eigvec)
                    if not np.array_equal((gate @ eigvec) % 2, eigvec):
                        print(col, eigvec, (gate @ eigvec) % 2)
                        raise Exception()
            n_evecs_t[timesteps,rep] = n_evecs

            for eigvec in eigvecs:
                ps = np.hstack((eigvec, [0]))

                for t in range(timesteps):
                    floquet = lst_gen(floquet_array)
                    for i in range(0, L ,2):
                        gate = allGates[next(floquet)]
                        twoQubitClif_ps(ps, gate, i, i+1)
                    for i in range(1, L+1, 2):
                        gate = allGates[next(floquet)]
                        twoQubitClif_ps(ps, gate, i, (i+1)%L)

                if (ps[-1] != 0):
                    allphase1[timesteps,rep] = False
                    break
          
    outfile = open(directory + '/L' + str(L) + '_evol' + str(max_t//L) + '_' + name_ext, 'wb')
    pickle.dump(np.stack((n_evecs_t, allphase1)), outfile)
    outfile.close()

#     outfile = open(directory + '/L' + str(L) + '_evol' + str(max_t//L) + '_' + name_ext + '_circuits', 'wb')
#     pickle.dump(circuits, outfile)
#     outfile.close()


def exp_sff_bruteverify(L, reps, max_t, directory, name_ext):
    circuits = choice(np.arange(576,len(allGates)), (L,reps))
    n_evecs_t = np.zeros((max_t,reps))
    allphase1 = np.full((max_t,reps), True)
    
    #-------------------------------------------------------------
    
    # load dictionary from index to decomposition
    infile = open('/Users/Shiye/Desktop/the/dictClif', 'rb')
    dictClif = pickle.load(infile)
    infile.close()
    
    # import methods for brute force unitary trace verification
    from routines_genclif import decompToUni
    
    permute = np.zeros((2**L,2**L),dtype='int8')
    for i in range(len(permute)-1): permute[2*i%(2**L-1),i] = 1
    permute[2**L-1,2**L-1] = 1
    # print(np.array_equal(np.linalg.inv(permute), permute.T))
    
    #-------------------------------------------------------------

    for rep in tqdm(range(reps)):
        
        pauli_strings = np.identity(2*L, dtype='int8')
        floquet = lst_gen(circuits[:,rep])
        
        for i in range(0, L ,2):
            gate = allGates[next(floquet)]
            twoQubitClif(pauli_strings, gate, i, i+1)
        for i in range(1, L+1, 2):
            gate = allGates[next(floquet)]
            twoQubitClif(pauli_strings, gate, i, (i+1)%L)
        
        for timesteps in range(max_t):
            
            gate = np.linalg.matrix_power(pauli_strings, timesteps) % 2
            
            eig1 = (gate - np.eye(2*L, dtype='int8')) % 2
            system = np.vstack((eig1, np.identity(2*L, dtype='int8')))
            system_solved = gf2RowRed(system.T.copy()).T
            

            n_evecs = 0
            eigvecs = []
            for col in range(2*L):
                if np.sum(system_solved[:2*L,col]) == 0:
                    n_evecs += 1
                    eigvec = system_solved[2*L:,col]
                    eigvecs.append(eigvec)
                    if not np.array_equal((gate @ eigvec) % 2, eigvec):
                        print(col, eigvec, (gate @ eigvec) % 2)
                        raise Exception()
            n_evecs_t[timesteps,rep] = n_evecs

            for eigvec in eigvecs:
                ps = np.hstack((eigvec, [0]))

                for t in range(timesteps):
                    floquet = lst_gen(floquet_array)
                    for i in range(0, L ,2):
                        gate = allGates[next(floquet)]
                        twoQubitClif_ps(ps, gate, i, i+1)
                    for i in range(1, L+1, 2):
                        gate = allGates[next(floquet)]
                        twoQubitClif_ps(ps, gate, i, (i+1)%L)

                if (ps[-1] != 0):
                    allphase1[timesteps,rep] = False
                    break
            
            #-------------------------------------------------------------
            
            U1 = decompToUni(dictClif[floquet_array[0]])
            for ind in range(1,len(floquet_array)//2):
                U1 = np.kron(U1, decompToUni(dictClif[floquet_array[ind]]))

            U2 = decompToUni(dictClif[floquet_array[len(floquet_array)//2]])
            for ind in range(len(floquet_array)//2+1,len(floquet_array)):
                U2 = np.kron(U2, decompToUni(dictClif[floquet_array[ind]]))
            
            U2 = permute.T @ U2 @ permute

            U = U2 @ U1
            U = np.linalg.matrix_power(U, timesteps)

            sff_brute = np.round(np.trace(U.H) * np.trace(U),3)

            
            if sff_brute != 2**len(eigvecs) * allphase1[timesteps, rep]:
                
                print(floquet_array)
                outfile = open('temp_floquet', 'wb')
                pickle.dump(floquet_array, outfile)
                outfile.close()
                
                print('timesteps:', timesteps)
                print('sff_brute:', sff_brute)
                print('sff_gf2:', 2**n_evecs * allphase1[timesteps, rep])
                print('len(eigvecs):', len(eigvecs))
                for eigvec in eigvecs:
                    print(eigvec)
                
                outfile = open('temp_ps', 'wb')
                pickle.dump(pauli_strings, outfile)
                outfile.close()
                
                outfile = open('temp_system', 'wb')
                pickle.dump(system, outfile)
                outfile.close()
                
                outfile = open('temp_syssolved', 'wb')
                pickle.dump(system_solved, outfile)
                outfile.close()
                
                raise Exception()
            #-------------------------------------------------------------

    outfile = open(directory + '/L' + str(L) + '_evol' + str(max_t//L) + '_' + name_ext, 'wb')
    pickle.dump(np.stack((n_evecs_t, allphase1)), outfile)
    outfile.close()
    

    
directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
max_t = int(sys.argv[4]) * L
name_ext = sys.argv[5]


start = time.time()
exp_sff_bruteverify(L=L, reps=reps, max_t=max_t, directory=directory, name_ext=name_ext)
print(time.time()-start)
