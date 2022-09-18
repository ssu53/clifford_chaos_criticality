import numpy as np
from random import randint, random
from numpy.random import choice
import time
from tqdm.auto import tqdm
import pickle


from routines_tab import allGates, Tab, Tab_clipped


def simulate_1D_obc(runtime = 20, L = 20, p = 0, verbose = False):
    """
    OBC
    """

    if verbose: start = time.time()
    
    state = Tab(L)
    entEnt = np.empty(runtime) 
    
    for t in (tqdm(range(runtime)) if verbose else range(runtime)):
        
        # compute entropy, from a single L/2 cut
        entEnt[t] = state.entEntropy(np.arange(L//2))
        
        # act with gates
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)

        # measure
        for i in range(0, L):
            if random() < p: state.measure(i)

        # act with gates on the alternate pairs
        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        
        # measure
        for i in range(0, L):
            if random() < p: state.measure(i)

    if verbose: print("runtime: {}".format(time.time() - start))
    
    return(entEnt, state)



def simulate_1D_obc_clipped(runtime = 20, L = 20, p = 0, verbose = False):
    """
    OBC
    in clipped gauge, cut at every point
    CAUTION: this is kind of stupid. averaging over different size cuts is not that reasonable, though 
    the transition is detectable this way 
    """

    if verbose: start = time.time()
    
    state = Tab(L)
    entEnt = np.empty(runtime) 
    
    for t in (tqdm(range(runtime)) if verbose else range(runtime)):

        # compute entropy
        entEnts = np.zeros(L-1, dtype=int)
        state_clipped = Tab_clipped(state.xzAdjacent())
        for i in range(L-1):
            entEnts[i] = state_clipped.entEntropy(i)
        entEnt[t] = np.mean(entEnts)
        
        """
        # compute entropy - sample cut_sample cuts of different sizes
        entEnts = np.empty(cut_samples, dtype=int)
        for i in range(cut_samples):
            #A = choice(L, size=cut_size, replace=False) # random geometry - no transition
            A = np.arange(0, randint(1,L-1)) # contiguous geometry OBC
            entEnts[i] = state.entEntropy(A)
        entEnt[t] = np.mean(entEnts)
        """
        
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(0, L):
            if random() < p: state.measure(i)
        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(0, L):
            if random() < p: state.measure(i)

    if verbose: print("runtime: {}".format(time.time() - start))
    
    return(entEnt, state)




def simulate_1D_pbc(runtime = 20, L = 20, p = 0, verbose = False):
    """
    PBC
    """
    
    if L % 2 != 0: raise Exception('L even only')

    if verbose: start = time.time()
    
    state = Tab(L)
    entEnt = np.empty(runtime) 
    
    for t in (tqdm(range(runtime)) if verbose else range(runtime)):
        
        # compute entropy - each of the L/2 inequivalent size L/2 cuts
        entEnts = []
        for site in range(L//2):
            entEnts.append(state.entEntropy(range(site,site+L//2)))
        entEnt[t] = np.mean(entEnts)
        
        """
        # compute entropy - sample cut_sample cuts of size cut_size
        entEnts = np.empty(cut_samples, dtype=int)
        for i in range(cut_samples):
            #A = choice(L, size=cut_size, replace=False) # random geometry - no transition
            A = np.roll(range(L), randint(0,L))[:cut_size] # contiguous geometry, PBC
            entEnts[i] = state.entEntropy(A)
        entEnt[t] = np.mean(entEnts)
        """
        
        for i in range(0, L, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
        for i in range(0, L):
            if random() < p: state.measure(i)
        for i in range(1, L+1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, (i+1)%L)
        for i in range(0, L):
            if random() < p: state.measure(i)

    if verbose: print("runtime: {}".format(time.time() - start))
    
    return(entEnt, state)



def exp_1D_obc(Ls, ps, reps, directory):
    """
    OBC experiment
    """

    start_tot = time.time()

    for L in Ls:
        for p in ps:
            print("L = ", L, " p = ", p)

            start = time.time()

            runtime = L
            ents = np.zeros((reps, runtime))

            for i in tqdm(range(reps)):
                ents[i], _ = simulate_1D_obc(runtime=runtime, L=L, p=p)

            outfile = open(directory + '/L' + str(L) + '_p' + str(p), 'wb')
            pickle.dump(ents, outfile)
            outfile.close()

            print("runtime: {}".format(time.time()-start))

    print("runtime total: {}".format(time.time()-start_tot))

    

    
def exp_1D_pbc(Ls, ps, reps, directory):
    """
    PBC experiment
    """

    start_tot = time.time()

    for L in Ls:
        for p in ps:
            print("L = ", L, " p = ", p)

            start = time.time()

            runtime = L
            ents = np.zeros((reps, runtime))

            for i in tqdm(range(reps)):
                ents[i], _ = simulate_1D_pbc(runtime=runtime, L=L, p=p)

            outfile = open(directory + '/L' + str(L) + '_p' + str(p), 'wb')
            pickle.dump(ents, outfile)
            outfile.close()

            print("runtime: {}".format(time.time()-start))

    print("runtime total: {}".format(time.time()-start_tot))

