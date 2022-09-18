import numpy as np
from random import random
from numpy.random import choice
import time
from tqdm.auto import tqdm
import pickle

from routines_tab import allGates, Tab


class TabAnc(Tab):
    """
    phaseless tableau (with stabilisers only, no phase bit)
    with ancilla, which are dynamically introduced by CNOT coupling 
    """
    
    def __init__(self, L):
        self.L = L
        self.n = L # number of qubits including ancilla
        self.tab = np.concatenate((np.zeros((L,L), dtype=np.int8), np.identity(L, dtype=np.int8)), axis=1)
        self.coupled_anc_t = []
        self.coupled_anc_q = []
        
        # qubit i is located at index i
        # ancilla j tacked onto end, indexed n+j

    
    def xzAdjacent(self):
        """
        overwrites parent xzAdjacent
        permute tableau representation from [x1, x2, ..., z1, z2, ...] -> [x1, z1, x2, z2, ...]
        as well as placing the ancilla after the qubit, according to qubit index
        e.g. coupled_anc_q=[0,0,2,0,1] says that after the qubit columns, there are 5 ancilla
        corresponding to CNOT coupling with qubits 0,0,2,0,1 respectively
        this does NOT change the instance tab
        """
        
        reorder = np.array(list(range(self.L)) + self.coupled_anc_q).argsort() # order to group qubit with respective ancillae
        perm = []
        for i in reorder:
            perm.append(i)
            perm.append(i+self.n)
        return self.tab[:,perm] # permute by columns
    
    
    def entEntropyAnc(self, A):
        """
        entanglement entropy for ancilla system, where cuts are made with ancilla attached to respective qubits
        A are the sites
        """
        B = np.hstack((A, self.L + np.where([site in A for site in self.coupled_anc_q])[0]))
        return self.entEntropy(B)
    
    
    def couple(self, a, t):
        """
        CNOT couples qubit a to an ancilla, which augments the tableau by one site
        updates coupled_anc_t and coupled_anc_q with the time and site of coupling respectively
        """
        
        # augment tableau
        self.tab = np.insert(self.tab, self.n, np.zeros(self.n), 1)
        self.tab = np.insert(self.tab, 2*self.n+1, np.zeros(self.n), 1)
        self.tab = np.insert(self.tab, self.n, np.zeros(2*(self.n+1)), 0) # extra stab before row n
        self.n = self.n+1
        self.tab[self.n-1,2*self.n-1] = 1
        
        # CNOT with control - qubit a, target - ancilla at site n-1
        b = self.n-1
        self.tab[:,b] = (self.tab[:,b] + self.tab[:,a]) % 2
        self.tab[:,self.n+a] = (self.tab[:,self.n+a] + self.tab[:,self.n+b]) % 2
        
        self.coupled_anc_t.append(t)
        self.coupled_anc_q.append(a)
        
    
    def measureAnc(self, f=1):
        """
        measure the ancilla in Z basis, and discard measurement outcome
        measures a fraction f of the ancilla projectively
        """

        for i in range(self.L, self.n):
            if random() < f: self.measure(i)


                
def simulate_anc_pbc(runtime, L, p, verbose = False):
    """
    PBC
    CNOT ancilla with probability p, augmenting the original tableau when there is an ancilla
    no measurements
    """

    if L % 2 != 0: raise Exception('L even only')
    
    if verbose: start = time.time()
    
    state = TabAnc(L)
    
    for t in (tqdm(range(runtime)) if verbose else range(runtime)):
        
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
    
        for i in range(L):
            if random() < p: state.couple(i,t)
                
        for i in range(1, L, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, (i+1)%L)
            
        for i in range(L):
            if random() < p: state.couple(i,t)
        
    if verbose: print("runtime: {}".format(time.time() - start))
    
    return(state)



def simulate_anc_obc(runtime, L, p, verbose = False):
    """
    OBC
    CNOT ancilla with probability p, augmenting the original tableau when there is an ancilla
    no measurements
    """
    
    if verbose: start = time.time()
    
    state = TabAnc(L)
    
    for t in (tqdm(range(runtime)) if verbose else range(runtime)):
        
        for i in range(0, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
    
        for i in range(L):
            if random() < p: state.couple(i,t)
                
        for i in range(1, L-1, 2):
            state.twoQubitClif(allGates[choice(len(allGates))], i, i+1)
            
        for i in range(L):
            if random() < p: state.couple(i,t)
        
    if verbose: print("runtime: {}".format(time.time() - start))
    
    return(state)



def exp_anc_obc(Ls, ps, reps, directory):
    """
    OBC ancilla experiment
    this one takes a cut only in the middle
    """
    
    start_tot = time.time()
    
    for L in Ls:
        A = np.arange(L//2)
        for p in ps:
            print("L = ", L, " p = ", p)
            
            start = time.time()
            
            runtime = L            
            ent_pres = np.empty((reps,))
            ent_posts = np.empty((reps,))
            
            for i in tqdm(range(reps)):
                
                state = simulate_anc_obc(runtime=runtime, L=L, p=p)
                ent_pres[i] = state.entEntropyAnc(A)
                state.measureAnc()
                ent_posts[i] = state.entEntropyAnc(A)

            outfile = open(directory + '/L' + str(L) + '_p' + str(p), 'wb')
            pickle.dump(np.vstack((np.array(ent_pres), np.array(ent_posts))), outfile)
            outfile.close()
            
            print("runtime: {}".format(time.time()-start))
    
    print("runtime total: {}".format(time.time()-start_tot))
    
    
    
def exp_anc_pbc(Ls, ps, reps, directory):
    """
    PBC ancilla experiment
    takes a cut everywhere
    """
        
    start_tot = time.time()
    
    for L in Ls:
        if L % 2 != 0: raise Exception('L even only')
        for p in ps:
            print("L = ", L, " p = ", p)
            
            start = time.time()
            
            runtime = L            
            ent_pres = np.zeros((reps,))
            ent_posts = np.zeros((reps,))
            
            for i in tqdm(range(reps)):
                
                state = simulate_anc_pbc(runtime=runtime, L=L, p=p)
                
                ents = []
                for site in range(L//2):
                    ents.append(state.entEntropyAnc(np.arange(site, site+L//2)))
                ent_pres[i] = np.mean(ents)
                
                state.measureAnc()
                
                ents = []
                for site in range(L//2):
                    ents.append(state.entEntropyAnc(np.arange(site, site+L//2)))                     
                ent_posts[i] = np.mean(ents)
                
            outfile = open(directory + '/L' + str(L) + '_p' + str(p), 'wb')
            pickle.dump(np.vstack((ent_pres, ent_posts)), outfile)
            outfile.close()
            
            print("runtime: {}".format(time.time()-start))
    
    print("runtime total: {}".format(time.time()-start_tot))
    
