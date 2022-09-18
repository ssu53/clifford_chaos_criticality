"""
routines for stabiliser states in Z2 tableau representation, following Aaronson-Gottesman algorithms
"""

import numpy as np
import pandas as pd
import pickle

#----------------------------------------------------------------------------------------

# load all elemnts of C2
infile = open('allGates', 'rb')
allGates = pickle.load(infile)
infile.close()

#----------------------------------------------------------------------------------------

class Tab:
    """
    phaseless tableau (with stabilisers only, no phase bit)
    """
    
    
    def __init__(self, n=None, tab=None):
        """
        when given n, initialises standard tableau for n-site system, stabilised by Z at each site
        when given tab, initialises tableau in stabiliser state given by the array
        """
        
        if n is not None:
            self.n = n
            self.tab = np.concatenate((np.zeros((n,n), dtype=np.int8), np.identity(n, dtype=np.int8)), axis=1)
        elif tab is not None:
            self.n = tab.shape[1]//2
            self.tab = tab
        else:
            raise Exception("invalid input")
            
    
    def twoQubitClif(self, gate, a, b):
        """
        applies two-qubit Clifford gate to stabiliser tableau, where a is the top qubit and b the bottom qubit
        following the definitions suggested by Martinis group's supplementary info for 
        “Superconducting quantum circuits at the surface code threshold for fault tolerance”
        """
        
        matches = self.tab[:,self.n+b] + 2*self.tab[:,self.n+a] + 4*self.tab[:,b] + 8*self.tab[:,a]
        self.tab[:,a] = gate[matches,0] # xa
        self.tab[:,b] = gate[matches,1] # xb
        self.tab[:,self.n+a] = gate[matches,2] # za
        self.tab[:,self.n+b] = gate[matches,3] # zb
    
    
    def measure(self, a):
        """
        measures qubit a in Z basis
        measurement outcome is not tracked (not maintaining phase bit)
        """
        
        # seek p such that x_{pa} = 1
        matches = np.where(self.tab[:,a] == 1)[0]
        
        if len(matches) != 0:
            # case I: random outcome
            p = matches[0]
            for i in range(self.n):
                if (i != p) and (self.tab[i,a] == 1):
                    self.tab[i,:] = (self.tab[p,:] + self.tab[i,:]) % 2 # this is the phase-indifferent rowsum operation
            self.tab[p,:] = 0
            self.tab[p, self.n+a] = 1

        # else is case II: deterministic outcome, no changes to tableau

        
    def entEntropy(self, A):
        """
        calculates entanglement entropy
        does NOT modify tab
        A is the qubit index. In ancilla model, indices for the corresponding ancilla should be enumerated here.
        """

        A_comp = [j for j in np.arange(self.n) if j not in A]

        if len(A_comp) == 0: # pure state
            return 0

        # slicing creates a new matrix tab_red, which is modified in-place by gf2RowRed, but does not affect tab 
        tab_red = self.tab[:, np.concatenate((np.array(A_comp), np.array(A_comp) + self.n))]

        # extract rank from row reduction
        nonzero_row = np.any(gf2RowRed(tab_red), axis=1)
        rbarA = np.sum(nonzero_row)

        rA = self.n - rbarA
        S_ent = len(A) - rA

        if S_ent != (rbarA - len(A_comp)):
            print("rbarA - len(A_comp)", rbarA - len(A_comp))
            raise Exception("inconsistent")

        return S_ent
    
    
    def xzAdjacent(self):
        """
        permute tableau representation from [x1, x2, ..., z1, z2, ...] -> [x1, z1, x2, z2, ...]
        this does NOT change the instance tab
        """
        
        perm = []
        for i in range(self.n):
            perm.append(i)
            perm.append(i+self.n)
        return self.tab[:,perm]
    
    
    def bellpair(self, a, b):
        """
        creates a Bell pair between spins at a and b, by hadamard (on a) and cnot (with a control, b target)
        assumes a and b are each initially deterministic in the local Z basis (i.e. 0 or 1)
        """
        
        # hadamard: X -> Z, Z -> X
        self.tab[:,[a,a+self.n]] = self.tab[:,[a+self.n,a]] 
        
        # cnot: XI -> XX, IX -> IX, ZI -> ZI, IZ -> ZZ
        self.tab[:,b] = (self.tab[:,b] + self.tab[:,a]) % 2
        self.tab[:,self.n+a] = (self.tab[:,self.n+a] + self.tab[:,self.n+b]) % 2
        
        
    def sym_inner_prod(self,h,i):
        """
        two generators commute iff their symplectic inner product is 0; anticommute iff it's 1
        returns the symplectic inner product of two rows in the tableau
        """
        xhs = self.tab[h,:self.n]
        zhs = self.tab[h,self.n:2*self.n]
        xis = self.tab[i,:self.n]
        zis = self.tab[i,self.n:2*self.n]
        return (np.sum(xhs * zis) + np.sum(xis * zhs)) % 2

    
    def check_invar(self):
        """
        check tableau commutation invariants for pure state evolution
        note: cannot check the full rank condition on this stabilisers-only tableau
        """
        
        com_invar = True
        # all stabilisers commute
        for a in range(self.n):
            for b in range(self.n):
                sip = self.sym_inner_prod(a,b)
                if (b == a + self.n):
                    if (sip % 2) != 1:
                        print(a, b, self.sym_inner_prod(a,b))
                        com_invar = False
                else:
                    if (sip % 2) != 0:
                        print(a, b, self.sym_inner_prod(a,b))
                        com_invar = False
        if com_invar == False:
            raise Exception("failed: commutation invariants")

        return com_invar

    
    def style(self):
        return pd.DataFrame(self.tab).style.apply(highlight_ones)    


#----------------------------------------------------------------------------------------

    
class Tab_clipped:
    """
    phaseless tableau (with stabilisers only, no phase bit)
    put in clipped gauge
    """
    
    def __init__(self, tab):
        """
        initialises the tableau from array representation
        tab should be in xz-adjacent form
        """
        
        self.m, self.n = tab.shape
        self.tab = Tab_clipped.clippedgauge(tab)
        self.l_ends = np.full((self.m,), -1)
        self.r_ends = np.full((self.m,), -1)
        self.getEndpoints()
    
    
    @staticmethod
    def clippedgauge(M):
        """
        M is the xz-adjacent matrix representation of a tableau with stabilisers only and no phase (instance of Tab)
        row reduce from left (gf2RowRed), and then from right combining with only shorter stabilisers, such that
        for every site x, the sum of the number of left endpoints and rightendpoints is 2 in this gauge
        modifies M in-place!
        """
        
        # deep copy would make it not modify in-place, however, for all applications, tab is first permuted by 
        # some xzAdjacent which makes new tableau
        #M = np.copy(M) 
        m, n = M.shape

        # from left
        i = 0
        j = 0
        while (i<m) and (j<n):

            # Find value of largest element in the remainder of column j.
            ones = np.where(M[i:m,j] == 1)[0]
            if len(ones) < 1:
                j += 1
                continue
            k = ones[0] + i

            # Swap ith and kth rows.
            M[[i,k],:] = M[[k,i],:]

            # Save the right hand side of the pivot row
            aijn = M[i,j:n]

            # Present column
            col = np.copy(M[:,j])

            # Never Xor the pivot row against itself
            col[i] = 0

            # Build matrix of bits to flip
            flip = np.outer(col, aijn)

            # Xor the right hand side of the pivot row with all the other rows
            M[:,j:n] = (M[:,j:n] + flip) % 2

            i += 1 
            j += 1

        # from right
        j = n-1
        done = [] # 'done' rows i.e. stabilisers whose right end is far along and should not be composed with others
        while (j>0):
            # find the lowest row with a 1 in this position
            ones = np.where(M[:,j] == 1)[0]
            ones = [one for one in ones if one not in done]

            if len(ones) == 0:
                j -= 1
                continue

            k = ones[-1]
            if len(ones) == 1:
                done.append(k)
                j -= 1
                continue
            done.append(k)

            # shorten other rows by composing with this short stabiliser in row k
            for row in ones[:-1]:
                M[row,:] = (M[row,:] + M[k,:]) % 2

            j -= 1

            # if all rows are done, break
            if len(done) == m:
                break

        return M
    
    
    def check_clippedgauge(self):
        """
        M is the xz-adjacent matrix representation of a tableau with stabilisers only and no phase (instance of Tab)
        returns true if the M satisfies the clipped gauge condition, else throws exception
        """
        
        if (self.n != 2*self.m):
            raise Exception('incorrect shape')
        
        ends = np.zeros((self.m,), dtype=np.int8)
        for row in range(self.m):
            ones = np.where(self.tab[row,:] == 1)[0]
            if len(ones) < 1: continue
            ends[ones[0] // 2] += 1
            ends[ones[-1] // 2] += 1

        if not all(ends == 2):
            raise Exception('fails clipped gauge condition')
        return True
    
    
    def getEndpoints(self):
        """
        for non ancilla
        returns left and right endpoints for an abbreviated tableau M (no destabilisers or phase) in clipped gauge
        also checks that endpoints satisfies clipped gauge, throwing exception if not
        this function's output is useful for entEntropyclipped
        """

        for row in range(self.m):
            ones = np.where(self.tab[row,:] == 1)[0]
            if len(ones) < 1:
                continue
            self.l_ends[row] = ones[0] // 2
            self.r_ends[row] = ones[-1] // 2

        ends = np.append(self.l_ends, self.r_ends)

        # l_ends and r_ends are initalised to -1, meaning that the stabiliser is trivial (zero row)
        n_zerorows = np.count_nonzero(self.l_ends < 0)
        if (n_zerorows > 0):
            print("there are trivial stabilisers in the tableau")
            return

        # check satisfies gauge condition
        for site in range(self.m):
            if (np.count_nonzero(ends == site) != 2):
                print(site)
                raise Exception('fails clipped gauge condition')


    def entEntropy(self, site):
        """
        calculates the entanglement entropy given left and right endpoints in clipped gauge
        with a cut after qubit at site
        this is linear time, unlike row reduction complexity for non clipped gauge Tab
        """
        if ((site < 0) or (site >= len(self.l_ends))):
            raise Exception("invalid site")
        return(np.sum((self.l_ends <= site) * (self.r_ends > site))//2)

    

#----------------------------------------------------------------------------------------


class Tab_phaseful:
    """
    phaseful tableau (with stabiliser and destabiliser rows, phase bit)
    """
    
    
    def __init__(self, n=None, tab=None):
        """
        when given n, initialises standard tableau, stabilised by Z at each site
        when given tab, initialises tableau in state given by the array
        """
        
        if n is not None:
            self.n = n
            self.tab = np.concatenate((np.identity(n=2*n, dtype=np.int8), np.zeros(shape=(2*n,1), dtype=np.int8)), axis=1)
        elif tab is not None:
            self.n = np.shape(tab)[1] // 2
            self.tab = tab
        else:
            raise Exception("invalid input")

            
    def twoQubitClif(self, gate, a, b):
        """
        applies two-qubit Clifford gate to stabiliser tableau tab, where a is the top qubit and b the bottom qubit
        following the definitions suggested by Martinis group's supplementary info for 
        “Superconducting quantum circuits at the surface code threshold for fault tolerance”
        """
        
        if (a < 0) or (b < 0) or (a >= self.n) or (b >= self.n):
            raise Exception('invalid site index')
        
        r = 2*self.n
        matches = self.tab[:,self.n+b] + 2*self.tab[:,self.n+a] + 4*self.tab[:,b] + 8*self.tab[:,a]
        self.tab[:,a] = gate[matches,0] # xa
        self.tab[:,b] = gate[matches,1] # xb
        self.tab[:,self.n+a] = gate[matches,2] # za
        self.tab[:,self.n+b] = gate[matches,3] # zb
        self.tab[:,r] = (self.tab[:,r] + gate[matches,4]) % 2 # phase

        
    def measure(self, a):
        """
        measures qubit a in Z basis, and returns measurement outcome
        """
        
        if (a < 0) or (a >= self.n):
            raise Exception('invalid site index')

        r = 2*self.n

        # seek p in {n+1, ..., 2n} such that x_{pa} = 1
        matches = np.where(self.tab[self.n:2*self.n,a] == 1)[0] + self.n

        if len(matches) != 0:
            # case I: random outcome
            p = matches[0]
    #         print("p = ", p)
            for i in range(0,2*self.n):
                if (i != p) and (i + self.n != p) and (self.tab[i,a] == 1):
                    self.rowsum(i,p)
            self.tab[p-self.n,:] = self.tab[p,:]
            self.tab[p,:] = 0
            self.tab[p, r] = (np.random.uniform() > 0.5) * 1
            self.tab[p, self.n+a] = 1
            return self.tab[p, r]
        else:
            # case II: deterministic outcome
            tab_scratch = Tab_phaseful(tab=np.append(self.tab, [np.zeros(2*self.n+1, dtype=np.int8)], axis=0)) # this is probably costly, bc append cannot happen in place
            for i in range(self.n):
                if (tab_scratch.tab[i,a] == 1):
                    tab_scratch.rowsum(2*self.n, i+self.n)
            return tab_scratch.tab[2*self.n,r] # and tab is unchanged in this case, since the outcome is determinate
    
    
    def measure_tossResult(tab, a):
        """
        measures qubit a in Z basis, discarding measurement outcome (for some efficiency boost)
        """
        
        if (a < 0) or (a >= self.n):
            raise Exception('invalid site index')
        
        # seek p in {n+1, ..., 2n} such that x_{pa} = 1
        matches = np.where(tab[self.n:2*self.n,a] == 1)[0] + self.n

        if len(matches) != 0:
            # case I: random outcome
            p = matches[0]
            for i in range(0,2*n):
                if (i != p) and (i + n != p) and (tab[i,a] == 1):
                    self.rowsum(i,p)
            self.tab[p-self.n,:] = tab[p,:]
            self.tab[p,:] = 0
            self.tab[p, self.n+a] = 1
        # else is case II: deterministic outcome, no changes to tableau
    
    
    def entEntropy(self, A):
        """
        calculates entanglement entropy
        does NOT modify tab
        A is the qubit index. In ancilla model, indices for the corresponding ancilla should be enumerated here.
        """

        A_comp = [j for j in np.arange(self.n) if j not in A]

        # slicing creates a new matrix tab_red, which is modified in-place by gf2RowRed, but does not affect tab 
        tab_red = self.tab[self.n:2*self.n, np.concatenate((np.array(A_comp), np.array(A_comp) + self.n))]

        # extract rank from row reduction
        nonzero_row = np.any(gf2RowRed(tab_red), axis=1)
        rbarA = np.sum(nonzero_row)

        rA = self.n - rbarA
        S_ent = len(A) - rA

        if S_ent != (rbarA - len(A_comp)):
            print("rbarA - len(A_comp)", rbarA - len(A_comp))
            raise Exception("inconsistent")

        return S_ent
    
    @staticmethod
    def gfunc(x1, z1, x2, z2):
        """
        g function, as defined by Aaronson-Gottesman
        """

        if ((x1 != 0) and (x1 != 1)) or ((z1 != 0) and (z1 != 1)) or ((x2 != 0) and (x2 != 1)) or ((z2 != 0) and (z2 != 1)):
            raise Exception('invalid inputs')
        if (x1 == 0) and (z1 == 0):
            return 0
        if (x1 == 1) and (z1 == 1):
            return (z2 - x2)
        if (x1 == 1) and (z1 == 0):
            return z2*(2*x2-1)
        if (x1 == 0) and (z1 == 1):
            return x2*(1-2*z2)

        
    def getDet(self, h, i):
        """
        returns 'determinant', i.e.
        twice phase bit of the product of generators h and i of the tableau is determinant modulo 4
        """
        
        r = 2*self.n
        xijs = self.tab[i,:self.n]
        zijs = self.tab[i,self.n:2*self.n]
        xhjs = self.tab[h,:self.n]
        zhjs = self.tab[h,self.n:2*self.n]
        return 2*self.tab[h,r] + 2*self.tab[i,r] + np.sum(list(map(Tab_phaseful.gfunc, xijs, zijs, xhjs, zhjs)))
  

    def rowsum(self, h, i):
        """
        rowsums, maintaining tableau invariant
        """

        r = 2*self.n
        det = self.getDet(h, i) % 4
        if (det == 0):
            self.tab[h,r] = 0
        elif (det == 2):
            self.tab[h,r] = 1
        else:
            raise Exception('bad: det is not equal to 0 or 2 mod 4')
        self.tab[h,:-1] = (self.tab[i,:-1] + self.tab[h,:-1]) % 2
       
    
    def sym_inner_prod(self,h,i):
        """
        two generators commute iff their symplectic inner product is 0; anticommute iff it's 1
        returns the symplectic inner product of two rows in the tableau
        """
        xhs = self.tab[h,:self.n]
        zhs = self.tab[h,self.n:2*self.n]
        xis = self.tab[i,:self.n]
        zis = self.tab[i,self.n:2*self.n]
        return (np.sum(xhs * zis) + np.sum(xis * zhs)) % 2

    
    def check_invar(self):
        """
        checks full rank and commutation invariants for pure state evolution
        """
        
        # necessary condition: full rank
        tab_copy = np.copy(self.tab)
        nonzero_row = np.any(gf2RowRed(tab_copy[:,:-1]), axis=1)
        if 2*self.n != np.sum(nonzero_row):
            #print(2*self.n, np.sum(nonzero_row))
            raise Exception("failed: matrix rank")

        com_invar = True
        # everything comutes, except the stabiliser and its corresponding destabiliser anticommute
        for a in range(0,2*self.n):
            for b in range(a, 2*self.n):
                sip = self.sym_inner_prod(a,b)
                if (b == a + self.n):
                    if (sip % 2) != 1:
                        print(a, b, self.sym_inner_prod(a,b))
                        com_invar = False
                else:
                    if (sip % 2) != 0:
                        print(a, b, self.sym_inner_prod(a,b))
                        com_invar = False
        if com_invar == False:
            raise Exception("failed: commutation invariants")

        return com_invar


    def style(self):
        return pd.DataFrame(self.tab).style.apply(highlight_ones)


#----------------------------------------------------------------------------------------


def highlight_ones(arr, color='darkorange'):
    '''
    highlight the ones in the array darkorange.
    '''
    is_one = arr == 1
    return ['background-color: darkorange' if v else '' for v in is_one]

#----------------------------------------------------------------------------------------


def gf2RowRed(M):
    """
    row reduction in GF2
    modifies M in-place!
    adapted from: https://gist.github.com/esromneb/652fed46ae328b17e104
    """
    m, n = M.shape
    i = 0
    j = 0
    while (i<m) and (j<n):
#         print(i, j)
#         print(M)

        # Find value of largest element in the remainder of column j.
        ones = np.where(M[i:m,j] == 1)[0]
        if len(ones) < 1:
            j += 1
            continue
        k = ones[0] + i
#         print("k", k)

        # Swap ith and kth rows.
        M[[i,k],:] = M[[k,i],:]

        # Save the right hand side of the pivot row
        aijn = M[i,j:n]
#         print('aijn', aijn)

        # Present column
        col = np.copy(M[:,j])
#         print('col', col)

        # Never Xor the pivot row against itself
        col[i] = 0

        # Build matrix of bits to flip
        flip = np.outer(col, aijn)
#         print('flip', flip)

        # Xor the right hand side of the pivot row with all the other rows
        M[:,j:n] = (M[:,j:n] + flip) % 2

        i += 1 
        j += 1
#         print()
    return M



# def gf2Rank(rows):
#     """
#     ------------------------------------------------------------------------
#     !!!!!!! SOMETHING IS WRONG HERE - DO NOT USE THIS !!!!!!!
#     ------------------------------------------------------------------------
    
#     Find rank of a matrix over GF2.

#     The rows of the matrix are given as nonnegative integers, thought
#     of as bit-strings.

#     This function modifies the input list. Use gf2_rank(rows.copy())
#     instead of gf2_rank(rows) to avoid modifying rows.
#     """
#     rank = 0
#     while rows:
#         pivot_row = rows.pop()
#         if pivot_row:
#             rank += 1
#             lsb = pivot_row & -pivot_row
#             for index, row in enumerate(rows):
#                 if row & lsb:
#                     rows[index] = row ^ pivot_row
#     return rank



# def gf2RowRep(M):
#     intRep = np.zeros(M.shape[0], dtype=int)
#     for i in range(len(intRep)):
#         j = 0
#         for bit in M[i]:
#             j = (j<<1) | bit
#         intRep[i] = j
#     return intRep



# # https://stackoverflow.com/questions/49287398/finding-null-space-of-binary-matrix-in-python
# # binary_rr
# # faster, the two row-reductions are different up to a non-unique row-echelon form
# def gf2RowRed_new(m):
#     """
#     ------------------------------------------------------------------------
#     !!!!!!! SOMETHING IS WRONG HERE - DO NOT USE THIS !!!!!!!
#     ------------------------------------------------------------------------
#     """
#     rows, cols = m.shape
#     l = 0
#     for k in range(min(rows, cols)):
#         if l >= cols: break
#         # Swap with pivot if m[k,l] is 0
#         if m[k,l] == 0:
#             found_pivot = False
#             while not found_pivot:
#                 if l >= cols: break
#                 for i in range(k+1, rows):
#                     if m[i,l]:
#                         m[[i,k]] = m[[k,i]]  # Swap rows
#                         found_pivot = True
#                         break

#                 if not found_pivot: l += 1

#         if l >= cols: break  # No more cols

#         # For rows below pivot, subtract row
#         for i in range(k+1, rows):
#             if m[i,l]: m[i] ^= m[k]
                
#         l += 1
        
#     return m
