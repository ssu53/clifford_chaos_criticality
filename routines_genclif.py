#----------------------------------------------------------------
# Generate from scratch, following Martinis reference
#----------------------------------------------------------------


import numpy as np


#----------------------------------------------------------------
# basic gates in unitary representation


paulix = np.matrix([[0,1],[1,0]])
pauliy = np.matrix([[0,-1j],[1j,0]])
pauliz = np.matrix([[1,0],[0,-1]])
identity = np.matrix([[1,0],[0,1]])

cz = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])

def xRot(theta):
    return np.matrix([[np.cos(theta/2), -1j*np.sin(theta/2)],[-1j*np.sin(theta/2), np.cos(theta/2)]])

def yRot(theta):
    return np.matrix([[np.cos(theta/2), -np.sin(theta/2)],[np.sin(theta/2), np.cos(theta/2)]])

x2p = xRot(+np.pi/2)
y2p = yRot(+np.pi/2)
x2m = xRot(-np.pi/2)
y2m = yRot(-np.pi/2)


#----------------------------------------------------------------
# gate characterisation
#----------------------------------------------------------------

def apply(U, N, compare=None):
    """
    applies unitary gate U to state described by operator N
    """
    result = U @ N @ U.H 
    result = result.round(decimals=10)
    if compare is not None:
        print(np.allclose(result, compare))
    return result


def check(U, N, compare):
    """
    returns True if unitary gate U applied to state described by operator N yields the operator state compare
    """
    result = U @ N @ U.H 
    return np.allclose(U @ N @ U.H, compare)

 
def characterise(gate, verbose=False):
    """
    characterises gate by its action on X,Y,Z states and returns the transformation
    in the form of a 3x3 binary array whose rows are the actions on X,Y,X respectively
    X=100, Y=110, Z=010, -X=101, -Y=111, -Z=011
    """
    
    gens = [paulix, pauliy, pauliz, -paulix, -pauliy, -pauliz]
    names = ['X', 'Y', 'Z', '-X', '-Y', '-Z']
    tab_rep = [[1,0,0], [1,1,0], [0,1,0], [1,0,1], [1,1,1], [0,1,1]]
    
    trans = np.zeros((3,3), dtype='int')
    
    out_gen = np.identity(2)
    out_name = -1
    
    for i in range(len(gens)):
        if check(gate, paulix, gens[i]):
            out_gen = gens[i]
            out_name = names[i]
            trans[0,:] = tab_rep[i]
            break
    if verbose: print('X -> ', out_name)
    
    for i in range(len(gens)):
        if check(gate, pauliy, gens[i]):
            out_gen = gens[i]
            out_name = names[i]
            trans[1,:] = tab_rep[i]
            break
    if verbose: print('Y -> ', out_name)
    
    for i in range(len(gens)):
        if check(gate, pauliz, gens[i]):
            out_gen = gens[i]
            out_name = names[i]
            trans[2,:] = tab_rep[i]
            break
    if verbose: print('Z -> ', out_name)
    
#     print(trans)
#     # verify that Y is formed from linear combination of X,Z, modulo the nonlinear phase transform
#     if np.sum((trans[0,0:2] + trans[2,0:2] - trans[1,0:2]) % 2) != 0:
#         raise Exception()
        
    return trans


#----------------------------------------------------------------
# single qubit Cliffords
#----------------------------------------------------------------

C1_names = ['I', 'X', 'Y', 'Y, X',
            'X2, Y2', 'X2, -Y2', '-X2, Y2', '-X2, -Y2', 
            'Y2, X2', 'Y2, -X2', '-Y2, X2', '-Y2, -X2',
            'X2', '-X2', 'Y2', '-Y2',
            '-X2, Y2, X2', '-X2, -Y2, X2',
            'X, Y2', 'X, -Y2', 'Y, X2', 'Y, -X2',
            'X2, Y2, X2', '-X2, Y2, -X2']

C1_gens = [identity, paulix, pauliy, paulix @ pauliy,
               y2p @ x2p, y2m @ x2p, y2p @ x2m, y2m @ x2m,
               x2p @ y2p, x2m @ y2p, x2p @ y2m, x2m @ y2m,
               x2p, x2m, y2p, y2m,
               x2p @ y2p @ x2m, x2p @ y2m @ x2m,
               y2p @ paulix, y2m @ paulix, x2p @ pauliy, x2m @ pauliy,
               x2p @ y2p @ x2p, x2m @ y2p @ x2m]

C1_trans = [None] * len(C1_names)

S1_names = ['I', 'Y2, X2', '-X2, -Y2']
S1_ind = [C1_names.index(name) for name in S1_names]

S1x_names = ['X2', 'X2, Y2, X2', '-Y2']
S1x_ind = [C1_names.index(name) for name in S1x_names]

S1y_names = ['Y2', 'Y, X2', '-X2, -Y2, X2']
S1y_ind = [C1_names.index(name) for name in S1y_names]


#----------------------------------------------------------------
# action of gates in GF2 representation
#----------------------------------------------------------------

def allGens():
    tab = [[0,0,0,0],
           [0,0,0,1],
           [0,0,1,0],
           [0,0,1,1],
           [0,1,0,0],
           [0,1,0,1],
           [0,1,1,0],
           [0,1,1,1],
           [1,0,0,0],
           [1,0,0,1],
           [1,0,1,0],
           [1,0,1,1],
           [1,1,0,0],
           [1,1,0,1],
           [1,1,1,0],
           [1,1,1,1]]
    tab = np.append(np.array(tab), np.zeros([16,1], dtype=int), 1)
    return tab

def CZ(tab, a, b):
    """
    applies phaseful control-Z gate to a phaseful binary tableau
    control a, target b
    """
    n = np.shape(tab)[1] // 2
    xa = a
    xb = b
    za = n+a
    zb = n+b
    r = 2*n
    tab[:,za] = (tab[:,za] + tab[:,xb]) % 2
    tab[:,zb] = (tab[:,zb] + tab[:,xa]) % 2
    tab[:,r] = (tab[:,r] + tab[:,xa] * tab[:,xb] * (tab[:,za] + tab[:,zb])) % 2
    

verbose = False # suppress printing
for i in range(len(C1_gens)):
    """
    populate C1_trans, the binary transformations of each Clifford 1-qubit
    """
    if verbose: print(C1_names[i])
    C1_trans[i] = characterise(C1_gens[i], verbose=verbose)
    if verbose: print()
    
def C1(tab, a, gate_ind):
    """
    applies phaseful C1 gate, indexed by gate_ind, to qubit a in the GF2 tableau tab
    """
    n = np.shape(tab)[1] // 2
    xa = a
    za = n+a
    xs = np.copy(tab[:,xa])
    zs = np.copy(tab[:,za])
    r = 2*n
    trans = C1_trans[gate_ind]
    tab[:,r] = (tab[:,r] + 
                xs * ((zs + 1) % 2) * trans[0,2] + 
                xs * zs * trans[1,2] + 
                ((xs + 1) % 2) * zs * trans[2,2]) % 2
    tab[:,xa] = (xs * trans[0,0] + zs * trans[2,0]) % 2
    tab[:,za] = (xs * trans[0,1] + zs * trans[2,1]) % 2
    
    
#----------------------------------------------------------------
# converting between GF2 and unitary representations of a gate
#----------------------------------------------------------------


def stabBinToUni_1(state):
    """
    state is a 2-element array of two bits 00->I, 10->X, 01->Z, 11->Y
    returns the 2x2 unitary
    """
    if np.array_equal(state, [0,0]):
        return np.identity(2)
    elif np.array_equal(state, [1,0]):
        return paulix
    elif np.array_equal(state, [0,1]):
        return pauliz
    elif np.array_equal(state, [1,1]):
        return pauliy
    else:
        raise Exception('invalid state')


def stabBinToUni_2(row):
    """
    row is a 5-element array of two qubits and overall phase
    returns the 4x4 unitary
    """
    a_state = stabBinToUni_1(row[[0,2]])
    b_state = stabBinToUni_1(row[[1,3]])
    phase = (-1)**row[4]
    return phase * np.kron(a_state, b_state)


def decompToUni(decomp):
    """
    recover the 4x4 unitary representation of a Clifford two-qubit gate from its indices into C1_gens
    note the order of gate concatenation: the first applied gate is RIGHTMOST in the matrix multiplication
    like function composition
    """
    if decomp[0] == 'single':
        return np.kron(C1_gens[decomp[1]], C1_gens[decomp[2]])
    elif decomp[0] == 'cnot':
        return np.kron(C1_gens[decomp[3]], C1_gens[decomp[4]]) @ cz @ \
        np.kron(C1_gens[decomp[1]], C1_gens[decomp[2]]) 
    elif decomp[0] == 'iswap':
        return np.kron(C1_gens[decomp[3]], C1_gens[decomp[4]]) @ cz @ \
               np.kron(C1_gens[C1_names.index('Y2')], C1_gens[C1_names.index('-X2')]) @ cz @ \
               np.kron(C1_gens[decomp[1]], C1_gens[decomp[2]])   
    elif decomp[0] == 'swap':
        return np.kron(np.identity(2), C1_gens[C1_names.index('Y2')]) @ cz @ \
               np.kron(C1_gens[C1_names.index('Y2')], C1_gens[C1_names.index('-Y2')]) @ cz @ \
               np.kron(C1_gens[C1_names.index('-Y2')], C1_gens[C1_names.index('Y2')]) @ cz @ \
               np.kron(C1_gens[decomp[1]], C1_gens[decomp[2]])
    else:
        raise Exception()