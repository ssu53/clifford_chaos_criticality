import numpy as np
import pickle

# load all elements of C2
infile = open('allGates', 'rb')
allGates = pickle.load(infile)
infile.close()

allGates_phaseless = np.empty((len(allGates), 4, 4), dtype='int')
for gate_ind in range(allGates.shape[0]):
    allGates_phaseless[gate_ind] = allGates[gate_ind, [8,4,2,1], :-1].T

inds = set(range(len(allGates_phaseless)))
allGates_phaseless_unique = []

while len(inds) > 0:
    i = inds.pop()
    gate_unique = [i]
    inds_remaining = list(inds)
    for j in inds_remaining:
        if np.array_equal(allGates_phaseless[i], allGates_phaseless[j]):
            gate_unique.append(j)
            inds.remove(j)
    allGates_phaseless_unique.append(gate_unique)
    if len(allGates_phaseless_unique) % 50 == 0:
        print(len(allGates_phaseless_unique))
    
outfile = open('unique_phaseless_gates', 'wb')
pickle.dump(allGates_phaseless_unique, outfile)
