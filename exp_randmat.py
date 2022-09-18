import numpy as np
import pickle
from tqdm import tqdm
import time
from numpy.fft import fft

from routines_tab import gf2RowRed, Tab_clipped

import sys
directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
max_t = int(sys.argv[4])*L

Us = []
count = 0
start = time.time()
while len(Us) < reps:
    rand = np.random.choice([0, 1], size=(2*L,2*L))
    if np.any(np.sum(gf2RowRed(rand.copy()), axis=1) == 0):
        count += 1
        continue
    Us.append(rand)
print(time.time()-start)
print(count)



n_evecs = np.empty((len(Us), max_t))
for i in tqdm(range(len(Us))):
    for t in range(1,max_t+1):
        gate = np.linalg.matrix_power(Us[i], t)
        eig1 = (gate - np.eye(2*L, dtype='int8')) % 2
        n_evecs[i, t-1] = np.sum(np.sum(Tab_clipped.clippedgauge(eig1.copy()), axis=1) == 0)
        
        
outfile = open(directory + '/L' + str(L) + '_evol' + str(max_t//L), 'wb')
pickle.dump(n_evecs, outfile)
outfile.close()