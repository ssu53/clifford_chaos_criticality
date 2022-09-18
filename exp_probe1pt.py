import pickle
import sys
import time

from routines_probe import exp_probe1pt

directory = sys.argv[1]
L = int(sys.argv[2])
reps = int(sys.argv[3])
name_ext = sys.argv[4]

ps = [0.05, 0.1, 0.15, 0.2, 0.25]

start = time.time()

for p in ps:
    refAs = exp_probe1pt(L=L, p=p, runtime_after=L, reps=reps)
    outfile = open(directory + '/L' + str(L) + '_p' + str(p) + '_' + name_ext, 'wb')
    pickle.dump(refAs, outfile)
    outfile.close()
    
print(time.time()-start)