import pickle
import sys
import time

from routines_probe import exp_probe2pt

directory = sys.argv[1]
L = int(sys.argv[2])
runtime = int(sys.argv[3]) * L
reps = int(sys.argv[4])
name_ext = sys.argv[5]

ps = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]

start = time.time()

for p in ps:
    ent = exp_probe2pt(L=L, runtime=runtime, p=p, reps=reps)
    outfile = open(directory + '/L' + str(L) + '_p' + str(p) + '_' + name_ext, 'wb')
    pickle.dump(ent, outfile)
    outfile.close()
    
print(time.time()-start)
