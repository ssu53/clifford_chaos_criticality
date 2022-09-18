from routines_anc import exp_anc_obc, exp_anc_pbc

import sys
directory = sys.argv[1]
bc = sys.argv[2]
reps = int(sys.argv[3])

#Ls = [8,12,16,20,24]
#ps = [p]

Ls = [12]
ps = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

if bc == 'obc':
    exp_anc_obc(Ls=Ls, ps=ps, reps=reps, directory=directory)
elif bc == 'pbc':
    exp_anc_pbc(Ls=Ls, ps=ps, reps=reps, directory=directory)
