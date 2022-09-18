from routines_1D import exp_1D_obc, exp_1D_pbc

import sys
directory = sys.argv[1]
bc = sys.argv[2]
reps = int(sys.argv[3])

# Ls = [16, 24, 32, 40, 48, 64, 128]
# ps = [0.17]

Ls = [12]
ps = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

if bc == 'obc':
    exp_1D_obc(Ls=Ls, ps=ps, reps=reps, directory=directory)
elif bc == 'pbc':
    exp_1D_pbc(Ls=Ls, ps=ps, reps=reps, directory=directory)
