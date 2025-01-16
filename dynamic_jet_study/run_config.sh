#!/bin/bash

# ReRuns: 15/01/2025
# nohup python3 sim_study.py 0 0.02 > smaller.out 2> smaller.err &

nohup python3 sim_study.py 2 0.02 '' > nolloydsmaller.out 2> nolloydsmaller.err
nohup python3 sim_study.py 2 0.01 '' > nolloydsmall.out 2> nolloydsmall.err
nohup python3 sim_study.py 2 0.005 '' > nolloydmid.out 2> nolloydmid.err

nohup python3 sim_study.py 2 0.01 'lloyd' > small.out 2> small.err
nohup python3 sim_study.py 2 0.005 'lloyd' > mid.out 2> mid.err



