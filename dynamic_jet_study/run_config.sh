#!/bin/bash

#nohup python3 sim_study.py 4 0.02 0 3 > methodcompare_0.out 2> methodcompare_0.err &
#nohup python3 sim_study.py 1 0.02 0.0001 3 > methodcompare_mid.out 2> methodcompare_mid.err &

#nohup python3 sim_study.py 1 0.01 0 1 > mid100_0.out 2> mid100_0.err &
#nohup python3 sim_study.py 6 0.01 0.0001 1 > mid100_strong.out 2> mid100_strong.err &
#nohup python3 sim_study.py 5 0.01 0.00005 1 > mid100_mid.out 2> mid100_mis.err &

nohup python3 sim_study.py 3 0.0025 0.00005 1 > hifi.out 2> hifi.err &

# nohup python3 sim_study.py 5 0.005 0 > nolloydmid.out 2> nolloydmid.err
# parser.add_argument("cuda", type=int, help="cuda index")
# parser.add_argument("epsilon", type=float, help="size of epsilon")
# parser.add_argument("strength", type=float, help="strength of perturbation")

