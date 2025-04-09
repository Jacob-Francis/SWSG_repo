import torch
import numpy as np
import matplotlib.pyplot as plt
from plotting_utils import plotting_one_step_swsg
from swsg_ot_algorithm import SWSGDynamcis
import matplotlib as mpl
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
import pickle
from time import perf_counter_ns
from geomloss import SamplesLoss
from tqdm import tqdm

# My python classes for loops we've defined so far
from swsg_ot_algorithm import SWSGDynamcis
from unbalancedsinkhorn import UnbalancedOT, DebiasedUOT
from utils import normal_pdf, jet_profile_initialisation, Sinkhorn_Divergence_balanced


####################################################################################
#               Main script
####################################################################################

# Are we using OT or S_e approx for the simulation?
ot_approx = True

cuda = 7
strength = 0.0001

global device
device = f'cuda:{cuda}'

for time, T in [(0,'T0'),(99,'T5'),(199,'T10')]:
    pre_fix='ot_approx'
    method = 'heun'


    ####################################################################################
    #            Main loop
    ####################################################################################

    s = []


    epsilons = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.015625/2, 0.015625/4]

    ### load in denser neigbout;
    i = 7
    epsilon = epsilons[i]
    N = 1/ epsilon**2
    n = int(np.sqrt(N))
    dt = 0.05
    print(epsilon, N, n, n**2)

    with open(f'data_store/OT_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
        data_fine = pickle.load(f)

    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    sigma_weights_dense /= sigma_weights_dense.sum()

    for i in range(len(epsilons)-1):
        epsilon = epsilons[i]
        N = 1/ epsilon**2
        n = int(np.sqrt(N))
        dt = 0.05
        print(epsilon, N, n, n**2)

            # Load in course data
        with open(f'data_store/OT_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
            data_course = pickle.load(f)

        _, _, _, sigma_weights, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)

        sigma_weights /= sigma_weights.sum()

        tic = perf_counter_ns()
        output, uotclass = Sinkhorn_Divergence_balanced(
                data_fine[0][:,:,time].to(device),
                sigma_weights_dense.to(device),
                data_course[0][:, :, time].to(device),
                sigma_weights.to(device),
                force_type="pykeops",
                tol=1e-10,
                epsilon=0.01,
                fullcompute=True,
                allow_annealing=False
            )  
        toc = perf_counter_ns()

        print('TIMING: ', tic-toc, output)
        s.append(output)

        with open(pre_fix+f'_sigma_{T}.pkl', 'wb') as f:
            pickle.dump(s, f)


            # plot the results
    # epsilons = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]

    ####################################################################################
    #            Height
    ####################################################################################
    # Load in fine data

    s1 = []

    i =7
    epsilon = epsilons[i]
    N = 1/ epsilon**2
    n = int(np.sqrt(N))
    dt = 0.05
    print(epsilon, N, n, n**2)
    method = 'heun'
    with open(f'data_store/OT_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
        data_fine = pickle.load(f)

    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    sigma_weights_dense /= sigma_weights_dense.sum()
    
    for i in range(len(epsilons)-1):
        epsilon = epsilons[i]
        N = 1/ epsilon**2
        n = int(np.sqrt(N))
        dt = 0.05
        print(epsilon, N, n, n**2)
        
        # Load in course data
        with open(f'data_store/OT_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
            data_course = pickle.load(f)

        X, _, _, sigma_weights, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)

        sigma_weights /= sigma_weights.sum()    
        
        tic = perf_counter_ns()
        output, uotclass = Sinkhorn_Divergence_balanced(
                X_dense.to(device),
                (data_fine[1][:,:,time]/data_fine[1][:,:,time].sum()).to(device).view(-1,1),
                X,
                (data_course[1][:, :, time]/data_course[1][:, :,time].sum()).to(device).view(-1,1),
                force_type="pykeops",
                tol=1e-10,
                epsilon=0.01,
                fullcompute=True,
                allow_annealing=False

            )  
        toc = perf_counter_ns()
        
        s1.append(output)
        with open(pre_fix+f'_heights_{T}.pkl', 'wb') as f:
            pickle.dump(s1, f)
    #  Look at solutions