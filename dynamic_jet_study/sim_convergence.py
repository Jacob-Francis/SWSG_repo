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
from SWSG_repo.dynamic_jet_study.localutils import normal_pdf, jet_profile_initialisation, Sinkhorn_Divergence_balanced


####################################################################################
#               Main script
####################################################################################
cuda = 1
strength = 0.0001

global device
device = f'cuda:{cuda}'
####################################################################################
#              load in fine data
####################################################################################


####################################################################################
#             Run or load symmetic potential calculation
####################################################################################


for time, T in [(0, 'T0'), (1414-1, 'T5')]:
    prefix = 'finer_full_'
    method = 'heun'


####################################################################################
#            Main loop
####################################################################################

    s = []
    epsilons=[]

    ### load in denser neigbout;
    i = 7 
    epsilon = (1/25) * np.sqrt(2)**(-i)
    N = 1/ epsilon**2
    n = int(np.sqrt(N))
    dt = (1/25) * np.sqrt(2)**(-7)
    print(epsilon, N, n, n**2)

    with open(f'data_store/simstudy_shorter_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
        data_fine = pickle.load(f)

    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    sigma_weights_dense /= sigma_weights_dense.sum()

    for i in range(6):
        epsilon = (1/25) * np.sqrt(2)**(-i)
        epsilons.append(epsilon)
        N = 1/ epsilon**2
        n = int(np.sqrt(N))
        dt = (1/25) * np.sqrt(2)**(-7)
        print(epsilon, N, n, n**2)

        # Load in course data
        with open(f'data_store/simstudy_shorter_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
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

        with open(prefix+f'sigma_{T}.pkl', 'wb') as f:
            pickle.dump(s, f)


            # plot the results

    s_values = np.array(s)  # Assuming 's' is defined elsewhere
    filtered_s = np.where(s_values > 1e-14, s_values, 0)  # Replace values below threshold with 0
    sqrt_s = np.sqrt(filtered_s)

    # Log-log plot
    plt.figure(figsize=(7, 5))
    plt.loglog(epsilons, sqrt_s, 'o:', label='Data points', markersize=8)

    # Polyfit for all data
    coeffs_all = np.polyfit(np.log(epsilons), np.log(sqrt_s), 1)
    fit_all = np.exp(coeffs_all[1]) * epsilons**coeffs_all[0]
    plt.loglog(epsilons, fit_all, '--', label=f'Fit (all data): slope={coeffs_all[0]:.2f}')

    plt.xticks(epsilons, [f"{eps:.6f}" for eps in epsilons])

    # Labels and formatting
    plt.title(f'Perturbed Jet comparison against finest resolution (sigma), at T={T}')
    plt.xlabel(r'$\varepsilon$ = 1/$\sqrt{N}$', fontsize=14)
    plt.ylabel(r'$\sqrt{S_{0.01}}$', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.savefig(prefix+f'sigma_{T}.png')


    ####################################################################################
    #            Height
    ####################################################################################
    # Load in fine data

    s1 = []
    epsilons = []

    i = 7
    epsilon = (1/25) * np.sqrt(2)**(-i)
    N = 1/ epsilon**2
    n = int(np.sqrt(N))
    dt = (1/25) * np.sqrt(2)**(-7)
    print(epsilon, N, n, n**2)

    with open(f'data_store/simstudy_shorter_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
        data_fine = pickle.load(f)

    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    sigma_weights_dense /= sigma_weights_dense.sum()
    
    for i in range(6):
        epsilon = (1/25) * np.sqrt(2)**(-i)
        epsilons.append(epsilon)
        N = 1/ epsilon**2
        n = int(np.sqrt(N))
        dt = (1/25) * np.sqrt(2)**(-7)
        print(epsilon, N, n, n**2)
        
        # Load in course data
        with open(f'data_store/simstudy_shorter_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
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
        with open(prefix+f'heights_{T}.pkl', 'wb') as f:
            pickle.dump(s1, f)
    #  Look at solutions

    s  = s1
    s_values = np.array(s)  # Assuming 's' is defined elsewhere
    filtered_s = np.where(s_values > 1e-14, s_values, 0)  # Replace values below threshold with 0
    sqrt_s = np.sqrt(filtered_s)

    # Log-log plot
    plt.figure(figsize=(7, 5))
    plt.loglog(epsilons, sqrt_s, 'o:', label='Data points', markersize=8)

    # Polyfit for all data
    coeffs_all = np.polyfit(np.log(epsilons), np.log(sqrt_s), 1)
    fit_all = np.exp(coeffs_all[1]) * epsilons**coeffs_all[0]
    plt.loglog(epsilons, fit_all, '--', label=f'Fit (all data): slope={coeffs_all[0]:.2f}')


    plt.xticks(epsilons, [f"{eps:.6f}" for eps in epsilons])

    # Labels and formatting
    plt.title(f'Perturbed Jet comparison against finest resolution (Height Error), at T={T}')
    plt.xlabel(r'$\varepsilon$ = 1/$\sqrt{N}$', fontsize=14)
    plt.ylabel(r'$\sqrt{S_{0.01}}$', fontsize=14)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)
    plt.savefig(prefix+f'heights_{T}.png')
