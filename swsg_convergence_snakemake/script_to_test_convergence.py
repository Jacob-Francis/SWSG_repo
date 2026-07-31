import pickle
import numpy as np
from unbalancedsinkhorn import UnbalancedOT, DebiasedUOT
import matplotlib.pyplot as plt
from utils import initialisation, jet_profile_initialisation


epsilon = 0.003125
n = int(1/epsilon)
# X, Y, G, h_true, mu = jet_profile_initialisation(epsilon, strength=0.0001, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
X, Y, G, h_true, mu = jet_profile_initialisation(epsilon, strength=0.0001, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)

with open("/home/jjf817/SWSG_repo/swsg_convergence_snakemake/data_store/four_epsilon_0.003125_profile_perturbedjet_results_nolloyd.pkl", "rb") as f:
    four3125 = pickle.load(f)

approx_height = four3125['h']

# normalise both cause that what i did before for some reason
approx_height /= approx_height.sum()
h_true /= h_true.sum()

uotclass = DebiasedUOT(pykeops=True, cuda_device='cuda:1')
uotclass.parameters(epsilon=0.01)
uotclass.densities(X, X, approx_height, h_true)

f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
    tol=1e-9,
    verbose=False,
    aprox="balanced",
    convergence_repeats=4,
    convergence_or_fail=False,
    epsilon_annealing=True,
    epsilon_annealing_const=0.995,
)

print("f update: ", f_update)
print("g update: ", g_update)
print("i sup: ", i_sup)

s = uotclass.sinkhorn_divergence(tol=1e-8, force_type='pykeops', return_type='dual')

print("Sinkhorn divergence: ", s)
try:
    print('sqrt', np.sqrt(s))
except:
    import torch
    print("torch", torch.sqrt(s))
