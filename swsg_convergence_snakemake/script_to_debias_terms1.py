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


# debiasing term 1
uotclass = DebiasedUOT(pykeops=True, cuda_device='cuda:4')
uotclass.parameters(epsilon=0.01)
uotclass.densities(X, X, approx_height, approx_height)

f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
    tol=1e-10,
    verbose=False,
    aprox="balanced",
    convergence_repeats=4,
    convergence_or_fail=True,
    # epsilon_annealing=True,
    # epsilon_annealing_const=0.995,
)

print("f update: ", f_update)
print("g update: ", g_update)
print("i sup: ", i_sup)

# calculate dual cost
dual_cost = uotclass.dual_cost(force_type='pykeops')

print("Dual cost: ", dual_cost)


# dict of things
results_dict = {
    'f_update': f_update,
    'g_update': g_update,
    'i_sup': i_sup,
    'dual_cost': sum(dual_cost).item(),
}

# save results to file
with open("/home/jjf817/SWSG_repo/swsg_convergence_snakemake/data_store/four_epsilon_0.003125_profile_perturbedjet_debiasing_term1_dual_cost.pkl", "wb") as f:
    pickle.dump(results_dict, f)