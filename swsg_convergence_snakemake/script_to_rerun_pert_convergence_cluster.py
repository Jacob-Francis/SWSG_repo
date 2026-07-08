import pickle
import numpy as np
from unbalancedsinkhorn import UnbalancedOT, DebiasedUOT
import matplotlib.pyplot as plt
from utils import initialisation, jet_profile_initialisation


# open the files
epsilons = [0.05, 0.025, 0.0125, 0.00625, 0.003125]
fine_epsilon = 0.003125/2

cuda_device = 'cuda:7'

save_dict = {}

for epsilon in epsilons:
    save_dict[epsilon] = {}
    n = int(1/epsilon)
    X, Y, G, h_true, mu = jet_profile_initialisation(fine_epsilon, strength=0.001, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)

    Xres, _, _, _, _ = jet_profile_initialisation(epsilon, strength=0.0001, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)

    with open(f"data_store/four_epsilon_{epsilon}_profile_perturbedjet_results_nolloyd.pkl", "rb") as f:
    # with open(f"/home/jjf817/SWSG_repo/swsg_convergence_snakemake/data_store/four_epsilon_{epsilon}_profile_perturbedjet_results_nolloyd.pkl", "rb") as f:
        four3125 = pickle.load(f)

    approx_height = four3125['h']/n**2

    # main term
    
    uotclass = DebiasedUOT(pykeops=True, cuda_device=cuda_device)
    uotclass.parameters(epsilon=0.01)
    uotclass.densities(Xres, X, approx_height, h_true)
    
    f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
        tol=1e-11,
        verbose=True,
        aprox="balanced",
        convergence_repeats=2,
        convergence_or_fail=False,
        # epsilon_annealing=True,
        # epsilon_annealing_const=0.995,
    )

    dual_cost = uotclass.dual_cost(force_type='pykeops')

    save_dict[epsilon]['dual_cost'] = sum(dual_cost).item()

    # temp save
    with open(f"data_store/four_pertjet_rerun_conv_00156.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    # debiasing term 1

    # debiating terms:

    uotclass1 = DebiasedUOT(pykeops=True, cuda_device=cuda_device)
    uotclass1.parameters(epsilon=0.01)
    uotclass1.densities(Xres, Xres, approx_height, approx_height)

    f_update, g_update, i_sup = uotclass1.sinkhorn_algorithm(
        tol=1e-11,
        verbose=True,
        aprox="balanced",
        convergence_repeats=2,
        convergence_or_fail=False,
        # epsilon_annealing=True,
        # epsilon_annealing_const=0.995,
    )

    dual_cost11 = uotclass1.dual_cost(force_type='pykeops')

    save_dict[epsilon]['dual_cost11'] = sum(dual_cost11).item()

    # temp save
    with open(f"data_store/four_pertjet_rerun_conv_00156.pkl", "wb") as f:
        pickle.dump(save_dict, f)

# Debiasing term 3: separate
# debiating terms:

uotclass2 = DebiasedUOT(pykeops=True, cuda_device=cuda_device)
uotclass2.parameters(epsilon=0.01)
uotclass2.densities(X, X, h_true, h_true)

f_update, g_update, i_sup = uotclass2.sinkhorn_algorithm(
    tol=1e-11,
    verbose=True,
    aprox="balanced",
    convergence_repeats=2,
    convergence_or_fail=False,
    # epsilon_annealing=True,
    # epsilon_annealing_const=0.995,
)

dual_cost22 = uotclass2.dual_cost(force_type='pykeops')

save_dict['dual_cost22'] = sum(dual_cost22).item()

# temp save
with open(f"data_store/four_pertjet_rerun_conv_00156.pkl", "wb") as f:
    pickle.dump(save_dict, f)

# plotting

# cyc;e through dict
error_list = []
for epsilon in epsilons:
    error_list.append(save_dict[epsilon]['dual_cost'] - 0.5 * save_dict[epsilon]['dual_cost11'] - 0.5* save_dict['dual_cost22'])

# plot log-log
plt.figure()
plt.loglog(epsilons, error_list, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Error')
plt.title('Convergence of Perturbed Jet Profile')
plt.grid(True, which="both", ls="--")

plt.savefig(f"data_store/four_pertjet_rerun_conv_00156.png", dpi=300)