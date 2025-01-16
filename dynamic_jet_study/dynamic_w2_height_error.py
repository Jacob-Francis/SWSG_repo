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


# My python classes for loops we've defined so far
from swsg_ot_algorithm import SWSGDynamcis

import argparse

parser = argparse.ArgumentParser(description="Parse config file")
parser.add_argument("cuda", type=int, help="cuda index")
parser.add_argument("epsilon", type=float, help="size of epsilon")
parser.add_argument("lloyd", type=str, help="lloyd or not")

# parser.add_argument("method", type=str, help="method; one of 'euler' 'heun' 'rk4'")

args = parser.parse_args()
cuda = args.cuda
epsilon = args.epsilon
lloyd = args.lloyd
# method = args.method


import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
from time import perf_counter_ns
from geomloss import SamplesLoss
from unbalancedsinkhorn import UnbalancedOT, DebiasedUOT

def Sinkhorn_Divergence_balanced(X, α, Y, β, dense_symmetric_potential=None,f0=None, g0=None, force_type='pykeops', tol=1e-12):
    '''
    # Run OT(a, b) on grid X, Y reusing the dense symmeric potential and cost
    # dense_symmetric_potential = dict(f=g, uot(dense, dense))
    # a,X has to be dense
    '''
    cuda = X.device
    uotclass = DebiasedUOT(pykeops=True, cuda_device=cuda)
    uotclass.parameters(epsilon=0.002)
    uotclass.densities(X, Y, α, β)

    tic = perf_counter_ns()
    if dense_symmetric_potential is None:

        f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            aprox="balanced",
            convergence_repeats=3,
            converge_or_fail=True
        )
        
        d = uotclass.dual_cost(force_type=force_type)
        
        print("DENSE symmetric update final convergence:", f_update, g_update, i_sup)
        return dict(f=uotclass.g.cpu(), dual=sum(d))
    else:

        # Run sinkhorn
        f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
            f0=f0, g0=g0, aprox="balanced", tol=tol
        )
        
        print("Sinkhorn update final convergence:", f_update, g_update, i_sup)


        # solve the new symmetric potential problem
        uotclass.debias_g = UnbalancedOT(
            set_fail=uotclass.set_fail,
            pykeops=uotclass.pykeops,
            debias=False,
            cuda_device=uotclass.device,
        )

        uotclass.debias_g.parameters(uotclass.epsilon, uotclass.rho, uotclass.cost_const)

        uotclass.debias_g.densities(uotclass.Y_t, uotclass.Y_t, uotclass.β_t, uotclass.β_t)

        f_update, g_update, i_sup = uotclass.debias_g.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=uotclass.right_div.print_type(),
            right_divergence=uotclass.right_div.print_type(),
            convergence_repeats=3,
        )
        toc = perf_counter_ns()

        print("Symmetric update final convergence:", f_update, g_update, i_sup)
        print(f"W2 Computed in {toc-tic} ns")

        return (
            sum(uotclass.dual_cost(force_type=force_type))
            - (
                dense_symmetric_potential['dual'].to(uotclass.device)
                + sum(uotclass.debias_g.dual_cost(force_type=force_type))
            )
            / 2
            + uotclass.epsilon * (uotclass.α_s.sum() - uotclass.β_t.sum()) ** 2 / 2
        ).cpu().item(), uotclass

############# Symmetric pot;
f = open('snakemake_pipeline/data_store/dense_sym_profile_jet_dict.pkl', 'rb')
dense_symmetric_potential = pickle.load(f)
f.close()


########## Parameters

dts = [0.1, 0.05]
epsilons = [0.02, 0.01, 0.005]
tf = lambda x : x.type(torch.float64).to(f'cuda:{cuda}')
tol=1e-11
loss = SamplesLoss('sinkhorn', p=2, blur=np.sqrt(0.002), scaling=0.999, potentials=True)
loss_list = []
for j, dt in enumerate(dts):
    for k, epsilon in enumerate(epsilons):
        with open(f'data_store/output_{method}_{dt}_{epsilon}_{lloyd}.pkl', 'rb') as f:
            data = pickle.load(f)
        times_steps = data[-1].shape[0]
        # data format; G_dt, height_dt, x_star, energy_list
        for t in range(times_steps):
            if t == 0:
                # Tile the 1D into a 2D profile
                m2=m1=int(1/0.002)
                X_dense = (
                    torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
                    torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
                )
                a=0.1
                b=10.0
                c=0.5
                d=1.0
                h_dense = a * np.tanh(b * (X_dense[:,1] - c)) + d
                h_dense /= h_dense.sum()
                
                # regular universe grid;
                n1=n2=int(1/epsilon)
                X = (
                    torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
                    torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
                )


                X_dense = tf(X_dense)
                h_dense = tf(h_dense).view(-1, 1)
                h = tf(data[1][:, :, t])
                h /= h.sum()
                X = tf(X)
                
                output = loss(h_dense, X_dense, h, X)

                cost, uotclass = Sinkhorn_Divergence_balanced(
                            X_dense,
                            h_dense,
                            X,
                            h,
                            dense_symmetric_potential=dense_symmetric_potential,
                            f0=output[0],
                            g0=output[1],
                            tol=tol)
            else:
                h = tf(data[1][:, :, t])
                h /= h.sum()
                cost, uotclass = Sinkhorn_Divergence_balanced(
                        X_dense,
                        h_dense,
                        X,
                        h,
                        dense_symmetric_potential=dense_symmetric_potential,
                        f0=uotclass.f,
                        g0=uotclass.g,
                        tol=tol)

            loss_list.append(cost)

            with open(f'data_store/height_loss_list_{method}_{dt}_{epsilon}_{lloyd}.pkl', 'wb') as f:
                pickle.dump(loss_list, f)





