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

def normal_pdf(x, y, mu_x, mu_y, sigma,alpha):
    """
    Calculate the PDF of a bivariate normal distribution.
    """
    # Constants
    sigma2 = torch.tensor([sigma**2])
    mu_x_ = torch.tensor([mu_x])
    mu_y_ = torch.tensor([mu_y])
    
    norm_factor = 1 / (2 * torch.pi * sigma2) #_x * sigma_y * torch.sqrt(1 - rho ** 2))   
    # Z computation
    z_x = (x - mu_x_) 
    z_y = (y - mu_y_) 
    
    z = z_x**2  + z_y ** 2 
    
    # PDF computation
    pdf = alpha*norm_factor * torch.exp(-0.5 * z/sigma2)
    pdfg0 = -alpha* z_x/sigma2 *norm_factor * torch.exp(-0.5 * z/sigma2)
    pdfg1 = -alpha* z_y/sigma2 *norm_factor * torch.exp(-0.5 * z/sigma2)
    
    return pdf.unsqueeze(-1), pdfg0  , pdfg1

def jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, pykeop=True):
    """
    Initialise a jet profile and associated object for solving the sWSG problem.
    """
    # Decide on parameters
    global n1, n2, m1, m2
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = int(1 / epsilon), int(1 / epsilon)

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return a * np.tanh(b * (x - c)) + d

    # integral of tanh is ln(cosh) so;
    def int_h(x):
        return a * np.log(np.cosh(b * (x - 0.5)) / np.cosh(-b * 0.5)) / b + d*x

    X_j = torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1)

    # Calculate nabla P: x + f^2 * g * partial h
    G_i = X_j + f**-2 * g * a * b * (1 - np.tanh(b * (X_j - 0.5)) ** 2)

    # Tile the 1D into a 2D profile
    X = torch.cartesian_prod(
        torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
        torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
    )
    Y = torch.cartesian_prod(
        torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2), torch.Tensor(X_j)
    )
    G = torch.cartesian_prod(
        torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2), torch.Tensor(G_i)
    )

    h_true = height_func(X[:, 1]).view(-1, 1)
    mu = torch.ones_like(h_true) * d  / len(X[:, 1])

    no, no0 , no1  = normal_pdf(X[:,0],X[:,1],0.5,0.3,0.1,strength)  ## 0 is stationnary 
    h_true = h_true  + no 
    h_true = h_true.div(torch.sum(h_true)) 
    G = G + torch.stack((no0, no1), dim=1)

    return X, Y, G, h_true, mu


def Sinkhorn_Divergence_balanced(
    X,
    α,
    Y,
    β,
    dense_symmetric_potential=None,
    f0=None,
    g0=None,
    force_type="pykeops",
    tol=1e-9,
    epsilon=0.01,
    fullcompute=False
):
    """
    # Run OT(a, b) on grid X, Y reusing the dense symmeric potential and cost
    # dense_symmetric_potential = dict(f=g, uot(dense, dense))
    # a,X has to be dense
    """
    cuda = α.device
    uotclass = DebiasedUOT(pykeops=True, cuda_device=cuda)
    uotclass.parameters(epsilon=epsilon)
    uotclass.densities(X, Y, α, β)

    tic = perf_counter_ns()
    if dense_symmetric_potential is None and not fullcompute:

        f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            aprox="balanced",
            convergence_repeats=3,
            convergence_or_fail=True,
        )

        d = uotclass.dual_cost(force_type=force_type)

        print("DENSE symmetric update final convergence:", f_update, g_update, i_sup)
        return dict(f=uotclass.g.view(-1, 1).cpu(), dual=sum(d))
    elif fullcompute:
        try:
            f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
                f0=f0, g0=g0, aprox="balanced", tol=tol, convergence_or_fail=True
            )
        except RuntimeWarning:
            f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
                f0=f0, g0=g0, aprox="balanced", tol=tol, convergence_or_fail=False, epsilon_annealing=True,
            )
        print("Sinkhorn full compute final convergence:", f_update, g_update, i_sup)
        s = uotclass.sinkhorn_divergence(tol=tol, force_type='pykeops', return_type='dual')
        return s.cpu().item(), uotclass
    else:


        f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
            f0=f0, g0=g0, aprox="balanced", tol=tol, convergence_or_fail=False,
        )

        print("Sinkhorn update final convergence:", f_update, g_update, i_sup)

        # solve the new symmetric potential problem
        uotclass.debias_g = UnbalancedOT(
            pykeops=uotclass.pykeops,
            debias=False,
            cuda_device=uotclass.device,
        )

        uotclass.debias_g.parameters(
            uotclass.epsilon, uotclass.rho, uotclass.cost_const
        )

        uotclass.debias_g.densities(
            uotclass.Y_t, uotclass.Y_t, uotclass.β_t, uotclass.β_t
        )
        
        # solve the new symmetric potential problem
        uotclass.debias_f = UnbalancedOT(
            pykeops=uotclass.pykeops,
            debias=False,
            cuda_device=uotclass.device,
        )

        uotclass.debias_f.parameters(
            uotclass.epsilon, uotclass.rho, uotclass.cost_const
        )

        uotclass.debias_f.densities(
            uotclass.X_s, uotclass.X_s, uotclass.α_s, uotclass.α_s
        )
        
        # load in known potential
        uotclass.debias_f.f = dense_symmetric_potential['f'].to(uotclass.α_s)
        uotclass.debias_f.g = dense_symmetric_potential['f'].to(uotclass.α_s)

        f_update, g_update, i_sup = uotclass.debias_g.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=uotclass.right_div.print_type(),
            right_divergence=uotclass.right_div.print_type(),
            convergence_repeats=3,
        )
        print("GGG update final convergence:", f_update, g_update, i_sup)

        f_update, g_update, i_sup = uotclass.debias_f.sinkhorn_algorithm(
            tol=tol,
            verbose=False,
            left_divergence=uotclass.right_div.print_type(),
            right_divergence=uotclass.right_div.print_type(),
            convergence_repeats=1,
        )
        toc = perf_counter_ns()

        print("FFFF update final convergence:", f_update, g_update, i_sup)
        print(f"W2 Computed in {toc-tic} ns")

        return (
            sum(uotclass.dual_cost(force_type=force_type))
            - (
                sum(uotclass.debias_f.dual_cost(force_type=force_type))
                + sum(uotclass.debias_g.dual_cost(force_type=force_type))
            )
            / 2
            + uotclass.epsilon * (uotclass.α_s.sum() - uotclass.β_t.sum()) ** 2 / 2
        ).cpu().item(), uotclass


####################################################################################
#               Main script
####################################################################################
cuda = 4
strength = 0.0001

global device
device = f'cuda:{cuda}'
####################################################################################
#              load in fine data
####################################################################################
i=7
epsilon = (1/25) * np.sqrt(2)**(-i)
N = 1/ epsilon**2
n = int(np.sqrt(N))
dt = (1/25) * np.sqrt(2)**(-7)
method = 'heun'

with open(f'data_store/simstudy_shorter_{method}_{epsilon}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
    data_fine = pickle.load(f)
    
X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
sigma_weights_dense /= sigma_weights_dense.sum()

####################################################################################
#             Run or load symmetic potential calculation
####################################################################################


time = 1414 -1
try:
    with open(f'data_store/simstudy_shorter_{method}_{epsilon}_{epsilon}_strength_{strength}_dense_potential_T5.pkl', 'rb') as f:
        d = pickle.load(f)
except FileNotFoundError:
    # Compute and save the symmtric potential;
    d = Sinkhorn_Divergence_balanced(
        data_fine[0][:,:,time].to(device),
        sigma_weights_dense.to(device),
        data_fine[0][:,:,time].to(device),
        sigma_weights_dense.to(device),
        dense_symmetric_potential=None,
        f0=None,
        g0=None,
        force_type="pykeops",
        tol=1e-7,
        epsilon=0.01,
        fullcompute=False
    )

    print(d)

    with open(f'data_store/simstudy_shorter_{method}_{epsilon}_{epsilon}_strength_{strength}_dense_potential_T5.pkl', 'wb') as f:
        pickle.dump(d,f)


####################################################################################
#            Main loop
####################################################################################

s = []
time = 1414 -1

for i in range(7):
    epsilon = (1/25) * np.sqrt(2)**(-i)
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
            dense_symmetric_potential=d,
            force_type="pykeops",
            tol=1e-10,
            epsilon=0.01,
            fullcompute=False
        )  
    toc = perf_counter_ns()

    print('TIMING: ', tic-toc, output)
    s.append(output)
    
    with open('saving_sigmashorter_T5.pkl', 'wb') as f:
        pickle.dump(s, f)


        # plot the results


epsilons = [(1/25) * np.sqrt(2)**(-i) for i in range(7)]
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

# # Polyfit for central points (excluding first and last)
# central_eps = epsilons[2:]
# central_sqrt_s = sqrt_s[2:]
# coeffs_central = np.polyfit(np.log(central_eps), np.log(central_sqrt_s), 1)
# fit_central = np.exp(coeffs_central[1]) * central_eps**coeffs_central[0]
# plt.loglog(central_eps, fit_central, '-.', label=f'Fit (central points): slope={coeffs_central[0]:.2f}')

plt.xticks(epsilons, [f"{eps:.6f}" for eps in epsilons])

# Labels and formatting
plt.title('Perturbed Jet comaprison against higher resolution sigma, at T=0')
plt.xlabel(r'$\varepsilon$ = 1/$\sqrt{N}$', fontsize=14)
plt.ylabel(r'$\sqrt{S_{0.01}}$', fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.6)
plt.savefig('sigma_convergence_T5.png')


####################################################################################
#            Height
####################################################################################
# Load in fine data
i=7
epsilon = (1/25) * np.sqrt(2)**(-i)
N = 1/ epsilon**2
n = int(np.sqrt(N))
dt = (1/25) * np.sqrt(2)**(-7)
method = 'heun'

with open(f'data_store/simstudy_shorter_{method}_{epsilon}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
    data_fine = pickle.load(f)
    
X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
sigma_weights_dense /= sigma_weights_dense.sum()

try:
    with open(f'data_store/simstudy_shorter_{method}_{epsilon}_{epsilon}_strength_{strength}_dense_potential_height_T5.pkl', 'rb') as f:
        d = pickle.load(f)
except FileNotFoundError:
    # Compute and save the symmtric potential;
    d = Sinkhorn_Divergence_balanced(
        X_dense.to(device),
        data_fine[1][:,:,-1].to(device),
        X_dense.to(device),
        data_fine[1][:,:,-1].to(device),
        dense_symmetric_potential=None,
        f0=None,
        g0=None,
        force_type="pykeops",
        tol=1e-9,
        epsilon=0.01,
        fullcompute=False
    )

    print(d)

    with open(f'data_store/simstudy_shorter_{method}_{epsilon}_{epsilon}_strength_{strength}_dense_potential_height_T5.pkl', 'wb') as f:
        pickle.dump(d,f)

s1 = []

for i in range(7):
    epsilon = (1/25) * np.sqrt(2)**(-i)
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
            (data_fine[1][:,:,-1]/data_fine[1][:,:,-1].sum()).to(device).view(-1,1),
            X,
            (data_course[1][:, :, -1]/data_course[1][:, :, -1].sum()).to(device).view(-1,1),
            dense_symmetric_potential=d,
            force_type="pykeops",
            tol=1e-10,
            epsilon=0.01,
            fullcompute=False
        )  
    toc = perf_counter_ns()
    
    s1.append(output)
    with open('shorter_heights.pkl', 'wb') as f:
        pickle.dump(s1, f)
#  Look at solutions



s  = s1
epsilons = [(1/25) * np.sqrt(2)**(-i) for i in range(7)]
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

# # Polyfit for central points (excluding first and last)
# central_eps = epsilons[2:]
# central_sqrt_s = sqrt_s[2:]
# coeffs_central = np.polyfit(np.log(central_eps), np.log(central_sqrt_s), 1)
# fit_central = np.exp(coeffs_central[1]) * central_eps**coeffs_central[0]
# plt.loglog(central_eps, fit_central, '-.', label=f'Fit (central points): slope={coeffs_central[0]:.2f}')

plt.xticks(epsilons, [f"{eps:.6f}" for eps in epsilons])

# Labels and formatting
plt.title('Perturbed Jet comaprison against higher resolution (Height Error), at T=5')
plt.xlabel(r'$\varepsilon$ = 1/$\sqrt{N}$', fontsize=14)
plt.ylabel(r'$\sqrt{S_{0.01}}$', fontsize=14)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.6)
plt.savefig('height_convergence_T5.png')
