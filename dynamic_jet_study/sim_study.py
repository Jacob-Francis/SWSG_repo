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


def jet_profile_initialisation(epsilon, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, pykeop=True):
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
    G_i = X_j + f**2 * g * a * b * (1 - np.tanh(b * (X_j - 0.5)) ** 2)

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

#     swsg_class = SWSGDynamcis(pykeops=pykeop, set_fail=False, cuda_device='cuda:0')
#     swsg_class.parameters(ε=epsilon, f=f, g=g)
#     # If unit mass (d=1) then mu is the expected 1/N
#     swsg_class.densities(source_points=G, target_points=X, source_density=mu, cost_type='periodic', L=1.0)

    return X, Y, G, h_true, mu

from geomloss import SamplesLoss
from tqdm import tqdm  # Import tqdm for the progress bar


def lloyd(blur, X, Y, alpha=None, beta=None, tol=1e-11, lr=0.9):
    """
    Llody fitting Y, beta to X, alpha, using geomloss. Tolerance at subsequent iterations aren't changing much.
    NOT a true convergence metric.
    """
    llody_loss = SamplesLoss('sinkhorn', p=2, blur=blur, scaling=0.9)
    err  = 1e3
    count = 0
    kmax = 200

    # Set up the tqdm progress bar
    with tqdm(total=kmax, desc="Llody Progress", unit="iter") as pbar:
        while err > tol and count < kmax:
            L_ = llody_loss(beta, Y, alpha, X)
            grad = torch.autograd.grad(L_, Y)[0]

            Y = Y - lr * grad * len(beta)  # Maybe times by beta here - though it is actually 1/N?
            err = torch.linalg.norm(grad)
            count += 1

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({"Error": err.item(), "Loss": L_.item()})
        
    print('Final W2 loss:', llody_loss(beta, Y, alpha, X))
    return Y

def jet2D_lloyd(device, dtype, epsilon=0.05, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, dense_scale=3, tol=1e-11):
    
    # Decide on parameters - baseed off of epsilon
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = dense_scale*int(1 / epsilon), dense_scale*int(1 / epsilon)
    # 2D Llody

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return a * torch.tanh(b * (x - c)) + d
    
    # Llody fit against dense sampling 
    Y = torch.rand((n1*n2, 2), device=device).type(dtype)*50 - 24  # Random box in [-4, 6]x[-4, 6] becuase I want solutions in [0, 1]x[0,1]
    Y = Y.requires_grad_(True)
    beta = torch.ones((n1*n2), device=device).type(dtype)
    beta /= beta.sum()
    
    # Uniform sample - maybe dense?
    X = torch.cartesian_prod(
        torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
        torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
    ).to(device).type(dtype)
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)
    alpha /= alpha.sum()
    
    Y = lloyd(epsilon**0.5, X, Y, alpha, beta, tol=tol)
    
    # generate regulr uniform grid of correct size for universe;
    # Overwriting dense grid used for lloyd sampling
    X = torch.cartesian_prod(
        torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
        torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
    ).to(device).type(dtype)
    # Not normalised to one for later purposes.
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)
    
    # Calculate nabla P: x + f^2 * g * partial h
    G = Y.detach().clone()
    G[:, 1] = G[:, 1] + f**2 * g * a * b * (1 - torch.tanh(b * (Y[:, 1] - 0.5)) ** 2) # X Y?

   
    h_true = alpha.view(-1, 1)
    mu = torch.ones_like(h_true) * d  / len(X[:, 1])

    return  X, Y, G, h_true, mu


# ######### Running cases
global device
device = f'cuda:{cuda}'

if lloyd=='lloyd':
    X_xy, Y, G_xy, h_true, mu = jet2D_lloyd(device=device, dtype=torch.float64, epsilon=epsilon, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, tol=1e-11)
else:
    X_xy, Y, G_xy, h_true, mu = jet_profile_initialisation(epsilon=epsilon, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    
for method in ['euler', 'heun', 'rk4']:
    for dt in [0.2, 0.1, 0.05]:
        print(f'Starting: output_{method}_{dt}_{epsilon}')
        swsg_class = SWSGDynamcis(pykeops=True, cuda_device=device)
        swsg_class.parameters(ε=epsilon, f=1.0, g=0.1)
        d = 1.0
        h_leb = torch.ones_like(X_xy[:,0]) * d / len(X_xy[:,0])
        
        if lloyd=='lloyd':
            sigma = torch.ones_like(G_xy[:,0]) * d / len(G_xy[:,0])
        else:
            sigma = h_true / h_true.sum()

        swsg_class.densities(source_points=G_xy.detach().clone(), target_points=X_xy.detach().clone(), source_density=sigma.detach(), target_density=h_leb.detach(), cost_type='periodic', L=1.0)

        time_steps = int(30/dt)
        tic = perf_counter_ns()
        output = swsg_class.stepping_scheme(debias=True, dt=dt, method=method, time_steps=time_steps, tol=1e-11, newton_tol=1e-11, geoverse_velocities=True, collect_x_star= True, sinkhorn_divergence=True)
        toc = perf_counter_ns()

        print('TIMING: ', toc-tic, f'data_store/output_{method}_{dt}_{epsilon}_{lloyd}.pkl')

        f = open(f'data_store/output_{method}_{dt}_{epsilon}_{lloyd}.pkl', 'wb')

        pickle.dump(output, f)

        f.close()
        
        del swsg_class, output
        torch.cuda.empty_cache()
