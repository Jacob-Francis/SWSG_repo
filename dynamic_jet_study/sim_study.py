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

import argparse

parser = argparse.ArgumentParser(description="Parse config file")
parser.add_argument("cuda", type=int, help="cuda index")
parser.add_argument("epsilon", type=float, help="size of epsilon")
parser.add_argument("strength", type=float, help="strength of perturbation")
parser.add_argument("method", type=int, help="1,2,3  methods?")

# parser.add_argument("method", type=str, help="method; one of 'euler' 'heun' 'rk4'")

args = parser.parse_args()
cuda = args.cuda
epsilon = args.epsilon
strength = args.strength
meth = args.method
g = 0.025
a = 0.1
b = 10

assert(3* np.sqrt(3) / (4*g) > a*b**2)

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

    no, no0 , no1  = normal_pdf(X[:,0],X[:,1],0.5,0.3,0.1,strength)  ## 0 is stationnary 
    h_true = h_true  + no 
    h_true = h_true.div(torch.sum(h_true)) 
    G = G + torch.stack((no0, no1), dim=1)

    return X, Y, G, h_true, mu


global device
device = f'cuda:{cuda}'

X_xy, Y, G_xy, h_true, mu = jet_profile_initialisation(epsilon, strength, f=1.0, g=g, a=a, b=b, c=0.5, d=1.0)

if meth == 2:
    methods = ['euler', 'heun']
elif meth ==  1:
    methods = ['heun']
else:
    methods = ['euler', 'heun','rk4']



for method in methods:
    for dt in [0.05]:   # 0.4, 0.2, 0.1, 
        print(f'Starting: output_{method}_{dt}_{epsilon}')
        swsg_class = SWSGDynamcis(pykeops=True, cuda_device=device)
        swsg_class.parameters(ε=epsilon, f=1.0, g=g)
        d = 1.0
        h_leb = torch.ones_like(X_xy[:,0]) * d / len(X_xy[:,0])

        sigma = h_true / h_true.sum()
        
#         # Load it in:
#         f = open(f'data_store/output_{method}_{dt}_{epsilon}_strength_{strength}_smallerg.pkl', 'rb')

#         output = pickle.load(f)

#         f.close()
        
#         G_xy = output[0][:, :, -1].view(-1, 2)

        swsg_class.densities(source_points=G_xy.detach().clone(), target_points=X_xy.detach().clone(), source_density=sigma.detach(), target_density=h_leb.detach(), cost_type='periodic', L=1.0)

        time_steps = int(100/dt)
        tic = perf_counter_ns()
        output = swsg_class.stepping_scheme(debias=True, dt=dt, method=method, time_steps=time_steps, tol=1e-11, newton_tol=1e-11, geoverse_velocities=True, collect_x_star= True, sinkhorn_divergence=True)
        toc = perf_counter_ns()

        print('TIMING: ', toc-tic, f'data_store/output_{method}_{dt}_{epsilon}_{strength}.pkl')

        f = open(f'data_store/output_{method}_{dt}_{epsilon}_strength_{strength}_smaller_g.pkl', 'wb')

        pickle.dump(output, f)

        f.close()

        del swsg_class, output
        torch.cuda.empty_cache()
