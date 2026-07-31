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

# parser.add_argument("method", type=str, help="method; one of 'euler' 'heun' 'rk4'")

cuda = 3
strength = 0.0001

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


global device
device = f'cuda:{cuda}'


for N in [25**2, 35**2, 50**2, 71**2, 100**2, 141**2]: #200**2
    epsilon = 1/np.sqrt(N)
    dt = 1/np.sqrt(200**2)
    method = 'heun'
    X_xy, Y, G_xy, h_true, mu = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    
    print(f' Generated profile for, {N}, {epsilon}, dt')
 
    print(f'Starting: output_{method}_{dt}_{epsilon}')
    swsg_class = SWSGDynamcis(pykeops=True, cuda_device=device)
    swsg_class.parameters(ε=epsilon, f=1.0, g=0.1)
    d = 1.0
    h_leb = torch.ones_like(X_xy[:,0]) * d / len(X_xy[:,0])

    sigma = h_true / h_true.sum()

#         # Load it in:
#         f = open(f'data_store/output_{method}_{dt}_{epsilon}_strength_{strength}_smallerg.pkl', 'rb')

#         output = pickle.load(f)

#         f.close()

#         G_xy = output[0][:, :, -1].view(-1, 2)

    swsg_class.densities(source_points=G_xy.detach().clone(), target_points=X_xy.detach().clone(), source_density=sigma.detach(), target_density=h_leb.detach(), cost_type='periodic', L=1.0)

    time_steps = int(20/dt)
    tic = perf_counter_ns()
    output = swsg_class.stepping_scheme(debias=True, dt=dt, method=method, time_steps=time_steps, tol=1e-11, newton_tol=1e-11, geoverse_velocities=True, collect_x_star= True, sinkhorn_divergence=True)
    toc = perf_counter_ns()

    print('TIMING: ', toc-tic, f'data_store/output_{method}_{dt}_{epsilon}_{strength}.pkl')

    f = open(f'data_store/simstudy_finer_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'wb')

    pickle.dump(output, f)

    f.close()

    del swsg_class, output
    torch.cuda.empty_cache()
    
    
    
# Load in fine data
E = 1 / np.sqrt(200**2)
method = 'heun'

with open(f'data_store/simstudy_{method}_{E}_{E}_strength_{strength}.pkl', 'rb') as f:
    data_fine = pickle.load(f)
    
times = data_fine[0].shape[-1]

X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(E, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
sigma_weights_dense /= sigma_weights_dense.sum()

s1 = []
uot_class = DebiasedUOT(pykeops=True,cuda_device=device)
uot_class.parameters(epsilon=1/200)
    
for N in [25**2, 35**2, 50**2, 71**2,100**2,141**2]: #, 141**2, 200**2
    epsilon = 1/np.sqrt(N)
    dt = 1/np.sqrt(200**2)
    method = 'heun'
    
    # Load in course data
    with open(f'data_store/simstudy_finer_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
        data_course = pickle.load(f)

    _, _, _, sigma_weights, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)

    sigma_weights /= sigma_weights.sum()
    

    uot_class.densities(source_points=data_course[0][:, :, -1],
                        target_points=data_fine[0][:,:, 3*times//2],
                        source_density=sigma_weights,
                        target_density=sigma_weights_dense)

    tic = perf_counter_ns()
    output = uot_class.sinkhorn_algorithm(aprox='balanced', tol=1e-8, epsilon_annealing=True)
    toc = perf_counter_ns()

    print('TIMING: ', tic-toc, output)
    
    tic = perf_counter_ns()
    output = uot_class.sinkhorn_divergence(return_type='dual', tol=1e-8,  force_type='pykeops', reuse_symmetric=True)
    toc = perf_counter_ns()

    print('TIMING: ', tic-toc, output)
    s1.append(output.item())
    with open('finer_fixed_eps_data.pkl', 'wb') as f:
        pickle.dump(s1, f)
        
plt.loglog([np.sqrt(1/n) for n in [25**2, 35**2, 50**2, 71**2,100**2, 141**2]], [np.sqrt(t) if t > 1e-14 else 0 for t in s], '.:')
plt.set_aspect('equal', adjustable='box')
plt.xlabel(r'$\varepsilon^2$')
plt.ylabel('$\sqrt{S_{\epsilon}}$')
plt.savefig('finer_fixed_eps_data.pdf')


s1 = []
for N in [25**2, 35**2, 50**2, 71**2,100**2,141**2]: #, 141**2, 200**2
    epsilon = 1/np.sqrt(N)
    dt = 1/np.sqrt(200**2)
    method = 'heun'
    
    # Load in course data
    with open(f'data_store/simstudy_finer_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
        data_course = pickle.load(f)

    X, _, _, sigma_weights, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)

    sigma_weights /= sigma_weights.sum()

    uot_class.densities(source_points=X,
                        target_points=X_dense,
                        source_density=data_course[1][:, :, -1],
                        target_density=data_fine[1][:,:, 3*times//2])

    tic = perf_counter_ns()
    output = uot_class.sinkhorn_algorithm(aprox='balanced', tol=1e-8, epsilon_annealing=True)
    toc = perf_counter_ns()

    print('TIMING: ', tic-toc, output)
    
    tic = perf_counter_ns()
    output = uot_class.sinkhorn_divergence(return_type='dual', tol=1e-8,  force_type='pykeops', reuse_symmetric=True)
    toc = perf_counter_ns()

    print('TIMING: ', tic-toc, output)
    s1.append(output.item())
    with open('finer_height_data.pkl', 'wb') as f:
        pickle.dump(s1, f)
#  Look at solutions

plt.loglog([np.sqrt(1/n) for n in [25**2, 35**2, 50**2, 71**2,100**2, 141**2]], [np.sqrt(t) if t > 1e-14 else 0 for t in s], '.:')
plt.set_aspect('equal', adjustable='box')
plt.xlabel(r'$\varepsilon^2$')
plt.ylabel('$ height \sqrt{S_{\epsilon}}$')
plt.savefig('finer_height_data.pdf')


