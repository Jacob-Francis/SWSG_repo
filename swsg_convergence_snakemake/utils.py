from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import cdist as scipy_cdist
from plotting_utils import colour_bar
from functools import partial
import pickle
from time import perf_counter_ns
from unbalancedsinkhorn import UnbalancedOT, DebiasedUOT

# My python classes for loops we've defined so far
from swsg_ot_algorithm import SWSGSinkNewton, SWSGDynamcis
from scipy.special import lambertw as lambertw_scipy
from lambertw import halley_lambertw
from lambertw import _residual as _lambw_res
from geomloss import SamplesLoss
from tqdm import tqdm  # Import tqdm for the progress bar


def lloyd(blur, X, Y, alpha=None, beta=None, tol=1e-11, lr=0.9):
    """
    Llody fitting Y, beta to X, alpha, using geomloss. Tolerance at subsequent iterations aren't changing much.
    NOT a true convergence metric.
    """
    llody_loss = SamplesLoss('sinkhorn', p=2, blur=blur, scaling=0.99, backend='multiscale')
    err  = 1e3
    kmax = 100
    count = 0

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

def incline2D_lloyd(device, dtype, epsilon=0.05, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, dense_scale=3, tol=1e-11):
    
    # Decide on parameters - baseed off of epsilon
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = dense_scale*int(1 / epsilon), dense_scale*int(1 / epsilon)
    # 2D Llody

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return a * b * (x - c) + d
    
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
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)
    
    # Calculate nabla P: x + f^2 * g * partial h
    G = Y.detach().clone()  
    G[:, 1] = G[:, 1] + f**-2 * g * a * b
   
    h_true = alpha.view(-1, 1) # Not normalised to one
    mu = torch.ones_like(h_true) * d  / len(X[:, 1])

    return  X, Y, G, h_true, mu

def uniform2D_lloyd(device, dtype, epsilon=0.05, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, dense_scale=3, tol=1e-11):
    
    # Decide on parameters - baseed off of epsilon
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = dense_scale*int(1 / epsilon), dense_scale*int(1 / epsilon)
    # 2D Llody
   
    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return torch.ones_like(x) / len(x)
    
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
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)

    # Calculate nabla P: x + f^2 * g * partial h
    G = Y.detach().clone() 

    h_true = alpha.view(-1, 1)
    mu = torch.ones_like(h_true) * d  / len(X[:, 1])

    return X, Y, G, h_true, mu


# pylint: disable=E1101
def incline(epsilon, f=1.0, g=0.1, a=0.2, b=1.0, c=0.5, d=1.0):
    """
    Initialise a jet profile and associated object for solving the sWSG problem.
    """
    global n1, n2, m1, m2

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return a * b * (x - c) + d

    # integral of tanh is ln(cosh) so;
    def int_h(x):
        return a * b * (x**2 / 2 - c * x) + d * x

    # find X_j s.t. universe weighting is 1/N
    # X_j = np.zeros(n1)
    # for k, ui in enumerate(
    #     d * np.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1, endpoint=True)
    # ):
    #     X_j[k] = optimize.root(lambda x: int_h(x) - ui, x0=0).x[0]
    X_j = np.zeros(n1)
    for k, ui in enumerate(d * np.linspace(1 / n1, 1, n1, endpoint=True)):
        X_j[k] = optimize.root(lambda x: int_h(x) - ui, x0=0).x[0]

    X_j[1:] = (X_j[:-1] + X_j[1:]) / 2
    X_j[0] = X_j[0] / 2

    # Calculate nabla P: x + f^2 * g * partial h
    G_i = X_j + f**-2 * g * a * b

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
    mu = torch.ones_like(h_true) * d / len(X[:, 1])

    return X, Y, G, h_true, mu


def incline_no_lloyd(epsilon, f=1.0, g=0.1, a=0.2, b=1.0, c=0.5, d=1.0):
    """
    Initialise a jet profile and associated object for solving the sWSG problem.
    """
    global n1, n2, m1, m2

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return a * b * (x - c) + d

    X_j = d * np.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1, endpoint=True)

    # Calculate nabla P: x + f^2 * g * partial h
    G_i = X_j + f**-2 * g * a * b

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
    mu = h_true * d / h_true.sum()

    return X, Y, G, h_true, mu


def initialisation(
    device, dtype, epsilon=0.05, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, profile_type="jet", cuda=0, tol=1e-11
):
    """
    Initialise a jet profile and associated object for solving the sWSG problem.
    """
    # Decide on parameters
    global n1, n2, m1, m2 
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = int(1 / epsilon), int(1 / epsilon)

    if profile_type == "jet":
        # Assigning uniform weighting to points - Lloyde type
        def height_func(x):
            return a * np.tanh(b * (x - c)) + d

        # integral of tanh is ln(cosh) so;
        def int_h(x):
            return a * np.log(np.cosh(b * (x - 0.5)) / np.cosh(-b * 0.5)) / b + d * x

        # find X_j s.t. universe weighting is 1/N
        # X_j = np.zeros(n1)
        # for k, ui in enumerate(
        #     d * np.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1, endpoint=True)
        # ):
        #     X_j[k] = optimize.root(lambda x: int_h(x) - ui, x0=0).x[0]
        X_j = np.zeros(n1)
        for k, ui in enumerate(d * np.linspace(1 / n1, 1, n1, endpoint=True)):
            X_j[k] = optimize.root(lambda x: int_h(x) - ui, x0=0).x[0]

        X_j[1:] = (X_j[:-1] + X_j[1:]) / 2
        X_j[0] = X_j[0] / 2

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
        mu = torch.ones_like(h_true) * d / len(X[:, 1])


        # Assert JET is correct;
          # Check that the jet profile starts off obeying the convexity principle
        assert (
            2 * f
            + g
            * (
                -2
                * a
                * b**2
                * torch.tanh(b * (G[:, 1] - c))
                * (1 - torch.tanh(b * (G[:, 1] - c)) ** 2)
            )
            > 0
        ).all()
        assert (
            f
            * (
                f
                + g
                * (
                    -2
                    * a
                    * b**2
                    * torch.tanh(b * (G[:, 1] - c))
                    * (1 - torch.tanh(b * (G[:, 1] - c)) ** 2)
                )
            )
        ).all()

    elif profile_type == "jet_no_lloyd":
        # Assigning uniform weighting to points - Lloyde type
        def height_func(x):
            return a * np.tanh(b * (x - c)) + d

        X_j = d * np.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1, endpoint=True)

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
        mu = h_true * d / h_true.sum()

    elif profile_type == "uniform":
        X = torch.cartesian_prod(
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        )
        Y = torch.cartesian_prod(
            torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
            torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
        )
        G = torch.cartesian_prod(
            torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
            torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
        )
        h_true = torch.ones_like(X[:, 1]).view(-1, 1)
        mu = torch.ones_like(h_true) * d / len(X[:, 1])
    elif profile_type == "thin":
        d = (epsilon / g) * 0.5
        print(f'Think fluid thickness {d}')
        X = torch.cartesian_prod(
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        )
        Y = torch.cartesian_prod(
            torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
            torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
        )
        G = torch.cartesian_prod(
            torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
            torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),s
        )
        h_true = d*torch.ones_like(X[:, 1]).view(-1, 1)
        mu = torch.ones_like(h_true) * d / len(X[:, 1])
    elif profile_type == "incline":
        X, Y, G, h_true, mu = incline(epsilon, f, g, a, b, c, d)
    elif profile_type == "incline_no_lloyd":
        X, Y, G, h_true, mu = incline_no_lloyd(
            epsilon, f, g, a, b, c, d
        )
    elif profile_type == "incline2D_lloyd":
        X, Y, G, h_true, mu = incline2D_lloyd(
            device, dtype,epsilon, f, g, a, b, c, d, tol=tol
        )
    elif profile_type == "jet2D_lloyd":
        X, Y, G, h_true, mu = jet2D_lloyd(
            device, dtype, epsilon, f, g, a, b, c, d, tol=tol
        )
    elif profile_type == "uniform2D_lloyd":
        X, Y, G, h_true, mu = uniform2D_lloyd(
            device, dtype, epsilon, f, g, a, b, c, d, tol=tol
        )

    return X, Y, G, h_true

def swsg_class_generate(
    X, Y, G, h_true, device, dtype, f=1.0, g=0.1, cuda=0, tol=1e-11, d=1.0, epsilon=0.05, 
):
    if cuda is None:
        swsg_class = SWSGSinkNewton(pykeops=True, set_fail=True)
    else:
        swsg_class = SWSGSinkNewton(
            pykeops=True, set_fail=False, cuda_device=f"cuda:{cuda}"
        )
    # torch.cuda.set_device(5)
    swsg_class.parameters(ε=epsilon, f=f, g=g)
    h_leb = torch.ones_like(X[:,0]) * d / len(X[:,0])
    sigma = torch.ones_like(G[:,0]) * d / len(G[:,0])
    swsg_class.densities(source_points=G.detach().cpu(), target_points=X.detach().cpu(), source_density=sigma.detach().cpu(), target_density=h_leb.detach().cpu())

    # assert (swsg_class.β_t == mu).all()
    torch.cuda.empty_cache()

    return X.detach().to(device), Y.detach().to(device), G.detach().to(device), swsg_class, h_true.detach().to(device)


def pykeops_approach_avoid_Wminus1(
    swsg_class, lambert_tolerance=1e-12, tolerance=1e-12, sym_update=True
):

    output = swsg_class.swsgsinkhorn_loop(
        tol=tolerance,
        verbose=False,
        newton_tol=tolerance,
        sinkhorn_divergence=True,
        energy=True,
    )

    print("Final convergence", output)

    #
    return (
        swsg_class.f,
        swsg_class.g,
        swsg_class.debias_f.f,
        swsg_class.g_s,
        output,
        swsg_class,
    )


def swsg_run_halley_sinkhorn(swsg_class, lambert_tolerance=1e-12, tolerance=1e-12):

    # initialise
    max_iterates = int(
        -1.5 / swsg_class.epsilon.cpu() * np.log(swsg_class.epsilon.cpu())
    )
    error_list = []

    output = swsg_class.swsgsinkhorn_loop(
        sinkhorn_steps=max_iterates,
        tol=tolerance,
        halley_updates=True,
        newton_tol=lambert_tolerance,
    )
    print("SWSG final:", output)
    swsg_class.debias_f = UnbalancedOT(
        set_fail=swsg_class.set_fail,
        pykeops=swsg_class.pykeops,
        debias=False,
        cuda_device=swsg_class.device,
    )
    swsg_class.debias_f.parameters(
        swsg_class.epsilon, swsg_class.rho, swsg_class.cost_const
    )
    swsg_class.debias_f.densities(
        swsg_class.X_s, swsg_class.X_s, swsg_class.α_s, swsg_class.α_s
    )

    f_update, g_update, i_sup = swsg_class.debias_f.sinkhorn_algorithm(
        tol=1e-12,
        verbose=False,
        aprox="balanced",
        convergence_repeats=3,
    )
    return (
        swsg_class.f,
        swsg_class.g,
        swsg_class.debias_f.f,
        torch.zeros_like(swsg_class.g),
        error_list,
        swsg_class,
    )



def swsg_solver(swsg_class, method="three", lambert_tolerance=1e-12, tolerance=1e-12):

    method_dict = dict(
        one=swsg_run_halley_sinkhorn,
        # two=swsg_run_debiased_newton,
        # three=pykeops_approach_W0Wminus1,
        four=pykeops_approach_avoid_Wminus1,
        # five=torch_cdist_approach,
        # six=partial(pykeops_approach_W0Wminus1, branch=0),
        # seven=partial(pykeops_approach_W0Wminus1, sym_update=False),
        # eight=partial(pykeops_approach_avoid_Wminus1, sym_update=False),
    )

    # Method:
    # 1/2/3
    tic = perf_counter_ns()
    φ, ψ, φ_s, ψ_s, error_list, swsg_class = method_dict[method](
        swsg_class, lambert_tolerance=lambert_tolerance, tolerance=tolerance
    )
    toc = perf_counter_ns()
    print('SWSGLOOP:', method, toc-tic)

    grad_phi = swsg_class.pykeops_formulas.barycentres(
        ψ,
        φ,
        swsg_class.X_s,
        swsg_class.Y_t,
        swsg_class.epsilon,
        swsg_class.α_s,
        swsg_class.β_t,
        swsg_class.Y_t,
        swsg_class.f_constant**2,
    )
    grad_phi_debias = swsg_class.pykeops_formulas.barycentres(
        φ_s,
        φ_s,
        swsg_class.X_s,
        swsg_class.X_s,
        swsg_class.epsilon,
        swsg_class.α_s,
        swsg_class.α_s,
        swsg_class.X_s,
        swsg_class.f_constant**2,
    )

    return φ, ψ, φ_s, ψ_s, grad_phi, grad_phi_debias, error_list


def Sinkhorn_Divergence_balanced(X, α, Y, β, uotclass=None):
    tic = perf_counter_ns()
    if uotclass is None:
        # Initialise it
        uotclass = DebiasedUOT(pykeops=True)
        uotclass.parameters(epsilon=0.001)
        uotclass.densities(X, Y, α, β)

        # solve the symmteric DENSE problem once
        uotclass.debias_f = UnbalancedOT(
            set_fail=uotclass.set_fail,
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

        f_update, g_update, i_sup = uotclass.debias_f.sinkhorn_algorithm(
            tol=1e-12,
            verbose=False,
            aprox="balanced",
            convergence_repeats=3,
        )

        print("DENSE symmetric update final convergence:", f_update, g_update, i_sup)
    else:
        # Else update the densities
        uotclass.densities(X, Y, α, β)

    # Run sinkhorn
    uotclass.sinkhorn_algorithm(
        f0=torch.zeros_like(α), g0=torch.zeros_like(β), aprox="balanced", tol=1e-14
    )

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
        tol=1e-12,
        verbose=False,
        left_divergence=uotclass.right_div.print_type(),
        right_divergence=uotclass.right_div.print_type(),
        convergence_repeats=3,
    )
    toc = perf_counter_ns()

    print("Symmetric update final convergence:", f_update, g_update, i_sup)
    print(f"W2 Computed in {toc-tic} ns")
    force_type = "pykeops"

    return (
        sum(uotclass.primal_cost(force_type=force_type))
        - (
            sum(uotclass.debias_f.primal_cost(force_type=force_type))
            + sum(uotclass.debias_g.primal_cost(force_type=force_type))
        )
        / 2
        + uotclass.epsilon * (uotclass.α_s.sum() - uotclass.β_t.sum()) ** 2 / 2
    ).cpu().item(), uotclass


def check_w_minus_one_residual(swsg_class, ψ_s, ψ, branch=-1, lambert_tolerance=1e-10):

    temp = swsg_class.pykeops_formulas.barycentres_bottom(
        ψ_s,
        swsg_class.Y_t,
        swsg_class.Y_t,
        swsg_class.epsilon,
        swsg_class.β_t,
        swsg_class.f_constant**2,
    )
    temp = (
        -swsg_class.g_constant
        * torch.exp(ψ / swsg_class.epsilon)
        * temp
        / swsg_class.epsilon
    )

    res = torch.linalg.norm(ψ_s.cpu()  - (
        ψ.cpu()
        - swsg_class.epsilon.cpu() * lambertw_scipy(temp.cpu(), k=branch, tol=lambert_tolerance).real
    ))

    return res.item()


# def run_sim(cuda=None, title="uniform_dense_", profile="uniform", myclass=True, d=1):
#     method_data = {}

#     global device, dtype
#     device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
#     dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor

#     for method in ["one", "four"]:
#         method_data[method] = {}
#         for epsilon in [0.05, 0.025, 0.0125, 0.00625, 0.003125]:
#             method_data[method][epsilon] = {}
#             method_data[method][epsilon]["bias"] = {}
#             method_data[method][epsilon]["debias"] = {}
#             method_data[method][epsilon]["h_error"] = {}

#     for epsilon in [0.05, 0.025, 0.0125, 0.00625, 0.003125]:
#         for method in ["four", "one"]:
#             print("Staring:", method, epsilon)

#             X, Y, G, swsg_class, h_true = initialisation(
#                 epsilon=epsilon, d=d, profile_type=profile, cuda=cuda
#             )

#             tic = perf_counter_ns()
#             φ, ψ, φ_s, ψ_s, grad_phi, grad_phi_debias, error_list = swsg_solver(
#                 swsg_class, method=method, tolerance=1e-11, lambert_tolerance=1e-11
#             )
#             toc = perf_counter_ns()
#             print(f'SWSG LOOP for method {method}:', toc-tic, ' ns')

#             h = (ψ_s - ψ) / 0.1
#             debias_x_star = grad_phi - (grad_phi_debias - G)
            
#             method_data[method][epsilon]["lambert_0"] = check_w_minus_one_residual(swsg_class, ψ_s, ψ, branch=0, lambert_tolerance=1e-11)
#             method_data[method][epsilon]["lambert_minus1"] = check_w_minus_one_residual(swsg_class, ψ_s, ψ, branch=-1, lambert_tolerance=1e-11)
#             print('BRANCH ERROR:', method_data[method][epsilon]["lambert_0"], method_data[method][epsilon]["lambert_minus1"])

#             print("h, psi_s, psi", h.sum(), ψ_s.sum(), ψ.sum())

#             l1 = torch.linalg.norm(h_true.view(-1, 1) - h, ord=1) * torch.unique(
#                 swsg_class.β_t
#             )
#             l1_h = l1.item()
#             l2 = (
#                 torch.linalg.norm(h_true.view(-1, 1) - h, ord=2)
#                 * torch.unique(swsg_class.β_t) ** 0.5
#             )
#             l2_h = l2.item()
#             linf_h = torch.linalg.norm(h_true.view(-1, 1) - h, ord=float("inf")).item()

#             f = open(f"data_store/{title}_convergence_data_store.pkl", "wb")
#             pickle.dump(method_data, f)
#             f.close()

#             # Saving profiles:
#             # ######## Saving h error
#             method_data[method][epsilon]["h_error"].update(dict(
#                 l1=l1_h, l2=l2_h, linf=linf_h
#             ))

#             # ######## Saving mesh error to fitted?

#             l1 = torch.linalg.norm(Y - grad_phi, ord=1) * torch.unique(swsg_class.β_t)
#             l1 = l1.item()
#             l2 = (
#                 torch.linalg.norm(Y - grad_phi, ord=2)
#                 * torch.unique(swsg_class.β_t) ** 0.5
#             )
#             l2 = l2.item()
#             linf = torch.linalg.norm(Y - grad_phi, ord=float("inf")).item()
#             method_data[method][epsilon]["bias"]["fit_mesh_error"] = dict(
#                 l1=l1, l2=l2, linf=linf
#             )

#             l1 = torch.linalg.norm(Y - debias_x_star, ord=1) * torch.unique(
#                 swsg_class.β_t
#             )
#             l1 = l1.item()
#             l2 = (
#                 torch.linalg.norm(Y - debias_x_star, ord=2)
#                 * torch.unique(swsg_class.β_t) ** 0.5
#             )
#             l2 = l2.item()
#             linf = torch.linalg.norm(Y - debias_x_star, ord=float("inf")).item()
#             method_data[method][epsilon]["debias"]["fit_mesh_error"] = dict(
#                 l1=l1, l2=l2, linf=linf
#             )

#             # ######## Saving mesh error to regular?
#             l1 = torch.linalg.norm(X - grad_phi, ord=1) * torch.unique(swsg_class.β_t)
#             l1 = l1.item()
#             l2 = (
#                 torch.linalg.norm(X - grad_phi, ord=2)
#                 * torch.unique(swsg_class.β_t) ** 0.5
#             )
#             l2 = l2.item()
#             linf = torch.linalg.norm(X - grad_phi, ord=float("inf")).item()
#             method_data[method][epsilon]["bias"]["regular_mesh_error"] = dict(
#                 l1=l1, l2=l2, linf=linf
#             )

#             l1 = torch.linalg.norm(X - debias_x_star, ord=1) * torch.unique(
#                 swsg_class.β_t
#             )
#             l1 = l1.item()
#             l2 = (
#                 torch.linalg.norm(X - debias_x_star, ord=2)
#                 * torch.unique(swsg_class.β_t) ** 0.5
#             )
#             l2 = l2.item()
#             linf = torch.linalg.norm(X - debias_x_star, ord=float("inf")).item()
#             method_data[method][epsilon]["debias"]["regular_mesh_error"] = dict(
#                 l1=l1, l2=l2, linf=linf
#             )

#             # ############### Wasserstien Error ###############
#             # Generate the dense mesh with 250 000 points.
#             if '2D_lloyd' in profile:
#                 X_dense, _, _, _, h_true_dense = initialisation(
#                     epsilon=epsilon / 3, d=d, profile_type=profile.split('2D')[0], cuda=cuda
#                 )
#                 X_dense = X_dense.type(dtype)
#                 h_true_dense = h_true_dense.type(dtype)
#             else:
#                 X_dense, _, _, _, h_true_dense = initialisation(
#                     epsilon=epsilon / 3, d=d, profile_type=profile, cuda=cuda
#                 )
#                 X_dense = X_dense.type(dtype)
#                 h_true_dense = h_true_dense.type(dtype)
            
            
#             N = len(X)
#             N_dense = len(X_dense)
#             if myclass:
   
#                 print(N, N_dense)
#                 print("Wass distanced ....")

#                 # Compute the Wass distance for the reconstructed height against dense sampling.

#                 s, wass_dist = Sinkhorn_Divergence_balanced(
#                     X_dense,
#                     h_true_dense / N_dense,
#                     swsg_class.Y_t,
#                     h / N,
#                 )

#                 method_data[method][epsilon]["h_error"]["W_error"] = s
#                 print("h error", s)

#                 s, wass_dist = Sinkhorn_Divergence_balanced(
#                     X_dense,
#                     h_true_dense / N_dense,
#                     Y,
#                     torch.ones_like(h) / N,
#                 )
#                 method_data[method][epsilon]["h_error"]["original"] = s

#                 # Regular debiased
#                 s, wass_dist = Sinkhorn_Divergence_balanced(
#                     X_dense,
#                     h_true_dense / N_dense,
#                     debias_x_star,
#                     torch.ones_like(h) / N,
#                 )
#                 method_data[method][epsilon]["debias"]["W_error_regular"] = s
#                 print("regular debiased", s)

#                 # Regular biased
#                 s, wass_dist = Sinkhorn_Divergence_balanced(
#                     X_dense,
#                     h_true_dense / N_dense,
#                     grad_phi,
#                     torch.ones_like(h) / N,
#                 )
#                 method_data[method][epsilon]["bias"]["W_error_regular"] = s
#                 print("regular biased", s)

#             else:
#                 # Dense processing to correct device
#                 dense_weights = swsg_class._torch_numpy_process(h_true_dense / N_dense)
#                 dense_points=  swsg_class._torch_numpy_process(X_dense)

#                 uni_weights = swsg_class._torch_numpy_process(torch.ones_like(h) / N)

#                 loss = SamplesLoss("sinkhorn", p=2, blur=np.sqrt(epsilon / 3), scaling=0.9999, backend='multiscale')
#                 N = len(X)
#                 N_dense = len(X_dense)

#                 s = loss(dense_weights, dense_points, uni_weights, swsg_class._torch_numpy_process(Y))
#                 method_data[method][epsilon]["h_error"]["original"] = s.detach().cpu()
#                 print('Oringal Se loss:', s)

#                 s = loss(dense_weights, dense_points, swsg_class._torch_numpy_process(h / N), swsg_class.Y_t)
#                 method_data[method][epsilon]["h_error"]["W_error"] = s.detach().cpu()
#                 print("h error", s)
                
#                 # REgular debiased
#                 s = loss(dense_weights, dense_points, uni_weights, swsg_class._torch_numpy_process(debias_x_star))
#                 method_data[method][epsilon]["debias"]["W_error_regular"] = s.detach().cpu()
#                 print("regular debiased", s)
                
#                 # Regular biased
#                 s = loss(dense_weights, dense_points, uni_weights, swsg_class._torch_numpy_process(grad_phi))
#                 print("regular bias", s)

#                 method_data[method][epsilon]["bias"]["W_error_regular"] = s.detach().cpu()

#             # ######## Saving error_list
#             method_data[method][epsilon]["error_list"] = error_list
#             # ########


#             f = open(f"data_store/{title}_convergence_data_store.pkl", "wb")
#             pickle.dump(method_data, f)
#             f.close()

#             print("Finishing:", method, epsilon)
#             torch.cuda.empty_cache()
#             print("cleared")

