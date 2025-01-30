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
    lloyd_loss = SamplesLoss(
        "sinkhorn", p=2, blur=blur, scaling=0.99, backend="multiscale"
    )
    err = 1e3
    kmax = 100
    count = 0

    # Set up the tqdm progress bar
    with tqdm(total=kmax, desc="Llody Progress", unit="iter") as pbar:
        while err > tol and count < kmax:
            L_ = lloyd_loss(beta, Y, alpha, X)
            grad = torch.autograd.grad(L_, Y)[0]

            Y = Y - lr * grad * len(
                beta
            )  # Maybe times by beta here - though it is actually 1/N?
            err = torch.linalg.norm(grad)
            count += 1

            # Update the progress bar
            pbar.update(1)
            pbar.set_postfix({"Error": err.item(), "Loss": L_.item()})

    print("Final W2 loss:", lloyd_loss(beta, Y, alpha, X))
    return Y


def jet2D_lloyd(
    device,
    dtype,
    epsilon=0.05,
    f=1.0,
    g=0.1,
    a=0.1,
    b=10.0,
    c=0.5,
    d=1.0,
    dense_scale=3,
    tol=1e-11,
):

    # Decide on parameters - baseed off of epsilon
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = dense_scale * int(1 / epsilon), dense_scale * int(1 / epsilon)
    # 2D Llody

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return a * torch.tanh(b * (x - c)) + d

    # Llody fit against dense sampling
    Y = (
        torch.rand((n1 * n2, 2), device=device).type(dtype) * 50 - 24
    )  # Random box in [-4, 6]x[-4, 6] becuase I want solutions in [0, 1]x[0,1]
    Y = Y.requires_grad_(True)
    beta = torch.ones((n1 * n2), device=device).type(dtype)
    beta /= beta.sum()

    # Uniform sample - maybe dense?
    X = (
        torch.cartesian_prod(
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        )
        .to(device)
        .type(dtype)
    )
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)
    alpha /= alpha.sum()

    Y = lloyd(epsilon**0.5, X, Y, alpha, beta, tol=tol)

    # generate regulr uniform grid of correct size for universe;
    # Overwriting dense grid used for lloyd sampling
    X = (
        torch.cartesian_prod(
            torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
            torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
        )
        .to(device)
        .type(dtype)
    )
    # Not normalised to one for later purposes.
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)

    # Calculate nabla P: x + f^2 * g * partial h
    G = Y.detach().clone()
    G[:, 1] = G[:, 1] + f**2 * g * a * b * (
        1 - torch.tanh(b * (Y[:, 1] - 0.5)) ** 2
    )  # X Y?

    h_true = alpha.view(-1, 1)
    mu = torch.ones_like(h_true) * d / len(X[:, 1])

    return X, Y, G, h_true, mu


def incline2D_lloyd(
    device,
    dtype,
    epsilon=0.05,
    f=1.0,
    g=0.1,
    a=0.1,
    b=10.0,
    c=0.5,
    d=1.0,
    dense_scale=3,
    tol=1e-11,
):

    # Decide on parameters - baseed off of epsilon
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = dense_scale * int(1 / epsilon), dense_scale * int(1 / epsilon)
    # 2D Llody

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return a * b * (x - c) + d

    # Llody fit against dense sampling
    Y = (
        torch.rand((n1 * n2, 2), device=device).type(dtype) * 50 - 24
    )  # Random box in [-4, 6]x[-4, 6] becuase I want solutions in [0, 1]x[0,1]
    Y = Y.requires_grad_(True)
    beta = torch.ones((n1 * n2), device=device).type(dtype)
    beta /= beta.sum()

    # Uniform sample - maybe dense?
    X = (
        torch.cartesian_prod(
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        )
        .to(device)
        .type(dtype)
    )

    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)
    alpha /= alpha.sum()

    Y = lloyd(epsilon**0.5, X, Y, alpha, beta, tol=tol)

    # generate regulr uniform grid of correct size for universe;
    # Overwriting dense grid used for lloyd sampling
    X = (
        torch.cartesian_prod(
            torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
            torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
        )
        .to(device)
        .type(dtype)
    )
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)

    # Calculate nabla P: x + f^2 * g * partial h
    G = Y.detach().clone()
    G[:, 1] = G[:, 1] + f**-2 * g * a * b

    h_true = alpha.view(-1, 1)
    mu = torch.ones_like(h_true) * d / len(X[:, 1])

    return X, Y, G, h_true, mu


def uniform2D_lloyd(
    device,
    dtype,
    epsilon=0.05,
    f=1.0,
    g=0.1,
    a=0.1,
    b=10.0,
    c=0.5,
    d=1.0,
    dense_scale=3,
    tol=1e-11,
):

    # Decide on parameters - baseed off of epsilon
    n1, n2 = int(1 / epsilon), int(1 / epsilon)
    m1, m2 = dense_scale * int(1 / epsilon), dense_scale * int(1 / epsilon)
    # 2D Llody

    # Assigning uniform weighting to points - Lloyde type
    def height_func(x):
        return torch.ones_like(x) / len(x)

    # Llody fit against dense sampling
    Y = (
        torch.rand((n1 * n2, 2), device=device).type(dtype) * 50 - 24
    )  # Random box in [-4, 6]x[-4, 6] becuase I want solutions in [0, 1]x[0,1]
    Y = Y.requires_grad_(True)
    beta = torch.ones((n1 * n2), device=device).type(dtype)
    beta /= beta.sum()

    # Uniform sample - maybe dense?
    X = (
        torch.cartesian_prod(
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        )
        .to(device)
        .type(dtype)
    )

    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)
    alpha /= alpha.sum()

    Y = lloyd(epsilon**0.5, X, Y, alpha, beta, tol=tol)

    # generate regulr uniform grid of correct size for universe;
    # Overwriting dense grid used for lloyd sampling
    X = (
        torch.cartesian_prod(
            torch.linspace(1 / (2 * n2), 1 - 1 / (2 * n2), n2),
            torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1),
        )
        .to(device)
        .type(dtype)
    )
    alpha = torch.Tensor(height_func(X[:, 1])).to(device).type(dtype)

    # Calculate nabla P: x + f^2 * g * partial h
    G = Y.detach().clone()

    h_true = alpha.view(-1, 1)
    mu = torch.ones_like(h_true) * d / len(X[:, 1])

    return X, Y, G, h_true, mu


# pylint: disable=E1101
def incline(epsilon, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0):
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

    # X_j = np.zeros(n1)
    # for k, ui in enumerate(d * np.linspace(1 / n1, 1, n1, endpoint=True)):
    #     X_j[k] = optimize.root(lambda x: int_h(x) - ui, x0=0).x[0]
    # X_j[1:] = (X_j[:-1] + X_j[1:]) / 2
    # X_j[0] = X_j[0] / 2

    X_j = torch.linspace(1 / (2 * n1), 1 - 1 / (2 * n1), n1)

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
    device,
    dtype,
    epsilon=0.05,
    f=1.0,
    g=0.1,
    a=0.1,
    b=10.0,
    c=0.5,
    d=1.0,
    profile_type="jet",
    cuda=0,
    tol=1e-11,
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

        # X_j = np.zeros(n1)
        # for k, ui in enumerate(d * np.linspace(1 / n1, 1, n1, endpoint=True)):
        #     X_j[k] = optimize.root(lambda x: int_h(x) - ui, x0=0).x[0]

        # X_j[1:] = (X_j[:-1] + X_j[1:]) / 2
        # X_j[0] = X_j[0] / 2

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
        print(f"Think fluid thickness {d}")
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
            s,
        )
        h_true = d * torch.ones_like(X[:, 1]).view(-1, 1)
        mu = torch.ones_like(h_true) * d / len(X[:, 1])
    elif profile_type == "incline":
        X, Y, G, h_true, mu = incline(epsilon, f, g, a, b, c, d)
    elif profile_type == "incline_no_lloyd":
        X, Y, G, h_true, mu = incline_no_lloyd(epsilon, f, g, a, b, c, d)
    elif profile_type == "incline2D_lloyd":
        X, Y, G, h_true, mu = incline2D_lloyd(
            device, dtype, epsilon, f, g, a, b, c, d, tol=tol
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
    X,
    Y,
    G,
    h_true,
    device,
    dtype,
    f=1.0,
    g=0.1,
    cuda=0,
    tol=1e-11,
    d=1.0,
    epsilon=0.05,
    lloyd=True,
):
    if cuda is None:
        swsg_class = SWSGSinkNewton(pykeops=True, cuda_device='cpu')
    else:
        swsg_class = SWSGSinkNewton(
            pykeops=True, cuda_device=f"cuda:{cuda}"
        )
    # torch.cuda.set_device(5)
    swsg_class.parameters(ε=epsilon, f=f, g=g)
    h_leb = torch.ones_like(X[:, 0]) * d / len(X[:, 0])

    if lloyd:
        sigma = torch.ones_like(G[:, 0]) * d / len(G[:, 0])
    else:
        sigma = h_true / h_true.sum()

    swsg_class.densities(
        source_points=G.detach().cpu(),
        target_points=X.detach().cpu(),
        source_density=sigma.detach().cpu(),
        target_density=h_leb.detach().cpu(),
    )

    # assert (swsg_class.β_t == mu).all()
    torch.cuda.empty_cache()

    return (
        X.detach().to(device),
        Y.detach().to(device),
        G.detach().to(device),
        swsg_class,
        h_true.detach().to(device),
    )


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

    print("SWSG Sinkhorn final:", output[0], " in ", output[2], " iterations ")

    #
    return (
        swsg_class.f,
        swsg_class.g,
        swsg_class.debias_f.f,
        swsg_class.g_s,
        output[3],
        swsg_class,
    )


def swsg_run_halley_sinkhorn(swsg_class, lambert_tolerance=1e-12, tolerance=1e-12):

    # initialise
    max_iterates = int(
        -1.5 / swsg_class.epsilon.cpu() * np.log(swsg_class.epsilon.cpu())
    )

    output = swsg_class.swsgsinkhorn_loop(
        sinkhorn_steps=max_iterates,
        tol=tolerance,
        halley_updates=True,
        newton_tol=lambert_tolerance,
    )

    error_list = output[3]

    print("SWSG halley final:", output[0], " in ", output[2], " iterations ")
    swsg_class.debias_f = UnbalancedOT(
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
    print("SWSGLOOP:", method, toc - tic)

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


def compute_dense_symmetric_potential(X, α, Y, β, cuda="cuda:0", force_type="pykeops"):
    """
    Compute the dense symmetric potential when no precomputed potential is provided.
    """
    uotclass = DebiasedUOT(pykeops=True, cuda_device=cuda)
    uotclass.parameters(epsilon=0.002)

    # We can tensorise:
    uotclass.densities(X, Y, α, β)

    tic = perf_counter_ns()
    f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
        tol=1e-12,
        verbose=False,
        aprox="balanced",
        convergence_repeats=3,
        convergence_or_fail=True,
    )
    toc = perf_counter_ns()

    d = uotclass.dual_cost(force_type=force_type)

    print("DENSE symmetric update final convergence:", f_update, g_update, i_sup)
    print(f"W2 Computed in {toc - tic} ns")

    return dict(f=uotclass.g.view(-1,1).cpu(), dual=sum(d))


def compute_sinkhorn_divergence(
    X, α, Y, β, dense_symmetric_potential, cuda="cuda:0", force_type="pykeops"
):
    """
    Compute the symmetric update using a precomputed potential.
    """
    uotclass = DebiasedUOT(pykeops=True, cuda_device=cuda)
    uotclass.parameters(epsilon=0.002)
    uotclass.densities(X, Y, α, β)

    # Run Sinkhorn
    tic = perf_counter_ns()
    uotclass.sinkhorn_algorithm(
        f0=dense_symmetric_potential["f"],
        g0=torch.zeros_like(uotclass.β_t),
        aprox="balanced",
        tol=1e-14,
    )

    # Solve the new symmetric potential problem
    uotclass.debias_g = UnbalancedOT(
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
    print(f"W2 Computed in {toc - tic} ns")

    return (
        (
            sum(uotclass.dual_cost(force_type=force_type))
            - (
                dense_symmetric_potential["dual"]
                + sum(uotclass.debias_g.dual_cost(force_type=force_type))
            )
            / 2
            + uotclass.epsilon * (uotclass.α_s.sum() - uotclass.β_t.sum()) ** 2 / 2
        )
        .cpu()
        .item()
    )


###################################
def Sinkhorn_Divergence_balanced(
    X,
    α,
    Y,
    β,
    dense_symmetric_potential=None,
    f0=None,
    g0=None,
    force_type="pykeops",
    tol=1e-12,
    epsilon=0.002,
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
        f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
            f0=f0, g0=g0, aprox="balanced", tol=tol, convergence_or_fail=True, convergence_repeats=3,


        )
        print("Sinkhorn full compute final convergence:", f_update, g_update, i_sup)
        s = uotclass.sinkhorn_divergence(tol=tol, force_type='pykeops', return_type='dual')
        return s.cpu().item(), uotclass
    else:

        # Run sinkhorn
        f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
            f0=f0, g0=g0, aprox="balanced", tol=tol, convergence_or_fail=True, convergence_repeats=3,
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
                dense_symmetric_potential["dual"].to(uotclass.device)
                + sum(uotclass.debias_g.dual_cost(force_type=force_type))
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

    res = torch.linalg.norm(
        ψ_s.cpu()
        - (
            ψ.cpu()
            - swsg_class.epsilon.cpu()
            * lambertw_scipy(temp.cpu(), k=branch, tol=lambert_tolerance).real
        )
    )

    return res.item()


