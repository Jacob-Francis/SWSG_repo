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


def normal_pdf(x, y, mu_x, mu_y, sigma, alpha):
    """
    Calculate the PDF of a bivariate normal distribution.
    """
    # Constants
    sigma2 = torch.tensor([sigma**2])
    mu_x_ = torch.tensor([mu_x])
    mu_y_ = torch.tensor([mu_y])

    norm_factor = 1 / (
        2 * torch.pi * sigma2
    )  # _x * sigma_y * torch.sqrt(1 - rho ** 2))
    # Z computation
    z_x = x - mu_x_
    z_y = y - mu_y_

    z = z_x**2 + z_y**2

    # PDF computation
    pdf = alpha * norm_factor * torch.exp(-0.5 * z / sigma2)
    pdfg0 = -alpha * z_x / sigma2 * norm_factor * torch.exp(-0.5 * z / sigma2)
    pdfg1 = -alpha * z_y / sigma2 * norm_factor * torch.exp(-0.5 * z / sigma2)

    return pdf.unsqueeze(-1), pdfg0, pdfg1


def jet_profile_initialisation(
    epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0, pykeop=True
):
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
        return a * np.log(np.cosh(b * (x - 0.5)) / np.cosh(-b * 0.5)) / b + d * x

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
    mu = torch.ones_like(h_true) * d / len(X[:, 1])

    no, no0, no1 = normal_pdf(
        X[:, 0], X[:, 1], 0.5, 0.3, 0.1, strength
    )  ## 0 is stationnary
    h_true = h_true + no
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
    fullcompute=False,
    allow_annealing=True,
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
                f0=f0,
                g0=g0,
                aprox="balanced",
                tol=tol,
                convergence_repeats=3,
                convergence_or_fail=allow_annealing,
            )
        except RuntimeWarning:
            print("EPSILON ANNEALING")
            f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
                f0=f0,
                g0=g0,
                aprox="balanced",
                tol=tol,
                convergence_or_fail=False,
                epsilon_annealing=True,
            )
        print("Sinkhorn full compute final convergence:", f_update, g_update, i_sup)
        s = uotclass.sinkhorn_divergence(
            tol=tol, force_type="pykeops", return_type="dual"
        )
        return s.cpu().item(), uotclass
    else:

        f_update, g_update, i_sup = uotclass.sinkhorn_algorithm(
            f0=f0,
            g0=g0,
            aprox="balanced",
            tol=tol,
            convergence_or_fail=False,
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
        uotclass.debias_f.f = dense_symmetric_potential["f"].to(uotclass.α_s)
        uotclass.debias_f.g = dense_symmetric_potential["f"].to(uotclass.α_s)

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


def compute_norms_and_plot(
    method="heun", strength=0.0001, dt=0.05, time_steps=[199], verbose=False
):
    """
    Computes L1, L2, and Linf norms for different grid resolutions and
    plots error trends for multiple time steps using NN-interpolation onto a finer grid.

    Parameters:
    - method (str): Numerical method used in simulation (e.g., "heun").
    - strength (float): Perturbation strength.
    - dt (float): Time step value.
    - time_steps (list of int): List of time indices to compare solutions.
    """

    # Define fine-grid parameters
    epsilon_fine = 0.0078125 / 2
    N_fine = int((1 / epsilon_fine) ** 2)
    n_fine = int(np.sqrt(N_fine))

    if verbose:
        print(
            f"Fine Grid -> ε: {epsilon_fine}, N: {N_fine}, n: {n_fine}, n²: {n_fine**2}"
        )

    # Load fine grid data
    fine_file = (
        f"data_store/output_{method}_{dt}_{epsilon_fine}_strength_{strength}.pkl"
    )
    with open(fine_file, "rb") as f:
        data_fine = pickle.load(f)

    # Initialise fine grid profile
    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(
        epsilon_fine, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0
    )
    sigma_weights_dense /= sigma_weights_dense.sum()
    X_dense_ = torch.linspace(1 / (2 * n_fine), 1 - 1 / (2 * n_fine), n_fine)

    # Figure settings: scale height by number of time steps
    fig, axes = plt.subplots(
        len(time_steps), 1, figsize=(8, 4 * len(time_steps)), dpi=200, sharex=True
    )

    if len(time_steps) == 1:
        axes = [axes]

    for ax, time_step in zip(axes, time_steps):
        l1_norms, l2_norms, linf_norms, epsilons = [], [], [], []

        # Iterate over different coarse grid resolutions
        for epsilon in [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]:
            N = int((1 / epsilon) ** 2)
            n = int(np.sqrt(N))
            epsilons.append(epsilon)

            if verbose:
                print(f"Coarse Grid -> ε: {epsilon}, N: {N}, n: {n}, n²: {n**2}")

            # Load coarse grid data
            coarse_file = (
                f"data_store/output_{method}_{dt}_{epsilon}_strength_{strength}.pkl"
            )
            with open(coarse_file, "rb") as f:
                data_course = pickle.load(f)

            # Initialize coarse grid profile
            X, _, _, sigma_weights, _ = jet_profile_initialisation(
                epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0
            )
            sigma_weights /= sigma_weights.sum()
            if verbose:
                print(
                    f"Coarse Sum: {data_course[1][:, :, time_step].sum()}, Fine Sum: {data_fine[1][:, :, time_step].sum()}"
                )

            # Interpolate coarse data onto fine grid
            coarse_data = data_course[1][:, :, time_step].reshape(n, n).numpy()
            interpolated = interpolate.interpn(
                (torch.linspace(1 / (2 * n), 1 - 1 / (2 * n), n).numpy(),) * 2,
                coarse_data,
                X_dense.numpy(),
                method="nearest",
                bounds_error=False,
                fill_value=None,
            )

            # Compute errors
            fine_data = (
                data_fine[1][:, :, time_step] / data_fine[1][:, :, time_step].sum()
            ).view(
                -1,
            )
            interpolated_data = torch.tensor(interpolated / interpolated.sum()).view(
                -1,
            )
            diff = (interpolated_data - fine_data) * len(X_dense)

            # Compute L1, L2, Linf norms
            l1_norms.append(
                (torch.linalg.norm(diff, ord=1) / len(sigma_weights_dense)).item()
            )
            l2_norms.append(
                (
                    torch.linalg.norm(diff, ord=2) / np.sqrt(len(sigma_weights_dense))
                ).item()
            )
            linf_norms.append((torch.linalg.norm(diff, ord=float("inf"))).item())

            if verbose:
                print(
                    f"Interpolated Sum: {interpolated.sum()}, Coarse Weight Sum: {sigma_weights.sum()}, Fine Weight Sum: {sigma_weights_dense.sum()}"
                )

        # Plot error norms for this time step
        ax.loglog(
            epsilons,
            l1_norms,
            color=CB_COLORS[0],
            marker="o",
            linestyle="-",
            label=r"$L_1$ norm",
        )
        ax.loglog(
            epsilons,
            l2_norms,
            color=CB_COLORS[1],
            marker="s",
            linestyle="-",
            label=r"$L_2$ norm",
        )
        ax.loglog(
            epsilons,
            linf_norms,
            color=CB_COLORS[2],
            marker="^",
            linestyle="-",
            label=r"$L_\infty$ norm",
        )

        # Fit reference slope line (Slope = 1)
        ref_x = epsilons[-3:]
        ref_y = l1_norms[-3:]
        C = ref_y[-1] / ref_x[-1] + 1e-1  # Compute scaling constant
        trend_line = C * np.array(epsilons) ** 1  # y = C * x^1

        ax.loglog(
            epsilons,
            trend_line,
            "--",
            color="black",
            label="Slope 1 Trend Line",
            alpha=0.6,
        )

        # Labels, title, and legend
        ax.set_xticks(epsilons)
        ax.set_xticklabels(
            [rf"$\epsilon$ = {k:.5g} | n = {1/k**2:3g}" for k in epsilons], rotation=45
        )
        ax.set_xlabel(r"$\epsilon$", fontsize=12)
        ax.set_ylabel("Norm Value", fontsize=12)
        ax.set_title(f"T={(time_step + 1)*dt}", fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Adjust layout to prevent overlap
    plt.suptitle("Height Error with NN-Interpolation")
    plt.tight_layout()

    # Save and show
    plt.savefig("nest_height_interp_multiple_T.pdf")
    plt.show()


def plot_nest_heights(
    title, epsilons, file_prefix, Times, save_filename, slope=1, return_s=False
):
    """
    Plots the perturbed jet comparison against neighboring resolution.

    Parameters:
    - title (str): The title of the plot.
    - epsilons (list): List of epsilon values.
    - file_prefix (str): Prefix for the data files.
    - Times (list): List of time steps (e.g., ["T5", "T10"]).
    - save_filename (str): Name of the file to save the plot.
    - slope (float): Slope for the trend line.
    - return_s (bool): Whether to return the loaded data.
    """

    s = {}

    # Load data for different time steps
    for T in Times:
        with open(f"pickle_folder/{file_prefix}_{T}.pkl", "rb") as f:
            s[T] = pickle.load(f)

    plt.figure(figsize=(7, 5), dpi=200)

    # Define styles
    time_styles = {
        "T0": {"marker": "x"},
        "T5": {"marker": "o"},
        "T10": {"marker": "s"},
    }  # Square  # Circle

    colors = ["#E69F00", "#56B4E9", "#009E73"]  # Colorblind-safe colors
    linestyles = ["dashdot", "dashdot", "dashdot"]  # Different line styles for norms

    # Plot each dataset
    for i, key in enumerate(s.keys()):
        s_ = np.array(s[key])
        if any(s_ < 0):
            print(s_[s_ < 0])
        sqrt_s = np.sqrt(s_)

        plt.loglog(
            epsilons,
            sqrt_s,
            marker=time_styles[key]["marker"],
            markerfacecolor="none",  # No fill
            linestyle=linestyles[i % len(linestyles)],  # Cycle through styles
            color=colors[i % len(colors)],  # Cycle through colors
            label=f'time={key.split("T")[1]}',
            markersize=10,
        )

    # Define trend line with slope 1
    ref_index = len(epsilons) // 2  # Use middle epsilon as reference
    C = sqrt_s[ref_index] / epsilons[ref_index] ** slope  # Compute scaling factor
    trend_line = C * np.array(epsilons) ** slope
    plt.loglog(
        epsilons, trend_line, "--", color="black", label=f"Trend line (slope={slope})"
    )

    plt.xticks(epsilons, [f"{eps:.6f}" for eps in epsilons])

    plt.legend(loc="best", fontsize=10)

    # Labels and formatting
    plt.title(title)
    plt.xlabel(r"$\varepsilon$ = 1/$\sqrt{N}$", fontsize=14)
    plt.ylabel(r"$\sqrt{S_{0.01}}$", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)

    # Save plot
    plt.savefig(save_filename)
    plt.show()

    if return_s:
        return s


def merged_plots(
    title="Perturbed Jet Height Convergence against fine solution",
    epsilons=[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.015625 / 2],
    file_prefix="nest_heights",
    Times=["T5", "T10"],
    save_filename="pertjet_all_height_against_finesolution.png",
    slope=1.5,
):

    s = {}

    # Load data for different time steps (T5, T10)
    Times = ["T5", "T10"]
    for T in Times:
        with open(f"pickle_folder/{file_prefix}_{T}.pkl", "rb") as f:
            s[T] = pickle.load(f)

    plt.figure(figsize=(7, 5), dpi=200)

    # Define different markers for T5 and T10
    time_markers = {"T5": "o", "T10": "s"}  # Circle  # Square

    # Define colors and line styles for L1, L2, and Linf
    norm_styles = {
        "l1": {"color": "#E69F00", "linestyle": "-"},  # Orange, solid
        "l2": {"color": "#56B4E9", "linestyle": "--"},  # Blue, dashed
        "linf": {"color": "#009E73", "linestyle": ":"},  # Green, dotted
    }

    # Use a distinct, colorblind-safe color for time-based curves
    time_color = "#CC79A7"  # Purple (Colorblind-safe)

    # Plot each dataset for T5 and T10
    for key in s.keys():
        sqrt_s = np.sqrt(np.array(s[key]))
        plt.loglog(
            epsilons,
            sqrt_s,
            marker=time_markers[key],  # Different markers for T5 and T10
            linestyle="dashdot",
            color=time_color,  # Use purple for time-based curves
            label=r"$\sqrt{S_{\epsilon}}$"
            + f" {key}",  # This will not go in the legend (handled separately)
            markerfacecolor="none",
            markeredgewidth=1.5,
            markersize=8,
        )

    # Add NN-based lines for l1, l2, linf
    times_lab = ["T5", "T10"]
    for i, key in enumerate(["99", "199"]):
        with open(f"pickle_folder/nn_height_heun_0.05_{key}.pkl", "rb") as f:
            lp_norm = pickle.load(f)

        # Plot L1, L2, and Linf norms
        for norm, style in norm_styles.items():
            plt.loglog(
                epsilons,
                lp_norm[norm],
                marker=time_markers[times_lab[i]],
                linestyle=style["linestyle"],
                color=style["color"],
                markersize=8,
                markerfacecolor="none",
                markeredgewidth=1.5,
            )

    # Trend line with slope 1
    ref_index = len(epsilons) // 2
    C = sqrt_s[ref_index] / epsilons[ref_index] ** slope
    trend_line = C * np.array(epsilons) ** slope
    plt.loglog(
        epsilons, trend_line, "--", color="black", label=f"Trend (slope={slope})"
    )

    plt.xticks(epsilons, [f"{eps:.6f}" for eps in epsilons])

    # Labels and formatting
    plt.title(title)
    plt.xlabel(r"$\varepsilon$ = 1/$\sqrt{N}$", fontsize=14)
    plt.ylabel(r"Error", fontsize=14)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6)

    # --- Custom Legend as a Key ---
    from matplotlib.lines import Line2D

    legend_elements = [
        # Markers for time steps
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="time=5",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="time=10",
        ),
        # Line styles for norms
        Line2D([0], [0], color="#E69F00", linestyle="-", label="L1"),
        Line2D([0], [0], color="#56B4E9", linestyle="--", label="L2 "),
        Line2D([0], [0], color="#009E73", linestyle=":", label="Linf "),
        Line2D(
            [0],
            [0],
            color="#CC79A7",
            linestyle="dashdot",
            label=r"$\sqrt{S_{\epsilon}}$ ",
        ),
        # Trend line
        Line2D(
            [0], [0], color="black", linestyle="--", label=f"Trend line (slope={slope})"
        ),
    ]

    plt.legend(handles=legend_elements, loc="best", fontsize=10, ncols=2)

    # Save plot
    plt.savefig(save_filename)
    plt.show()


def compute_sigma_distance(time, epsilons, device, strength=0.0001, method='heun', dt=0.05, prefix='ot_approx', T='T0'):

    s = []

    # Fine epsilon setup
    fine_index = len(epsilons) - 1
    epsilon_fine = epsilons[fine_index]
    with open(f'data_store/output_{method}_{dt}_{epsilon_fine}_strength_{strength}.pkl', 'rb') as f:
        data_fine = pickle.load(f)

    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon_fine, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    sigma_weights_dense /= sigma_weights_dense.sum()

    for i in range(fine_index):
        epsilon = epsilons[i]
        with open(f'data_store/output_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
            data_course = pickle.load(f)

        _, _, _, sigma_weights, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
        sigma_weights /= sigma_weights.sum()

        tic = perf_counter_ns()
        output, _ = Sinkhorn_Divergence_balanced(
            data_fine[0][:,:,time].to(device),
            sigma_weights_dense.to(device),
            data_course[0][:,:,time].to(device),
            sigma_weights.to(device),
            force_type="pykeops",
            tol=1e-10,
            epsilon=0.01,
            fullcompute=True,
            allow_annealing=False
        )
        toc = perf_counter_ns()

        print(f'[sigma] epsilon={epsilon:.5f}, time={tic - toc}, output={output}')
        s.append(output)

    with open(prefix + f'_sigma_{T}.pkl', 'wb') as f:
        pickle.dump(s, f)

    return s


def compute_heights_distance(time, epsilons, strength, device, method='heun', dt=0.05, prefix='ot_approx', T='T0'):

    s1 = []

    fine_index = len(epsilons) - 1
    epsilon_fine = epsilons[fine_index]
    with open(f'data_store/OT_{method}_{dt}_{epsilon_fine}_strength_{strength}.pkl', 'rb') as f:
        data_fine = pickle.load(f)

    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(epsilon_fine, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
    sigma_weights_dense /= sigma_weights_dense.sum()

    for i in range(fine_index):
        epsilon = epsilons[i]
        with open(f'data_store/OT_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
            data_course = pickle.load(f)

        X, _, _, sigma_weights, _ = jet_profile_initialisation(epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0)
        sigma_weights /= sigma_weights.sum()

        tic = perf_counter_ns()
        output, _ = Sinkhorn_Divergence_balanced(
            X_dense.to(device),
            (data_fine[1][:,:,time]/data_fine[1][:,:,time].sum()).to(device).view(-1,1),
            X,
            (data_course[1][:,:,time]/data_course[1][:,:,time].sum()).to(device).view(-1,1),
            force_type="pykeops",
            tol=1e-10,
            epsilon=0.01,
            fullcompute=True,
            allow_annealing=False
        )
        toc = perf_counter_ns()

        print(f'[height] epsilon={epsilon:.5f}, time={tic - toc}, output={output}')
        s1.append(output)

    with open(prefix + f'_heights_{T}.pkl', 'wb') as f:
        pickle.dump(s1, f)

    return s1
