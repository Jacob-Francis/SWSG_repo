import matplotlib as mpl
from scipy import optimize, interpolate
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.animation as animation
import pickle
from tqdm import tqdm
import torch

# Define colorblind-friendly colors (Tol's vibrant)
CB_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]



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
    

def compute_norms_and_plot(method='heun', strength=0.0001, dt=0.05, time_steps=[199], verbose=False, prefix='output', save_file='nest_height_interp_multiple_T.pdf'):
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
        print(f"Fine Grid -> ε: {epsilon_fine}, N: {N_fine}, n: {n_fine}, n²: {n_fine**2}")

    # Load fine grid data

    fine_file = f'data_store/{prefix}_{method}_{dt}_{epsilon_fine}_strength_{strength}.pkl'
    with open(fine_file, 'rb') as f:
        data_fine = pickle.load(f)

    # Initialise fine grid profile
    X_dense, _, _, sigma_weights_dense, _ = jet_profile_initialisation(
        epsilon_fine, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0
    )
    sigma_weights_dense /= sigma_weights_dense.sum()
    X_dense_ = torch.linspace(1 / (2 * n_fine), 1 - 1 / (2 * n_fine), n_fine)

    # Figure settings: scale height by number of time steps
    fig, axes = plt.subplots(len(time_steps), 1, figsize=(8, 4 * len(time_steps)), dpi=200, sharex=True)

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
            coarse_file = f'data_store/{prefix}_{method}_{dt}_{epsilon}_strength_{strength}.pkl'
            with open(coarse_file, 'rb') as f:
                data_course = pickle.load(f)

            # Initialize coarse grid profile
            X, _, _, sigma_weights, _ = jet_profile_initialisation(
                epsilon, strength, f=1.0, g=0.1, a=0.1, b=10.0, c=0.5, d=1.0
            )
            sigma_weights /= sigma_weights.sum()
            if verbose:
                print(f"Coarse Sum: {data_course[1][:, :, time_step].sum()}, Fine Sum: {data_fine[1][:, :, time_step].sum()}")

            # Interpolate coarse data onto fine grid
            coarse_data = data_course[1][:, :, time_step].reshape(n, n).numpy()
            interpolated = interpolate.interpn(
                (torch.linspace(1 / (2 * n), 1 - 1 / (2 * n), n).numpy(),) * 2,
                coarse_data,
                X_dense.numpy(),
                method='nearest',
                bounds_error=False,
                fill_value=None
            )

            # Compute errors
            fine_data = (data_fine[1][:, :, time_step] / data_fine[1][:, :, time_step].sum()).view(-1,)
            interpolated_data = torch.tensor(interpolated / interpolated.sum()).view(-1,)
            diff = (interpolated_data - fine_data) * len(X_dense)

            # Compute L1, L2, Linf norms
            l1_norms.append((torch.linalg.norm(diff, ord=1) / len(sigma_weights_dense)).item())
            l2_norms.append((torch.linalg.norm(diff, ord=2) / np.sqrt(len(sigma_weights_dense))).item())
            linf_norms.append((torch.linalg.norm(diff, ord=float('inf'))).item())
            
            if verbose:
                print(f"Interpolated Sum: {interpolated.sum()}, Coarse Weight Sum: {sigma_weights.sum()}, Fine Weight Sum: {sigma_weights_dense.sum()}")
                
                
        with open(f'pickle_folder/{prefix}_nn_height_{method}_{dt}_{time_step}.pkl', 'wb') as f:
            pickle.dump({'l1':l1_norms, 'l2':l2_norms, 'linf':linf_norms} , f)

        # Plot error norms for this time step
        ax.loglog(epsilons, l1_norms, color=CB_COLORS[0], marker='o', linestyle='-', label=r'$L_1$ norm')
        ax.loglog(epsilons, l2_norms, color=CB_COLORS[1], marker='s', linestyle='-', label=r'$L_2$ norm')
        ax.loglog(epsilons, linf_norms, color=CB_COLORS[2], marker='^', linestyle='-', label=r'$L_\infty$ norm')

        # Fit reference slope line (Slope = 1)
        ref_x = epsilons[-3:]
        ref_y = l1_norms[-3:]
        C = ref_y[-1] / ref_x[-1] + 1e-1  # Compute scaling constant
        trend_line = C * np.array(epsilons)**1  # y = C * x^1

        ax.loglog(epsilons, trend_line, '--', color='black', label='Slope 1 Trend Line', alpha=0.6)

        # Labels, title, and legend
        ax.set_xticks(epsilons)
        ax.set_xticklabels(
            [fr'$\epsilon$ = {k:.5g} | n = {1/k**2:3g}' for k in epsilons],
            rotation=45
        )
        ax.set_xlabel(r'$\epsilon$', fontsize=12)
        ax.set_ylabel('Norm Value', fontsize=12)
        ax.set_title(f'T={(time_step + 1)*dt}', fontsize=14)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

    # Adjust layout to prevent overlap
    plt.suptitle('Height Error with NN-Interpolation')
    plt.tight_layout()

    # Save and show
    plt.savefig(save_file)
    plt.show()
    

def plot_energy_conservation(data_dir='data_store', epsilon=0.02, strength=0.0, save_path='energy_conservation_for_jet.pdf'):
    """Plot kinetic, potential, and total energy conservation for different integrators and dt values."""

    # Define colorblind-friendly colors (Tol's vibrant)
    CB_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
    DT_VALUES = [0.4, 0.2, 0.1, 0.05]  # Time steps to test
    LINE_STYLES = {
        "Kinetic": "-",
        "Potential": ":",
        "Total Energy": "--"
    }
    methods = ["heun", "rk4"]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=300, sharex=True)

    for row, method in enumerate(methods):
        ax1 = axes[row, 0]  # Left plot: Kinetic vs Potential
        ax2 = ax1  # Using same axis for both kinetic & potential
        ax3 = axes[row, 1]  # Right plot: Total Energy error

        for i, dt in enumerate(DT_VALUES):
            with open(f'{data_dir}/output_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
                data = pickle.load(f)

            time = np.linspace(0, 60, int(60/dt))
            entropic = np.array(data[3][0]['terms']['d3']) + np.array(data[3][0]['terms']['d31']) + np.array(data[3][0]['terms']['d32'])
            potential = np.array(data[3][0]['terms']['p5'])
            kinetic = np.array(data[3][0]['dual']) - potential - entropic
            potential -= potential[0]
            kinetic -= kinetic[0]
            total_dual = np.array(data[3][0]['dual']) - 0.05
            abs_energy_error = abs(total_dual - total_dual[0]) / total_dual[0]

            color = CB_COLORS[i % len(CB_COLORS)]

            # Plot kinetic and potential energy
            ax1.plot(time, kinetic, linestyle=LINE_STYLES["Kinetic"], color=color)
            ax2.plot(time, potential, linestyle=LINE_STYLES["Potential"], color=color)

            # Plot total energy error
            ax3.semilogy(time, abs_energy_error, linestyle=LINE_STYLES["Total Energy"], color=color)

        # Axis labels and titles
        if row == 1:
            ax1.set_xlabel("Time")
        ax2.set_ylabel("Energy Variation")
        ax2.set_title(f"{method.upper()} - Potential vs Kinetic Energy")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Energy Error (log scale)")
        ax3.set_title(f"{method.upper()} - Normalised Energy Error")

    # Legends (combined and stacked vertically)
    custom_legend_lines = [
        plt.Line2D([0], [0], color="black", linestyle=LINE_STYLES["Kinetic"], label="Kinetic Energy"),
        plt.Line2D([0], [0], color="black", linestyle=LINE_STYLES["Potential"], label="Potential Energy"),
        plt.Line2D([0], [0], color="black", linestyle=LINE_STYLES["Total Energy"], label="Abs. Energy Error"),
    ]

    color_legend_lines = [
        plt.Line2D([0], [0], color=CB_COLORS[i], linestyle="-", label=f"dt = {DT_VALUES[i]}")
        for i in range(len(DT_VALUES))
    ]

    # Stack both legends in one location vertically
    fig.legend(
        handles=custom_legend_lines + color_legend_lines,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,  # stack vertically
        framealpha=0.5,
        title="Legend",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

    
    import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_energy_conservation(data_dir='data_store', epsilon=0.02, strength=0.0001, save_path='energy_conservation_for_jet_pertubed.pdf'):
    """Plot kinetic, potential, and total energy conservation for different integrators and dt values."""

    # Define colorblind-friendly colors (Tol's vibrant)
    CB_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7"]
    DT_VALUES = [0.4, 0.2, 0.1, 0.05]  # Time steps to test
    LINE_STYLES = {
        "Kinetic": "-",
        "Potential": ":",
        "Total Energy": "--"
    }
    methods = ["heun", "rk4"]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=300, sharex=True)

    for row, method in enumerate(methods):
        ax1 = axes[row, 0]  # Left plot: Kinetic vs Potential
        ax2 = ax1  # Using same axis for both kinetic & potential
        ax3 = axes[row, 1]  # Right plot: Total Energy error

        for i, dt in enumerate(DT_VALUES):
            with open(f'{data_dir}/output_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
                data = pickle.load(f)

            time = np.linspace(0, 60, int(60/dt))
            entropic = np.array(data[3][0]['terms']['d3']) + np.array(data[3][0]['terms']['d31']) + np.array(data[3][0]['terms']['d32'])
            potential = np.array(data[3][0]['terms']['p5'])
            kinetic = np.array(data[3][0]['dual']) - potential - entropic
            potential -= potential[0]
            kinetic -= kinetic[0]
            total_dual = np.array(data[3][0]['dual']) #- 0.05
            abs_energy_error = abs(total_dual - total_dual[0]) / total_dual[0]

            color = CB_COLORS[i % len(CB_COLORS)]

            # Plot kinetic and potential energy
            ax1.plot(time, kinetic, linestyle=LINE_STYLES["Kinetic"], color=color)
            ax2.plot(time, potential, linestyle=LINE_STYLES["Potential"], color=color)

            # Plot total energy error
            ax3.semilogy(time, abs_energy_error, linestyle=LINE_STYLES["Total Energy"], color=color)

        # Axis labels and titles
        if row == 1:
            ax1.set_xlabel("Time")
        ax2.set_ylabel("Energy Variation")
        ax2.set_title(f"{method.upper()}"+r" - $\mathcal{P}(h)$ vs $\mathcal{K}_{\sigma}(h)$")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Energy Error (log scale)")
        ax3.set_title(f"{method.upper()} - Default Energy Conservation")

    # Legends (combined and stacked vertically)
    custom_legend_lines = [
        plt.Line2D([0], [0], color="black", linestyle=LINE_STYLES["Kinetic"], label=r"$\mathcal{K}_{\sigma}(h)$ (Kinetic Energy)"),
        plt.Line2D([0], [0], color="black", linestyle=LINE_STYLES["Potential"], label=r"$\mathcal{P}(h)$ (Potential Energy)"),
        plt.Line2D([0], [0], color="black", linestyle=LINE_STYLES["Total Energy"], label="Default Energy Conservation"),
    ]

    color_legend_lines = [
        plt.Line2D([0], [0], color=CB_COLORS[i], linestyle="-", label=f"dt = {DT_VALUES[i]}")
        for i in range(len(DT_VALUES))
    ]

    # Stack both legends in one location vertically
    fig.legend(
        handles=custom_legend_lines + color_legend_lines,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=7,  # stack vertically
        framealpha=0.5,
        title="Legend",
        fontsize=9
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for legend
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()








