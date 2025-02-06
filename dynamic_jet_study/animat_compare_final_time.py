import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
from scipy import optimize
import matplotlib.cm as cm

def periodic_velocity_reconstruction(Gt, Xtilde, dt, L=1.0):
    """
    Periodic boundary reconstruction for velocity calculation.
    Assumes periodic boundary conditions in x direction.
    """
    x_component = -(Gt[:, 1] - Xtilde[:, 1])  # x-velocity component
    y_component = (Gt[:, 0] - Xtilde[:, 0])  # y-velocity component
    
    # Stack the x_component and apply periodic boundary condition
    stacked_x = torch.stack([x_component, x_component + L, x_component - L], dim=-1)
    
    # Find the closest value with the minimum absolute difference
    min_diff_indices = torch.abs(stacked_x).min(dim=-1).indices
    
    # Select the correct x_component based on the minimum absolute difference
    x_component = stacked_x.gather(dim=-1, index=min_diff_indices.unsqueeze(-1)).squeeze(-1)
    
    return torch.stack([x_component, y_component], dim=-1) / dt

def compute_vectors_for_methods(methods, dt, time_steps, epsilon):
    """
    Load and compute vectors for all methods over all time steps.
    """
    output_data = {}
    precomputed_vectors = {}
    precomputed_x_tilde_vectors = {}

    for method in methods:
        # Load data for each method
        with open(f'output_{method}_{dt}_{epsilon}_strength_0.0.pkl', 'rb') as f:
            output = pickle.load(f)
        output_data[method] = output

        # Compute velocity vectors for each method over time steps
        precomputed_vectors[method] = [
            periodic_velocity_reconstruction(output[0][:, :, i], output[0][:, :, i + 1], dt, L=1.0).cpu()
            for i in range(time_steps - 1)
        ]
        precomputed_x_tilde_vectors[method] = [
            periodic_velocity_reconstruction(output[2][:, :, i], output[2][:, :, i + 1], dt, L=1.0).cpu()
            for i in range(time_steps - 1)
        ]

    return output_data, precomputed_vectors, precomputed_x_tilde_vectors

def plot_single_method(ax, output_data, vectors, method, i, method_idx, norm, title):
    """
    Helper function to plot data for a single method (Geoverse velocity, Jet height, or Barycentre).
    """
    vector = vectors[method][i]
    sc = ax.scatter(
        output_data[method][0][:, 0, i].cpu(), output_data[method][0][:, 1, i].cpu(),
        c=vector.norm(dim=1), cmap='seismic', norm=norm, s=5, alpha=0.5
    )
    ax.quiver(
        output_data[method][0][::5, 0, i].cpu(), output_data[method][0][::5, 1, i].cpu(),
        vector[::5, 0], vector[::5, 1]
    )
    ax.set_title(f'{title}: {method}', fontsize=14)
    ax.set_xlabel(r'$y_1$', fontsize=12)
    ax.set_ylabel(r'$y_2$', fontsize=12)
    return sc

def plot_comparison(output_data, precomputed_vectors, precomputed_x_tilde_vectors, methods, time_steps, dt, epsilon):
    """
    Plot the comparison of methods (Euler, Heun, RK4) for various parameters.
    """
    n1 = n2 = 1
    # Create subplots
    fig, axes = plt.subplots(len(methods), 3, figsize=(15, 5 * len(methods)), dpi=200, sharex='all', sharey='all')

    # Set column titles
    axes[0, 0].set_title('Geoverse Velocities', fontsize=14)
    axes[0, 1].set_title('Universe Jet Height', fontsize=14)
    axes[0, 2].set_title(r'Universe Barycentre ($\tilde{X}$)', fontsize=14)

    # Determine color normalization
    vmax = max(
        [precomputed_vectors[method][-1].norm(dim=1).max() for method in methods] +
        [precomputed_x_tilde_vectors[method][-1].norm(dim=1).max() for method in methods]
    )
    velocity_norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    height_norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # Loop over methods and create plots for each method
    for row_idx, method in enumerate(methods):
        ax1, ax2, ax3 = axes[row_idx]  # Unpack subplots for the current method

        # Final time step index
        i = time_steps - 2  # We want the penultimate time step

        # Plot Geoverse Velocities
        sc1 = plot_single_method(ax1, output_data, precomputed_vectors, method, i, row_idx, velocity_norm, 'Geoverse Velocities')

        # Plot Universe Jet Height (3D)
        ax2 = fig.add_subplot(len(methods), 3, row_idx * 3 + 2, projection='3d')
        sc2 = ax2.scatter(
            output_data[method][2][:, 0, i].cpu(), output_data[method][2][:, 1, i].cpu(),
            n1 * n2 * output_data[method][1][:, :, i],
            c=n1 * n2 * output_data[method][1][:, :, i], cmap='viridis', norm=height_norm
        )
        ax2.set_xlabel(r'$x_1$', fontsize=12)
        ax2.set_ylabel(r'$x_2$', fontsize=12)
        ax2.set_zlabel(r'$h$', fontsize=12)
        ax2.view_init(elev=25, azim=225)  # Rotate 3D plot for better visualization

        # Plot Universe Barycentre
        sc3 = plot_single_method(ax3, output_data, precomputed_x_tilde_vectors, method, i, row_idx, velocity_norm, 'Universe Barycentre')

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.25, 0.04, 0.25, 0.02])  # (left, bottom, width, height)
    cbar1 = fig.colorbar(mpl.cm.ScalarMappable(norm=velocity_norm, cmap='seismic'), cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(r'Velocity Magnitude', fontsize=12)

    cbar_ax2 = fig.add_axes([0.55, 0.04, 0.25, 0.02])  # Adjust position for the second colorbar
    cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=height_norm, cmap='viridis'), cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label(r'Jet Height ($h$)', fontsize=12)

    # Set overall title and layout
    plt.suptitle(
        fr'Comparison of Methods: Euler, Heun, RK4 \n $\epsilon$={epsilon}, dt={dt}',
        fontsize=14, y=0.95
    )
    plt.tight_layout()
    plt.savefig('comparison_fig.png')

# Main script
methods = ['euler', 'heun', 'rk4']
epsilon = 0.02
dt = 0.05
time_steps = int(60 / dt)  # Calculate the number of time steps

# Load data and compute vectors
output_data, precomputed_vectors, precomputed_x_tilde_vectors = compute_vectors_for_methods(methods, dt, time_steps, epsilon)

# Plot comparison of methods at the final time step
plot_comparison(output_data, precomputed_vectors, precomputed_x_tilde_vectors, methods, time_steps, dt, epsilon)
