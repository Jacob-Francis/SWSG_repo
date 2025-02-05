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
import matplotlib.colors as colors
import matplotlib.cm as cm


# My python classes for loops we've defined so far
from swsg_ot_algorithm import SWSGDynamcis

# f = open(f'data_store/output_{method}_{dt}_{epsilon}_{lloyd}.pkl', 'wb')
# 5e-05
for strength in [0.0]:#, 0.0001, 5e-5]:
    # strength = 0.0
    method= 'heun'
    epsilon = 0.01
    dt = 0.05
    # /home/jacob/SWSG_repo/dynamic_jet_study/output_heun_0.05_0.01_strength_0.0.pkl
    with open(f'output_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
        output = pickle.load(f)


    a = 0.1
    b = 10.0
    c = 0.5
    d = 1.0

    def height_func(x):
        return a * torch.tanh(b * (x - c)) + d

    def periodic_g_x_vel(Gt, Xtilde, dt, L=1.0):
        """
        Periodic boundary reconsruciton give, periodic in x
        """
        # rotate
        x_component = -(Gt[:, 1] - Xtilde[:, 1])  
        y_component  = (Gt[:, 0] - Xtilde[:, 0])
        
        # Stack the x_component and apply periodic boundary condition
        stacked_x = torch.stack([x_component, x_component + L, x_component - L], dim=-1)
        
        # Find the closest value with the minimum absolute difference
        min_diff_indices = torch.abs(stacked_x).min(dim=-1).indices
        
        # Select the correct x_component based on the minimum absolute difference
        x_component = stacked_x.gather(dim=-1, index=min_diff_indices.unsqueeze(-1)).squeeze(-1)

        
        return torch.stack([x_component, y_component], dim=-1) /dt

    def periodic_vec_reconstruction(Gt, Gt_plus1, dt, L=1.0):
        """
        Periodic boundary reconsruciton give, periodic in x
        """
        x_component = (Gt_plus1[:, 0] - Gt[:, 0])  
        y_component  = (Gt_plus1[:, 1] - Gt[:, 1])
        
        # Stack the x_component and apply periodic boundary condition
        stacked_x = torch.stack([x_component, x_component + L, x_component - L], dim=-1)
        
        # Find the closest value with the minimum absolute difference
        min_diff_indices = torch.abs(stacked_x).min(dim=-1).indices
        
        # Select the correct x_component based on the minimum absolute difference
        x_component = stacked_x.gather(dim=-1, index=min_diff_indices.unsqueeze(-1)).squeeze(-1)

        
        return torch.stack([x_component, y_component], dim=-1) /dt


    m1 = m2 = int(1/epsilon)

    X = torch.cartesian_prod(
        torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
        torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
    )
    alpha = torch.Tensor(height_func(X[:, 1]))
    h_true = alpha/ alpha.sum()
    time_steps = int(60/dt)

    animation_tpye = False

    fig = plt.figure(figsize=(28,7), dpi=200)
    ax1 = fig.add_subplot(143)
    ax2 = fig.add_subplot(142, projection='3d')
    ax3 = fig.add_subplot(141)
    ax4 = fig.add_subplot(144)


    n1 = n2 = 1 # because the potentials should have the correct scaling
    h_true = h_true.cpu()

    # Add color bar outside the animation function
    cbar1 = plt.colorbar(ax1.scatter([], []), ax=ax1, orientation='horizontal', cmap='seismic')
    cbar1.set_label('Geoverse Velocity')

    cbar2 = plt.colorbar(ax2.scatter([], []), ax=ax2, orientation='horizontal')
    cbar2.set_label('Water Height')

    cbar3 = plt.colorbar(ax3.scatter([], []), ax=ax3, orientation='horizontal', cmap='seismic')
    cbar3.set_label('Universe Approximate Velocity')

    cbar4 = plt.colorbar(ax4.scatter([], []), ax=ax4, orientation='horizontal', cmap='seismic')
    cbar4.set_label('Ageostrophic realtive Error (\%)')

    print('running vectors')

    # precomputed_vectors = [periodic_g_x_vel(output[0][:, :, i], output[2][:, :, i], dt, L=1.0).cpu() for i in range(0, time_steps - 1)]
    precomputed_vectors = [periodic_vec_reconstruction(output[0][:, :, i], output[0][:, :, i+1], dt, L=1.0).cpu() for i in range(0, time_steps - 1)]

    print('ran ...')

    global_vmin = torch.min(torch.cat([v.norm(dim=1) for v in precomputed_vectors]))
    global_vmax = torch.max(torch.cat([v.norm(dim=1) for v in precomputed_vectors]))
    norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='seismic')
    sm.set_array([])

    cbar1.update_normal(sm)

    precomputed_x_tilde_vectors = [periodic_vec_reconstruction(output[2][:, :, i], output[2][:, :, i + 1], dt, L=1.0).cpu() for i in range(0, time_steps - 1)]

    print('finishing...')

    global_vmin = torch.min(torch.cat([v.norm(dim=1) for v in precomputed_x_tilde_vectors]))
    global_vmax = torch.max(torch.cat([v.norm(dim=1) for v in precomputed_x_tilde_vectors]))
    norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='seismic')
    sm.set_array([])

    cbar3.update_normal(sm)

    print('finished.')


    # Compute global min and max across all time steps
    global_vmin = float('inf')
    global_vmax = float('-inf')

    # print(time_steps, len(precomputed_x_tilde_vectors), len(precomputed_vectors), len(range(1, time_steps - 1)) )

    for i in range(0, time_steps - 2):
        vector = 100*((precomputed_x_tilde_vectors[i] - precomputed_vectors[i]).norm(dim=1)/precomputed_vectors[i].norm(dim=1))
    #     vector = (output[0][:, :, i] - output[2][:, :, i] - precomputed_x_tilde_vectors[i]).norm(dim=1)
    #     print('shape check',precomputed_x_tilde_vectors[i].shape, precomputed_vectors[i].norm(dim=1).shape, vector.shape)
        global_vmin = min(global_vmin, torch.min(vector).item())
        global_vmax = max(global_vmax, torch.max(vector).item())

    # Define the color normalization
    norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap='Reds')
    sm.set_array([])

    # Update colorbar
    cbar4.update_normal(sm)
    animation_tpye = False
    if animation_tpye:
        def update_frame(i):
            # Clear previous frames
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()

            # Top-left plot: Geoverse Velocities
            vector = precomputed_vectors[i]
            ax1.scatter(output[0][:, 0, i].cpu(), output[0][:, 1, i].cpu(), c=vector.norm(dim=1), cmap='seismic', s=10, alpha=0.5)
            ax1.quiver(output[0][:, 0, i].cpu(), output[0][:, 1, i].cpu(), vector[:, 0], vector[:, 1])
            ax1.set(title=f'Geoverse Velocities, energy={output[-1][0]["dual"][i]}')

            # Top-right plot: Universe Jet Height
            ax2.scatter(X[:, 0].cpu(), X[:, 1].cpu(), n1 * n2 * output[1][:, :, i], c=n1 * n2 * output[1][:, :, i], cmap='viridis')
            ax2.scatter(output[0][:, 0, i].cpu(), output[0][:, 1, i].cpu(), 0 * output[0][:, 0, i].cpu(), c=output[0][:, 0, 0].cpu(), s=0.5, cmap='rainbow')
            ax2.set(title=f"Universe Jet Height, \n Linf = {torch.linalg.norm(n1 * n2 * output[1][:, :, i].view(-1,) - h_true.view(-1), ord=float('inf'))}")
            ax2.azim = -15

            vector = precomputed_x_tilde_vectors[i]
            # Bottom-left plot: Same as top-left (Geoverse Velocities) or could be something different
            ax3.scatter(output[2][:, 0, i].cpu(), output[2][:, 1, i].cpu(), c=vector.norm(dim=1), cmap='seismic', s=10, alpha=0.5)
            ax3.quiver(output[2][:, 0, i].cpu(), output[2][:, 1, i].cpu(), vector[:, 0], vector[:, 1])
            ax3.set(title=fr'Universe Barycentre, ($\tilde X$)')

            # Bottom-left plot: Same as top-left (Geoverse Velocities) or could be something different
            vector = precomputed_x_tilde_vectors[i] - precomputed_vectors[i]
            ax4.scatter(output[2][:, 0, i].cpu(), output[2][:, 1, i].cpu(), c=100*(vector.norm(dim=1)/precomputed_vectors[i].norm(dim=1)), cmap='Reds', s=10, alpha=1.0)
            ax4.quiver(output[2][:, 0, i].cpu(), output[2][:, 1, i].cpu(), vector[:, 0], vector[:, 1])
            ax4.set(title=f'Ageostrophic Velocities (quiver)')

            vector = 100*(vector.norm(dim=1)/precomputed_vectors[i].norm(dim=1))
            global_vmin = torch.min(vector).item()
            global_vmax = torch.max(vector).item()

            # Define the color normalization
            norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
            sm = plt.cm.ScalarMappable(norm=norm, cmap='Reds')
            sm.set_array([])

            # Update colorbar
            cbar4.update_normal(sm)

            # Update the title
            fig.suptitle(f'Jet Profile ($S_\epsilon$, PBC), RK4 stepping, Frame {i} \n f,g,$\epsilon$,N,M,dt = 1.0, 0.1, {epsilon}, {m1}$^2$ {m1}$^2$, {dt}')
            plt.tight_layout()

        print(time_steps)
        time_steps=203
        ani = animation.FuncAnimation(fig, update_frame, frames=range(1, time_steps-2, 10), blit=False)


        # ani = animation.FuncAnimation(fig, function, frames=range(NUM_FRAMES), blit=False)

        # Save or display the animation
    #     ani.save(f'test.mp4', writer='ffmpeg', fps=5)

        ani.save(f'animation_output_heun_{dt}_{epsilon}_{strength}_{method}.mp4', writer='ffmpeg', fps=5)
    else:


        # Define time stamps for plotting
        time_stamps = [0, int((3*time_steps) // 6), int((4.5*time_steps)//6), int((6*time_steps)//6)-2]  # Modify as needed

        # time_stamps = [0, int((3*time_steps) // 6), int((6*time_steps)//6 - 2)]  # Modify as needed
        num_rows = len(time_stamps)

        # Create figure and subplots
        # fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), dpi=200)
        fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5 * num_rows), dpi=200, sharex='all', sharey='all')

        # Convert to 2D array if there's only one row
        if num_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        # Convert second column (ax2) to 3D plots
        for row_idx in range(num_rows):
            fig.delaxes(axes[row_idx, 2])

            axes[row_idx, 2] = fig.add_subplot(num_rows, 3, row_idx * 3 + 3, projection='3d')

        # Labels for the columns (only on the top row)
        axes[0, 0].set_title('Geoverse Velocities', fontsize=14)
        axes[0, 1].set_title('Universe Jet Height', fontsize=14)
        axes[0, 2].set_title(r'Universe Barycentre ($\tilde{X}$)', fontsize=14)

        # Determine color normalization based on all time steps
        vmax = max([precomputed_vectors[i].norm(dim=1).max() for i in time_stamps] + 
            [precomputed_x_tilde_vectors[i].norm(dim=1).max() for i in time_stamps])

        velocity_norm = colors.Normalize(vmin=0, vmax=vmax)
        height_norm = colors.Normalize(vmin=min((n1 * n2 * output[1][:, :, i]).min() for i in time_stamps), vmax=max((n1 * n2 * output[1][:, :, i]).max() for i in time_stamps))

        # Create colorbars
        cmap_velocity = cm.ScalarMappable(norm=velocity_norm, cmap='seismic')
        cmap_height = cm.ScalarMappable(norm=height_norm, cmap='viridis')

        # Loop through selected time steps
        for row_idx, i in enumerate(time_stamps):
            skip=7
            ax1, ax3, ax2 = axes[row_idx]  # Unpack subplots for this row

            # Time label for each row
            ax1.text(
                0.75, 0.1, f't = {i * dt:.2f}',  # Positioning in axis coordinates (0 to 1)
                transform=ax1.transAxes,            # Transform to axis coordinate system
                fontsize=14,                        # Font size
                va='top',                           # Vertical alignment
                ha='left',                          # Horizontal alignment
                color='white',                      # Label color
            )

            # First plot: Geoverse Velocities
            vector = precomputed_vectors[i]
            sc1 = ax1.scatter(output[0][:, 0, i].cpu(), output[0][:, 1, i].cpu(), 
                            c=vector.norm(dim=1), cmap='seismic', norm=velocity_norm, s=5/(n1**2), alpha=0.5)
            ax1.quiver(output[0][::skip, 0, i].cpu(), output[0][::skip, 1, i].cpu(), vector[::skip, 0], vector[::skip, 1])
            ax1.set_xlabel(r'$y_1$', fontsize=12)
            ax1.set_ylabel(r'$y_2$', fontsize=12)

            # Second plot: Universe Jet Height (3D)
            sc2 = ax2.scatter(X[:, 0].cpu(), X[:, 1].cpu(), n1 * n2 * output[1][:, :, i], 
                            c=n1 * n2 * output[1][:, :, i], cmap='viridis', norm=height_norm)
            ax2.set_xlabel(r'$x_1$', fontsize=12)
            ax2.set_ylabel(r'$x_2$', fontsize=12)
            ax2.set_zlabel(r'$h$', fontsize=12)
            ax2.view_init(elev=25, azim=225)  # Rotate 3D plot for better visualization
            
            # Third plot: Universe Barycentre
            vector = precomputed_x_tilde_vectors[i]
            sc3 = ax3.scatter(output[2][:, 0, i].cpu(), output[2][:, 1, i].cpu(), 
                            c=vector.norm(dim=1), cmap='seismic', norm=velocity_norm, s=5/(n1**2), alpha=0.5)
            ax3.quiver(output[2][::skip, 0, i].cpu(), output[2][::skip, 1, i].cpu(), vector[::skip, 0], vector[::skip, 1])
            ax3.set_xlabel(r'$y_1$', fontsize=12)
            ax3.set_ylabel(r'$y_2$', fontsize=12)

        # Define position for colorbars at the bottom
        cbar_ax1 = fig.add_axes([0.25, 0.04, 0.25, 0.02])  # (left, bottom, width, height)
        cbar1 = fig.colorbar(cmap_velocity, cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label(r'Velocity Magnitude', fontsize=12)

        cbar_ax2 = fig.add_axes([0.55, 0.04, 0.25, 0.02])  # Adjust position for the second colorbar
        cbar2 = fig.colorbar(cmap_height, cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label(r'Jet Height ($h$)', fontsize=12)


        # Add a global title
        plt.suptitle(
            fr'Jet Profile, N = {int(np.sqrt(m1*m2))}$^2$, $\alpha$={strength}, method={method}' +
            f' \n f,g,$\epsilon$, N,M,dt = 1.0, 0.1, {epsilon}, {m1}$^2$ {m1}$^2$, {dt}',
            fontsize=14, y=0.95
        )
        # plt.suptitle(f'Jet Profile ($S_\epsilon$, PBC), RK4 stepping, Frame {i} ')

        # plt.tight_layout()    # plt.show()


        plt.savefig(f'figure_heun_{dt}_{epsilon}_{strength}_{method}.png', pad_inches=0.5)


# # ##############################################################
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from swsg_ot_algorithm import SWSGDynamcis
# import matplotlib as mpl
# import pickle

# # The function to plot comparison across different methods (e.g. 'heun', 'rk4', etc.)
# def plot_comparison_across_methods(methods, dt, epsilon, strengths):
#     fig, axes = plt.subplots(len(methods), len(strengths), figsize=(20, 15), dpi=200)
    
#     # Loop over each method and strength for plotting
#     for method_idx, method in enumerate(methods):
#         for strength_idx, strength in enumerate(strengths):
#             # Load the data for this specific method and strength combination
#             with open(f'output_{method}_{dt}_{epsilon}_strength_{strength}.pkl', 'rb') as f:
#                 output = pickle.load(f)

#             # Here you can define your model, X, alpha, etc. as you've already done earlier
#             m1 = m2 = int(1 / epsilon)
#             X = torch.cartesian_prod(
#                 torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
#                 torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
#             )
#             alpha = torch.Tensor(height_func(X[:, 1]))
#             h_true = alpha / alpha.sum()
#             time_steps = int(60 / dt)
            
#             # Create subplots for each method & strength combination
#             ax = axes[method_idx, strength_idx]

#             # Plot Geoverse Velocity (assuming 'output' contains the velocity info in the first element)
#             vector = precomputed_vectors[i]
#             sc1 = ax.scatter(output[0][:, 0, i].cpu(), output[0][:, 1, i].cpu(), c=vector.norm(dim=1), cmap='seismic', s=10, alpha=0.5)
#             ax.quiver(output[0][:, 0, i].cpu(), output[0][:, 1, i].cpu(), vector[:, 0], vector[:, 1])
#             ax.set_title(f'Method: {method}, Strength: {strength}')
#             ax.set_xlabel(r'$y_1$', fontsize=12)
#             ax.set_ylabel(r'$y_2$', fontsize=12)

#     # Add colorbars and other customizations outside the main loop
#     cbar_ax1 = fig.add_axes([0.25, 0.04, 0.25, 0.02])  # (left, bottom, width, height)
#     cbar1 = fig.colorbar(sc1, cax=cbar_ax1, orientation='horizontal')
#     cbar1.set_label(r'Velocity Magnitude', fontsize=12)

#     # Add a global title
#     plt.suptitle(f'Comparison across methods (dt={dt}, epsilon={epsilon})', fontsize=14, y=1.05)
    
#     # Adjust layout to avoid overlap
#     plt.tight_layout()
#     plt.show()

#     # Optionally, save the figure
#     plt.savefig(f'comparison_across_methods_{dt}_{epsilon}.png', pad_inches=0.5)

# # Define methods and strengths to test
# methods = ['euler', 'heun', 'rk4']
# strengths = [0.0, 0.0001, 5e-5]
# dt = 0.05
# epsilon = 0.01

# # Call the plotting function
# plot_comparison_across_methods(methods, dt, epsilon, strengths)
