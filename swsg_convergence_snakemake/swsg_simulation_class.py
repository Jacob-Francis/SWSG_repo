import torch
import pickle
from utils import (
    initialisation,
    swsg_solver,
    swsg_class_generate,
    compute_dense_symmetric_potential,
    Sinkhorn_Divergence_balanced,
    normal_pdf
)
from time import perf_counter_ns
from geomloss import SamplesLoss
import numpy as np


class SWSGSimulation:
    def __init__(self, cuda=None, profile="uniform", d=1, b=10, tol=1e-11, suff="_lloyd"):
        if cuda is None:
            self.device = "cpu"
            self.dtype = torch.DoubleTensor
        else:
            self.device = torch.device(
                f"cuda:{cuda}" if torch.cuda.is_available() else "cpu"
            )
            self.dtype = torch.float64
        self.profile = profile
        self.d = d
        self.b = b
        if self.profile == "shallowjet":
            self.b = 5
        self.tol = tol
        if suff == "_lloyd":
            self.lloyd = True
        elif suff == "_nolloyd":
            self.lloyd = False
    
    def u_g(self, x, a=0.1, c=0.5, g0=0.1):
        if self.profile == 'uniform':
            return -g0*torch.zeros_like(x)
        elif self.profile=='shallowjet' or self.profile=='jet':
            temp = torch.zeros_like(x)
            temp[:, 0] = a*self.b*(1 - torch.tanh(self.b*(x[:, 1]-c))**2) 
            return -g0*temp
        elif self.profile=='incline':
            temp = torch.ones_like(x)*self.b*a
            temp[:, 1] = 0 
            return -g0*temp
        elif self.profile == 'perturbedjet':
            temp = torch.zeros_like(x)
            temp[:, 0] = a*self.b*(1 - torch.tanh(self.b*(x[:, 1]-c))**2) 

            no, no0 , no1  = normal_pdf(x[:,0],x[:,1],0.5,0.3,0.1,strength=0.0001)  ## 0 is stationnary 
            temp = temp + torch.stack((no0, no1), dim=1)
        
            raise NotImplementedError
        #######################################################################################################################

    def generate_case(self, epsilon, output_dir):
        """Run the simulation for a given method and ε, saving intermediate results."""
        print(f"LLoyd:, ε={epsilon}, profile={self.profile}")

        tic = perf_counter_ns()

        if self.lloyd:
            suffix = "2D_lloyd"
        else:
            suffix = ""

        if self.profile == "shallowjet":
            X, Y, G, h_true = initialisation(
                self.device,
                self.dtype,
                epsilon=epsilon,
                b=self.b,
                d=self.d,
                profile_type="jet" + suffix,
                cuda=self.device.index,
                tol=self.tol,
            )
        else:
            X, Y, G, h_true = initialisation(
                self.device,
                self.dtype,
                epsilon=epsilon,
                b=self.b,
                d=self.d,
                profile_type=self.profile + suffix,
                cuda=self.device.index,
                tol=self.tol,
            )
        toc = perf_counter_ns()

        if self.lloyd:
            print(f"Lloyd generation completed for {self.profile}: {toc - tic} ns")

        # Save intermediate results to file
        result = {
            "X": X.cpu(),
            "Y": Y.cpu(),
            "G": G.cpu(),
            "h_true": h_true.cpu(),
        }

        output_path = f"{output_dir}/epsilon_{epsilon}_profile_{self.profile}_lloyd.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        print(f"SimLlody evlation saved to {output_path}")

    def lloyd_or_not(self, result_file, epsilon):
        # Load saved simulation results
        if self.lloyd:
            # Load pre-ran data of Lloyd fitting
            with open(result_file, "rb") as f:
                result = pickle.load(f)

            h_true = result["h_true"]
            h_true /= h_true.sum()
            Y = result["Y"]
            X = result["X"]
            G = result["G"]
        else:
            X, Y, G, h_true = initialisation(
                self.device,
                self.dtype,
                epsilon=epsilon,
                b=self.b,
                d=self.d,
                profile_type="jet" if self.profile == "shallowjet" else self.profile,
                cuda=self.device.index,
                tol=self.tol,
            )
            h_true /= h_true.sum()

        return X, Y, G, h_true

    def run_simulation(self, method, epsilon, result_file, output_dir):
        """Run the simulation for a given method and ε, saving intermediate results."""
        print(f"Running simulation: {method}, ε={epsilon}, profile={self.profile}")

        X, Y, G, h_true = self.lloyd_or_not(result_file, epsilon)

        X, Y, G, swsg_class, h_true = swsg_class_generate(
            X,
            Y,
            G,
            h_true,
            self.device,
            self.dtype,
            epsilon=epsilon,
            cuda=self.device.index,
            tol=self.tol,
            lloyd=self.lloyd,
        )

        tic = perf_counter_ns()
        φ, ψ, φ_s, ψ_s, grad_phi, grad_phi_debias, error_list = swsg_solver(
            swsg_class, method=method, tolerance=self.tol, lambert_tolerance=self.tol
        )
        toc = perf_counter_ns()
        print(f"SWSG solver completed for {method}: {toc - tic} ns")

        # Calculate intermediate results
        h = (ψ_s - ψ) / 0.1
        debias_x_star = grad_phi - (grad_phi_debias - G)

        # Save intermediate results to file
        result = {
            "h": h.cpu(),
            "grad_phi": grad_phi.cpu(),
            "debias_x_star": debias_x_star.cpu(),
            "φ": φ.cpu(),
            "ψ": ψ.cpu(),
            "ψ_s": ψ_s.cpu(),
            "error_list": error_list,
        }

        suffix = "_lloyd" if self.lloyd else "_nolloyd"
        output_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_results{suffix}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        print(f"Simulation results saved to {output_path}")

    def compute_errors(self, method, epsilon, result_file, lloyd_file, output_dir):
        """Compute norms and Wasserstein distances using saved simulation results."""

        #################################################################################################################### Correct this
        print(f"Computing errors for: {method}, ε={epsilon}")

        # Load saved simulation results
        with open(result_file, "rb") as f:
            result = pickle.load(f)

        h = result["h"]
        grad_phi = result["grad_phi"]
        debias_x_star = result["debias_x_star"]

        X, Y, G, h_true = self.lloyd_or_not(lloyd_file, epsilon)

        N = len(X[:, 0])
        h = (h / N).view(-1, 1)

        # Norm errors for h
        l1_h = torch.linalg.norm(h_true - h, ord=1)
        l2_h = torch.linalg.norm(h_true - h, ord=2)
        linf_h = torch.linalg.norm(h_true - h, ord=float("inf")).item()

        error_data = {
            "h_error": {
                "l1": l1_h.item(),
                "l2": l2_h.item(),
                "linf": linf_h,
            },
            "bias": {},
            "debias": {},
        }

        # Regular mesh error
        for key, target in {
            "bias": grad_phi,
            "debias": debias_x_star,
        }.items():
            l1 = torch.linalg.norm(Y - target, ord=1) / N
            l2 = torch.linalg.norm(Y - target, ord=2) / N**0.5
            linf = torch.linalg.norm(Y - target, ord=float("inf")).item()
            error_data[key]["fit_mesh_error"] = dict(
                l1=l1.item(), l2=l2.item(), linf=linf
            )

            l1 = torch.linalg.norm(X - target, ord=1) / N
            l2 = torch.linalg.norm(X - target, ord=2) / N**0.5
            linf = torch.linalg.norm(X - target, ord=float("inf")).item()
            error_data[key]["regular_mesh_error"] = dict(
                l1=l1.item(), l2=l2.item(), linf=linf
            )

        # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        suffix = "_lloyd" if self.lloyd else "_nolloyd"
        error_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_lnormerrors{suffix}.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(error_data, f)
        print(f"Error data saved to {error_path}")

    def height_func(
        self, x, f=1.0, g=0.1, a=0.1, b=10, c=0.5, d=1.0, profile_type="jet"
    ):
        if profile_type == "jet":
            return a * np.tanh(self.b * (x[:, 1] - c)) + d
        elif profile_type == "shallowjet":
            return a * np.tanh(self.b * (x[:, 1] - c)) + d
        elif profile_type == "incline":
            return a * self.b * (x[:, 1] - c) + d
        elif profile_type == "uniform":
            return torch.ones_like(x[:, 1]) / len(x[:, 1])
        elif profile_type == "perturbedjet":
            temp = a * np.tanh(self.b * (x[:, 1] - c)) + d
            no, no0 , no1  = normal_pdf(x[:,0],x[:,1],0.5,0.3,0.1,strength=0.0001)  ## 0 is stationnary 
            temp = temp  + no
            raise temp
        else:
            raise KeyError("Unknown profile type")

    def compute_dense_samples(self, a=0.1, b=10, c=0.5, d=1.0, full=True):
        dense_epsilon = 0.002
        m1 = m2 = int(1 / 0.002)

        # Initialise the regular denisty - don't need to save as we can rerun anything
  
        X = torch.cartesian_prod(
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        )

        h_density = self.height_func(
            X, a=a, b=self.b, c=c, d=d, profile_type=self.profile
        ).view(-1, 1)
        h_density /= h_density.sum()

        if full:
            return X, h_density
        else:
            return (
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        ), h_density
    def compute_density_symmetric_potential(self, output_dir):
        ###############################################################
        ##############################################################
        X, h_density = self.compute_dense_samples(a=0.1, c=0.5, d=self.d, full=False)

        # compute symmetric OT problem (balanced) and  sav full class.
        dense_symmetric_dict = compute_dense_symmetric_potential(
            X, h_density, X, h_density, self.device
        )

        # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        error_path = f"{output_dir}/dense_sym_profile_{self.profile}_dict.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(dense_symmetric_dict, f)
        print(f"Dense data saved to {error_path}")

    def compute_W2_errors(
        self, method, epsilon, result_file, lloyd_file, l_errors, output_dir, which=1
    ):
        """Compute norms and Wasserstein distances using saved simulation results."""
        print(f"Computing errors for: {method}, ε={epsilon}")

        # Load saved simulation results
        with open(result_file, "rb") as f:
            result = pickle.load(f)

        h = result["h"]
        grad_phi = result["grad_phi"]
        debias_x_star = result["debias_x_star"]

        X, Y, G, h_true = self.lloyd_or_not(lloyd_file, epsilon)

        # Load dense sym pot results
        error_path = f"{output_dir}/dense_sym_profile_{self.profile}_dict.pkl"
        with open(error_path, "rb") as f:
            dense_symmetric_dict = pickle.load(f)

        ############### Wasserstien Error ###############


        N = len(X[:, 0])

        _torch_numpy_process = lambda x: torch.tensor(
            x, dtype=torch.float64, device=self.device
        )

        if self.lloyd:
            uni_weights = _torch_numpy_process(torch.ones_like(h))
            uni_weights /= uni_weights.sum()
        else:
            uni_weights = h_true / h_true.sum()

        # Load data;
        with open(l_errors, "rb") as f:
            method_data = pickle.load(f)

        N = len(X)
        n = int(np.sqrt(N))
        def periodic_g_x_vel(Gt, Xtilde, dt, L=1.0, periodic=True):
            """
            Periodic boundary reconsruciton give, periodic in x
            """
            # rotate
            x_component = -(Gt[:, 1] - Xtilde[:, 1])  
            y_component  = (Gt[:, 0] - Xtilde[:, 0])

            if periodic:
                # Stack the x_component and apply periodic boundary condition
                stacked_x = torch.stack([x_component, x_component + L, x_component - L], dim=-1)

                # Find the closest value with the minimum absolute difference
                min_diff_indices = torch.abs(stacked_x).min(dim=-1).indices

                # Select the correct x_component based on the minimum absolute difference
                x_component = stacked_x.gather(dim=-1, index=min_diff_indices.unsqueeze(-1)).squeeze(-1)


            return torch.stack([x_component, y_component], dim=-1) /dt

        # 1: u_g [l1, l2, linf]
        print("WHICH",  which)
        if which == 1:
            for key, target in {
                "bias": grad_phi,
                "debias": debias_x_star,
            }.items():
                diff = G - target - self.u_g(target)
                l1 = torch.linalg.norm(diff, ord=1) / N
                l2 = torch.linalg.norm(diff, ord=2) / N**0.5
                linf = torch.linalg.norm(diff, ord=float("inf")).item()
                method_data[key]["u_g_error"] = dict(
                    l1=l1.item(), l2=l2.item(), linf=linf
                )
                diff = G - target - self.u_g(X)
                l1 = torch.linalg.norm(diff, ord=1) / N
                l2 = torch.linalg.norm(diff, ord=2) / N**0.5
                linf = torch.linalg.norm(diff, ord=float("inf")).item()
                method_data[key]["u_g_error_true"] = dict(
                    l1=l1.item(), l2=l2.item(), linf=linf
                )
                
                diff = periodic_g_x_vel(G, target, 1, L=1.0, periodic=False) - self.u_g(X)
                l1 = torch.linalg.norm(diff, ord=1) / N
                l2 = torch.linalg.norm(diff, ord=2) / N**0.5
                linf = torch.linalg.norm(diff, ord=float("inf")).item()
                method_data[key]["u_g_error_j_true"] = dict(
                    l1=l1.item(), l2=l2.item(), linf=linf
                )
        # ?: X true [l1, l2, linf] (Doesn't make sense for later time steps)
        # 2: h reconstruction, S_eps (__, dense) [with orginal too?]
        if which == 2:
             # Generate the dense mesh with 250 000 points.
            loss = lambda a, x, b, y, f0, g0: Sinkhorn_Divergence_balanced(
                x,
                a,
                y,
                b,
                f0=f0,
                g0=g0,
                dense_symmetric_potential=dense_symmetric_dict,
                tol=1e-12,
            )
            X_dense, h_true_dense = self.compute_dense_samples(
                a=0.1, b=self.b, c=0.5, d=self.d, full=True
            )
            dense_weights = _torch_numpy_process(h_true_dense)
            dense_weights /= dense_weights.sum()
            print('DENSE weight device', dense_weights.device, self.device)

            dense_points = _torch_numpy_process(X_dense)
            N_dense = len(X_dense)
            n_dense = int(np.sqrt(N_dense))
            print("here")
            if self.lloyd:
                mesh = _torch_numpy_process(Y)
                dense = dense_points
            else:
                mesh = (_torch_numpy_process(Y[::n, 0]), _torch_numpy_process(Y[:n, 1]))
                dense = (_torch_numpy_process(X_dense[::n_dense, 0]), _torch_numpy_process(X_dense[:n_dense, 1]))
            s, uotclass = loss(
                dense_weights,
                dense,
                uni_weights,
                mesh,
                None,
                None,
            )
            method_data["h_error"]["dense_original"] = s
            print("Oringal Se loss:", s)

            print("nope")
            # This can always be tensorised
            s, uotclass = loss(
                dense_weights,
                (_torch_numpy_process(X_dense[::n_dense, 0]), _torch_numpy_process(X_dense[:n_dense, 1])),                                                  #dense_weights,
                _torch_numpy_process(h / N),
                (_torch_numpy_process(X[::n, 0]), _torch_numpy_process(X[:n, 1])),                                  # _torch_numpy_process(X),
                uotclass.f,
                uotclass.g,
            )
            method_data["h_error"]["dense_W_error"] = s
            print("h error", s)

        # 3: h reconstruction, S_eps (__, fine mesh) [with orginal too?]
        if which == 3:
            loss = lambda a, x, b, y, f0, g0: Sinkhorn_Divergence_balanced(
                    x,
                    a,
                    y,
                    b,
                    f0=f0,
                    g0=g0,
                    dense_symmetric_potential=None,
                    tol=1e-12,
                    epsilon=epsilon,
                    fullcompute=True
                )
            if self.lloyd:
                mesh = _torch_numpy_process(Y)
            else:
                mesh = (_torch_numpy_process(Y[::n, 0]), _torch_numpy_process(Y[:n, 1]))
            s, uotclass = loss(
                uni_weights,
                mesh,
                uni_weights,
                mesh,
                None,
                None,
            )
            method_data["h_error"]["fine_original"] = s
            print("Oringal Se loss:", s)

            print("nope")
            # This can always be tensorised
            s, uotclass = loss(
                uni_weights,
                mesh,                                                  #dense_weights,
                _torch_numpy_process(h / N),
                (_torch_numpy_process(X[::n, 0]), _torch_numpy_process(X[:n, 1])),                                  # _torch_numpy_process(X),
                None,
                None,
            )
            method_data["h_error"]["fine_W_error"] = s
            print("h error", s)
            
        # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        suffix = "_lloyd" if self.lloyd else "_nolloyd"
        error_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_errors_{which}_which{suffix}.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(method_data, f)
        print(f"Error data saved to {error_path}")

    def combine_dicts(self, method, epsilon, output_dir):
        def merge_dicts(dict1, dict2):
            """Recursively merges two dictionaries, keeping the value from dict1 if keys overlap."""
            for key, value in dict2.items():
                if key in dict1:
                    if isinstance(dict1[key], dict) and isinstance(value, dict):
                        # If both values are dictionaries, merge them recursively
                        dict1[key] = merge_dicts(dict1[key], value)
                    else:
                        # If not both are dictionaries, keep the value from dict1 (or dict2, depending on preference)
                        continue  # This keeps dict1's value and discards dict2's value for this key
                else:
                    dict1[key] = value
            return dict1

        suffix = "_lloyd" if self.lloyd else "_nolloyd"
        main_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_errors"
        which = 1
        with open(main_path + f"_{which}_which{suffix}.pkl", "rb") as f:
            dict0 = pickle.load(f)

        for which in [2, 3]:
            with open(main_path + f"_{which}_which{suffix}.pkl", "rb") as f:
                dict1 = pickle.load(f)
            dict0 = merge_dicts(dict0, dict1)

        error_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_errors{suffix}.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(dict0, f)
        print(f"Error data saved to {error_path}")
