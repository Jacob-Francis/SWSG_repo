
import torch
import pickle
from utils import initialisation, swsg_solver, swsg_class_generate, compute_dense_symmetric_potential, Sinkhorn_Divergence_balanced
from time import perf_counter_ns
from geomloss import SamplesLoss
import numpy as np

class SWSGSimulation:
    def __init__(self, cuda=None, profile="uniform", d=1, b=10, tol=1e-11):
        if cuda is None:
            self.device = 'cpu'
            self.dtype = torch.DoubleTensor
        else:
            self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
        self.profile = profile
        self.d = d
        self.b = b
        self.tol = tol

    def generate_case(self, epsilon, output_dir):
        """Run the simulation for a given method and ε, saving intermediate results."""
        print(f"LLoyd:, ε={epsilon}, profile={self.profile}")

        tic = perf_counter_ns()
        X, Y, G, h_true = initialisation(
            self.device, self.dtype, epsilon=epsilon, b=self.b, d=self.d, profile_type=self.profile+'2D_lloyd', cuda=self.device.index, tol=self.tol
        )
        toc = perf_counter_ns()
    
        print(f"Lloyd generation completed for {self.profile}: {toc - tic} ns")

        # Save intermediate results to file
        result = {
            "X": X.cpu(),
            "Y": Y.cpu(),
            "G": G.cpu(),
            "h_true": h_true.cpu(),
        }

        output_path = f"{output_dir}/epsilon_{epsilon}_profile_{self.profile}_llody.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        print(f"SimLlody evlation saved to {output_path}")

    def run_simulation(self, method, epsilon, result_file, output_dir):
        """Run the simulation for a given method and ε, saving intermediate results."""
        print(f"Running simulation: {method}, ε={epsilon}, profile={self.profile}")


                # Load saved simulation results
        with open(result_file, "rb") as f:
            result = pickle.load(f)

        h_true = result["h_true"]
        h_true /= h_true.sum()
        Y = result["Y"]
        X = result["X"]
        G = result["G"]
        X, Y, G, swsg_class, h_true = swsg_class_generate(
            X, Y, G, h_true, self.device, self.dtype, epsilon=epsilon, b=self.b, d=self.d, cuda=self.device.index, tol=self.tol
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

        output_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_results.pkl"
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
        
        with open(lloyd_file, "rb") as f:
            result = pickle.load(f)

        h_true = result["h_true"].view(-1, 1)
        h_true /= h_true.sum()
        Y = result["Y"]
        X = result["X"]
        G = result["G"]
        
        N = len(X[:, 0])
        h = (h/N).view(-1,1)                                                    

        # Norm errors for h
        l1_h = torch.linalg.norm(h_true - h, ord=1)
        l2_h = (
            torch.linalg.norm(h_true - h, ord=2)
        )
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
            l2 = (
                torch.linalg.norm(Y - target, ord=2) / N ** 0.5
            )
            linf = torch.linalg.norm(Y - target, ord=float("inf")).item()
            error_data[key]["fit_mesh_error"] = dict(
                l1=l1.item(), l2=l2.item(), linf=linf
            )

            l1 = torch.linalg.norm(X - target, ord=1) / N 
            l2 = (
                torch.linalg.norm(X - target, ord=2) / N** 0.5
            )
            linf = torch.linalg.norm(X - target, ord=float("inf")).item()
            error_data[key]["regular_mesh_error"] = dict(
                l1=l1.item(), l2=l2.item(), linf=linf
            )

        # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        error_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_lnormerrors.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(error_data, f)
        print(f"Error data saved to {error_path}")
    
    def height_func(self, x, f=1.0, g=0.1, a=0.1, b=10, c=0.5, d=1.0, profile_type="jet"):
        if profile_type == 'jet':
            return a * np.tanh(self.b * (x - c)) + d
        elif profile_type== 'incline': 
            return a * self.b * (x - c) + d
        elif profile_type == 'uniform':
            return torch.ones_like(x) / len(x)
        else:
            raise KeyError('Unknown profiel type')
     
    def compute_dense_samples(self, a=0.1, b=10, c=0.5, d=1.0):
        dense_epsilon = 0.002
        m1 = m2 = int(1/0.002)
        
        # Initialise the regular denisty - don't need to save as we can rerun anything
        X = torch.cartesian_prod(
            torch.linspace(1 / (2 * m2), 1 - 1 / (2 * m2), m2),
            torch.linspace(1 / (2 * m1), 1 - 1 / (2 * m1), m1),
        )
        
        h_density = self.height_func(X[:, 1], a=a, b=self.b, c=c, d=d, profile_type=self.profile).view(-1, 1)
        h_density /= h_density.sum()
        
        return X, h_density
         
    def compute_density_symmetric_potential(self, output_dir):
        ###############################################################
        ##############################################################
        X, h_density = self.compute_dense_samples(a=0.1, c=0.5, d=self.d)
        
        # compute symmetric OT problem (balanced) and  sav full class.
        dense_symmetric_dict = compute_dense_symmetric_potential(X, h_density, X, h_density, self.device)
                  
         # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        error_path = f"{output_dir}/dense_sym_profile_{self.profile}_dict.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(dense_symmetric_dict, f)
        print(f"Dense data saved to {error_path}")
                                                          

    def compute_W2_errors(self, method, epsilon, result_file, lloyd_file, l_errors, output_dir, which='1'):
        """Compute norms and Wasserstein distances using saved simulation results."""
        print(f"Computing errors for: {method}, ε={epsilon}")

        # Load saved simulation results
        with open(result_file, "rb") as f:
            result = pickle.load(f)

        h = result["h"]
        grad_phi = result["grad_phi"]
        debias_x_star = result["debias_x_star"]
        
        with open(lloyd_file, "rb") as f:
            result = pickle.load(f)

        h_true = result["h_true"]
        Y = result["Y"]
        X = result["X"]
        G = result["G"]
        
        # Load dense sym pot results
        error_path = f"{output_dir}/dense_sym_profile_{self.profile}_dict.pkl"
        with open(error_path, "rb") as f:
            dense_symmetric_dict = pickle.load(f)

        ############### Wasserstien Error ###############
        # Generate the dense mesh with 250 000 points.
        X_dense, h_true_dense = self.compute_dense_samples(a=0.1, b=self.b, c=0.5, d=self.d)
        
        N = len(X[:, 0])
        N_dense = len(X_dense[:, 0])

        _torch_numpy_process = lambda x : torch.tensor(x, dtype=torch.float64, device=self.device) 

        dense_weights = _torch_numpy_process(h_true_dense)
        dense_weights /= dense_weights.sum()
        dense_points=  _torch_numpy_process(X_dense)

        uni_weights = _torch_numpy_process(torch.ones_like(h))
        uni_weights /= uni_weights.sum()

        # Load data;
        with open(l_errors, "rb") as f:
            method_data = pickle.load(f)

        loss = lambda a,x,b,y,f0,g0 : Sinkhorn_Divergence_balanced(x,a,y,b,f0=f0, g0=g0, dense_symmetric_potential=dense_symmetric_dict, tol=1e-12)
        N = len(X)
        N_dense = len(X_dense)

        if which == '1':
            s, uotclass = loss(dense_weights, dense_points, uni_weights, _torch_numpy_process(Y), None, None)
            method_data["h_error"]["original"] = s
            print('Oringal Se loss:', s)
        elif which == '2':
            s,uotclass = loss(dense_weights, dense_points, _torch_numpy_process(h / N), _torch_numpy_process(X), uotclass.f, uotclass.g)
            method_data["h_error"]["W_error"] = s
            print("h error", s)
        elif which == '3':
            # REgular debiased
            s, uotclass = loss(dense_weights, dense_points, uni_weights, _torch_numpy_process(debias_x_star), uotclass.f, uotclass.g)
            method_data["debias"]["W_error_regular"] = s
            print("regular debiased", s)
        elif which == '4':
            # Regular biased
            s, uotclass = loss(dense_weights, dense_points, uni_weights, _torch_numpy_process(grad_phi), uotclass.f, uotclass.g)
            method_data["bias"]["W_error_regular"] = s
            print("regular bias", s)

        # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        error_path = f"{output_dir}/temp/{method}_epsilon_{epsilon}_profile_{self.profile}_errors_{which}_which.pkl"
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

        main_path  = f"{output_dir}/temp/{method}_epsilon_{epsilon}_profile_{self.profile}_errors"
        which =  1
        with open(main_path+f"_{which}_which.pkl", 'rb') as f:
            dict0 = pickle.load(f) 
        
        for which in [2, 3, 4]:
            with open(main_path+f"_{which}_which.pkl", 'rb') as f:
                dict1 = pickle.load(f) 
            dict0 = merge_dicts(dict0,dict1)
            
        error_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_errors.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(dict0, f)
        print(f"Error data saved to {error_path}")



