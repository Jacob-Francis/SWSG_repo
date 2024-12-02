
import torch
import pickle
from utils import initialisation, swsg_solver, swsg_class_generate
from time import perf_counter_ns
from geomloss import SamplesLoss

class SWSGSimulation:
    def __init__(self, cuda=None, profile="uniform", d=1, tol=1e-11):
        if cuda is None:
            self.device = 'cpu'
            self.dtype = torch.DoubleTensor
        else:
            self.device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
        self.profile = profile
        self.d = d
        self.tol = tol

    def generate_case(self, epsilon, output_dir):
        """Run the simulation for a given method and ε, saving intermediate results."""
        print(f"LLoyd:, ε={epsilon}, profile={self.profile}")

        tic = perf_counter_ns()
        X, Y, G, h_true = initialisation(
            self.device, self.dtype, epsilon=epsilon, d=self.d, profile_type=self.profile+'2D_lloyd', cuda=self.device.index, tol=self.tol
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
        Y = result["Y"]
        X = result["X"]
        G = result["G"]
        X, Y, G, swsg_class, h_true = swsg_class_generate(
            X, Y, G, h_true, self.device, self.dtype, epsilon=epsilon, d=self.d, cuda=self.device.index, tol=self.tol
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

        # Norm errors for h
        l1_h = torch.linalg.norm(h_true.view(-1, 1) - h, ord=1) / len(h_true) 
        l2_h = (
            torch.linalg.norm(h_true.view(-1, 1) - h, ord=2) / len(h_true) ** 0.5
        )
        linf_h = torch.linalg.norm(h_true.view(-1, 1) - h, ord=float("inf")).item()

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
            l1 = torch.linalg.norm(Y - target, ord=1) / len(h_true)
            l2 = (
                torch.linalg.norm(Y - target, ord=2) / len(h_true) ** 0.5
            )
            linf = torch.linalg.norm(Y - target, ord=float("inf")).item()
            error_data[key]["fit_mesh_error"] = dict(
                l1=l1.item(), l2=l2.item(), linf=linf
            )

            l1 = torch.linalg.norm(X - target, ord=1) / len(h_true) 
            l2 = (
                torch.linalg.norm(X - target, ord=2) / len(h_true)** 0.5
            )
            linf = torch.linalg.norm(X - target, ord=float("inf")).item()
            error_data[key]["regular_mesh_error"] = dict(
                l1=l1.item(), l2=l2.item(), linf=linf
            )

        # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        error_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_l_errors.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(error_data, f)
        print(f"Error data saved to {error_path}")
    

    def compute_W2_errors(self, method, epsilon, result_file, lloyd_file, l_errors, output_dir):
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

        ############### Wasserstien Error ###############
        # Generate the dense mesh with 250 000 points.
        if '2D_lloyd' in self.profile:
            X_dense, _, _, _, h_true_dense = initialisation(
                epsilon=epsilon / 3, d=d, profile_type=profile.split('2D')[0], cuda=cuda
            )
            X_dense = X_dense.type(dtype)
            h_true_dense = h_true_dense.type(dtype)
        else:
            X_dense, _, _, _, h_true_dense = initialisation(
                epsilon=epsilon / 3, d=d, profile_type=profile, cuda=cuda
            )
            X_dense = X_dense.type(dtype)
            h_true_dense = h_true_dense.type(dtype)
        
        N = len(X)
        N_dense = len(X_dense)

        _torch_numpy_process = lambda x : torch.tensor(x, stype=self.dtype, device=self.device) 

        dense_weights = _torch_numpy_process(h_true_dense / N_dense)
        dense_points=  _torch_numpy_process(X_dense)

        uni_weights = _torch_numpy_process(torch.ones_like(h) / N)

        # Load data;
        with open(l_errors, "rb") as f:
            method_data = pickle.load(f)

        loss = SamplesLoss("sinkhorn", p=2, blur=np.sqrt(epsilon / 3), scaling=0.9999, backend='multiscale')
        N = len(X)
        N_dense = len(X_dense)

        s = loss(dense_weights, dense_points, uni_weights, _torch_numpy_process(Y))
        method_data["h_error"]["original"] = s.detach().cpu()
        print('Oringal Se loss:', s)

        s = loss(dense_weights, dense_points, _torch_numpy_process(h / N), _torch_numpy_process(X))
        method_data["h_error"]["W_error"] = s.detach().cpu()
        print("h error", s)

        # REgular debiased
        s = loss(dense_weights, dense_points, uni_weights, _torch_numpy_process(debias_x_star))
        method_data["debias"]["W_error_regular"] = s.detach().cpu()
        print("regular debiased", s)
        
        # Regular biased
        s = loss(dense_weights, dense_points, uni_weights, _torch_numpy_process(grad_phi))
        method_data["bias"]["W_error_regular"] = s.detach().cpu()

        print("regular bias", s)

        # Save error data to file # {method}_epsilon_{epsilon}_profile_{profile}_errors
        error_path = f"{output_dir}/{method}_epsilon_{epsilon}_profile_{self.profile}_errors.pkl"
        with open(error_path, "wb") as f:
            pickle.dump(error_data, f)
        print(f"Error data saved to {error_path}")
