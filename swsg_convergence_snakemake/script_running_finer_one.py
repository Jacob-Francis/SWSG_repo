from swsg_simulation_class import SWSGSimulation
sim = SWSGSimulation(cuda=4, profile="perturbedjet", d=1, suff="_nolloyd")
sim.run_simulation("one", 0.003125/2, None, output_dir="data_store")

