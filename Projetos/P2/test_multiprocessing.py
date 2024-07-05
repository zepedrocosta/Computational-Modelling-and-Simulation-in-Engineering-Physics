import numpy as np
from colorama import Fore
from multiprocessing import Pool


def run_simulation_wrapper(params):
    v0, alpha, mode = params
    return v0, alpha, run_simulation(v0, alpha, mode)


def run_automatic(mode):
    """
    Run the simulation with a range of values for v0 and alpha

    Parameters
    ----------
    mode : str
        The mode of the simulation
    """
    v0_range = list(range(0, 15001, 100))  # should start in 0
    alpha_range = list(range(16))

    success_count = 0
    simulation_count = 0
    valid_parameters = np.empty((0, 2), dtype=int)

    parameters = [(v0, alpha, mode) for v0 in v0_range for alpha in alpha_range]

    with Pool(5) as pool:
        for v0, alpha, success in pool.imap(run_simulation_wrapper, parameters):
            print(
                "Sucessos: {} | Simulações: {}".format(success_count, simulation_count),
                end="\r",
            )
            simulation_count += 1
            if success:
                success_count += 1
                valid_parameters = np.append(valid_parameters, [[v0, alpha]], axis=0)

    print(
        "Sucessos: {} | Simulações: {}".format(success_count, simulation_count),
        end="\r",
    )
    print(Fore.GREEN + f"Sucessos: {success_count}" + Fore.RESET)


# Dummy implementation: consider all simulations successful
def run_simulation(v0, alpha, mode):
    print(f"Running simulation with v0={v0} and alpha={alpha}")
    return True


# Example usage
if __name__ == "__main__":
    run_automatic("fast")
