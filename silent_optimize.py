import sys
import os
import logging
from contextlib import redirect_stdout
import tqdm


def silent_optimize(network, solver_name="gurobi", solver_options=None):
    """
    Optimize a PyPSA network silently: suppress logs, progress bars, and Gurobi messages.
    
    Parameters:
    - network: PyPSA Network object
    - solver_name: Solver name (default: "gurobi")
    - solver_options: Dictionary of solver options (default: {"OutputFlag": 0})
    """
    if solver_options is None:
        solver_options = {"OutputFlag": 0}

    # Suppress PyPSA, Linopy, Gurobi logging output
    for name in ["pypsa", "linopy", "gurobipy"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    # Monkey-patch tqdm to disable progress bars
    tqdm.tqdm = lambda *args, **kwargs: iter(args[0]) if args else iter([])

    # Redirect stdout to suppress Gurobi messages
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            network.optimize(solver_name=solver_name, solver_options=solver_options)
            #print("Objective value:", network.objective)

