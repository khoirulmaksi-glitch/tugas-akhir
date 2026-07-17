import numpy as np
from pypower.idx_bus import VM
from pypower.idx_brch import PF, PT
from power_flow import run_power_flow
from control_variables import apply_controls
from constraints import calculate_penalties
import config

def calculate_fitness(X, base_ppc, ctrl):
    ppc = apply_controls(base_ppc, X, ctrl)
    results, success = run_power_flow(ppc)
    if not success:
        return config.LAMBDA_CONV, np.inf, np.inf, False
    branch, bus = results['branch'], results['bus']
    ploss = np.sum(branch[:, PF] + branch[:, PT])
    vd = np.sum(np.abs(bus[ctrl.pq_idx, VM] - 1.0))
    penalty = calculate_penalties(results)
    overall = ploss + vd + penalty
    return overall, ploss, vd, (penalty == 0)