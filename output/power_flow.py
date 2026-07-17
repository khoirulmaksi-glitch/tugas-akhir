import pandas as pd
import numpy as np
import copy
from pypower.api import runpf, ppoption
import config

def load_system():
    bus_df = pd.read_csv(config.DATASET_DIR / "bus.csv")
    branch_df = pd.read_csv(config.DATASET_DIR / "branch.csv").drop(columns=['branch_id'], errors='ignore')
    gen_df = pd.read_csv(config.DATASET_DIR / "generator.csv").drop(columns=['gen_id'], errors='ignore')
    branch_vals = branch_df.values.copy()
    if np.all(branch_vals[:, 9] == 0):
        for t in [10, 11, 14, 35]: branch_vals[t, 9] = 1.0
    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus_df.values.copy(),
        "branch": branch_vals,
        "gen": gen_df.values.copy()
    }

def run_power_flow(ppc):
    opt = ppoption(VERBOSE=0, OUT_ALL=0)
    return runpf(copy.deepcopy(ppc), opt)