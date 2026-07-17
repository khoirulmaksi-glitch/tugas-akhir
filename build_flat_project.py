import os
import shutil
from pathlib import Path

BASE_DIR = Path('/Users/supermaxxxzz/tugas-akhir')
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = {}

files["../requirements.txt"] = """numpy
pandas
scipy
matplotlib
openpyxl
pypower
"""

files["config.py"] = """
import os
from pathlib import Path

# Parameter Utama
POPULATION_SIZE = 100
MAX_ITERATIONS = 400
NUMBER_OF_RUNS = 30

# Batas Variabel
V_G_MIN, V_G_MAX = 0.95, 1.05
T_MIN, T_MAX = 0.90, 1.10
Q_C_MIN, Q_C_MAX = 0.0, 40.0

# Penalti
LAMBDA_V = 1e6
LAMBDA_Q = 1e6
LAMBDA_CONV = 1e8

# Parameter HWOA-GWO
GAMMA_DEFAULT = 1.0
EARLY_STOPPING_TOLERANCE = 1e-6
EARLY_STOPPING_PATIENCE = 30

ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "output"

SEEDS = list(range(1000, 1000 + NUMBER_OF_RUNS))
"""

files["power_flow.py"] = """
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
"""

files["constraints.py"] = """
from pypower.idx_bus import VM, VMAX, VMIN
from pypower.idx_gen import QG, QMAX, QMIN
import config

def calculate_penalties(results):
    bus, gen = results['bus'], results['gen']
    penalty = 0.0
    qg, qmax, qmin = gen[:, QG], gen[:, QMAX], gen[:, QMIN]
    for i in range(len(qg)):
        if qg[i] > qmax[i]: penalty += config.LAMBDA_Q * (qg[i] - qmax[i])**2
        elif qg[i] < qmin[i]: penalty += config.LAMBDA_Q * (qmin[i] - qg[i])**2
    v, vmax, vmin = bus[:, VM], bus[:, VMAX], bus[:, VMIN]
    for i in range(len(v)):
        if v[i] > vmax[i]: penalty += config.LAMBDA_V * (v[i] - vmax[i])**2
        elif v[i] < vmin[i]: penalty += config.LAMBDA_V * (vmin[i] - v[i])**2
    return penalty
"""

files["control_variables.py"] = """
import numpy as np
from pypower.idx_bus import BUS_TYPE
from pypower.idx_brch import TAP
import config

class ControlVariables:
    def __init__(self, ppc):
        bus, branch, gen = ppc['bus'], ppc['branch'], ppc['gen']
        self.gen_idx = gen[:, 0].astype(int) - 1
        self.pq_idx = np.where(bus[:, BUS_TYPE] == 1)[0]
        self.tap_idx = np.where((branch[:, TAP] != 0) & (branch[:, TAP] != 1))[0]
        self.qc_idx = np.array([9, 11, 14, 16, 19, 20, 22, 23, 28])
        
        self.num_V = len(self.gen_idx)
        self.num_T = len(self.tap_idx)
        self.num_Q = len(self.qc_idx)
        self.dim = self.num_V + self.num_T + self.num_Q
        
        lb_v = np.ones(self.num_V) * config.V_G_MIN
        ub_v = np.ones(self.num_V) * config.V_G_MAX
        lb_t = np.ones(self.num_T) * config.T_MIN
        ub_t = np.ones(self.num_T) * config.T_MAX
        lb_q = np.ones(self.num_Q) * config.Q_C_MIN
        ub_q = np.ones(self.num_Q) * config.Q_C_MAX
        
        self.lb = np.concatenate([lb_v, lb_t, lb_q])
        self.ub = np.concatenate([ub_v, ub_t, ub_q])

def apply_controls(ppc, X, ctrl):
    from pypower.idx_bus import VM, BS
    from pypower.idx_gen import VG
    from pypower.idx_brch import TAP
    V = X[:ctrl.num_V]
    T = X[ctrl.num_V : ctrl.num_V+ctrl.num_T]
    Qc = X[ctrl.num_V+ctrl.num_T :]
    for i in range(ctrl.num_V):
        ppc['bus'][ctrl.gen_idx[i], VM] = V[i]
        ppc['gen'][i, VG] = V[i]
    for i in range(ctrl.num_T):
        ppc['branch'][ctrl.tap_idx[i], TAP] = T[i]
    ppc['bus'][:, BS] = 0.0
    for i in range(ctrl.num_Q):
        ppc['bus'][ctrl.qc_idx[i], BS] = Qc[i]
    return ppc
"""

files["objective_function.py"] = """
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
"""

files["hybrid_woa_gwo.py"] = """
import numpy as np
import time
from objective_function import calculate_fitness
import config

def run(base_ppc, ctrl, early_stopping=False, seed=0):
    np.random.seed(seed)
    dim, pop_size, max_iter = ctrl.dim, config.POPULATION_SIZE, config.MAX_ITERATIONS
    lb, ub = ctrl.lb, ctrl.ub
    
    Positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    Alpha_pos, Beta_pos, Delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    Alpha_score, Beta_score, Delta_score = np.inf, np.inf, np.inf
    
    cg_curve = np.zeros(max_iter)
    Ploss_best = np.inf; VD_best = np.inf; is_feasible = False
    patience_cnt = 0; stop_iter = max_iter; stop_reason = 'maximum_iterations'
    t0 = time.time()
    
    for t in range(max_iter):
        for i in range(pop_size):
            fit, pl, vd, feas = calculate_fitness(Positions[i,:], base_ppc, ctrl)
            if fit < Alpha_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = Alpha_score, Alpha_pos.copy()
                Alpha_score, Alpha_pos = fit, Positions[i,:].copy()
                Ploss_best, VD_best, is_feasible = pl, vd, feas
            elif fit < Beta_score and fit > Alpha_score:
                Delta_score, Delta_pos = Beta_score, Beta_pos.copy()
                Beta_score, Beta_pos = fit, Positions[i,:].copy()
            elif fit < Delta_score and fit > Beta_score:
                Delta_score, Delta_pos = fit, Positions[i,:].copy()
                
        a = 2 - t * (2 / max_iter)
        a2 = -1 + t * (-1 / max_iter)
        w_t = (1 - t / max_iter) ** config.GAMMA_DEFAULT
        
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            A1, C1 = 2 * a * r1 - a, 2 * r2
            p, l = np.random.rand(), (a2 - 1) * np.random.rand() + 1
            if p < 0.5:
                if abs(A1) >= 1:
                    X_rand = Positions[np.random.randint(0, pop_size), :]
                    X_WOA = X_rand - A1 * abs(C1 * X_rand - Positions[i,:])
                else:
                    X_WOA = Alpha_pos - A1 * abs(C1 * Alpha_pos - Positions[i,:])
            else:
                X_WOA = abs(Alpha_pos - Positions[i,:]) * np.exp(l) * np.cos(l*2*np.pi) + Alpha_pos
                
            X1 = Alpha_pos - (2*a*np.random.rand()-a) * abs(2*np.random.rand() * Alpha_pos - Positions[i,:])
            X2 = Beta_pos - (2*a*np.random.rand()-a) * abs(2*np.random.rand() * Beta_pos - Positions[i,:])
            X3 = Delta_pos - (2*a*np.random.rand()-a) * abs(2*np.random.rand() * Delta_pos - Positions[i,:])
            X_GWO = (X1 + X2 + X3) / 3.0
            
            Positions[i,:] = np.clip(w_t * X_WOA + (1 - w_t) * X_GWO, lb, ub)
            
        cg_curve[t] = Alpha_score
        if (t+1)%10==0 or t==max_iter-1: print(f"\\rIter {t+1:03d}/{max_iter} | Ploss: {Ploss_best:.4f} | VD: {VD_best:.4f} | Fit: {Alpha_score:.4f}", end="", flush=True)
        if early_stopping and t > 0:
            imp = abs(cg_curve[t-1] - cg_curve[t]) / max(1, abs(cg_curve[t-1]))
            if imp < config.EARLY_STOPPING_TOLERANCE:
                patience_cnt += 1
                if patience_cnt >= config.EARLY_STOPPING_PATIENCE:
                    stop_iter = t + 1; stop_reason = 'early_stopping_no_improvement'
                    cg_curve[t+1:] = cg_curve[t]
                    break
            else: patience_cnt = 0
    print()
    return {"fit": Alpha_score, "cg": cg_curve.tolist(), "pos": Alpha_pos.tolist(), "ploss": Ploss_best, "vd": VD_best, "feas": bool(is_feasible), "iter": stop_iter, "time": time.time()-t0, "reason": stop_reason}
"""

files["woa_tunggal.py"] = """
import numpy as np
import time
from objective_function import calculate_fitness
import config

def run(base_ppc, ctrl, early_stopping=False, seed=0):
    np.random.seed(seed)
    dim, pop_size, max_iter = ctrl.dim, config.POPULATION_SIZE, config.MAX_ITERATIONS
    lb, ub = ctrl.lb, ctrl.ub
    Positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    cg_curve = np.zeros(max_iter)
    best_score = np.inf; best_pos = None; Ploss_best = np.inf; VD_best = np.inf; is_feasible = False
    t0 = time.time()
    
    for t in range(max_iter):
        for i in range(pop_size):
            fit, pl, vd, feas = calculate_fitness(Positions[i,:], base_ppc, ctrl)
            if fit < best_score:
                best_score, best_pos = fit, Positions[i,:].copy()
                Ploss_best, VD_best, is_feasible = pl, vd, feas
        
        a = 2 - t * (2 / max_iter)
        a2 = -1 + t * (-1 / max_iter)
        for i in range(pop_size):
            r1, r2 = np.random.rand(), np.random.rand()
            A, C = 2 * a * r1 - a, 2 * r2
            p, l = np.random.rand(), (a2 - 1) * np.random.rand() + 1
            if p < 0.5:
                if abs(A) >= 1:
                    X_rand = Positions[np.random.randint(0, pop_size), :]
                    Positions[i,:] = X_rand - A * abs(C * X_rand - Positions[i,:])
                else:
                    Positions[i,:] = best_pos - A * abs(C * best_pos - Positions[i,:])
            else:
                Positions[i,:] = abs(best_pos - Positions[i,:]) * np.exp(l) * np.cos(l*2*np.pi) + best_pos
            Positions[i,:] = np.clip(Positions[i,:], lb, ub)
            
        cg_curve[t] = best_score
        if (t+1)%10==0 or t==max_iter-1: print(f"\\rIter {t+1:03d}/{max_iter} | Ploss: {Ploss_best:.4f} | VD: {VD_best:.4f} | Fit: {best_score:.4f}", end="", flush=True)
    print()
    return {"fit": best_score, "cg": cg_curve.tolist(), "pos": best_pos.tolist(), "ploss": Ploss_best, "vd": VD_best, "feas": bool(is_feasible), "iter": max_iter, "time": time.time()-t0, "reason": "maximum_iterations"}
"""

files["pso.py"] = """
import numpy as np
import time
from objective_function import calculate_fitness
import config

def run(base_ppc, ctrl, early_stopping=False, seed=0):
    np.random.seed(seed)
    dim, pop_size, max_iter = ctrl.dim, config.POPULATION_SIZE, config.MAX_ITERATIONS
    lb, ub = ctrl.lb, ctrl.ub
    Positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    Velocity = np.zeros_like(Positions)
    PBest, PBestScore = Positions.copy(), np.full(pop_size, np.inf)
    cg_curve = np.zeros(max_iter)
    best_score = np.inf; best_pos = None; Ploss_best = np.inf; VD_best = np.inf; is_feasible = False
    t0 = time.time()
    
    for t in range(max_iter):
        w = 0.9 - (0.9 - 0.4) * (t / max_iter)
        for i in range(pop_size):
            fit, pl, vd, feas = calculate_fitness(Positions[i,:], base_ppc, ctrl)
            if fit < PBestScore[i]:
                PBestScore[i], PBest[i,:] = fit, Positions[i,:].copy()
            if fit < best_score:
                best_score, best_pos = fit, Positions[i,:].copy()
                Ploss_best, VD_best, is_feasible = pl, vd, feas
        cg_curve[t] = best_score
        r1, r2 = np.random.rand(pop_size, dim), np.random.rand(pop_size, dim)
        Velocity = w*Velocity + 2.0*r1*(PBest - Positions) + 2.0*r2*(best_pos - Positions)
        Positions = np.clip(Positions + Velocity, lb, ub)
        if (t+1)%10==0 or t==max_iter-1: print(f"\\rIter {t+1:03d}/{max_iter} | Ploss: {Ploss_best:.4f} | VD: {VD_best:.4f} | Fit: {best_score:.4f}", end="", flush=True)
    print()
    return {"fit": best_score, "cg": cg_curve.tolist(), "pos": best_pos.tolist(), "ploss": Ploss_best, "vd": VD_best, "feas": bool(is_feasible), "iter": max_iter, "time": time.time()-t0, "reason": "maximum_iterations"}
"""

files["ga.py"] = """
import numpy as np
import time
from objective_function import calculate_fitness
import config

def run(base_ppc, ctrl, early_stopping=False, seed=0):
    np.random.seed(seed)
    dim, pop_size, max_iter = ctrl.dim, config.POPULATION_SIZE, config.MAX_ITERATIONS
    lb, ub = ctrl.lb, ctrl.ub
    Positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    cg_curve = np.zeros(max_iter)
    best_score = np.inf; best_pos = None; Ploss_best = np.inf; VD_best = np.inf; is_feasible = False
    t0 = time.time()
    
    for t in range(max_iter):
        for i in range(pop_size):
            fit, pl, vd, feas = calculate_fitness(Positions[i,:], base_ppc, ctrl)
            if fit < best_score:
                best_score, best_pos = fit, Positions[i,:].copy()
                Ploss_best, VD_best, is_feasible = pl, vd, feas
        cg_curve[t] = best_score
        
        NewPositions = np.zeros_like(Positions)
        NewPositions[0,:] = best_pos
        for i in range(1, pop_size, 2):
            p1, p2 = Positions[np.random.randint(0, pop_size),:], Positions[np.random.randint(0, pop_size),:]
            c1, c2 = p1.copy(), p2.copy()
            if np.random.rand() < 0.8:
                alpha = np.random.rand(dim)
                c1, c2 = alpha*p1 + (1-alpha)*p2, alpha*p2 + (1-alpha)*p1
            if np.random.rand() < 0.1: c1 += np.random.randn(dim) * (ub-lb) * 0.1
            if np.random.rand() < 0.1: c2 += np.random.randn(dim) * (ub-lb) * 0.1
            NewPositions[i,:] = np.clip(c1, lb, ub)
            if i+1 < pop_size: NewPositions[i+1,:] = np.clip(c2, lb, ub)
        Positions = NewPositions
        if (t+1)%10==0 or t==max_iter-1: print(f"\\rIter {t+1:03d}/{max_iter} | Ploss: {Ploss_best:.4f} | VD: {VD_best:.4f} | Fit: {best_score:.4f}", end="", flush=True)
    print()
    return {"fit": best_score, "cg": cg_curve.tolist(), "pos": best_pos.tolist(), "ploss": Ploss_best, "vd": VD_best, "feas": bool(is_feasible), "iter": max_iter, "time": time.time()-t0, "reason": "maximum_iterations"}
"""

files["main.py"] = """
import argparse
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
from power_flow import load_system, run_power_flow
from control_variables import ControlVariables
import hybrid_woa_gwo
import woa_tunggal
import pso
import ga

def run_experiment(name, runner_func, ppc, ctrl, early_stop=False):
    print(f"\\n--- Running {name} ---")
    runs = []
    with open(config.OUTPUT_DIR / "run_log.txt", "a") as f:
        f.write(f"\\n--- Running {name} ---\\n")
        
    for i, seed in enumerate(config.SEEDS):
        print(f"[{name}] Run {i+1:02d}/{config.NUMBER_OF_RUNS} : ", end="")
        res = runner_func(ppc, ctrl, early_stopping=early_stop, seed=seed)
        res['name'] = name
        runs.append(res)
        
    best_run = min(runs, key=lambda x: x['fit'])
    with open(config.OUTPUT_DIR / f"{name}_best_solution.json", "w") as f: json.dump(best_run, f, indent=4)
    
    # Save convergence history
    pd.DataFrame([r['cg'] for r in runs]).to_csv(config.OUTPUT_DIR / f"{name}_convergence_data.csv", index=False)
    return runs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="all", choices=["hybrid", "woa", "pso", "ga", "all"])
    parser.add_argument("--mode", type=str, default="final", help="use 'debug' to test quickly")
    args = parser.parse_args()

    if args.mode == "debug":
        config.POPULATION_SIZE = 10
        config.MAX_ITERATIONS = 10
        config.NUMBER_OF_RUNS = 2
        config.SEEDS = config.SEEDS[:2]
        
    with open(config.OUTPUT_DIR / "run_log.txt", "w") as f: f.write(f"Starting ORPD Simulation (Mode: {args.mode})\\n")
        
    print(f"Loading IEEE 30-Bus System Dataset from {config.DATASET_DIR}")
    base_ppc = load_system()
    ctrl = ControlVariables(base_ppc)
    print(f"Variables: {ctrl.dim} (V: {ctrl.num_V}, T: {ctrl.num_T}, Q: {ctrl.num_Q})")
    
    res, success = run_power_flow(base_ppc)
    if success:
        from pypower.idx_brch import PF, PT; from pypower.idx_bus import VM
        pl = res['branch'][:, PF].sum() + res['branch'][:, PT].sum()
        vd = abs(res['bus'][ctrl.pq_idx, VM] - 1.0).sum()
        print(f"Base Case -> Ploss: {pl:.4f} MW, VD: {vd:.4f}")
    
    all_results = {}
    
    if args.algorithm in ["ga", "all"]: all_results["GA"] = run_experiment("ga", ga.run, base_ppc, ctrl)
    if args.algorithm in ["pso", "all"]: all_results["PSO"] = run_experiment("pso", pso.run, base_ppc, ctrl)
    if args.algorithm in ["woa", "all"]: all_results["WOA"] = run_experiment("woa", woa_tunggal.run, base_ppc, ctrl)
    if args.algorithm in ["hybrid", "all"]:
        all_results["Hybrid_WOA_GWO_NoES"] = run_experiment("hybrid_woa_gwo_noes", hybrid_woa_gwo.run, base_ppc, ctrl, False)
        all_results["Hybrid_WOA_GWO"] = run_experiment("hybrid_woa_gwo", hybrid_woa_gwo.run, base_ppc, ctrl, True)

    if args.algorithm == "all":
        # Build master table
        rows = []
        plt.figure(figsize=(10,6))
        for algo_name, runs in all_results.items():
            ploss = [r['ploss'] for r in runs]; vd = [r['vd'] for r in runs]
            fit = [r['fit'] for r in runs]; time_ = [r['time'] for r in runs]
            feas = [r['feas'] for r in runs]
            
            rows.append({
                "Method": algo_name,
                "Best Ploss": round(np.min(ploss), 4), "Mean Ploss": round(np.mean(ploss), 4), "Std Ploss": round(np.std(ploss), 4),
                "Best VD": round(np.min(vd), 4), "Mean VD": round(np.mean(vd), 4), "Std VD": round(np.std(vd), 4),
                "Best Overall": round(np.min(fit), 4), "Mean Overall": round(np.mean(fit), 4), "Std Overall": round(np.std(fit), 4),
                "Mean Iter": round(np.mean([r['iter'] for r in runs]), 1),
                "Mean Time": round(np.mean(time_), 4),
                "Feasibility": round(sum(feas)/len(feas), 4)
            })
            best_run = min(runs, key=lambda x: x['fit'])
            if "NoES" not in algo_name: plt.plot(best_run['cg'], label=algo_name)
            
        df = pd.DataFrame(rows)
        df.to_csv(config.OUTPUT_DIR / "hasil_perbandingan.csv", index=False)
        df.to_excel(config.OUTPUT_DIR / "hasil_perbandingan.xlsx", index=False)
        
        plt.yscale('log'); plt.xlabel('Iteration'); plt.ylabel('Fitness'); plt.title('Convergence Curves')
        plt.legend(); plt.savefig(config.OUTPUT_DIR / "convergence_curves.png"); plt.close()
        
        print("\\nSimulation complete. All tables, graphs, and csvs saved to output/")

if __name__ == "__main__":
    main()
"""

for filepath, content in files.items():
    with open(OUTPUT_DIR / filepath, 'w') as f:
        f.write(content.strip() + '\\n')
