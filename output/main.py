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

def run_experiment(name, runner_func, ppc, ctrl, early_stop=False):
    banner = f"""
============================================================
Running {name}
Population     : {config.POPULATION_SIZE}
Max Iteration  : {config.MAX_ITERATIONS}
Total Runs     : {config.NUMBER_OF_RUNS}
============================================================"""
    print(banner)
    runs = []
    with open(config.OUTPUT_DIR / "run_log.txt", "a") as f:
        f.write(banner + "\\n")
        
    for i, seed in enumerate(config.SEEDS):
        res = runner_func(ppc, ctrl, algo_name=name, run_idx=i+1, early_stopping=early_stop, seed=seed)
        res['name'] = name
        runs.append(res)
        
        # End of run prints
        stop_iter = res.get('iter', config.MAX_ITERATIONS)
        best_fit = res['fit']
        print(f"\n{name} | Run {i+1:02d}/{config.NUMBER_OF_RUNS} Completed | Best Fitness: {best_fit:.6f} | Stopped Iteration: {stop_iter}\n")
        
    best_run = min(runs, key=lambda x: x['fit'])
    with open(config.OUTPUT_DIR / f"{name}_best_solution.json", "w") as f: json.dump(best_run, f, indent=4)
    
    # Save convergence history
    pd.DataFrame([r['cg'] for r in runs]).to_csv(config.OUTPUT_DIR / f"{name}_convergence_data.csv", index=False)
    return runs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="all", choices=["hybrid", "all"])
    parser.add_argument("--mode", type=str, default="final", help="use 'debug' to test quickly")
    args = parser.parse_args()

    if args.mode == "debug":
        config.POPULATION_SIZE = 10
        config.MAX_ITERATIONS = 10
        config.NUMBER_OF_RUNS = 2
        config.SEEDS = config.SEEDS[:2]
        
    with open(config.OUTPUT_DIR / "run_log.txt", "w") as f: f.write(f"Starting ORPD Simulation (Mode: {args.mode})")
        
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
        
        # Analisis Sensitivitas (Commented out because sensitivity.py is missing)
        # from sensitivity import calculate_lsf, print_top_sensitive_buses
        # sens = calculate_lsf(base_ppc)
        # if sens:
        #     print_top_sensitive_buses(sens, top_n=9) # karena default ada 9 variabel QC
    
    all_results = {}
    
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
            label_name = "Hybrid (Tanpa Early Stop)" if "NoES" in algo_name else f"Hybrid (Early Stop di iter {best_run['iter']})"
            plt.plot(best_run['cg'], label=label_name)
            
        df = pd.DataFrame(rows)
        df.to_csv(config.OUTPUT_DIR / "hasil_perbandingan.csv", index=False)
        df.to_excel(config.OUTPUT_DIR / "hasil_perbandingan.xlsx", index=False)
        
        plt.yscale('log'); plt.xlabel('Iteration'); plt.ylabel('Fitness'); plt.title('Perbandingan Konvergensi (Early Stop vs Tanpa Early Stop)')
        plt.legend(); plt.savefig(config.OUTPUT_DIR / "hybrid_early_stop_comparison.png"); plt.close()
        
        print("Simulation complete. All tables, graphs, and csvs saved to output/")

if __name__ == "__main__":
    main()