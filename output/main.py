import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypower.api import runpf, ppoption
from pypower.idx_bus import BUS_I, BUS_TYPE, VM, VA, BS, PD, QD, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, PF, PT, QF, QT
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, VG, GEN_STATUS

from woatunggal import run_woa
from gwotunggal import run_gwo

# Penalty coefficients
LAMBDA_V = 100000.0
LAMBDA_Q = 100000.0
LAMBDA_S = 100000.0
LARGE_FITNESS = 1e9

def load_system_data(dataset_dir):
    bus_df = pd.read_csv(os.path.join(dataset_dir, 'bus.csv'))
    branch_df = pd.read_csv(os.path.join(dataset_dir, 'branch.csv'))
    gen_df = pd.read_csv(os.path.join(dataset_dir, 'generator.csv'))
    
    gencost_path = os.path.join(dataset_dir, 'gencost.csv')
    if os.path.exists(gencost_path):
        gencost_df = pd.read_csv(gencost_path, header=None)
        gencost_arr = gencost_df.values
    else:
        gencost_arr = np.zeros((len(gen_df), 7))
        gencost_arr[:, 0] = 2
        gencost_arr[:, 3] = 3

    bus = bus_df.values
    branch = branch_df.values
    gen = gen_df.values
    
    ppc = {
        'version': '2',
        'baseMVA': 100.0,
        'bus': bus.copy(),
        'branch': branch.copy(),
        'gen': gen.copy(),
        'gencost': gencost_arr.copy()
    }
    
    constraint_df = pd.read_csv(os.path.join(dataset_dir, 'constraint.csv'))
    variables_metadata = []
    constraints_metadata = {'Qg': {}, 'S': {}, 'Voltage': (0.95, 1.05)}
    
    def find_branch_index(bus_a, bus_b):
        for idx, br in enumerate(branch):
            f_bus, t_bus = int(br[F_BUS]), int(br[T_BUS])
            if (f_bus == bus_a and t_bus == bus_b) or (f_bus == bus_b and t_bus == bus_a):
                return idx
        raise ValueError(f"No branch found connecting bus {bus_a} and bus {bus_b}")

    for idx, row in constraint_df.iterrows():
        if pd.isna(row['Category']): continue
        cat = str(row['Category']).strip()
        var_name = str(row['Variable']).strip()
        bus_branch = str(row['Bus/Branch']).strip()
        val_min = float(row['Min'])
        val_max = float(row['Max'])
        
        if cat == 'CONTROL_VARIABLE':
            if var_name.startswith('Vg'):
                bus_id = int(bus_branch.replace('Bus', ''))
                variables_metadata.append({'name': var_name, 'type': 'generator_voltage', 'target_id': bus_id, 'lower_bound': val_min, 'upper_bound': val_max})
            elif var_name.startswith('Tap'):
                parts = bus_branch.split('-')
                br_idx = find_branch_index(int(parts[0]), int(parts[1]))
                variables_metadata.append({'name': var_name, 'type': 'transformer_tap', 'target_id': br_idx, 'lower_bound': val_min, 'upper_bound': val_max})
            elif var_name.startswith('Qc'):
                bus_id = int(bus_branch.replace('Bus', ''))
                variables_metadata.append({'name': var_name, 'type': 'shunt_capacitor', 'target_id': bus_id, 'lower_bound': val_min, 'upper_bound': val_max})
        elif cat == 'CONSTRAINT':
            if var_name == 'VoltageLimit':
                constraints_metadata['Voltage'] = (val_min, val_max)
            elif var_name.startswith('Qg'):
                bus_id = int(bus_branch.replace('Bus', ''))
                constraints_metadata['Qg'][bus_id] = (val_min, val_max)
            elif var_name.startswith('S'):
                parts = bus_branch.replace('Line', '').split('-')
                br_idx = find_branch_index(int(parts[0]), int(parts[1]))
                constraints_metadata['S'][br_idx] = val_max
                
    return ppc, variables_metadata, constraints_metadata

def evaluate_fitness(x, base_ppc, variables_metadata, constraints_metadata=None, return_details=False):
    if constraints_metadata is None:
        constraints_metadata = {'Qg': {}, 'S': {}, 'Voltage': (0.95, 1.05)}
        
    new_ppc = {
        'version': base_ppc['version'], 'baseMVA': base_ppc['baseMVA'],
        'bus': base_ppc['bus'].copy(), 'branch': base_ppc['branch'].copy(),
        'gen': base_ppc['gen'].copy(), 'gencost': base_ppc['gencost'].copy()
    }
    
    bus_map = {int(new_ppc['bus'][i, BUS_I]): i for i in range(new_ppc['bus'].shape[0])}
    gen_map = {int(new_ppc['gen'][i, GEN_BUS]): i for i in range(new_ppc['gen'].shape[0])}
    
    for val, meta in zip(x, variables_metadata):
        var_type = meta['type']
        target_id = int(meta['target_id'])
        val_clipped = np.clip(val, meta['lower_bound'], meta['upper_bound'])
        
        if var_type == 'generator_voltage':
            if target_id in gen_map: new_ppc['gen'][gen_map[target_id], VG] = val_clipped
            if target_id in bus_map: new_ppc['bus'][bus_map[target_id], VM] = val_clipped
        elif var_type == 'transformer_tap':
            new_ppc['branch'][target_id, TAP] = val_clipped
        elif var_type == 'shunt_capacitor':
            if target_id in bus_map: new_ppc['bus'][bus_map[target_id], BS] = val_clipped
                
    opt = ppoption(VERBOSE=0, OUT_ALL=0)
    results, success = runpf(new_ppc, opt)
    
    if not success:
        if return_details:
            return LARGE_FITNESS, {'success': False, 'p_loss': LARGE_FITNESS, 'voltage_deviation': LARGE_FITNESS}
        return LARGE_FITNESS
    
    p_losses = results['branch'][:, PF] + results['branch'][:, PT]
    p_loss = np.sum(p_losses)
    
    v_min, v_max = constraints_metadata['Voltage']
    v_penalty = 0.0
    voltages = results['bus'][:, VM]
    bus_types = results['bus'][:, BUS_TYPE]
    pq_indices = np.where(bus_types == 1)[0]
    
    for idx in pq_indices:
        v = voltages[idx]
        if v < v_min: v_penalty += (v_min - v) ** 2
        elif v > v_max: v_penalty += (v - v_max) ** 2
            
    voltage_deviation = np.sum(np.abs(voltages[pq_indices] - 1.0))
    
    q_penalty = 0.0
    qg = results['gen'][:, QG]
    base_mva = results['baseMVA']
    for g_idx in range(len(qg)):
        bus_id = int(results['gen'][g_idx, GEN_BUS])
        qmin, qmax = constraints_metadata['Qg'].get(bus_id, (results['gen'][g_idx, QMIN], results['gen'][g_idx, QMAX]))
        q_gen_pu = qg[g_idx] / base_mva
        q_min_pu = qmin / base_mva
        q_max_pu = qmax / base_mva
        if q_gen_pu < q_min_pu: q_penalty += (q_min_pu - q_gen_pu) ** 2
        elif q_gen_pu > q_max_pu: q_penalty += (q_gen_pu - q_max_pu) ** 2
            
    s_penalty = 0.0
    pf, pt, qf, qt = results['branch'][:, PF], results['branch'][:, PT], results['branch'][:, QF], results['branch'][:, QT]
    s_from, s_to = np.sqrt(pf**2 + qf**2), np.sqrt(pt**2 + qt**2)
    s_max_branch = np.maximum(s_from, s_to)
    for br_idx, s_limit in constraints_metadata['S'].items():
        if s_max_branch[br_idx] > s_limit:
            s_penalty += ((s_max_branch[br_idx] - s_limit) / base_mva) ** 2

    fitness = p_loss + voltage_deviation + LAMBDA_V * v_penalty + LAMBDA_Q * q_penalty + LAMBDA_S * s_penalty
    
    if return_details:
        return fitness, {'success': True, 'p_loss': p_loss, 'voltage_deviation': voltage_deviation}
    return fitness

def run_hybrid_woa_gwo(base_ppc, variables_metadata, constraints_metadata, eval_func, pop_size=200, transition_iter=250, patience=50, tol=1e-6, verbose=True):
    t_start = time.time()
    num_vars = len(variables_metadata)
    lb = np.array([m['lower_bound'] for m in variables_metadata])
    ub = np.array([m['upper_bound'] for m in variables_metadata])
    
    X = np.random.uniform(lb, ub, (pop_size, num_vars))
    fitness = np.zeros(pop_size)
    
    if verbose:
        print(f"Hybrid Estafet: Inisialisasi {pop_size} agen. WOA s/d iter {transition_iter}, lalu GWO...")
        
    for i in range(pop_size):
        fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
        while fitness[i] >= LARGE_FITNESS:
            X[i] = np.random.uniform(lb, ub)
            fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
            
    sorted_idx = np.argsort(fitness)
    X_alpha, fit_alpha = X[sorted_idx[0]].copy(), fitness[sorted_idx[0]]
    X_beta, fit_beta = X[sorted_idx[1]].copy(), fitness[sorted_idx[1]]
    X_delta, fit_delta = X[sorted_idx[2]].copy(), fitness[sorted_idx[2]]
    
    convergence_curve = []
    best_fit_history = fit_alpha
    no_improve_count = 0
    max_iter_safety = 5000
    it = 0
    
    while it < max_iter_safety:
        nominal_max_iter = transition_iter * 2 # Misal total iterasi efektif 500
        a = 2.0 - min(it, nominal_max_iter) * (2.0 / nominal_max_iter)
        
        is_woa_phase = it < transition_iter
        
        for i in range(pop_size):
            if is_woa_phase:
                # WOA Update Mechanism (Eksplorasi)
                r1, r2 = np.random.rand(num_vars), np.random.rand(num_vars)
                A = 2 * a * r1 - a
                C = 2 * r2
                p = np.random.rand()
                l = np.random.uniform(-1, 1)
                
                if p < 0.5:
                    if np.any(np.abs(A) >= 1):
                        # Exploration
                        rand_idx = np.random.randint(0, pop_size)
                        X_rand = X[rand_idx].copy()
                        D_X_rand = np.abs(C * X_rand - X[i])
                        X_new = np.where(np.abs(A) >= 1, X_rand - A * D_X_rand, X_alpha - A * np.abs(C * X_alpha - X[i]))
                    else:
                        # Exploitation (Encircling)
                        D_Leader = np.abs(C * X_alpha - X[i])
                        X_new = X_alpha - A * D_Leader
                else:
                    # Exploitation (Spiral)
                    b_const = 1.0
                    D_Leader = np.abs(X_alpha - X[i])
                    X_new = D_Leader * np.exp(b_const * l) * np.cos(2 * np.pi * l) + X_alpha
            else:
                # GWO Update Mechanism (Eksploitasi)
                r1_a, r2_a = np.random.rand(num_vars), np.random.rand(num_vars)
                A1 = 2 * a * r1_a - a; C1 = 2 * r2_a
                D_alpha = np.abs(C1 * X_alpha - X[i])
                X1 = X_alpha - A1 * D_alpha
                
                r1_b, r2_b = np.random.rand(num_vars), np.random.rand(num_vars)
                A2 = 2 * a * r1_b - a; C2 = 2 * r2_b
                D_beta = np.abs(C2 * X_beta - X[i])
                X2 = X_beta - A2 * D_beta
                
                r1_d, r2_d = np.random.rand(num_vars), np.random.rand(num_vars)
                A3 = 2 * a * r1_d - a; C3 = 2 * r2_d
                D_delta = np.abs(C3 * X_delta - X[i])
                X3 = X_delta - A3 * D_delta
                
                X_new = (X1 + X2 + X3) / 3.0
                
            # Boundary checks
            mask_lower = X_new < lb
            if np.any(mask_lower): X_new[mask_lower] = lb[mask_lower] + np.random.rand(np.sum(mask_lower)) * (ub[mask_lower] - lb[mask_lower]) * 0.1
            mask_upper = X_new > ub
            if np.any(mask_upper): X_new[mask_upper] = ub[mask_upper] - np.random.rand(np.sum(mask_upper)) * (ub[mask_upper] - lb[mask_upper]) * 0.1
            X_new = np.clip(X_new, lb, ub)
            
            X[i] = X_new
            fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
            
        # Update Alpha, Beta, Delta
        for i in range(pop_size):
            if fitness[i] < fit_alpha:
                fit_alpha, X_alpha = fitness[i], X[i].copy()
            elif fitness[i] < fit_beta:
                fit_beta, X_beta = fitness[i], X[i].copy()
            elif fitness[i] < fit_delta:
                fit_delta, X_delta = fitness[i], X[i].copy()
                
        convergence_curve.append(fit_alpha)
        if verbose and (it + 1) % 10 == 0: 
            fase = "WOA" if is_woa_phase else "GWO"
            print(f"Hybrid ({fase}): Iterasi {it+1} - Fitness Terbaik: {fit_alpha:.6f}")
            
        if it == transition_iter - 1:
            no_improve_count = 0 # Reset kesabaran saat transisi fase
            if verbose: print("--- Memulai Fase Eksploitasi (GWO) ---")
            
        if fit_alpha < best_fit_history - tol:
            best_fit_history, no_improve_count = fit_alpha, 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience and not is_woa_phase:
            if verbose: print(f"Hybrid konvergen pada iterasi {it+1} di fase GWO")
            break
        it += 1
            
    return X_alpha, fit_alpha, np.array(convergence_curve), time.time() - t_start

def main():
    np.random.seed(42)
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
    
    print("=" * 60)
    print("  Optimal Reactive Power Dispatch (ORPD) Optimization")
    print("=" * 60)
    
    base_ppc, variables_metadata, constraints_metadata = load_system_data(dataset_dir)
    print(f"Dataset dimuat. Jumlah variabel kontrol: {len(variables_metadata)}")
    
    pop_size = 100
    patience = 50
    
    print("\n--- Running Whale Optimization Algorithm (WOA) ---")
    x_woa, fit_woa, curve_woa, time_woa = run_woa(base_ppc, variables_metadata, constraints_metadata, evaluate_fitness, pop_size=pop_size, patience=patience)
    _, det_woa = evaluate_fitness(x_woa, base_ppc, variables_metadata, constraints_metadata, return_details=True)
    
    print("\n--- Running Grey Wolf Optimizer (GWO) ---")
    x_gwo, fit_gwo, curve_gwo, time_gwo = run_gwo(base_ppc, variables_metadata, constraints_metadata, evaluate_fitness, pop_size=pop_size, patience=patience)
    _, det_gwo = evaluate_fitness(x_gwo, base_ppc, variables_metadata, constraints_metadata, return_details=True)
    
    print("\n--- Running Hybrid WOA-GWO (Estafet) ---")
    trans_iter = 250
    x_hyb, fit_hyb, curve_hyb, time_hyb = run_hybrid_woa_gwo(base_ppc, variables_metadata, constraints_metadata, evaluate_fitness, pop_size=200, transition_iter=trans_iter, patience=patience)
    _, det_hyb = evaluate_fitness(x_hyb, base_ppc, variables_metadata, constraints_metadata, return_details=True)
    
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(f"WOA    | Fit: {fit_woa:.5f} | Ploss: {det_woa['p_loss']:.5f} MW | VD: {det_woa['voltage_deviation']:.5f} p.u. | Waktu: {time_woa:.2f} s")
    print(f"GWO    | Fit: {fit_gwo:.5f} | Ploss: {det_gwo['p_loss']:.5f} MW | VD: {det_gwo['voltage_deviation']:.5f} p.u. | Waktu: {time_gwo:.2f} s")
    print(f"HYBRID | Fit: {fit_hyb:.5f} | Ploss: {det_hyb['p_loss']:.5f} MW | VD: {det_hyb['voltage_deviation']:.5f} p.u. | Waktu: {time_hyb:.2f} s")
    print("=" * 60)
    
    # ---------------------------
    # Plotting Graphs
    # ---------------------------
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Bar Chart Comparison
    labels = ['WOA Tunggal', 'GWO Tunggal', 'Hybrid WOA-GWO']
    ploss = [det_woa['p_loss'], det_gwo['p_loss'], det_hyb['p_loss']]
    vd = [det_woa['voltage_deviation'], det_gwo['voltage_deviation'], det_hyb['voltage_deviation']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(x - width/2, ploss, width, label='Power Loss (MW)', color='skyblue')
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, vd, width, label='Voltage Deviation (p.u.)', color='salmon')
    
    ax1.set_ylabel('Power Loss (MW)', fontsize=12)
    ax2.set_ylabel('Voltage Deviation (p.u.)', fontsize=12)
    ax1.set_title('Perbandingan Hasil Fungsi Objektif (Ploss & VD)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    
    # Add values on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', ha='center', va='bottom')
        
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    chart_path = os.path.join(output_dir, 'objective_comparison_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Grafik perbandingan tersimpan di {chart_path}")
    
    # 2. Convergence Curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(curve_woa) + 1), curve_woa, label='WOA Tunggal', color='blue')
    plt.plot(range(1, len(curve_gwo) + 1), curve_gwo, label='GWO Tunggal', color='red')
    plt.plot(range(1, len(curve_hyb) + 1), curve_hyb, label='Hybrid WOA-GWO', color='green')
    plt.title('Kurva Konvergensi Algoritma', fontsize=14)
    plt.xlabel('Iterasi', fontsize=12)
    plt.ylabel('Fitness Terbaik', fontsize=12)
    
    # Custom x-ticks interval 5
    max_iter = max(len(curve_woa), len(curve_gwo), len(curve_hyb))
    plt.xticks(np.arange(1, max_iter + 5, 5), rotation=90, fontsize=8)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    conv_path = os.path.join(output_dir, 'convergence_curves.png')
    plt.savefig(conv_path, dpi=300, bbox_inches='tight')
    print(f"Kurva konvergensi tersimpan di {conv_path}")

    # 3. Waktu Eksekusi Bar Chart
    times = [time_woa, time_gwo, time_hyb]
    
    plt.figure(figsize=(8, 5))
    bars3 = plt.bar(labels, times, color=['blue', 'red', 'green'], alpha=0.7)
    plt.title('Perbandingan Waktu Eksekusi', fontsize=14)
    plt.ylabel('Waktu (detik)', fontsize=12)
    
    for bar in bars3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f} s', ha='center', va='bottom')
        
    time_chart_path = os.path.join(output_dir, 'execution_time_chart.png')
    plt.savefig(time_chart_path, dpi=300, bbox_inches='tight')
    print(f"Grafik waktu eksekusi tersimpan di {time_chart_path}")

    # 4. Grafik Estafet / Fase Hybrid (Eksplorasi & Eksploitasi)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(curve_hyb) + 1), curve_hyb, color='black', linewidth=2, label='Fitness Hybrid Estafet')
    
    max_x = len(curve_hyb)
    
    # Shade for WOA
    if max_x > 0:
        end_woa = min(max_x, trans_iter)
        plt.axvspan(1, end_woa, color='lightblue', alpha=0.5, label=f'Fase Eksplorasi: WOA\n(Iterasi 1-{end_woa})')
    
    # Shade for GWO
    if max_x > trans_iter:
        plt.axvspan(trans_iter, max_x, color='lightgreen', alpha=0.5, label=f'Fase Eksploitasi: GWO\n(Iterasi {trans_iter}-{max_x})')
        plt.axvline(x=trans_iter, color='red', linestyle='--', linewidth=2, label='Titik Transisi (Estafet)')
        
    plt.title('Kurva Konvergensi Algoritma Hybrid (Eksplorasi → Eksploitasi)', fontsize=14)
    plt.xlabel('Iterasi', fontsize=12)
    plt.ylabel('Fitness Terbaik', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    phase_path = os.path.join(output_dir, 'hybrid_phases.png')
    plt.savefig(phase_path, dpi=300, bbox_inches='tight')
    print(f"Grafik fase hybrid tersimpan di {phase_path}")

if __name__ == '__main__':
    main()
