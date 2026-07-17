import numpy as np
import pandas as pd
import os
from pypower.api import runpf, ppoption
from pypower.idx_bus import BUS_I, BUS_TYPE, VM, VA, BS, PD, QD, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, PF, PT, QF, QT
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, VG, GEN_STATUS

# Load dataset2
dataset_dir = '/Users/supermaxxxzz/tugas-akhir/dataset2'

bus_df = pd.read_csv(os.path.join(dataset_dir, 'bus.csv'))
branch_df = pd.read_csv(os.path.join(dataset_dir, 'branch.csv'))
gen_df = pd.read_csv(os.path.join(dataset_dir, 'generator.csv'))

bus = bus_df.values
branch = branch_df.values
gen = gen_df.values
gencost_arr = np.zeros((len(gen), 7))

base_ppc = {
    'version': '2', 'baseMVA': 100.0,
    'bus': bus.copy(), 'branch': branch.copy(),
    'gen': gen.copy(), 'gencost': gencost_arr.copy()
}

# Variable bounds
constraint_df = pd.read_csv(os.path.join(dataset_dir, 'constraint.csv'))
variables_metadata = []
constraints_metadata = {'Qg': {}, 'S': {}, 'Voltage': (0.95, 1.05)}

def find_branch_index(bus_a, bus_b):
    for idx, br in enumerate(branch):
        f_bus, t_bus = int(br[F_BUS]), int(br[T_BUS])
        if (f_bus == bus_a and t_bus == bus_b) or (f_bus == bus_b and t_bus == bus_a):
            return idx
    return 0

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

from pso2 import run_pso

LAMBDA_V = 100000.0
LAMBDA_Q = 100000.0
LAMBDA_S = 100000.0

def evaluate_fitness_single(x, base_ppc, variables_metadata, constraints_metadata=None, return_details=False):
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
    LARGE_FITNESS = 1e9
    if not success:
        return (LARGE_FITNESS, {'success': False}) if return_details else LARGE_FITNESS
    
    p_losses = results['branch'][:, PF] + results['branch'][:, PT]
    p_loss = np.sum(p_losses)
    
    # Penalties
    v_penalty = 0.0
    voltages = results['bus'][:, VM]
    bus_types = results['bus'][:, BUS_TYPE]
    pq_indices = np.where(bus_types == 1)[0]
    
    v_min, v_max = constraints_metadata['Voltage']
    for idx in pq_indices:
        v = voltages[idx]
        if v < v_min: v_penalty += (v_min - v) ** 2
        elif v > v_max: v_penalty += (v - v_max) ** 2
            
    voltage_deviation = np.sum(np.abs(voltages[pq_indices] - 1.0))
    
    # ONLY LOSS as objective + penalties
    fitness = p_loss + LAMBDA_V * v_penalty
    
    if return_details:
        return fitness, {'success': True, 'p_loss': p_loss, 'voltage_deviation': voltage_deviation}
    return fitness

print("Running PSO with SINGLE OBJECTIVE (Minimize Power Loss)...")
x_pso, fit_pso, _, _, _ = run_pso(base_ppc, variables_metadata, constraints_metadata, evaluate_fitness_single, pop_size=30, max_iter=50, early_stop=False, verbose=True)
_, det_pso = evaluate_fitness_single(x_pso, base_ppc, variables_metadata, constraints_metadata, return_details=True)
print(f"Optimal Loss: {det_pso['p_loss']:.4f} MW | VD: {det_pso['voltage_deviation']:.4f}")
