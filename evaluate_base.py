import pandas as pd
import numpy as np
import os
from pypower.api import runpf, ppoption
from pypower.idx_bus import BUS_I, BUS_TYPE, VM, VA, BS, PD, QD, VMAX, VMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, PF, PT, QF, QT
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, VG, GEN_STATUS

dataset_dir = '/Users/supermaxxxzz/tugas-akhir/dataset'

bus_df = pd.read_csv(os.path.join(dataset_dir, 'bus.csv'))
branch_df = pd.read_csv(os.path.join(dataset_dir, 'branch.csv'))
gen_df = pd.read_csv(os.path.join(dataset_dir, 'generator.csv'))

bus = bus_df.values
branch = branch_df.values
gen = gen_df.values

ppc = {
    'version': '2',
    'baseMVA': 100.0,
    'bus': bus.copy(),
    'branch': branch.copy(),
    'gen': gen.copy(),
}

opt = ppoption(VERBOSE=0, OUT_ALL=0)
results, success = runpf(ppc, opt)

if success:
    p_losses = results['branch'][:, PF] + results['branch'][:, PT]
    p_loss = np.sum(p_losses)
    
    voltages = results['bus'][:, VM]
    bus_types = results['bus'][:, BUS_TYPE]
    pq_indices = np.where(bus_types == 1)[0]
    voltage_deviation = np.sum(np.abs(voltages[pq_indices] - 1.0))
    
    print(f"Base Case Power Loss: {p_loss:.4f} MW")
    print(f"Base Case Voltage Deviation: {voltage_deviation:.4f}")
else:
    print("Power flow failed for base case.")
