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