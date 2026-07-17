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