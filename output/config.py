import os
from pathlib import Path

# Parameter Utama
POPULATION_SIZE = 100
MAX_ITERATIONS = 400
NUMBER_OF_RUNS = 1

# Batas Variabel
V_G_MIN, V_G_MAX = 0.95, 1.05
T_MIN, T_MAX = 0.90, 1.10
Q_C_MIN, Q_C_MAX = 0.0, 40.0

# Penalti
LAMBDA_V = 1e6
LAMBDA_Q = 1e6
LAMBDA_CONV = 1e8

# Parameter HWOA-GWO & Early Stopping
GAMMA_DEFAULT = 1.0
EARLY_STOPPING_TOLERANCE = 1e-6
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_TOLERANCE_MW = 0.0001

ROOT_DIR = Path(__file__).parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_DIR = ROOT_DIR / "output"

SEEDS = list(range(1000, 1000 + NUMBER_OF_RUNS))