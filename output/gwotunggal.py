import time
import numpy as np

def run_gwo(base_ppc, variables_metadata, constraints_metadata, eval_func, pop_size=100, patience=50, tol=1e-6, verbose=True):
    t_start = time.time()
    num_vars = len(variables_metadata)
    lb = np.array([m['lower_bound'] for m in variables_metadata])
    ub = np.array([m['upper_bound'] for m in variables_metadata])
    
    X = np.random.uniform(lb, ub, (pop_size, num_vars))
    fitness = np.zeros(pop_size)
    LARGE_FITNESS = 1e9
    
    if verbose:
        print(f"GWO Tunggal: Inisialisasi {pop_size} agen pencarian...")
        
    for i in range(pop_size):
        fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
        while fitness[i] >= LARGE_FITNESS:
            X[i] = np.random.uniform(lb, ub)
            fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
            
    sorted_idx = np.argsort(fitness)
    X_alpha = X[sorted_idx[0]].copy()
    fit_alpha = fitness[sorted_idx[0]]
    X_beta = X[sorted_idx[1]].copy()
    fit_beta = fitness[sorted_idx[1]]
    X_delta = X[sorted_idx[2]].copy()
    fit_delta = fitness[sorted_idx[2]]
    
    convergence_curve = []
    best_fit_history = fit_alpha
    no_improve_count = 0
    
    max_iter_safety = 5000
    it = 0
    
    while it < max_iter_safety:
        nominal_max_iter = 1000
        a = 2.0 - min(it, nominal_max_iter) * (2.0 / nominal_max_iter)
        
        for i in range(pop_size):
            r1_a, r2_a = np.random.rand(num_vars), np.random.rand(num_vars)
            A1 = 2 * a * r1_a - a
            C1 = 2 * r2_a
            D_alpha = np.abs(C1 * X_alpha - X[i])
            X1 = X_alpha - A1 * D_alpha
            
            r1_b, r2_b = np.random.rand(num_vars), np.random.rand(num_vars)
            A2 = 2 * a * r1_b - a
            C2 = 2 * r2_b
            D_beta = np.abs(C2 * X_beta - X[i])
            X2 = X_beta - A2 * D_beta
            
            r1_d, r2_d = np.random.rand(num_vars), np.random.rand(num_vars)
            A3 = 2 * a * r1_d - a
            C3 = 2 * r2_d
            D_delta = np.abs(C3 * X_delta - X[i])
            X3 = X_delta - A3 * D_delta
            
            X_new = (X1 + X2 + X3) / 3.0
            
            mask_lower = X_new < lb
            if np.any(mask_lower): X_new[mask_lower] = lb[mask_lower] + np.random.rand(np.sum(mask_lower)) * (ub[mask_lower] - lb[mask_lower]) * 0.1
            mask_upper = X_new > ub
            if np.any(mask_upper): X_new[mask_upper] = ub[mask_upper] - np.random.rand(np.sum(mask_upper)) * (ub[mask_upper] - lb[mask_upper]) * 0.1
            X_new = np.clip(X_new, lb, ub)
            
            X[i] = X_new
            fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
            
        for i in range(pop_size):
            if fitness[i] < fit_alpha:
                fit_alpha = fitness[i]
                X_alpha = X[i].copy()
            elif fitness[i] < fit_beta:
                fit_beta = fitness[i]
                X_beta = X[i].copy()
            elif fitness[i] < fit_delta:
                fit_delta = fitness[i]
                X_delta = X[i].copy()
                
        convergence_curve.append(fit_alpha)
        
        if verbose and (it + 1) % 10 == 0:
            print(f"GWO: Iterasi {it+1} - Fitness Terbaik: {fit_alpha:.6f}")
            
        if fit_alpha < best_fit_history - tol:
            best_fit_history = fit_alpha
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            if verbose:
                print(f"GWO konvergen pada iterasi {it+1} setelah tidak ada perbaikan selama {patience} iterasi.")
            break
        it += 1
            
    run_time = time.time() - t_start
    return X_alpha, fit_alpha, np.array(convergence_curve), run_time
