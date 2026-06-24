import time
import numpy as np

def run_woa(base_ppc, variables_metadata, constraints_metadata, eval_func, pop_size=100, patience=50, tol=1e-6, verbose=True):
    t_start = time.time()
    num_vars = len(variables_metadata)
    lb = np.array([m['lower_bound'] for m in variables_metadata])
    ub = np.array([m['upper_bound'] for m in variables_metadata])
    
    # Initialize population
    X = np.random.uniform(lb, ub, (pop_size, num_vars))
    fitness = np.zeros(pop_size)
    LARGE_FITNESS = 1e9
    
    if verbose:
        print(f"WOA Tunggal: Inisialisasi {pop_size} agen pencarian...")
        
    for i in range(pop_size):
        fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
        while fitness[i] >= LARGE_FITNESS:
            X[i] = np.random.uniform(lb, ub)
            fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
            
    best_idx = np.argmin(fitness)
    global_best_X = X[best_idx].copy()
    global_best_fit = fitness[best_idx]
    
    convergence_curve = []
    best_fit_history = global_best_fit
    no_improve_count = 0
    b_const = 1.0
    
    max_iter_safety = 5000
    it = 0
    
    while it < max_iter_safety:
        nominal_max_iter = 1000
        a = 2.0 - min(it, nominal_max_iter) * (2.0 / nominal_max_iter) 
        
        for i in range(pop_size):
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
                    X_new = X_rand - A * D_X_rand
                else:
                    # Exploitation
                    D_Leader = np.abs(C * global_best_X - X[i])
                    X_new = global_best_X - A * D_Leader
            else:
                # Exploitation spiral
                D_Leader = np.abs(global_best_X - X[i])
                X_new = D_Leader * np.exp(b_const * l) * np.cos(2 * np.pi * l) + global_best_X
                
            # Boundary
            mask_lower = X_new < lb
            if np.any(mask_lower): X_new[mask_lower] = lb[mask_lower] + np.random.rand(np.sum(mask_lower)) * (ub[mask_lower] - lb[mask_lower]) * 0.1
            mask_upper = X_new > ub
            if np.any(mask_upper): X_new[mask_upper] = ub[mask_upper] - np.random.rand(np.sum(mask_upper)) * (ub[mask_upper] - lb[mask_upper]) * 0.1
            X_new = np.clip(X_new, lb, ub)
            
            X[i] = X_new
            fitness[i] = eval_func(X[i], base_ppc, variables_metadata, constraints_metadata)
            
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < global_best_fit:
            global_best_X = X[best_idx].copy()
            global_best_fit = fitness[best_idx]
            
        convergence_curve.append(global_best_fit)
        
        if verbose and (it + 1) % 10 == 0:
            print(f"WOA: Iterasi {it+1} - Fitness Terbaik: {global_best_fit:.6f}")
            
        if global_best_fit < best_fit_history - tol:
            best_fit_history = global_best_fit
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            if verbose:
                print(f"WOA konvergen pada iterasi {it+1} setelah tidak ada perbaikan selama {patience} iterasi.")
            break
        it += 1
            
    run_time = time.time() - t_start
    return global_best_X, global_best_fit, np.array(convergence_curve), run_time
