import numpy as np
import time
from objective_function import calculate_fitness
import config

def run(base_ppc, ctrl, algo_name="", run_idx=1, early_stopping=False, seed=0):
    np.random.seed(seed)
    dim, pop_size, max_iter = ctrl.dim, config.POPULATION_SIZE, config.MAX_ITERATIONS
    lb, ub = ctrl.lb, ctrl.ub
    
    Positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    Alpha_pos, Beta_pos, Delta_pos = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    Alpha_score, Beta_score, Delta_score = np.inf, np.inf, np.inf
    
    cg_curve = np.zeros(max_iter)
    Ploss_best = np.inf; VD_best = np.inf; is_feasible = False
    patience_cnt = 0; stop_iter = max_iter; stop_reason = 'maximum_iterations'; prev_ploss = np.inf
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
        if (t + 1) % 10 == 0 or t == max_iter - 1:
            print(f"{algo_name} | Run {run_idx:02d}/{config.NUMBER_OF_RUNS} | Iteration {t+1}/{max_iter} | Ploss: {Ploss_best:.6f} | VD: {VD_best:.6f} | Best Fitness: {Alpha_score:.6f}")
        if early_stopping:
            if is_feasible:
                if (prev_ploss - Ploss_best) < config.EARLY_STOPPING_TOLERANCE_MW:
                    patience_cnt += 1
                    if patience_cnt >= config.EARLY_STOPPING_PATIENCE:
                        stop_iter = t + 1; stop_reason = 'early_stopping_converged'
                        print(f"\n\n{algo_name} | Run {run_idx:02d}/{config.NUMBER_OF_RUNS} | Early stopping at iteration {stop_iter}/{max_iter} | Best Ploss: {Ploss_best:.6f} MW")
                        break
                else:
                    patience_cnt = 0
                    prev_ploss = Ploss_best
            else:
                patience_cnt = 0
    return {"fit": Alpha_score, "cg": cg_curve.tolist(), "pos": Alpha_pos.tolist(), "ploss": Ploss_best, "vd": VD_best, "feas": bool(is_feasible), "iter": stop_iter, "time": time.time()-t0, "reason": stop_reason}