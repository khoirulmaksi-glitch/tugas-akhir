import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

import config
from power_flow import load_system
from control_variables import ControlVariables
import hybrid_woa_gwo

def run_iteration_sensitivity():
    print("\n=== Memulai Analisis Sensitivitas Jumlah Iterasi ===")
    
    base_ppc = load_system()
    ctrl = ControlVariables(base_ppc)
    
    # Kita kunci Population Size di angka 40 
    # (Kenapa 40? Karena dari riset sebelumnya, populasi 40 memberikan keseimbangan 
    # terbaik antara hasil optimal dan waktu eksekusi yang tidak terlalu lama).
    config.POPULATION_SIZE = 40
    config.NUMBER_OF_RUNS = 2 # Diulang 2 kali biar dirata-rata
    
    # Skenario Max Iteration yang akan diuji kelipatan 5 (dari 5 sampai 50)
    iter_sizes = list(range(5, 51, 5))
    
    results = []
    
    print(f"Menguji Sensitivitas Max Iterations: {iter_sizes}")
    print(f"Population Size fixed at: {config.POPULATION_SIZE}")
    print("-" * 50)
    
    for max_iter in iter_sizes:
        print(f"Running for Max Iterations = {max_iter}...")
        config.MAX_ITERATIONS = max_iter
        
        start_time = time.time()
        # Jalankan algoritma
        res = hybrid_woa_gwo.run(base_ppc, ctrl, algo_name="HWOA_GWO", run_idx=1, early_stopping=False, seed=42)
        exec_time = time.time() - start_time
        
        results.append({
            'Max_Iterations': max_iter,
            'Best_Fitness': res['fit'],
            'Best_Ploss': res['ploss'],
            'Time_Seconds': exec_time
        })
        print(f"  -> Best Fitness: {res['fit']:.6f} | Ploss: {res['ploss']:.6f} MW | Time: {exec_time:.2f} s")
        
    df = pd.DataFrame(results)
    df.to_csv(config.OUTPUT_DIR / "sensitivity_iteration.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('Best Fitness (Fungsi Objektif)', color=color)
    ax1.plot(df['Max_Iterations'], df['Best_Fitness'], marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Execution Time (seconds)', color=color)  
    ax2.plot(df['Max_Iterations'], df['Time_Seconds'], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Sensitivitas Iterasi (5 - 50) terhadap Fitness & Waktu')
    fig.tight_layout()  
    
    plot_path = config.OUTPUT_DIR / "sensitivity_iteration_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nData disimpan di: output/sensitivity_iteration.csv")
    print(f"Grafik visualisasi disimpan di: {plot_path}")
    print("=== Analisis Selesai ===")

def run_population_sensitivity():
    print("\n=== Memulai Analisis Sensitivitas Ukuran Populasi ===")
    
    base_ppc = load_system()
    ctrl = ControlVariables(base_ppc)
    
    # Kita kunci Max Iterations di angka 100 
    # (Kenapa 100? Karena algoritma umumnya sudah mulai konvergen di titik ini)
    config.MAX_ITERATIONS = 100
    config.NUMBER_OF_RUNS = 2 # Diulang 2 kali biar dirata-rata
    
    # Skenario Population Size yang akan diuji kelipatan 5 (dari 5 sampai 50)
    pop_sizes = list(range(5, 51, 5))
    
    results = []
    
    print(f"Menguji Sensitivitas Population Size: {pop_sizes}")
    print(f"Max Iterations fixed at: {config.MAX_ITERATIONS}")
    print("-" * 50)
    
    for pop in pop_sizes:
        print(f"Running for Population Size = {pop}...")
        config.POPULATION_SIZE = pop
        
        start_time = time.time()
        # Jalankan algoritma
        res = hybrid_woa_gwo.run(base_ppc, ctrl, algo_name="HWOA_GWO", run_idx=1, early_stopping=False, seed=42)
        exec_time = time.time() - start_time
        
        results.append({
            'Population_Size': pop,
            'Best_Fitness': res['fit'],
            'Best_Ploss': res['ploss'],
            'Time_Seconds': exec_time
        })
        print(f"  -> Best Fitness: {res['fit']:.6f} | Ploss: {res['ploss']:.6f} MW | Time: {exec_time:.2f} s")
        
    df = pd.DataFrame(results)
    df.to_csv(config.OUTPUT_DIR / "sensitivity_population.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:green'
    ax1.set_xlabel('Population Size')
    ax1.set_ylabel('Best Fitness (Fungsi Objektif)', color=color)
    ax1.plot(df['Population_Size'], df['Best_Fitness'], marker='^', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Execution Time (seconds)', color=color)  
    ax2.plot(df['Population_Size'], df['Time_Seconds'], marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Sensitivitas Populasi (10 - 60) terhadap Fitness & Waktu')
    fig.tight_layout()  
    
    plot_path = config.OUTPUT_DIR / "sensitivity_population_plot.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"\nData disimpan di: output/sensitivity_population.csv")
    print(f"Grafik visualisasi disimpan di: {plot_path}")
    print("=== Analisis Selesai ===")

if __name__ == "__main__":
    # run_iteration_sensitivity()
    run_population_sensitivity()
