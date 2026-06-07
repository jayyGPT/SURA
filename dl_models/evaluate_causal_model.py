import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("Evaluating Strict Causal Hybrid DL model...")
    
    csv_path = '../Datasets/Metrics_S9_Causal_Hybrid.csv'
    if not os.path.exists(csv_path):
        print("Predictions file not found!")
        return
        
    df_dl = pd.read_csv(csv_path)
    
    # Calculate Metrics
    mae = df_dl['Error'].mean()
    rmse = np.sqrt(np.mean(df_dl['Error']**2))
    max_err = df_dl['Error'].max()
    
    print(f"Causal Hybrid PDR-DL Model (S9+ Test)")
    print(f"MAE:  {mae:.2f} meters")
    print(f"RMSE: {rmse:.2f} meters")
    print(f"Max:  {max_err:.2f} meters")
    
    # Trajectory Plot
    plt.figure(figsize=(10, 8))
    plt.plot(df_dl['True_X'], df_dl['True_Y'], label='Ground Truth Trajectory', color='blue', linewidth=2, marker='o', markersize=3, alpha=0.6)
    plt.plot(df_dl['Pred_X'], df_dl['Pred_Y'], label='Causal Hybrid Prediction', color='green', linewidth=2, linestyle='dashed', alpha=0.8)
    
    # Highlight Start
    plt.scatter([df_dl['True_X'].iloc[0]], [df_dl['True_Y'].iloc[0]], color='gold', s=150, label='Known Start Position (P0)', edgecolors='black', zorder=5)
    
    plt.title('2D Spatial Trajectory: Strict Causal Hybrid vs Ground Truth (S9+)', fontsize=14)
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    traj_out = '../Datasets/trajectory_causal_s9.png'
    plt.savefig(traj_out, dpi=300, bbox_inches='tight')
    print(f"Saved trajectory plot to {traj_out}")
    
    # CDF Plot
    plt.figure(figsize=(8, 6))
    sorted_err_dl = np.sort(df_dl['Error'])
    p_dl = 1. * np.arange(len(df_dl)) / (len(df_dl) - 1)
    plt.plot(sorted_err_dl, p_dl, label=f'Causal Hybrid (MAE: {mae:.2f}m)', color='green', linewidth=2.5)
    
    plt.title('CDF of Positioning Error (Strict Causal constraints)', fontsize=14)
    plt.xlabel('Distance Error (meters)')
    plt.ylabel('Probability')
    plt.xlim(0, 15) 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    cdf_out = '../Datasets/cdf_causal_s9.png'
    plt.savefig(cdf_out, dpi=300, bbox_inches='tight')
    print(f"Saved CDF plot to {cdf_out}")

if __name__ == '__main__':
    main()
