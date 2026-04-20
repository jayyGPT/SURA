import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_fusion():
    metrics_path = os.path.join('Datasets', 'Metrics_A8_Fusion.csv')
    df = pd.read_csv(metrics_path)
    
    # 1. Map Plot
    plt.figure(figsize=(12, 8))
    plt.plot(df['True_X'], df['True_Y'], label='Ground Truth Tracing', color='green', linewidth=4, alpha=0.6)
    plt.plot(df['Pred_X'], df['Pred_Y'], label='EKF Kalman Tracking', color='red', linestyle='--', linewidth=2)
    plt.title("A8 Multi-Modal Tracking using Coarse-to-Fine EKF Fusion")
    plt.xlabel('Physical X (m)')
    plt.ylabel('Physical Y (m)')
    plt.legend()
    plt.grid(True)
    
    map_out = os.path.join('Datasets', 'Kalman_Map_A8.png')
    plt.savefig(map_out, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cumulative Distribution Function (CDF) of Error
    errors = np.sort(df['Error'])
    p = 1. * np.arange(len(errors)) / (len(errors) - 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(errors, p, color='blue', linewidth=3)
    plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'Mean Error ({np.mean(errors):.2f}m)')
    plt.title('CDF of Localization Error (A8 EKF)')
    plt.xlabel('Positioning Error (meters)')
    plt.ylabel('Cumulative Probability')
    plt.xlim(0, max(errors) + 5)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    cdf_out = os.path.join('Datasets', 'Kalman_CDF_A8.png')
    plt.savefig(cdf_out, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    plot_fusion()
