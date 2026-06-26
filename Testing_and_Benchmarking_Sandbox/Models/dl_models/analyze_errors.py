import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df_dl = pd.read_csv('../Datasets/Metrics_S9_DeepLearning.csv')
    
    # 1. Plot Error over Time
    plt.figure(figsize=(10, 5))
    plt.plot(df_dl.index, df_dl['Error'], label='Prediction Error (m)', color='red')
    plt.axhline(y=df_dl['Error'].mean(), color='b', linestyle='--', label=f"Mean Error ({df_dl['Error'].mean():.2f}m)")
    plt.title('Error Magnitude over Sequential Timesteps (S9+)')
    plt.xlabel('Timestep Index')
    plt.ylabel('Error (meters)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Datasets/error_over_time.png', bbox_inches='tight')
    
    # 2. Analyze the worst 5% of errors
    threshold = df_dl['Error'].quantile(0.95)
    worst_errors = df_dl[df_dl['Error'] >= threshold]
    
    print("--- ANALYSIS OF WORST ERRORS ---")
    print(f"95th Percentile Error Threshold: {threshold:.2f} meters")
    print(f"Number of frames above threshold: {len(worst_errors)}")
    print("\nIndices of worst errors:")
    print(worst_errors.index.tolist())
    
    # 3. Check the "end of the motion"
    print("\n--- ANALYSIS OF THE FINAL 10 TIMESTEPS ---")
    end_df = df_dl.tail(10)
    print(end_df)
    
    # Let's also look at the physical coordinates to see what's happening geometrically
    plt.figure(figsize=(10, 8))
    plt.plot(df_dl['True_X'], df_dl['True_Y'], label='Ground Truth Trajectory', color='blue', alpha=0.6)
    
    # Highlight the end of the trajectory
    plt.scatter(end_df['True_X'], end_df['True_Y'], color='cyan', s=100, label='Ground Truth (End)', edgecolors='black', zorder=5)
    plt.plot(end_df['Pred_X'], end_df['Pred_Y'], label='Prediction (End Jumps)', color='red', marker='x', markersize=8, linewidth=2, linestyle='dotted')
    
    # Draw connecting lines for the bad jumps
    for idx, row in end_df.iterrows():
        plt.plot([row['True_X'], row['Pred_X']], [row['True_Y'], row['Pred_Y']], color='gray', linestyle=':', alpha=0.5)
        
    plt.title('Focus on End of Trajectory Jumps')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Datasets/end_trajectory_focus.png', bbox_inches='tight')

if __name__ == '__main__':
    main()
