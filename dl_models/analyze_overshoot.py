import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('../Datasets/Metrics_S9_Causal_Hybrid.csv')
    
    # Calculate True Deltas and Predicted Deltas
    # dx, dy for each step
    true_dx = np.diff(df['True_X'])
    true_dy = np.diff(df['True_Y'])
    
    pred_dx = np.diff(df['Pred_X'])
    pred_dy = np.diff(df['Pred_Y'])
    
    # Calculate Heading Angles (in degrees)
    true_heading = np.degrees(np.arctan2(true_dy, true_dx))
    pred_heading = np.degrees(np.arctan2(pred_dy, pred_dx))
    
    # Unwrap angles to prevent 360 to 0 jumps
    true_heading = np.unwrap(np.radians(true_heading)) * (180/np.pi)
    pred_heading = np.unwrap(np.radians(pred_heading)) * (180/np.pi)
    
    # Find the index of the sharpest turn
    # The sharpest turn has the largest change in true heading
    heading_change = np.abs(np.diff(true_heading))
    turn_idx = np.argmax(heading_change)
    
    print(f"Sharpest turn detected around frame index {turn_idx}")
    
    # Plot Heading over time, zoomed in around the turn
    window = 20 # frames before and after
    start_idx = max(0, turn_idx - window)
    end_idx = min(len(true_heading), turn_idx + window)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(start_idx, end_idx), true_heading[start_idx:end_idx], label='True Motion Heading', color='blue', marker='o')
    plt.plot(range(start_idx, end_idx), pred_heading[start_idx:end_idx], label='Predicted DL Heading', color='green', marker='x', linestyle='dashed')
    
    plt.axvline(x=turn_idx, color='red', linestyle=':', label='Exact Turn Point')
    
    plt.title('Heading Angle Analysis (True vs Predicted)')
    plt.xlabel('Frame Index')
    plt.ylabel('Heading Angle (Degrees)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Datasets/heading_lag_analysis.png', bbox_inches='tight')
    print("Saved heading analysis plot.")

if __name__ == '__main__':
    main()
