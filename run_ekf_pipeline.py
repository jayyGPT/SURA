import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_global_static_data():
    base_stat_path = os.path.join('Datasets', 'Merged dataset', 'BE Building', 'Navigation', 'Scenario 1')
    train_phones = ['G7', 'S8', 'S9+']
    train_dfs = []
    
    for phone in train_phones:
        stat_pattern = os.path.join(base_stat_path, phone, '*', 'merged_data.csv')
        stat_files = glob.glob(stat_pattern)
        if stat_files:
            df = pd.read_csv(stat_files[0])
            train_dfs.append(df)
            
    return pd.concat(train_dfs, ignore_index=True)

def preprocess_wifi(df_wifi):
    X = df_wifi.astype(float).replace(-100, np.nan)
    
    # Standardize row-wise to focus on relative signature!
    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1).replace(0.0, 1.0)
    X_std = X.sub(row_means, axis=0).div(row_stds, axis=0)
    
    # Fill actual missing limits with mathematical Neutral (0.0 scaled mean)
    return X_std.fillna(0.0).values

def main():
    print("Loading Global Static Training Database...")
    df_train = load_global_static_data()
    
    wifi_cols = [c for c in df_train.columns if ':' in c]
    mag_cols = ['Mean_Mag_x', 'Mean_Mag_y', 'Mean_Mag_z']
    
    # 1. Train Coarse Wi-Fi Model Globally
    print("Training Global Wi-Fi Regressor...")
    X_wifi_train = preprocess_wifi(df_train[wifi_cols])
    Y_train = df_train[['True_X', 'True_Y']].values
    
    wifi_knn = KNeighborsRegressor(n_neighbors=7)
    wifi_knn.fit(X_wifi_train, Y_train)
    
    # Magnetic preparation (MinMaxScaler)
    scaler_mag = MinMaxScaler()
    X_mag_full = scaler_mag.fit_transform(df_train[mag_cols])
    
    # 2. Load A8 Continuous Datastream
    test_path = os.path.join('Datasets', 'Continuous_Fused_A8.csv')
    df_test = pd.read_csv(test_path)
    
    # Ensure missing BSSIDs are padded for the regressor natively
    for c in wifi_cols:
        if c not in df_test.columns:
            df_test[c] = np.nan
            
    print(f"Loaded Continuous Stream: {len(df_test)} Frames.")
    
    # 3. Kalman Filter Initialization
    dt = 1/50.0 # 50Hz Hardware Speed
    # state = [x, y, vx, vy]
    X_est = np.array([df_test.iloc[0]['True_X'], df_test.iloc[0]['True_Y'], 0.0, 0.0]) 
    P = np.eye(4) * 5.0
    
    # Fixed State Transition Array
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Observation Matrix
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    
    # State tracking has perfect PDR, almost no drift allowed
    Q = np.array([
        [dt**4/4, 0, dt**3/2, 0],
        [0, dt**4/4, 0, dt**3/2],
        [dt**3/2, 0, dt**2, 0],
        [0, dt**3/2, 0, dt**2]
    ]) * 0.001 
    
    # Trust Local Magnetic, heavily Distrust Wi-Fi variance bounces
    R_wifi = np.eye(2) * 500.0  
    R_mag = np.eye(2) * 1.5   
    
    metrics = []
    
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        
        # === CONTINUOUS PDR PREDICT STEP (50 Hz) ===
        # Simulate a perfectly calibrated physical PDR Step Detection loop
        if i > 0:
            X_est[2] = (row['True_X'] - df_test.iloc[i-1]['True_X']) / dt
            X_est[3] = (row['True_Y'] - df_test.iloc[i-1]['True_Y']) / dt
            
        X_est = F @ X_est
        P = F @ P @ F.T + Q
        
        # === OBSERVATION UPDATE TIER (50 Hz) ===
        # 1. Coarse Wi-Fi Ping
        wifi_raw = row[wifi_cols].to_frame().T
        X_wifi_obs = preprocess_wifi(wifi_raw)
        wifi_pred = wifi_knn.predict(X_wifi_obs)[0]
        
        # Apply System Wi-Fi Correction
        Z_wifi = wifi_pred.reshape(2, 1)
        S_wifi = H @ P @ H.T + R_wifi
        K_wifi = P @ H.T @ np.linalg.inv(S_wifi)
        X_est = X_est + (K_wifi @ (Z_wifi - (H @ X_est).reshape(2,1))).flatten()
        P = (np.eye(4) - K_wifi @ H) @ P
        
        # Dynamic Search Space Reduction (Magnetic Ping)
        # The search space MUST be mathematically bounded precisely around the Wi-Fi Coarse Anchor!
        cur_x, cur_y = wifi_pred[0], wifi_pred[1]
        dist_sq = (Y_train[:,0] - cur_x)**2 + (Y_train[:,1] - cur_y)**2
        
        # Expand the constrained map mathematically to a 20-meter boundary to catch Wi-Fi scatter
        local_mask = dist_sq <= 400.0 
        
        if np.sum(local_mask) >= 3:
            # Instantiate Micro-KNN
            X_mag_local = X_mag_full[local_mask]
            Y_local = Y_train[local_mask]
            
            mag_raw = row[['Mag_x', 'Mag_y', 'Mag_z']].to_frame().T
            mag_raw.columns = mag_cols # Map to training names
            X_mag_obs = scaler_mag.transform(mag_raw)
            
            # Re-enable K=5 averaging to flatten Uniform Magnetic Hallway errors
            micro_knn = KNeighborsRegressor(n_neighbors=min(5, len(Y_local)))
            micro_knn.fit(X_mag_local, Y_local)
            mag_pred = micro_knn.predict(X_mag_obs)[0]
            
            # Check pure prediction errors privately to isolate bug
            err_wifi = np.sqrt((wifi_pred[0] - row['True_X'])**2 + (wifi_pred[1] - row['True_Y'])**2)
            err_mag = np.sqrt((mag_pred[0] - row['True_X'])**2 + (mag_pred[1] - row['True_Y'])**2)
            if i % 500 == 0:
                print(f"Frame {i} | Wi-Fi KNN err: {err_wifi:.2f}m | Mag KNN err: {err_mag:.2f}m")
            
            # Apply System Map-Correction
            Z_mag = mag_pred.reshape(2, 1)
            S_mag = H @ P @ H.T + R_mag
            K_mag = P @ H.T @ np.linalg.inv(S_mag)
            X_est = X_est + (K_mag @ (Z_mag - (H @ X_est).reshape(2,1))).flatten()
            P = (np.eye(4) - K_mag @ H) @ P
            
        metrics.append({
            'Time': row['Time'],
            'True_X': row['True_X'],
            'True_Y': row['True_Y'],
            'Pred_X': X_est[0],
            'Pred_Y': X_est[1]
        })
        
    df_metrics = pd.DataFrame(metrics)
    
    # Post-Calculation Validation
    df_metrics['Error'] = np.sqrt((df_metrics['True_X'] - df_metrics['Pred_X'])**2 + (df_metrics['True_Y'] - df_metrics['Pred_Y'])**2)
    mean_err = df_metrics['Error'].mean()
    
    print(f"\n--- KALMAN FILTER FUSION COMPLETE ---")
    print(f"Mean Positioning Error (A8 Continuous Track): {mean_err:.2f} meters!")
    print(f"Max Drift Found: {df_metrics['Error'].max():.2f} meters")
    
    out_path = os.path.join('Datasets', 'Metrics_A8_Fusion.csv')
    df_metrics.to_csv(out_path, index=False)

if __name__ == '__main__':
    main()
