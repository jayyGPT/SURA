import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_wifi(df, wifi_cols):
    X = df[wifi_cols].astype(float).replace(-100, np.nan)
    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1).replace(0.0, 1.0)
    X_std = X.sub(row_means, axis=0).div(row_stds, axis=0)
    return X_std.fillna(0.0).values

def create_sequences(wifi_data, imu_data, labels, time_steps=10):
    X_w, X_i, y = [], [], []
    # If the file is too short, return empty
    if len(wifi_data) <= time_steps:
        return np.array([]), np.array([]), np.array([])
        
    for i in range(len(wifi_data) - time_steps):
        X_w.append(wifi_data[i : i + time_steps])
        X_i.append(imu_data[i : i + time_steps])
        # Predict the position for the entire sequence
        y.append(labels[i : i + time_steps])
        
    return np.array(X_w), np.array(X_i), np.array(y)

def main():
    print("Starting Deep Learning Preprocessing Pipeline...")
    
    # 1. Load Data
    base_dir = '../Datasets'
    train_phones = ['A8', 'G7', 'S8']
    test_phone = 'S9+'
    
    train_dfs = []
    for p in train_phones:
        path = os.path.join(base_dir, f'Continuous_Fused_{p}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Phone'] = p
            train_dfs.append(df)
            print(f"Loaded Train: {p} - {len(df)} frames")
            
    df_train = pd.concat(train_dfs, ignore_index=True)
    
    path_test = os.path.join(base_dir, f'Continuous_Fused_{test_phone}.csv')
    df_test = pd.read_csv(path_test)
    df_test['Phone'] = test_phone
    print(f"Loaded Test: {test_phone} - {len(df_test)} frames")
    
    # 2. Reconcile WiFi columns (Ensure all BSSIDs exist in both train and test)
    wifi_cols_train = [c for c in df_train.columns if ':' in c]
    wifi_cols_test = [c for c in df_test.columns if ':' in c]
    all_wifi_cols = list(set(wifi_cols_train + wifi_cols_test))
    
    for c in all_wifi_cols:
        if c not in df_train.columns:
            df_train[c] = np.nan
        if c not in df_test.columns:
            df_test[c] = np.nan
            
    # Sort columns to maintain order
    all_wifi_cols.sort()
    
    imu_cols = ['Mag_x', 'Mag_y', 'Mag_z', 'Acc_x', 'Acc_y', 'Acc_z', 'Gyro_x', 'Gyro_y', 'Gyro_z', 'Orn_x', 'Orn_y', 'Orn_z']
    # Sometimes Pressure is missing, verify
    if 'Pressure' in df_train.columns and 'Pressure' in df_test.columns:
        imu_cols.append('Pressure')
        
    print(f"Total WiFi Features: {len(all_wifi_cols)}")
    print(f"Total IMU Features: {len(imu_cols)}")
    
    # 3. Process Per Phone (Downsampling to 2Hz -> Every 25th row)
    DOWNSAMPLE_RATE = 25 
    TIME_STEPS = 10 # 5 seconds of context at 2Hz
    
    scaler = MinMaxScaler()
    
    # Fit scaler only on Train Data
    scaler.fit(df_train[imu_cols])
    
    X_train_w_list, X_train_i_list, y_train_list = [], [], []
    X_test_w_list, X_test_i_list, y_test_list = [], [], []
    
    # Process Train Sets Separately so we don't bleed sequences across phones
    for p in train_phones:
        df_p = df_train[df_train['Phone'] == p].reset_index(drop=True)
        # Downsample
        df_p_ds = df_p.iloc[::DOWNSAMPLE_RATE].reset_index(drop=True)
        
        wifi_p = preprocess_wifi(df_p_ds, all_wifi_cols)
        imu_p = scaler.transform(df_p_ds[imu_cols])
        labels_p = df_p_ds[['True_X', 'True_Y']].values
        
        X_w, X_i, y = create_sequences(wifi_p, imu_p, labels_p, time_steps=TIME_STEPS)
        if len(X_w) > 0:
            X_train_w_list.append(X_w)
            X_train_i_list.append(X_i)
            y_train_list.append(y)
            
    # Process Test Set
    df_test_ds = df_test.iloc[::DOWNSAMPLE_RATE].reset_index(drop=True)
    wifi_test = preprocess_wifi(df_test_ds, all_wifi_cols)
    imu_test = scaler.transform(df_test_ds[imu_cols])
    labels_test = df_test_ds[['True_X', 'True_Y']].values
    
    X_w_test, X_i_test, y_test = create_sequences(wifi_test, imu_test, labels_test, time_steps=TIME_STEPS)
    X_test_w_list.append(X_w_test)
    X_test_i_list.append(X_i_test)
    y_test_list.append(y_test)
    
    # 4. Concatenate Final Arrays
    X_train_wifi = np.concatenate(X_train_w_list, axis=0)
    X_train_imu = np.concatenate(X_train_i_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    X_test_wifi = np.concatenate(X_test_w_list, axis=0)
    X_test_imu = np.concatenate(X_test_i_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    print(f"Train Shape -> WiFi: {X_train_wifi.shape}, IMU: {X_train_imu.shape}, Labels: {y_train.shape}")
    print(f"Test Shape  -> WiFi: {X_test_wifi.shape}, IMU: {X_test_imu.shape}, Labels: {y_test.shape}")
    
    # 5. Save Data
    out_dir = os.path.join(base_dir, 'dl_processed')
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, 'X_train_wifi.npy'), X_train_wifi)
    np.save(os.path.join(out_dir, 'X_train_imu.npy'), X_train_imu)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    
    np.save(os.path.join(out_dir, 'X_test_wifi.npy'), X_test_wifi)
    np.save(os.path.join(out_dir, 'X_test_imu.npy'), X_test_imu)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    
    print(f"Successfully saved processed tensors to {out_dir}")

if __name__ == '__main__':
    main()
