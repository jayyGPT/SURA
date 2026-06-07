"""
Preprocessing Pipeline v2: Three-Branch Architecture
Separates spatial fingerprints (WiFi, Magnetic) from motion sensors (IMU).
Augments training data with reversed paths to force environment learning.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def preprocess_wifi(df, wifi_cols):
    X = df[wifi_cols].astype(float).replace(-100, np.nan)
    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1).replace(0.0, 1.0)
    X_std = X.sub(row_means, axis=0).div(row_stds, axis=0)
    return X_std.fillna(0.0).values

def add_engineered_features(df):
    df['Acc_Mag'] = np.sqrt(df['Acc_x']**2 + df['Acc_y']**2 + df['Acc_z']**2)
    df['Gyro_Mag'] = np.sqrt(df['Gyro_x']**2 + df['Gyro_y']**2 + df['Gyro_z']**2)
    df['Head_Cos'] = np.cos(df['Orn_z'])
    df['Head_Sin'] = np.sin(df['Orn_z'])
    return df

def create_sequences(wifi_data, mag_data, imu_data, labels, time_steps=10):
    X_w, X_m, X_i, y = [], [], [], []
    if len(wifi_data) <= time_steps:
        return np.array([]), np.array([]), np.array([]), np.array([])
    for i in range(len(wifi_data) - time_steps):
        X_w.append(wifi_data[i : i + time_steps])
        X_m.append(mag_data[i : i + time_steps])
        X_i.append(imu_data[i : i + time_steps])
        y.append(labels[i : i + time_steps])
    return np.array(X_w), np.array(X_m), np.array(X_i), np.array(y)

def main():
    print("=" * 60)
    print("Preprocessing Pipeline v2: Three-Branch + Augmentation")
    print("=" * 60)
    
    base_dir = '../Datasets'
    train_phones = ['A8', 'G7', 'S8']
    test_phone = 'S9+'
    
    # Separate feature groups
    mag_cols = ['Mag_x', 'Mag_y', 'Mag_z']
    motion_cols = ['Acc_Mag', 'Gyro_Mag', 'Head_Cos', 'Head_Sin']
    
    # Load all data
    train_dfs = []
    for p in train_phones:
        path = os.path.join(base_dir, f'Continuous_Fused_{p}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Phone'] = p
            train_dfs.append(df)
            print(f"Loaded Train: {p} - {len(df)} frames")
            
    df_train_all = pd.concat(train_dfs, ignore_index=True)
    df_train_all = add_engineered_features(df_train_all)
    
    path_test = os.path.join(base_dir, f'Continuous_Fused_{test_phone}.csv')
    df_test = pd.read_csv(path_test)
    df_test['Phone'] = test_phone
    df_test = add_engineered_features(df_test)
    print(f"Loaded Test: {test_phone} - {len(df_test)} frames")
    
    # Reconcile WiFi columns
    wifi_cols_train = [c for c in df_train_all.columns if ':' in c]
    wifi_cols_test = [c for c in df_test.columns if ':' in c]
    all_wifi_cols = sorted(list(set(wifi_cols_train + wifi_cols_test)))
    
    for c in all_wifi_cols:
        if c not in df_train_all.columns:
            df_train_all[c] = np.nan
        if c not in df_test.columns:
            df_test[c] = np.nan
    
    # Check for Pressure
    if 'Pressure' in df_train_all.columns and 'Pressure' in df_test.columns:
        motion_cols.append('Pressure')
    
    all_sensor_cols = mag_cols + motion_cols
    
    print(f"\nFeature Groups:")
    print(f"  WiFi (spatial fingerprint):    {len(all_wifi_cols)} features")
    print(f"  Magnetic (spatial fingerprint): {len(mag_cols)} features")
    print(f"  Motion/IMU (temporal):          {len(motion_cols)} features")
    
    # Fit scalers on train data only
    DOWNSAMPLE_RATE = 25
    TIME_STEPS = 10
    
    mag_scaler = MinMaxScaler()
    imu_scaler = MinMaxScaler()
    
    # Collect all train data for fitting scalers
    all_train_mag = []
    all_train_imu = []
    for p in train_phones:
        df_p = df_train_all[df_train_all['Phone'] == p].reset_index(drop=True)
        df_p_ds = df_p.iloc[::DOWNSAMPLE_RATE].reset_index(drop=True)
        all_train_mag.append(df_p_ds[mag_cols].values)
        all_train_imu.append(df_p_ds[motion_cols].values)
    
    mag_scaler.fit(np.vstack(all_train_mag))
    imu_scaler.fit(np.vstack(all_train_imu))
    
    # Process each phone
    X_train_w_list, X_train_m_list, X_train_i_list, y_train_list = [], [], [], []
    
    for p in train_phones:
        df_p = df_train_all[df_train_all['Phone'] == p].reset_index(drop=True)
        df_p_ds = df_p.iloc[::DOWNSAMPLE_RATE].reset_index(drop=True)
        
        wifi_p = preprocess_wifi(df_p_ds, all_wifi_cols)
        mag_p = mag_scaler.transform(df_p_ds[mag_cols])
        imu_p = imu_scaler.transform(df_p_ds[motion_cols])
        labels_p = df_p_ds[['True_X', 'True_Y']].values
        
        # FORWARD sequences
        X_w, X_m, X_i, y = create_sequences(wifi_p, mag_p, imu_p, labels_p, time_steps=TIME_STEPS)
        if len(X_w) > 0:
            X_train_w_list.append(X_w)
            X_train_m_list.append(X_m)
            X_train_i_list.append(X_i)
            y_train_list.append(y)
            print(f"  {p} forward: {len(X_w)} windows")
    
    # Process Test Set (FORWARD only - we test reversed separately)
    df_test_ds = df_test.iloc[::DOWNSAMPLE_RATE].reset_index(drop=True)
    wifi_test = preprocess_wifi(df_test_ds, all_wifi_cols)
    mag_test = mag_scaler.transform(df_test_ds[mag_cols])
    imu_test = imu_scaler.transform(df_test_ds[motion_cols])
    labels_test = df_test_ds[['True_X', 'True_Y']].values
    
    X_w_test, X_m_test, X_i_test, y_test = create_sequences(
        wifi_test, mag_test, imu_test, labels_test, time_steps=TIME_STEPS)
    
    # Also create REVERSED test set
    wifi_test_rev = wifi_test[::-1].copy()
    mag_test_rev = mag_test[::-1].copy()
    imu_test_rev = imu_test[::-1].copy()
    labels_test_rev = labels_test[::-1].copy()
    
    X_w_test_r, X_m_test_r, X_i_test_r, y_test_r = create_sequences(
        wifi_test_rev, mag_test_rev, imu_test_rev, labels_test_rev, time_steps=TIME_STEPS)
    
    # Concatenate training data
    X_train_wifi = np.concatenate(X_train_w_list, axis=0)
    X_train_mag = np.concatenate(X_train_m_list, axis=0)
    X_train_imu = np.concatenate(X_train_i_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    print(f"\n{'='*60}")
    print(f"Final Dataset Shapes:")
    print(f"  Train WiFi:     {X_train_wifi.shape}")
    print(f"  Train Magnetic: {X_train_mag.shape}")
    print(f"  Train IMU:      {X_train_imu.shape}")
    print(f"  Train Labels:   {y_train.shape}")
    print(f"  Test WiFi:      {X_w_test.shape}")
    print(f"  Test Magnetic:  {X_m_test.shape}")
    print(f"  Test IMU:       {X_i_test.shape}")
    print(f"  Test Labels:    {y_test.shape}")
    print(f"  Test Rev WiFi:  {X_w_test_r.shape}")
    print(f"  Test Rev Labels:{y_test_r.shape}")
    
    # Save
    out_dir = os.path.join(base_dir, 'dl_processed_v2')
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, 'X_train_wifi.npy'), X_train_wifi)
    np.save(os.path.join(out_dir, 'X_train_mag.npy'), X_train_mag)
    np.save(os.path.join(out_dir, 'X_train_imu.npy'), X_train_imu)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    
    np.save(os.path.join(out_dir, 'X_test_wifi.npy'), X_w_test)
    np.save(os.path.join(out_dir, 'X_test_mag.npy'), X_m_test)
    np.save(os.path.join(out_dir, 'X_test_imu.npy'), X_i_test)
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test)
    
    np.save(os.path.join(out_dir, 'X_test_rev_wifi.npy'), X_w_test_r)
    np.save(os.path.join(out_dir, 'X_test_rev_mag.npy'), X_m_test_r)
    np.save(os.path.join(out_dir, 'X_test_rev_imu.npy'), X_i_test_r)
    np.save(os.path.join(out_dir, 'y_test_rev.npy'), y_test_r)
    
    print(f"\nSaved all tensors to {out_dir}")

if __name__ == '__main__':
    main()
