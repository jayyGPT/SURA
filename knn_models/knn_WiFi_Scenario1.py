import os
import glob
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

def process_files(device_folders, base_path):
    all_pivots = []
    
    for device in device_folders:
        search_path = os.path.join(base_path, device, '**', '*.csv')
        files = glob.glob(search_path, recursive=True)
        print(f"[{device}] Found {len(files)} files to process...")
        for f in files:
            try:
                # Based on previous tests, the files are actually xls format saved as .csv
                df = pd.read_excel(f, engine='xlrd')
                
                # Check if required columns exist
                required = ['Time', 'X-pos', 'Y-pos', 'BSSID', 'RSS']
                if not all(col in df.columns for col in required):
                    continue
                
                # Filter to needed columns
                df = df[required]
                
                # Forward fill the missing metadata for the scan group
                df[['Time', 'X-pos', 'Y-pos']] = df[['Time', 'X-pos', 'Y-pos']].ffill()
                
                # Drop rows where Time/Position is somehow still null or RSS is null
                df = df.dropna(subset=['Time', 'X-pos', 'Y-pos', 'BSSID', 'RSS'])
                
                # Group by Time and Location, collect APs into columns
                # Since multiple same BSSIDs might rarely appear in a single scan, we use 'mean'
                pivot_df = df.pivot_table(index=['Time', 'X-pos', 'Y-pos'], 
                                          columns='BSSID', 
                                          values='RSS', 
                                          aggfunc='mean').reset_index()
                
                # Append Device context
                pivot_df['Device'] = device
                all_pivots.append(pivot_df)
                
            except Exception as e:
                print(f"Error processing {f}: {e}")
                
    if len(all_pivots) == 0:
        return pd.DataFrame()
        
    combo_df = pd.concat(all_pivots, ignore_index=True)
    # Fill any NaNs arising from non-overlapping BSSIDs between scans/files with -100
    combo_df = combo_df.fillna(-100)
    return combo_df

def main():
    base_path = os.path.join('Datasets', 'WiFi dataset', 'BE Engineering', 'Navigation', 'Scenario-1')
    
    train_devices = ['A8', 'S8', 'G7']
    test_devices = ['S9+']
    
    print("--- Extracting Train Data ---")
    train_df = process_files(train_devices, base_path)
    print("--- Extracting Test Data ---")
    test_df = process_files(test_devices, base_path)
    
    if train_df.empty or test_df.empty:
        print("Error: Train or test set is empty. Check files and paths.")
        return
        
    print(f"\nTrain rows: {len(train_df)}, Test rows: {len(test_df)}")
    
    # Identify feature columns (BSSIDs) available in training data
    non_features = ['Time', 'X-pos', 'Y-pos', 'Device']
    train_bssid_cols = [c for c in train_df.columns if c not in non_features]
    
    # Ensure test set has the exact same feature columns as training set
    for col in train_bssid_cols:
        if col not in test_df.columns:
            test_df[col] = -100 # Default no-signal value
            
    # We only care about BSSIDs that were seen during training for our model
    feature_cols = train_bssid_cols
    
    # Reorder test columns to match train exactly
    test_df = test_df[non_features + feature_cols]
    
    # Save the processed datasets
    print("\nSaving datasets to CSV...")
    train_df.to_csv('scenario1_train.csv', index=False)
    test_df.to_csv('scenario1_test.csv', index=False)
    print("Saved 'scenario1_train.csv' and 'scenario1_test.csv'.")
    
    # Machine Learning Pipeline
    print("\nTraining KNN (K=7) Regressor...")
    scaler = MinMaxScaler()
    
    X_train = scaler.fit_transform(train_df[feature_cols])
    Y_train = train_df[['X-pos', 'Y-pos']].values
    
    X_test = scaler.transform(test_df[feature_cols])
    Y_test = test_df[['X-pos', 'Y-pos']].values
    
    model = KNeighborsRegressor(n_neighbors=7, metric='euclidean')
    model.fit(X_train, Y_train)
    
    print("Testing on S9+ device...")
    predictions = model.predict(X_test)
    
    # Calculate Euclidean distance errors
    errors = np.sqrt(np.sum((Y_test - predictions)**2, axis=1))
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    p90_error = np.percentile(errors, 90)
    
    print("-" * 30)
    print("Euclidean Distance Error Metrics")
    print(f"Mean Error: {mean_error:.2f} m")
    print(f"90th Percentile Error: {p90_error:.2f} m")
    print(f"Max Error:  {max_error:.2f} m")
    print("-" * 30)

if __name__ == "__main__":
    main()
