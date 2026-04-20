import os
import glob
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

def load_building_data(devices, base_path):
    all_data = []
    
    for device in devices:
        search_path = os.path.join(base_path, device, '**', 'merged_data.csv')
        for f in glob.glob(search_path, recursive=True):
            try:
                df = pd.read_csv(f)
                required_cols = ['True_X', 'True_Y']
                if all(col in df.columns for col in required_cols):
                    df['Device'] = device
                    all_data.append(df)
            except: pass
                
    if not all_data: return pd.DataFrame()
    # Filling NaN with -100 here to unify the dataframe sizes, will process out later
    combo_df = pd.concat(all_data, ignore_index=True).fillna(-100)
    return combo_df

def process_wifi_features(df, feature_cols):
    X = df[feature_cols].copy()
    # Replace default hardcoded -100 with np.nan for valid statistical logic
    X = X.replace(-100, np.nan)
    
    # Standardize row-wise to focus on relative signature instead of absolute dBm limits
    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1).replace(0.0, 1.0)
    X_std = X.sub(row_means, axis=0).div(row_stds, axis=0)
    
    # Fill any signals that the device missed/dropped entirely with the neutral median (0.0)
    return X_std.fillna(0.0).values

def main():
    base_path = os.path.join('Datasets', 'Merged dataset', 'BE Building', 'Navigation', 'Scenario 1')
    
    train_df = load_building_data(['A8', 'S8', 'G7'], base_path)
    test_df = load_building_data(['S9+'], base_path)
    
    if train_df.empty or test_df.empty:
        print("Data load failed.")
        return
        
    feature_masks = ['Timestamp', 'True_X', 'True_Y', 'Device']
    bssid_train_cols = [c for c in train_df.columns if c not in feature_masks and not c.startswith('Mean_')]

    # Ensure test matches train dimension structure
    for col in bssid_train_cols:
        if col not in test_df.columns:
            test_df[col] = -100
            
    feature_cols = bssid_train_cols
    
    print("\nApplying custom Row-Wise scaling and Zero-Imputation...")
    X_train = process_wifi_features(train_df, feature_cols)
    Y_train = train_df[['True_X', 'True_Y']].values
    
    X_test = process_wifi_features(test_df, feature_cols)
    Y_test = test_df[['True_X', 'True_Y']].values
    
    print("Optimizing K via GridSearchCV...")
    params = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20]}
    grid = GridSearchCV(KNeighborsRegressor(metric='euclidean'), params, cv=5, scoring='neg_mean_absolute_error')
    grid.fit(X_train, Y_train)
    best_k = grid.best_params_['n_neighbors']
    
    model = grid.best_estimator_
    
    predictions = model.predict(X_test)
    errors = np.sqrt(np.sum((Y_test - predictions)**2, axis=1))
    
    print("-" * 40)
    print("OPTIMIZED WIFI MODEL (RELATIVE SCALING)")
    print("-" * 40)
    print(f"Optimal K-Value Found: {best_k}")
    print(f"Mean Euclidean Error:  {np.mean(errors):.2f} m")
    print(f"90th Percentile Error: {np.percentile(errors, 90):.2f} m")
    print(f"Maximum Error Value:   {np.max(errors):.2f} m")
    print("-" * 40)

if __name__ == "__main__":
    main()
