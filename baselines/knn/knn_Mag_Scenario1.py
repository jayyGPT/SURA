import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
                required_cols = ['True_X', 'True_Y', 'Mean_Mag_x', 'Mean_Mag_y', 'Mean_Mag_z']
                if all(col in df.columns for col in required_cols):
                    df['Device'] = device
                    all_data.append(df)
            except: pass
    if not all_data: return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

def main():
    base_path = os.path.join('Datasets', 'Merged dataset', 'BE Building', 'Navigation', 'Scenario 1')
    
    train_df = load_building_data(['A8', 'S8', 'G7'], base_path)
    test_df = load_building_data(['S9+'], base_path)
    
    if train_df.empty or test_df.empty:
        print("Data load failed.")
        return
        
    feature_cols = ['Mean_Mag_x', 'Mean_Mag_y', 'Mean_Mag_z']
    important_cols = feature_cols + ['True_X', 'True_Y']
    train_df = train_df.dropna(subset=important_cols)
    test_df = test_df.dropna(subset=important_cols)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    Y_train = train_df[['True_X', 'True_Y']].values
    X_test = scaler.transform(test_df[feature_cols])
    Y_test = test_df[['True_X', 'True_Y']].values
    
    # Optimize K using K-Fold CV
    print("Optimizing K via GridSearchCV...")
    params = {'n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20]}
    grid = GridSearchCV(KNeighborsRegressor(metric='euclidean'), params, cv=5, scoring='neg_mean_absolute_error')
    grid.fit(X_train, Y_train)
    best_k = grid.best_params_['n_neighbors']
    
    model = grid.best_estimator_
    
    predictions = model.predict(X_test)
    errors = np.sqrt(np.sum((Y_test - predictions)**2, axis=1))
    
    print("-" * 40)
    print("OPTIMIZED PURE MAGNETIC MODEL")
    print("-" * 40)
    print(f"Optimal K-Value Found: {best_k}")
    print(f"Mean Euclidean Error:  {np.mean(errors):.2f} m")
    print(f"90th Percentile Error: {np.percentile(errors, 90):.2f} m")
    print(f"Maximum Error Value:   {np.max(errors):.2f} m")
    print("-" * 40)

if __name__ == "__main__":
    main()
