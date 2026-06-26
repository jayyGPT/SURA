import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import warnings

warnings.filterwarnings('ignore')
plt.switch_backend('Agg') # Safe for headless environments

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
    return pd.concat(all_data, ignore_index=True).fillna(-100)

def process_wifi(df, feature_cols):
    X = df[feature_cols].copy().replace(-100, np.nan)
    row_means = X.mean(axis=1)
    row_stds = X.std(axis=1).replace(0.0, 1.0)
    return X.sub(row_means, axis=0).div(row_stds, axis=0).fillna(0.0).values

def main():
    base_path = os.path.join('Datasets', 'Merged dataset', 'BE Building', 'Navigation', 'Scenario 1')
    
    train_df = load_building_data(['A8', 'S8', 'G7'], base_path)
    test_df = load_building_data(['S9+'], base_path)
    
    important_cols = ['True_X', 'True_Y', 'Mean_Mag_x', 'Mean_Mag_y', 'Mean_Mag_z']
    train_df = train_df.dropna(subset=important_cols)
    test_df = test_df.dropna(subset=important_cols)
        
    bssid_train_cols = [c for c in train_df.columns if c not in ['Timestamp', 'True_X', 'True_Y', 'Device'] and not c.startswith('Mean_')]
    for col in bssid_train_cols:
        if col not in test_df.columns: test_df[col] = -100 

    Y_train = train_df[['True_X', 'True_Y']].values
    Y_test = test_df[['True_X', 'True_Y']].values

    # 1. Pure Magnetic (k=5)
    mag_cols = ['Mean_Mag_x', 'Mean_Mag_y', 'Mean_Mag_z']
    scaler = MinMaxScaler()
    X_mag_tr = scaler.fit_transform(train_df[mag_cols])
    X_mag_te = scaler.transform(test_df[mag_cols])

    m_mag = KNeighborsRegressor(n_neighbors=5, metric='euclidean').fit(X_mag_tr, Y_train)
    err_mag = np.sqrt(np.sum((Y_test - m_mag.predict(X_mag_te))**2, axis=1))

    # 2. Pure WiFi (k=5)
    X_wifi_tr = process_wifi(train_df, bssid_train_cols)
    X_wifi_te = process_wifi(test_df, bssid_train_cols)

    m_wifi = KNeighborsRegressor(n_neighbors=5, metric='euclidean').fit(X_wifi_tr, Y_train)
    err_wifi = np.sqrt(np.sum((Y_test - m_wifi.predict(X_wifi_te))**2, axis=1))

    # 3. Hybrid (k=7)
    X_hy_tr = np.hstack((X_mag_tr, X_wifi_tr))
    X_hy_te = np.hstack((X_mag_te, X_wifi_te))
    
    m_hybrid = KNeighborsRegressor(n_neighbors=7, metric='euclidean').fit(X_hy_tr, Y_train)
    err_hybrid = np.sqrt(np.sum((Y_test - m_hybrid.predict(X_hy_te))**2, axis=1))

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Bar Chart of Mean & 90th Percentile
    metrics = {
        'Model': ['Pure Magnetic', 'Pure Magnetic', 'Pure Wi-Fi', 'Pure Wi-Fi', 'Hybrid (Mag+WiFi)', 'Hybrid (Mag+WiFi)'],
        'Metric': ['Mean Error', '90th Pct Error'] * 3,
        'Error Rate (m)': [
            np.mean(err_mag), np.percentile(err_mag, 90),
            np.mean(err_wifi), np.percentile(err_wifi, 90),
            np.mean(err_hybrid), np.percentile(err_hybrid, 90)
        ]
    }
    df_metrics = pd.DataFrame(metrics)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Model', y='Error Rate (m)', hue='Metric', data=df_metrics, palette=['#3498db', '#e74c3c'])
    plt.title('Positioning Validation Accuracy (Samsung S9+ Test Device)', fontsize=16, fontweight='bold')
    plt.ylabel('Distance Error (meters)', fontsize=12)
    plt.xlabel('Algorithm Baseline', fontsize=12)
    
    # Add values on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fm', padding=3)

    plt.tight_layout()
    plt.savefig('bar_chart_metrics.png', dpi=300)
    plt.close()

    # Chart 2: Empirical Cumulative Distribution Function (CDF)
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=err_mag, label=f'Magnetic Only (Mean: {np.mean(err_mag):.2f}m)', color='red', linewidth=2)
    sns.ecdfplot(data=err_wifi, label=f'Wi-Fi Only (Mean: {np.mean(err_wifi):.2f}m)', color='blue', linewidth=2)
    sns.ecdfplot(data=err_hybrid, label=f'Hybrid Fusion (Mean: {np.mean(err_hybrid):.2f}m)', color='green', linewidth=3)

    plt.title('Cumulative Distribution Function (CDF) of Positioning Errors', fontsize=16, fontweight='bold')
    plt.xlabel('Positioning Error (meters)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Zoomed limit to make curves readable (errors > 40 aren't as important visually)
    plt.xlim(0, 40)
    plt.tight_layout()
    plt.savefig('cdf_comparison.png', dpi=300)
    plt.close()

    print("Graphs generated successfully as 'bar_chart_metrics.png' and 'cdf_comparison.png'.")

if __name__ == "__main__":
    main()
