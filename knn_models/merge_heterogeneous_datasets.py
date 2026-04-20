import os
import glob
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def main():
    mag_base_dir = os.path.join('Datasets', 'Magnetic field dataset', 'Static Data')
    wifi_base_dir = os.path.join('Datasets', 'WiFi dataset')
    merged_base_dir = os.path.join('Datasets', 'Merged dataset')

    # Collect ALL WiFi files globally to map Timestamp -> filepath
    print("Indexing WiFi dataset...")
    wifi_files = glob.glob(os.path.join(wifi_base_dir, '**', '*.csv'), recursive=True)
    wifi_map = {}
    for f in wifi_files:
        basename = os.path.basename(f)
        # Handle potential multiple "Navigation_" components or format inconsistencies
        # The timestamp is always the last component of the filename before extension
        # e.g. "WiFi_BE Building_Scenario 1_User 2 (M-174cm)_Navigation_2021.05.19 171520.csv"
        timestamp = basename.split('_')[-1].replace('.csv', '')
        wifi_map[timestamp] = f

    print(f"Indexed {len(wifi_map)} unique WiFi timestamps.")

    # Collect ALL Magnetic files globally
    print("Indexing Magnetic dataset...")
    mag_files = glob.glob(os.path.join(mag_base_dir, '**', '*.csv'), recursive=True)
    print(f"Discovered {len(mag_files)} static IMU recordings to process.")

    matched_count = 0
    mismatched_count = 0

    # Group output rows by relative sub-directory path
    data_by_folder = {}

    for f_mag in mag_files:
        basename = os.path.basename(f_mag)
        timestamp = basename.split('_')[-1].replace('.csv', '')

        if timestamp not in wifi_map:
            mismatched_count += 1
            # Could not find a matched Wi-Fi file
            continue

        f_wifi = wifi_map[timestamp]

        try:
            # Load IMU Data (some are incorrectly saved as XLS)
            try:
                df_mag = pd.read_csv(f_mag)
            except Exception:
                df_mag = pd.read_excel(f_mag, engine='xlrd')

            # Extract True Coordinate Location (constant across file)
            # Take mean of X-cord, Y-cord representing static ground truth
            # Remove Time & unnamed columns before averaging IMU data
            numeric_mag = df_mag.select_dtypes(include=[np.number])
            if 'Unnamed: 16' in numeric_mag.columns:
                numeric_mag = numeric_mag.drop(columns=['Unnamed: 16'])
            
            # The mean of constant values (e.g. X-cord) is simply the value itself
            mean_imu = numeric_mag.mean(numeric_only=True)
            
            # Load WiFi Data
            df_wifi = pd.read_excel(f_wifi, engine='xlrd')
            df_wifi = df_wifi[['BSSID', 'RSS']].dropna()

            # Average RSS readings per BSSID across the multiple scans
            bssid_means = df_wifi.groupby('BSSID')['RSS'].mean()

            # Construct row
            row = {'Timestamp': timestamp}
            
            # Use X-cord and Y-cord as X-pos and Y-pos properly
            row['True_X'] = mean_imu.get('X-cord', np.nan)
            row['True_Y'] = mean_imu.get('Y-cord', np.nan)
            
            # Append other mean IMU features
            for col in ['Mag_x', 'Mag_y', 'Mag_z', 'Acc_x', 'Acc_y', 'Acc_z', 
                        'Gyro_x', 'Gyro_y', 'Gyro_z', 'Orn_x', 'Orn_y', 'Orn_z', 'Pressure']:
                row[f'Mean_{col}'] = mean_imu.get(col, np.nan)
            
            # Append all BSSID RSS values
            for bssid, rss in bssid_means.items():
                row[bssid] = rss

            # Determine original relative path (e.g. CS Engineering\Navigation\A8\User 2)
            rel_dir = os.path.relpath(os.path.dirname(f_mag), mag_base_dir)
            if rel_dir not in data_by_folder:
                data_by_folder[rel_dir] = []
            
            data_by_folder[rel_dir].append(row)
            matched_count += 1

        except Exception as e:
            # Handle rare load/format errors
            print(f"Failed parsing files for {timestamp}: {e}")
            mismatched_count += 1

    # Analysis output
    total_files = len(mag_files)
    success_rate = (matched_count / total_files * 100) if total_files > 0 else 0.0

    print("-" * 40)
    print("Dataset Processing Summary")
    print(f"Total Magnetic Static Files : {total_files}")
    print(f"Total Successful Merges     : {matched_count}")
    print(f"Total Orphan/Mismatches     : {mismatched_count}")
    print(f"Processing Success Rate     : {success_rate:.2f}%")
    print("-" * 40)

    # Save compiled sets dynamically
    print("Generating structured CSV matrices...")
    saved_files = 0
    for rel_dir, rows in data_by_folder.items():
        if len(rows) == 0: continue
        
        # Consolidate per-folder DataFrame
        df_folder = pd.DataFrame(rows)
        
        # Sort columns to enforce canonical placement
        fixed_cols = ['Timestamp', 'True_X', 'True_Y']
        imu_cols = [c for c in df_folder.columns if c.startswith('Mean_')]
        bssid_cols = [c for c in df_folder.columns if c not in fixed_cols and c not in imu_cols]
        
        df_folder = df_folder[fixed_cols + imu_cols + bssid_cols]
        
        # Missing BSSIDs become NaN over multiple timestamps. Pad with -100.
        df_folder[bssid_cols] = df_folder[bssid_cols].fillna(-100)

        # Output pathing logic
        out_dir = os.path.join(merged_base_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "merged_data.csv")
        
        df_folder.to_csv(out_file, index=False)
        saved_files += 1

    print(f"Complete! Saved {saved_files} distinct matrices across the directory structure.")

if __name__ == "__main__":
    main()
