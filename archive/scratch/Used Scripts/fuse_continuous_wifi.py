import pandas as pd
import numpy as np
import os
import glob
import math

def fuse_continuous_data():
    base_cont_path = os.path.join('Datasets', 'Magnetic field dataset', 'Continuous Data', 'BE Building', 'Navigation', 'Scenario 1')
    base_stat_path = os.path.join('Datasets', 'Merged dataset', 'BE Building', 'Navigation', 'Scenario 1')
    
    phones = ['A8', 'G7', 'S8', 'S9+']
    
    for phone in phones:
        print(f"\nProcessing Phone: {phone}")
        
        # 1. Find Continuous IMU File
        cont_pattern = os.path.join(base_cont_path, phone, '*', 'IMU_*.csv')
        cont_files = glob.glob(cont_pattern)
        if not cont_files:
            print(f"Skipping {phone} - No continuous file found.")
            continue
        cont_file = cont_files[0]
        df_cont = pd.read_csv(cont_file)
        
        # 2. Find Static Merged Data
        stat_pattern = os.path.join(base_stat_path, phone, '*', 'merged_data.csv')
        stat_files = glob.glob(stat_pattern)
        if not stat_files:
            print(f"Skipping {phone} - No static file found.")
            continue
        stat_file = stat_files[0]
        df_stat = pd.read_csv(stat_file)
        
        # Sort Static temporally to reconstruct walk geometry
        df_stat['Time_Obj'] = pd.to_datetime(df_stat['Timestamp'], format='%Y.%m.%d %H%M%S', errors='coerce')
        df_stat = df_stat.sort_values(by='Time_Obj').reset_index(drop=True)
        
        wifi_cols = [c for c in df_stat.columns if ':' in c]
        total_static = len(df_stat)
        
        # 3. Synchronize Temporal Unique Seconds for 1Hz Wi-Fi calculation
        unique_times = df_cont['Time'].unique()
        total_secs = len(unique_times)
        
        # We pre-calculate the WiFi vectors for each unique second (1Hz Ping)
        wifi_lookup = {}
        for t_idx, time_str in enumerate(unique_times):
            f_time = t_idx / max(1, (total_secs - 1))
            grid_idx = f_time * (total_static - 1)
            idx0 = int(math.floor(grid_idx))
            idx1 = int(math.ceil(grid_idx))
            frac = grid_idx - idx0
            
            pt_start = df_stat.iloc[idx0]
            pt_end = df_stat.iloc[idx1]
            
            wifi_vector = {}
            for bssid in wifi_cols:
                v1 = pt_start[bssid]
                v2 = pt_end[bssid]
                
                v1_missing = math.isnan(v1)
                v2_missing = math.isnan(v2)
                
                if v1_missing and v2_missing:
                    val = np.nan
                elif v1_missing:
                    val = -100 + frac * (v2 - (-100))
                elif v2_missing:
                    val = v1 + frac * ((-100) - v1)
                else:
                    val = v1 + frac * (v2 - v1)
                    
                if not math.isnan(val):
                    val += np.random.normal(0, 4.0) # Gaussian 1Hz scattering 
                    if val < -95: val = np.nan
                    elif val > -30: val = -30
                        
                wifi_vector[bssid] = val
                
            wifi_lookup[time_str] = wifi_vector
            
        # 4. Process all rows natively for 50Hz
        total_rows = len(df_cont)
        
        fused_rows = []
        for i in range(total_rows):
            row_dict = df_cont.iloc[i].to_dict()
            time_str = row_dict['Time']
            
            # Smoothly calculate True_X and True_Y 50 times a second
            f_row = i / max(1, (total_rows - 1))
            grid_idx = f_row * (total_static - 1)
            idx0 = int(math.floor(grid_idx))
            idx1 = int(math.ceil(grid_idx))
            frac = grid_idx - idx0
            
            x0, y0 = df_stat.iloc[idx0]['True_X'], df_stat.iloc[idx0]['True_Y']
            x1, y1 = df_stat.iloc[idx1]['True_X'], df_stat.iloc[idx1]['True_Y']
            
            row_dict['True_X'] = x0 + frac * (x1 - x0)
            row_dict['True_Y'] = y0 + frac * (y1 - y0)
            
            # Bind the 1Hz locked Wi-Fi vector to this 50Hz frame
            wf_ping = wifi_lookup[time_str]
            row_dict.update(wf_ping)
            
            fused_rows.append(row_dict)
            
        df_fused = pd.DataFrame(fused_rows)
        out_filename = f'Continuous_Fused_{phone}.csv'
        out_path = os.path.join('Datasets', out_filename)
        df_fused.to_csv(out_path, index=False)
        print(f"Generated successfully: {out_filename}")
        print(f"Rows: {len(df_fused)} | Columns: {len(df_fused.columns)}")

if __name__ == "__main__":
    fuse_continuous_data()
