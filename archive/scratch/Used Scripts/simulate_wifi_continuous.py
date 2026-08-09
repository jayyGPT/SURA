import pandas as pd
import numpy as np
import os
import random
import math

def simulate_walking_path():
    # We will simulate for BE Building, Scenario 1, S8, User 2
    data_path = os.path.join('Datasets', 'Merged dataset', 'BE Building', 'Navigation', 'Scenario 1', 'S8', 'User 2', 'merged_data.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # 1. Sort by Timestamp to create chronological Walking Path
    df['Time_Obj'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d %H%M%S', errors='coerce')
    df = df.sort_values(by='Time_Obj').reset_index(drop=True)
    
    # Find Wi-Fi BSSID columns (They are MAC Addresses containing colons)
    wifi_cols = [c for c in df.columns if ':' in c]
    
    simulated_rows = []
    
    # Starting simulated time at a fake distinct epoch for continuous runs
    current_sim_time = pd.to_datetime("2026-01-01 12:00:00")
    
    for i in range(len(df) - 1):
        pt_start = df.iloc[i]
        pt_end = df.iloc[i+1]
        
        # 2Hz Resolution
        steps = 2
        
        for step in range(steps):
            frac = step / steps
            
            interp_X = pt_start['True_X'] + frac * (pt_end['True_X'] - pt_start['True_X'])
            interp_Y = pt_start['True_Y'] + frac * (pt_end['True_Y'] - pt_start['True_Y'])
            
            row = {
                'Simulated_Time': current_sim_time,
                'True_X': interp_X,
                'True_Y': interp_Y
            }
            
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
                    # Gaussian Injection
                    val += np.random.normal(0, 4.0)
                    
                    # Physical limits
                    if val < -95:
                        val = np.nan
                    elif val > -30:
                        val = -30
                        
                row[bssid] = val
                
            simulated_rows.append(row)
            current_sim_time += pd.Timedelta(milliseconds=500) 
            
    # Add final point
    pt_end = df.iloc[-1]
    row = {
        'Simulated_Time': current_sim_time,
        'True_X': pt_end['True_X'],
        'True_Y': pt_end['True_Y']
    }
    for bssid in wifi_cols:
        v1 = pt_end[bssid]
        if not math.isnan(v1):
            val = v1 + np.random.normal(0, 4.0)
            if val < -95:
                val = np.nan
            elif val > -30:
                val = -30
            row[bssid] = val
        else:
            row[bssid] = np.nan
            
    simulated_rows.append(row)
    
    df_sim = pd.DataFrame(simulated_rows)
    output_path = os.path.join('Datasets', 'Simulated_Continuous_WiFi.csv')
    df_sim.to_csv(output_path, index=False)
    
    print(f"--- Continuous Wi-Fi Simulation Complete ---")
    print(f"Generated {len(df_sim)} vectors over {len(df)} static map anchors.")
    print(f"Resolution: 2 Hz")
    print(f"Total Unique Wi-Fi Networks simulated: {len(wifi_cols)}")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    simulate_walking_path()
