import os
import glob
import pandas as pd
import numpy as np

def main():
    # 1. Load Static Merged Data
    static_pattern = os.path.join('Datasets', 'Merged dataset', 'BE Building', 'Navigation', 'Scenario 1', '*', '*', 'merged_data.csv')
    static_files = glob.glob(static_pattern)
    
    static_dfs = []
    for f in static_files:
        try:
            df = pd.read_csv(f)
            if 'Timestamp' in df.columns and 'Mean_Mag_x' in df.columns:
                df['Time_Obj'] = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d %H%M%S', errors='coerce')
                static_dfs.append(df[['Time_Obj', 'Mean_Mag_x', 'Mean_Mag_y', 'Mean_Mag_z', 'True_X', 'True_Y']])
        except Exception as e:
            print(f"Error static file {f}: {e}")
            
    if not static_dfs:
        print("No static data found or all had errors.")
    else:
        df_static = pd.concat(static_dfs, ignore_index=True)
        print("Length of df_static before dropna:", len(df_static))
        df_static = df_static.dropna(subset=['Time_Obj'])
        print("Length of df_static after dropna:", len(df_static))
    
    # 2. Load Continuous Data
    cont_pattern = os.path.join('Datasets', 'Magnetic field dataset', 'Continuous Data', 'BE Building', 'Navigation', 'Scenario 1', '*', '*', 'IMU_*.csv')
    cont_files = glob.glob(cont_pattern)
    
    cont_dfs = []
    for f in cont_files:
        try:
            df = pd.read_csv(f)
            if 'Time' in df.columns and 'Mag_x' in df.columns:
                # Need to handle exact format '2021.05.19 17:37:43'
                df['Time_Obj'] = pd.to_datetime(df['Time'], format='%Y.%m.%d %H:%M:%S', errors='coerce')
                cont_dfs.append(df[['Time_Obj', 'Mag_x', 'Mag_y', 'Mag_z']])
        except Exception as e:
            print(f"Error cont file {f}: {e}")

    if not cont_dfs:
        print("No continuous data found.")
        return
        
    df_cont_raw = pd.concat(cont_dfs, ignore_index=True)
    print("Length of df_cont before dropna:", len(df_cont_raw))
    df_cont_raw = df_cont_raw.dropna(subset=['Time_Obj'])
    print("Length of df_cont after dropna:", len(df_cont_raw))
    
    if len(df_cont_raw) == 0 or 'df_static' not in locals() or len(df_static) == 0:
        print("Missing required valid data")
        return
        
    df_cont = df_cont_raw.groupby('Time_Obj').mean().reset_index()

    # 3. Match Data via Inner Merge on Exact Second
    matched_df = pd.merge(df_static, df_cont, on='Time_Obj', how='inner')
    
    print(f"Total Unique Static Timestamps: {len(df_static)}")
    print(f"Total Unique Continuous Seconds: {len(df_cont)}")
    print(f"Total Matches Found: {len(matched_df)}")
    
    if len(matched_df) > 0:
        # Compute Error
        err_mag = np.sqrt(
            (matched_df['Mean_Mag_x'] - matched_df['Mag_x'])**2 +
            (matched_df['Mean_Mag_y'] - matched_df['Mag_y'])**2 +
            (matched_df['Mean_Mag_z'] - matched_df['Mag_z'])**2
        )
        matched_df['Mag_Error'] = err_mag
        
        print("\n--- MAGNETIC FIELD ERROR ANALYSIS (Matched Timestamps) ---")
        print(f"Mean Euclidean Error: {np.mean(err_mag):.2f} μT")
        print(f"Max Euclidean Error: {np.max(err_mag):.2f} μT")
        print(f"Min Euclidean Error: {np.min(err_mag):.2f} μT")
        print(matched_df[['Time_Obj', 'Mag_Error', 'True_X', 'True_Y']].head())
    else:
        print("\nNo overlapping timestamps were found between the Static and Continuous datasets.")

if __name__ == "__main__":
    main()
