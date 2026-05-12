import os
import glob

def check_wifi_continuous_overlap():
    # Load Continuous IMU File timestamps (filename suffixes)
    cont_pattern = os.path.join('Datasets', 'Magnetic field dataset', 'Continuous Data', 'BE Building', 'Navigation', 'Scenario 1', '*', '*', 'IMU_*.csv')
    cont_files = glob.glob(cont_pattern)
    
    cont_timestamps = set()
    for f in cont_files:
        # e.g. IMU_BE Building_Scenario 1_User 2 (M-174cm)_Navigation_2021.05.19 173743.csv
        ts = f.split('_')[-1].replace('.csv', '')
        cont_timestamps.add(ts)
        
    print(f"Total Continuous IMU Sessions: {len(cont_timestamps)}")
    
    # Load all Wi-Fi files for Scenario-1
    wifi_pattern = os.path.join('Datasets', 'WiFi dataset', 'BE Engineering', 'Navigation', 'Scenario-1', '*', '*', 'WiFi_*.csv')
    wifi_files = glob.glob(wifi_pattern)
    
    wifi_timestamps = set()
    for f in wifi_files:
        ts = f.split('_')[-1].replace('.csv', '')
        wifi_timestamps.add(ts)
        
    print(f"Total Wi-Fi Sessions: {len(wifi_timestamps)}")
    
    # Check overlap
    overlap = cont_timestamps.intersection(wifi_timestamps)
    print(f"Overlap count (Matches between Continuous IMU and Wi-Fi): {len(overlap)}")
    if overlap:
        print("Matching Timestamps:", overlap)
        
if __name__ == "__main__":
    check_wifi_continuous_overlap()
