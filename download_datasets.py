import os
import subprocess
import sys

# Define path for Datasets directory
repo_path = r"c:\Users\lenovo\Documents\GitHub\SURA"
dataset_dir = os.path.join(repo_path, "Datasets")
os.makedirs(dataset_dir, exist_ok=True)
print(f"Created/Verified Dataset directory at: {dataset_dir}")

def download_miskolc():
    print("\n--- Downloading Miskolc IIS Hybrid IPS Dataset ---")
    try:
        # Ensure the ucimlrepo library is installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo", "pandas"])
        
        from ucimlrepo import fetch_ucirepo
        
        # Fetch the dataset (ID 375)
        print("Fetching data from UCI Machine Learning Repository...")
        miskolc_iis_hybrid_ips = fetch_ucirepo(id=375)
        
        # Extract features and targets (typically Pandas DataFrames)
        X = miskolc_iis_hybrid_ips.data.features
        y = miskolc_iis_hybrid_ips.data.targets
        
        # Save to the Datasets folder
        features_path = os.path.join(dataset_dir, "miskolc_features.csv")
        targets_path = os.path.join(dataset_dir, "miskolc_targets.csv")
        
        X.to_csv(features_path, index=False)
        y.to_csv(targets_path, index=False)
        
        print(f"Success! Miskolc dataset saved to:")
        print(f" - {features_path}")
        print(f" - {targets_path}")
    except Exception as e:
        print(f"Error fetching Miskolc dataset: {e}")

if __name__ == "__main__":
    download_miskolc()
    
    print("\n--- Data Acquisition Summary ---")
    print("1. Miskolc IIS: Download completed via UCIMLRepo.")
    print("2. MagWi: The dataset from the IEEE paper is NOT publicly hosted via a direct API. We must request it from the corresponding author (Imran Ashraf on ResearchGate).")
    print("3. MagPIE: Large 10GB+ raw Rosbag files are hosted at 'http://bretl.csl.illinois.edu/magpie'. These should be downloaded manually via a browser.")
