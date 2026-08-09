import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# Use a clean style
plt.style.use('seaborn-v0_8-darkgrid')

def generate_pdfs():
    base_dir = Path(r"c:\Users\lenovo\Documents\GitHub\SURA\Datasets\Magnetic field dataset")
    out_dir = Path(r"c:\Users\lenovo\Documents\GitHub\SURA\WorkSpace\Plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use os.walk to find all CSVs
    groups = {} 
    for root, dirs, files in os.walk(base_dir):
        if 'COEX' in root: continue 
        csvs = [f for f in files if f.endswith('.csv')]
        if csvs:
            groups[root] = [os.path.join(root, f) for f in csvs]
            
    # Group the directories by building
    building_groups = {}
    for root, csv_files in groups.items():
        rel_path = os.path.relpath(root, base_dir)
        parts = rel_path.split(os.sep)
        bldg = parts[1].strip() if len(parts) > 1 else 'Unknown_Building'
        if bldg not in building_groups:
            building_groups[bldg] = []
        building_groups[bldg].append((root, csv_files, parts))

    # All trajectories PDF
    all_pdf_path = out_dir / "All_Trajectories.pdf"
    
    print(f"Generating PDFs. Total buildings to process: {len(building_groups)}")
    
    with PdfPages(all_pdf_path) as all_pdf:
        for bldg, items in sorted(building_groups.items()):
            bldg_clean = bldg.replace(' ', '_')
            bldg_pdf_path = out_dir / f"{bldg_clean}_Trajectories.pdf"
            
            with PdfPages(bldg_pdf_path) as bldg_pdf:
                # Sort items so they appear logically
                items.sort(key=lambda x: x[0])
                
                for root, csv_files, parts in items:
                    dt = parts[0].strip()
                    sub_parts = parts[2:]
                    
                    if len(sub_parts) == 3:
                        scen, phone, user = sub_parts
                    elif len(sub_parts) == 2:
                        scen, phone, user = sub_parts[0], sub_parts[1], "Unknown_User"
                    elif len(sub_parts) == 1:
                        scen, phone, user = sub_parts[0], "Unknown_Phone", "Unknown_User"
                    else:
                        scen, phone, user = "Unknown_Scenario", "Unknown_Phone", "Unknown_User"
                        
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_title(f"Building: {bldg} | Type: {dt}\nScenario: {scen} | Phone: {phone} | User: {user}", fontsize=14, pad=15)
                    ax.set_xlabel("X Coordinate (meters)", fontsize=12)
                    ax.set_ylabel("Y Coordinate (meters)", fontsize=12)
                    
                    # Force uniform scale! 1 unit on X = 1 unit on Y
                    ax.set_aspect('equal', adjustable='datalim')
                    
                    has_data = False
                    all_x, all_y = [], []
                    scmappable = None
                    
                    for f in csv_files:
                        try:
                            df = pd.read_csv(f, usecols=['X-cord', 'Y-cord'], on_bad_lines='skip')
                            df['X-cord'] = pd.to_numeric(df['X-cord'], errors='coerce')
                            df['Y-cord'] = pd.to_numeric(df['Y-cord'], errors='coerce')
                            df = df.dropna(subset=['X-cord', 'Y-cord'])
                            if not df.empty:
                                has_data = True
                                if 'Continuous' in dt:
                                    idx = np.linspace(0, 1, len(df))
                                    scmappable = ax.scatter(df['X-cord'], df['Y-cord'], c=idx, cmap='viridis', s=15, alpha=0.8, edgecolor='none')
                                    ax.plot(df['X-cord'], df['Y-cord'], color='gray', linewidth=0.5, alpha=0.5)
                                else:
                                    all_x.extend(df['X-cord'].values)
                                    all_y.extend(df['Y-cord'].values)
                        except Exception:
                            pass
                            
                    if has_data:
                        if 'Static' in dt:
                            ax.scatter(all_x, all_y, color='crimson', s=40, alpha=0.6, edgecolor='white', linewidth=0.5)
                        elif 'Continuous' in dt and scmappable is not None:
                            cbar = fig.colorbar(scmappable, ax=ax, fraction=0.046, pad=0.04)
                            cbar.set_label('Trajectory Progression (Start ➔ End)', rotation=270, labelpad=15)
                        
                        plt.tight_layout()
                        bldg_pdf.savefig(fig)
                        all_pdf.savefig(fig)
                    
                    plt.close(fig)
            print(f"Generated PDF: {bldg_pdf_path}")
            
    print(f"Generated Master PDF: {all_pdf_path}")

if __name__ == "__main__":
    generate_pdfs()
