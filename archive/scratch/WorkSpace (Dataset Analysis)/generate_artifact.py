import os
import glob
from pathlib import Path

def generate_artifact():
    plots_dir = Path(r"c:\Users\lenovo\Documents\GitHub\SURA\WorkSpace\Plots")
    artifact_path = Path(r"C:\Users\lenovo\.gemini\antigravity-ide\brain\f649f5bc-f8a8-408e-b9d7-415aabf882c9\trajectories.md")
    
    # Structure: Plots / Building / DataType / Scenario_Phone_User.png
    # Find all PNGs recursively
    png_files = sorted(glob.glob(str(plots_dir / "**" / "*.png"), recursive=True))
    
    # Group by Building and DataType
    groups = {}
    for f in png_files:
        rel_path = os.path.relpath(f, str(plots_dir))
        parts = rel_path.split(os.sep)
        if len(parts) < 3:
            continue
            
        bldg = parts[0]
        dt = parts[1]
        
        bldg_clean = bldg.replace('_', ' ')
        dt_clean = dt.replace('_', ' ')
        
        if bldg_clean not in groups:
            groups[bldg_clean] = {}
        if dt_clean not in groups[bldg_clean]:
            groups[bldg_clean][dt_clean] = []
            
        groups[bldg_clean][dt_clean].append(f)
        
    with open(artifact_path, "w", encoding="utf-8") as out:
        out.write("# 📍 Dataset Trajectories & Grids\n\n")
        out.write("Below are the plotted coordinate spaces for each scenario, phone, and user combination. **Continuous Data** shows trajectories (colored by progression from start to end) while **Static Data** shows the precise grid node locations recorded.\n\n")
        out.write("> [!NOTE]\n> **COEX** is omitted as its files contain no valid coordinates.\n\n")
        
        for bldg, dts in sorted(groups.items()):
            out.write(f"## {bldg}\n\n")
            for dt, files in sorted(dts.items()):
                out.write(f"### {dt}\n\n")
                out.write("````carousel\n")
                for i, f in enumerate(files):
                    filename = os.path.basename(f)
                    caption = filename.replace('.png', '').replace('_', ' ')
                    
                    # Ensure path uses forward slashes
                    f_fwd = f.replace('\\', '/')
                    out.write(f"![{caption}]({f_fwd})\n")
                    if i < len(files) - 1:
                        out.write("<!-- slide -->\n")
                out.write("````\n\n")
                
    print(f"Generated artifact at {artifact_path} with {len(png_files)} images.")

if __name__ == "__main__":
    generate_artifact()
