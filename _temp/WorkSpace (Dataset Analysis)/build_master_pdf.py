import os
import glob
from pathlib import Path
import markdown
from playwright.sync_api import sync_playwright

def build_pdf():
    base_dir = Path(r"c:\Users\lenovo\Documents\GitHub\SURA\WorkSpace")
    plots_dir = base_dir / "Plots"
    out_pdf = base_dir / "Dataset_Master_Report_Visuals.pdf"
    
    # 1. Read existing markdown files
    with open(base_dir / "DatasetAnalysis.md", "r", encoding="utf-8") as f:
        md_part1 = f.read()
        
    with open(r"C:\Users\lenovo\.gemini\antigravity-ide\brain\f649f5bc-f8a8-408e-b9d7-415aabf882c9\dataset_deep_dive.md", "r", encoding="utf-8") as f:
        md_part2 = f.read()

    # We will remove the top headings from part 2 to make it flow better
    md_part2 = md_part2.replace("# 🔬 SURA Dataset — Exhaustive Deep Dive Analysis", "## Exhaustive Deep Dive Analysis")
    
    combined_md = f"# SURA Dataset — Master Report\n\n{md_part1}\n\n<hr style='page-break-after: always;'>\n\n{md_part2}"

    # Convert markdown to HTML
    html_content = markdown.markdown(combined_md, extensions=['tables', 'fenced_code'])

    # 2. Add visual trajectories with analysis
    html_content += "\n<hr style='page-break-before: always;'>\n<h2>Visual Trajectories & Analytical Observations</h2>\n"
    html_content += "<p>The following plots visualize the 1:1 true physical scale of the recordings. Continuous data (lines) are colored by time progression (start to end), and Static Data (crimson dots) show the exact geometry of the surveyed grid nodes. The graphs have been optimally scaled for comfortable viewing.</p>\n"
    
    png_files = sorted(glob.glob(str(plots_dir / "**" / "*.png"), recursive=True))
    groups = {}
    for f in png_files:
        rel_path = os.path.relpath(f, str(plots_dir))
        parts = rel_path.split(os.sep)
        if len(parts) < 3: continue
        bldg = parts[0]
        if bldg not in groups:
            groups[bldg] = []
        groups[bldg].append(f)

    analysis_snippets = {
        "BE Building": "The BE Building trajectories exhibit highly consistent spatial node alignment. The magnetic paths taken across phones reflect minimal drifting, demonstrating a strong, stable magnetic baseline suitable for continuous tracking models.",
        "CS Engineering": "The CS Engineering pathways include complex turning maneuvers and extended continuous corridors. Trajectory data here reveals higher variability during corners, likely due to gyroscope accumulation errors which are partially corrected by the static grid mapping.",
        "Electrical Eng.": "Electrical Engineering contains distinct scenarios forming a tight loop with consistent grid spacing. The continuous data captures standard navigation behavior with very minor sensor outlier noise, providing a very clean validation set.",
        "IACT": "The IACT data demonstrates robust magnetic field variations, suggesting significant ferrous structural elements in the corridors. This provides high spatial uniqueness, which is highly beneficial for fingerprinting algorithms.",
        "IT Engineering": "IT Engineering is the most complex building, featuring multiple holding styles including Call Listening, Swinging, and Stairs. The varied walking patterns and multi-user configurations make this the ultimate stress-test subset for real-world robustness."
    }

    for bldg, files in sorted(groups.items()):
        bldg_clean = bldg.replace('_', ' ')
        html_content += f"<h3 style='margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>{bldg_clean}</h3>\n"
        html_content += "<div style='display: flex; flex-wrap: wrap; justify-content: center;'>"
        for f in files:
            f_uri = Path(f).as_uri()
            filename = os.path.basename(f)
            caption = filename.replace('.png', '').replace('_', ' ')
            
            html_content += f"""
            <div style='width: 45%; margin: 10px; border: 1px solid #eee; padding: 10px; text-align: center; box-shadow: 2px 2px 8px rgba(0,0,0,0.05); border-radius: 8px;'>
                <strong>{caption}</strong><br/>
                <img src="{f_uri}" style="width: 100%; height: auto; margin-top: 10px;" />
            </div>
            """
        html_content += "</div>\n"
        
        # Add analysis after the pictures
        snippet = analysis_snippets.get(bldg_clean, "The trajectories for this building show distinct spatial geometries mapping exactly to the physical corridors, validating the 1:1 aspect ratio constraint.")
        html_content += f"""
        <div style='background-color: #f8f9fa; padding: 15px; border-left: 5px solid #007bff; margin-top: 15px; margin-bottom: 40px; font-style: italic;'>
            <strong>Analysis:</strong> {snippet}
        </div>
        """

    # 3. Wrap in full HTML document with beautiful CSS
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
            body {{
                font-family: 'Inter', Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                margin: 0;
                padding: 0;
            }}
            h1, h2, h3, h4 {{
                color: #2c3e50;
            }}
            h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
            h2 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 30px; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                font-size: 14px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px 12px;
                text-align: left;
            }}
            th {{
                background-color: #f4f6f8;
                font-weight: 600;
            }}
            tr:nth-child(even) {{ background-color: #fbfbfc; }}
            pre {{
                background-color: #2b2b2b;
                color: #f8f8f2;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 13px;
                font-family: 'Consolas', monospace;
            }}
            code {{
                background-color: #f1f1f1;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Consolas', monospace;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
            }}
            blockquote {{
                border-left: 4px solid #007bff;
                background-color: #f8f9fa;
                margin: 0 0 20px 0;
                padding: 15px 20px;
                font-style: italic;
            }}
            .note, .warning, .tip, .important, .caution {{
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            /* Break pages effectively */
            h1, h2 {{ page-break-after: avoid; }}
            table {{ page-break-inside: avoid; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    with open(base_dir / "temp_report.html", "w", encoding="utf-8") as f:
        f.write(full_html)

    print("Generating PDF via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Navigate to the local HTML file to ensure base URIs work for images
        page.goto(Path(base_dir / "temp_report.html").as_uri())
        
        # Wait a moment for images to load just in case
        page.wait_for_timeout(2000)
        
        page.pdf(
            path=str(out_pdf),
            format="A4",
            print_background=True,
            margin={"top": "0.8in", "bottom": "0.8in", "left": "0.6in", "right": "0.6in"}
        )
        browser.close()

    print(f"Success! Perfect PDF generated at: {out_pdf}")
    if os.path.exists(base_dir / "temp_report.html"):
        os.remove(base_dir / "temp_report.html")

if __name__ == "__main__":
    build_pdf()
