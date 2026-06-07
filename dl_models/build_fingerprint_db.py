"""
Build the static fingerprint database for one building (default: IT Engineering).

Each output row = one node-visit fingerprint:
  - Ground-truth (X, Y) taken from the MAGNETIC file (the WiFi file's own coords
    are a bogus sequential counter -- see memory: wifi-coords-are-bogus).
  - Rotation-invariant magnetic features (orientation-independent, so the
    fingerprint is a property of the ENVIRONMENT, not the walking direction):
        |M|        magnitude
        M_vert     component along gravity (world-vertical), signed
        M_horiz    horizontal magnitude
        dip        magnetic inclination angle (rad)
    Gravity direction is estimated per-frame from the accelerometer (valid here
    because static recordings are held still, so Acc ~= gravity).
    Each is aggregated over the node's ~119 readings as mean + std (8 features).
  - WiFi RSS vector indexed by a global BSSID vocabulary (missing AP = -100 dBm).

WiFi files are matched to mag files by filename (identical except IMU_/WiFi_ prefix).

Outputs (to Datasets/fingerprint_db/<building_slug>/):
  - nodes.csv          : metadata + 8 mag features + N_ap + WiFi RSS columns
  - bssid_vocab.json   : ordered list of BSSIDs (defines the WiFi vector layout)
  - coverage.png       : node coverage scatter (sanity check)
"""
import os
import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Known phone tokens as they appear in folder names
PHONE_TOKENS = ["S9+", "A8", "G7", "S8", "LG G6", "LG Q6", "G6", "Q6"]
WIFI_FLOOR = -100.0  # RSS sentinel for an AP that was not seen at a node


def parse_meta_from_path(path, building):
    """Extract (mode, scenario, phone, user) from the folder components after the building name."""
    parts = os.path.normpath(path).split(os.sep)
    try:
        b_idx = parts.index(building)
        comps = parts[b_idx + 1:-1]  # folders between building and filename
    except ValueError:
        comps = []
    mode = comps[0] if comps else "Unknown"
    scenario = next((c for c in comps if "Scenario" in c), "NA")
    phone = next((c for c in comps if c in PHONE_TOKENS), "Unknown")
    user = next((c for c in comps if c.replace(" ", "").lower().startswith("user")), "Unknown")
    return mode, scenario, phone, user


def mag_rotation_invariant(df):
    """Compute orientation-invariant magnetic features per row, return mean+std aggregates."""
    M = df[["Mag_x", "Mag_y", "Mag_z"]].to_numpy(dtype=float)
    A = df[["Acc_x", "Acc_y", "Acc_z"]].to_numpy(dtype=float)

    M_norm = np.linalg.norm(M, axis=1)
    a_norm = np.linalg.norm(A, axis=1)
    a_norm[a_norm == 0] = np.nan
    g_hat = A / a_norm[:, None]                      # unit gravity direction (body frame)

    M_vert = np.sum(M * g_hat, axis=1)               # signed vertical component
    horiz_sq = np.maximum(M_norm ** 2 - M_vert ** 2, 0.0)
    M_horiz = np.sqrt(horiz_sq)
    dip = np.arctan2(M_vert, M_horiz)                # inclination angle (rad)

    feats = {
        "magN": M_norm, "magV": M_vert, "magH": M_horiz, "dip": dip,
    }
    out = {}
    for name, arr in feats.items():
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[f"{name}_mean"], out[f"{name}_std"] = np.nan, np.nan
        else:
            out[f"{name}_mean"], out[f"{name}_std"] = float(arr.mean()), float(arr.std())
    return out


def wifi_basename_for(mag_basename):
    """Map an IMU_/mag filename to its matching WiFi filename."""
    if mag_basename.startswith("IMU_"):
        return "WiFi_" + mag_basename[len("IMU_"):]
    return None


def main(building="IT Engineering"):
    base = ".."
    mag_root = os.path.join(base, "Datasets", "Magnetic field dataset", "Static Data", building)
    wifi_root = os.path.join(base, "Datasets", "WiFi dataset", building)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", building).strip("_").lower()
    out_dir = os.path.join(base, "Datasets", "fingerprint_db", slug)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 64)
    print(f"Building fingerprint DB for: {building}")
    print("=" * 64)

    mag_files = glob.glob(os.path.join(mag_root, "**", "*.csv"), recursive=True)
    wifi_files = glob.glob(os.path.join(wifi_root, "**", "*.csv"), recursive=True)
    print(f"Static mag files: {len(mag_files)}")
    print(f"WiFi files:       {len(wifi_files)}")

    # Index WiFi files by basename for 1:1 matching
    wifi_by_name = {os.path.basename(p): p for p in wifi_files}

    # ---- Pass 1: parse every mag file + its matched WiFi scan ----
    records = []          # one dict per node-visit (meta + mag feats)
    wifi_scans = []       # parallel list: dict {bssid: rss} or None
    bssid_counter = {}    # global AP frequency

    n_paired = 0
    for i, mf in enumerate(mag_files):
        if (i + 1) % 250 == 0:
            print(f"  ...{i+1}/{len(mag_files)} mag files")
        try:
            dfm = pd.read_csv(mf)
        except Exception:
            continue
        if "X-cord" not in dfm.columns or "Mag_x" not in dfm.columns:
            continue
        dfm = dfm.dropna(subset=["X-cord", "Y-cord", "Mag_x", "Mag_y", "Mag_z",
                                 "Acc_x", "Acc_y", "Acc_z"])
        if len(dfm) == 0:
            continue

        x = float(dfm["X-cord"].iloc[0])
        y = float(dfm["Y-cord"].iloc[0])
        mode, scenario, phone, user = parse_meta_from_path(mf, building)
        rec = {
            "x": x, "y": y, "mode": mode, "scenario": scenario,
            "phone": phone, "user": user, "n_mag_rows": len(dfm),
            "file": os.path.basename(mf),
        }
        rec.update(mag_rotation_invariant(dfm))

        # Matched WiFi scan
        scan = None
        wbn = wifi_basename_for(os.path.basename(mf))
        if wbn and wbn in wifi_by_name:
            try:
                dfw = pd.read_excel(wifi_by_name[wbn])
                dfw.columns = [str(c).strip() for c in dfw.columns]
                if "BSSID" in dfw.columns and "RSS" in dfw.columns:
                    dfw = dfw.dropna(subset=["BSSID", "RSS"])
                    # keep strongest RSS if an AP appears twice
                    scan = dfw.groupby("BSSID")["RSS"].max().astype(float).to_dict()
                    for b in scan:
                        bssid_counter[b] = bssid_counter.get(b, 0) + 1
                    n_paired += 1
            except Exception:
                scan = None

        records.append(rec)
        wifi_scans.append(scan)

    print(f"\nParsed node-visits: {len(records)} | WiFi-paired: {n_paired}")

    # ---- Build BSSID vocabulary (sorted by frequency desc, then name) ----
    bssid_vocab = sorted(bssid_counter.keys(), key=lambda b: (-bssid_counter[b], b))
    print(f"Unique BSSIDs (AP vocabulary): {len(bssid_vocab)}")
    bssid_index = {b: j for j, b in enumerate(bssid_vocab)}

    # ---- Assemble final table ----
    meta_df = pd.DataFrame(records)
    n_ap = [0 if s is None else len(s) for s in wifi_scans]
    meta_df["n_ap"] = n_ap
    meta_df["has_wifi"] = [s is not None for s in wifi_scans]

    wifi_mat = np.full((len(records), len(bssid_vocab)), WIFI_FLOOR, dtype=np.float32)
    for r, scan in enumerate(wifi_scans):
        if scan is None:
            continue
        for b, rss in scan.items():
            wifi_mat[r, bssid_index[b]] = rss
    wifi_cols = [f"AP_{j}" for j in range(len(bssid_vocab))]
    wifi_df = pd.DataFrame(wifi_mat, columns=wifi_cols)

    full = pd.concat([meta_df.reset_index(drop=True), wifi_df], axis=1)

    # ---- Save ----
    nodes_path = os.path.join(out_dir, "nodes.csv")
    full.to_csv(nodes_path, index=False)
    with open(os.path.join(out_dir, "bssid_vocab.json"), "w") as f:
        json.dump({"bssid_vocab": bssid_vocab,
                   "wifi_floor": WIFI_FLOOR,
                   "ap_columns": wifi_cols}, f, indent=2)

    # ---- Coverage / sanity summary ----
    uniq_nodes = meta_df[["x", "y"]].round(1).drop_duplicates()
    print("\n" + "-" * 64)
    print("SUMMARY")
    print("-" * 64)
    print(f"Node-visits (rows):     {len(full)}")
    print(f"Unique (X,Y) nodes:     {len(uniq_nodes)}")
    print(f"X range: [{meta_df['x'].min():.1f}, {meta_df['x'].max():.1f}]  "
          f"Y range: [{meta_df['y'].min():.1f}, {meta_df['y'].max():.1f}]")
    print(f"WiFi-paired visits:     {meta_df['has_wifi'].sum()} / {len(meta_df)} "
          f"({100*meta_df['has_wifi'].mean():.1f}%)")
    print(f"APs/scan (paired):      median {np.median([a for a in n_ap if a>0]):.0f}")
    print(f"Phones:    {sorted(meta_df['phone'].unique())}")
    print(f"Users:     {sorted(meta_df['user'].unique())}")
    print(f"Modes:     {sorted(meta_df['mode'].unique())}")
    print(f"Scenarios: {sorted(meta_df['scenario'].unique())}")
    print(f"\nSaved -> {nodes_path}")

    # ---- Coverage plot: visits per unique node ----
    fig, ax = plt.subplots(figsize=(12, 4))
    cnt = meta_df.groupby([meta_df["x"].round(1), meta_df["y"].round(1)]).size().reset_index(name="visits")
    sc = ax.scatter(cnt["x"], cnt["y"], c=cnt["visits"], cmap="viridis", s=60, edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="# visits (phones x users x scenarios)")
    ax.set_title(f"{building} static fingerprint coverage ({len(uniq_nodes)} unique nodes)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "coverage.png"), dpi=200, bbox_inches="tight")
    print(f"Saved -> {os.path.join(out_dir, 'coverage.png')}")


if __name__ == "__main__":
    main()
