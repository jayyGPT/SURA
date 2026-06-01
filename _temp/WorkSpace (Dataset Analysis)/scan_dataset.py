"""
Exhaustive Dataset Scanner for SURA Indoor Positioning Datasets.
Walks every file in the Magnetic Field (Static + Continuous) and WiFi directories.
Collects: datapoints, nulls, coordinate ranges, time ranges, sensor stats,
trajectory nodes, file counts, file sizes, and per-file row distributions.
"""

import os
import csv
import json
import sys
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(r"c:\Users\lenovo\Documents\GitHub\SURA\Datasets")
MAG_BASE = BASE / "Magnetic field dataset"
WIFI_BASE = BASE / "WiFi dataset"

SENSOR_COLS = [
    "Time", "X-cord", "Y-cord",
    "Mag_x", "Mag_y", "Mag_z",
    "Acc_x", "Acc_y", "Acc_z",
    "Gyro_x", "Gyro_y", "Gyro_z",
    "Orn_x", "Orn_y", "Orn_z",
    "Pressure"
]

def safe_float(val):
    """Try to parse a float, return None if fails."""
    if val is None:
        return None
    val = str(val).strip().rstrip(",")
    if val == "" or val.lower() == "nan" or val.lower() == "null":
        return None
    try:
        return float(val)
    except ValueError:
        return None

def analyze_csv(filepath):
    """Analyze a single magnetic field CSV file exhaustively."""
    result = {
        "filepath": str(filepath),
        "filename": os.path.basename(filepath),
        "filesize_bytes": os.path.getsize(filepath),
        "total_rows": 0,
        "null_counts": {},       # col_name -> count of nulls/empty
        "coord_pairs": set(),    # unique (x, y) pairs
        "x_range": [float('inf'), float('-inf')],
        "y_range": [float('inf'), float('-inf')],
        "time_first": None,
        "time_last": None,
        "sensor_ranges": {},     # col_name -> [min, max]
        "has_header": False,
        "actual_columns": [],
        "parse_errors": 0,
    }

    # Initialize sensor ranges and null counts
    for col in SENSOR_COLS:
        result["sensor_ranges"][col] = [float('inf'), float('-inf')]
        result["null_counts"][col] = 0

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                # Clean header - strip whitespace and trailing commas
                header = [h.strip().rstrip(",") for h in header if h.strip().rstrip(",")]
                result["actual_columns"] = header
                result["has_header"] = any(col in header for col in ["Time", "X-cord", "Mag_x"])

            if not result["has_header"]:
                # If no header, reset and treat first row as data
                f.seek(0)
                reader = csv.reader(f)
                header = SENSOR_COLS  # assume standard columns

            row_count = 0
            for row in reader:
                if not row or all(c.strip() == "" for c in row):
                    continue
                row_count += 1

                # Map row values to column names
                for i, col in enumerate(SENSOR_COLS):
                    if i >= len(row):
                        result["null_counts"][col] += 1
                        continue

                    val_str = row[i].strip().rstrip(",")

                    if col == "Time":
                        if val_str == "" or val_str.lower() in ("nan", "null", "none"):
                            result["null_counts"][col] += 1
                        else:
                            if result["time_first"] is None:
                                result["time_first"] = val_str
                            result["time_last"] = val_str
                        continue

                    val = safe_float(val_str)
                    if val is None:
                        result["null_counts"][col] += 1
                        continue

                    # Update ranges
                    if val < result["sensor_ranges"][col][0]:
                        result["sensor_ranges"][col][0] = val
                    if val > result["sensor_ranges"][col][1]:
                        result["sensor_ranges"][col][1] = val

                    # Track coordinates
                    if col == "X-cord":
                        if val < result["x_range"][0]: result["x_range"][0] = val
                        if val > result["x_range"][1]: result["x_range"][1] = val
                    elif col == "Y-cord":
                        if val < result["y_range"][0]: result["y_range"][0] = val
                        if val > result["y_range"][1]: result["y_range"][1] = val

                # Track unique coordinate pairs
                x_val = safe_float(row[1].strip().rstrip(",")) if len(row) > 1 else None
                y_val = safe_float(row[2].strip().rstrip(",")) if len(row) > 2 else None
                if x_val is not None and y_val is not None:
                    result["coord_pairs"].add((x_val, y_val))

            result["total_rows"] = row_count

    except Exception as e:
        result["parse_errors"] += 1
        result["error"] = str(e)

    # Convert set to count for JSON serialization
    result["unique_coords"] = len(result["coord_pairs"])
    result["coord_pairs_list"] = sorted(list(result["coord_pairs"]))
    del result["coord_pairs"]

    # Clean up infinity values
    if result["x_range"][0] == float('inf'):
        result["x_range"] = [None, None]
    if result["y_range"][0] == float('inf'):
        result["y_range"] = [None, None]

    for col in SENSOR_COLS:
        if result["sensor_ranges"][col][0] == float('inf'):
            result["sensor_ranges"][col] = [None, None]

    return result


def parse_path_parts(filepath, data_type):
    """Extract building, scenario, phone, user from filesystem path."""
    parts = Path(filepath).parts
    # Find the data_type marker (Static Data / Continuous Data)
    info = {
        "data_type": data_type,
        "building": "Unknown",
        "mode": "Unknown",
        "scenario": "Unknown",
        "phone": "Unknown",
        "user": "Unknown",
    }

    try:
        if data_type in ("Static", "Continuous"):
            # Path: .../Static Data/Building/Mode/Scenario/Phone/User/file.csv
            # or for some: .../Static Data/Building/Navigation/Scenario/Phone/User/file.csv
            marker = "Static Data" if data_type == "Static" else "Continuous Data"
            idx = parts.index(marker)
            if idx + 1 < len(parts): info["building"] = parts[idx + 1]
            if idx + 2 < len(parts): info["mode"] = parts[idx + 2]
            if idx + 3 < len(parts): info["scenario"] = parts[idx + 3]
            if idx + 4 < len(parts): info["phone"] = parts[idx + 4]
            if idx + 5 < len(parts): info["user"] = parts[idx + 5]
    except (ValueError, IndexError):
        pass

    return info


def walk_magnetic_data():
    """Walk all magnetic field CSV files and analyze them."""
    all_results = []

    for data_type_folder, data_type_label in [("Static Data", "Static"), ("Continuous Data", "Continuous")]:
        data_dir = MAG_BASE / data_type_folder
        if not data_dir.exists():
            print(f"  [SKIP] {data_dir} does not exist")
            continue

        csv_count = 0
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.lower().endswith(".csv"):
                    fpath = os.path.join(root, fname)
                    csv_count += 1
                    if csv_count % 100 == 0:
                        print(f"  Processed {csv_count} CSVs in {data_type_label}...", flush=True)

                    analysis = analyze_csv(fpath)
                    path_info = parse_path_parts(fpath, data_type_label)
                    analysis.update(path_info)
                    all_results.append(analysis)

        print(f"  Finished {data_type_label}: {csv_count} CSV files analyzed.", flush=True)

    return all_results


def walk_wifi_data():
    """Walk WiFi dataset - these are binary xls files, so we gather file metadata."""
    wifi_results = []

    for root, dirs, files in os.walk(WIFI_BASE):
        for fname in files:
            fpath = os.path.join(root, fname)
            parts = Path(fpath).parts

            info = {
                "filepath": str(fpath),
                "filename": fname,
                "filesize_bytes": os.path.getsize(fpath),
                "extension": os.path.splitext(fname)[1].lower(),
                "building": "Unknown",
                "mode": "Unknown",
                "scenario": "Unknown",
                "phone": "Unknown",
                "user": "Unknown",
            }

            # Try to parse path: WiFi dataset/Building/[Navigation/]Scenario/Phone/User/file
            # Or for COEX: WiFi dataset/COEX/Phone/User/file (no Navigation/Scenario)
            try:
                idx = parts.index("WiFi dataset")
                remaining = parts[idx+1:]
                # Check various structures
                if len(remaining) >= 1: info["building"] = remaining[0]

                # COEX has no Navigation/Scenario level - directly Phone/User
                if info["building"] == "COEX":
                    if len(remaining) >= 2: info["phone"] = remaining[1]
                    if len(remaining) >= 3: info["user"] = remaining[2]
                    info["mode"] = "Navigation"
                    info["scenario"] = "Scenario-1"
                else:
                    if len(remaining) >= 2: info["mode"] = remaining[1]
                    if len(remaining) >= 3: info["scenario"] = remaining[2]
                    if len(remaining) >= 4: info["phone"] = remaining[3]
                    if len(remaining) >= 5: info["user"] = remaining[4]
            except (ValueError, IndexError):
                pass

            wifi_results.append(info)

    return wifi_results


def aggregate_magnetic(results):
    """Aggregate magnetic data results by multiple levels."""
    # Grand totals
    grand = {
        "total_files": len(results),
        "total_rows": sum(r["total_rows"] for r in results),
        "total_filesize_bytes": sum(r["filesize_bytes"] for r in results),
        "total_null_counts": defaultdict(int),
        "all_unique_coords": set(),
        "global_x_range": [float('inf'), float('-inf')],
        "global_y_range": [float('inf'), float('-inf')],
        "global_sensor_ranges": {col: [float('inf'), float('-inf')] for col in SENSOR_COLS},
        "rows_per_file": [],
        "time_range_global_first": None,
        "time_range_global_last": None,
    }

    # Per building
    by_building = defaultdict(lambda: {
        "total_files": 0, "total_rows": 0, "total_filesize_bytes": 0,
        "null_counts": defaultdict(int), "unique_coords": set(),
        "x_range": [float('inf'), float('-inf')], "y_range": [float('inf'), float('-inf')],
        "rows_per_file": [], "data_types": set(), "scenarios": set(),
        "phones": set(), "users": set(), "modes": set(),
        "sensor_ranges": {col: [float('inf'), float('-inf')] for col in SENSOR_COLS},
        "time_first": None, "time_last": None,
    })

    # Per data_type
    by_data_type = defaultdict(lambda: {
        "total_files": 0, "total_rows": 0, "total_filesize_bytes": 0,
        "null_counts": defaultdict(int), "unique_coords": set(),
        "rows_per_file": [],
    })

    # Per building+data_type+scenario
    by_bds = defaultdict(lambda: {
        "total_files": 0, "total_rows": 0, "null_counts": defaultdict(int),
        "unique_coords": set(), "phones": set(), "users": set(),
        "rows_per_file": [],
        "x_range": [float('inf'), float('-inf')], "y_range": [float('inf'), float('-inf')],
        "time_first": None, "time_last": None,
    })

    # Per building+data_type+scenario+phone
    by_bdsp = defaultdict(lambda: {
        "total_files": 0, "total_rows": 0, "null_counts": defaultdict(int),
        "unique_coords": set(), "users": set(), "rows_per_file": [],
    })

    for r in results:
        b = r["building"]
        d = r["data_type"]
        s = r["scenario"]
        p = r["phone"]
        u = r["user"]
        m = r["mode"]
        rows = r["total_rows"]

        # Grand
        grand["total_null_counts"] = dict(grand["total_null_counts"])
        for col in SENSOR_COLS:
            grand["total_null_counts"][col] = grand["total_null_counts"].get(col, 0) + r["null_counts"].get(col, 0)
        for coord in r.get("coord_pairs_list", []):
            grand["all_unique_coords"].add(tuple(coord))
        grand["rows_per_file"].append(rows)

        # Update global ranges
        if r["x_range"][0] is not None:
            if r["x_range"][0] < grand["global_x_range"][0]: grand["global_x_range"][0] = r["x_range"][0]
            if r["x_range"][1] > grand["global_x_range"][1]: grand["global_x_range"][1] = r["x_range"][1]
        if r["y_range"][0] is not None:
            if r["y_range"][0] < grand["global_y_range"][0]: grand["global_y_range"][0] = r["y_range"][0]
            if r["y_range"][1] > grand["global_y_range"][1]: grand["global_y_range"][1] = r["y_range"][1]

        for col in SENSOR_COLS:
            sr = r["sensor_ranges"].get(col, [None, None])
            if sr[0] is not None:
                if sr[0] < grand["global_sensor_ranges"][col][0]:
                    grand["global_sensor_ranges"][col][0] = sr[0]
                if sr[1] > grand["global_sensor_ranges"][col][1]:
                    grand["global_sensor_ranges"][col][1] = sr[1]

        if r["time_first"]:
            if grand["time_range_global_first"] is None or r["time_first"] < grand["time_range_global_first"]:
                grand["time_range_global_first"] = r["time_first"]
        if r["time_last"]:
            if grand["time_range_global_last"] is None or r["time_last"] > grand["time_range_global_last"]:
                grand["time_range_global_last"] = r["time_last"]

        # Per building
        bb = by_building[b]
        bb["total_files"] += 1
        bb["total_rows"] += rows
        bb["total_filesize_bytes"] += r["filesize_bytes"]
        for col in SENSOR_COLS:
            bb["null_counts"][col] += r["null_counts"].get(col, 0)
        for coord in r.get("coord_pairs_list", []):
            bb["unique_coords"].add(tuple(coord))
        bb["rows_per_file"].append(rows)
        bb["data_types"].add(d)
        bb["scenarios"].add(s)
        bb["phones"].add(p)
        bb["users"].add(u)
        bb["modes"].add(m)
        if r["x_range"][0] is not None:
            if r["x_range"][0] < bb["x_range"][0]: bb["x_range"][0] = r["x_range"][0]
            if r["x_range"][1] > bb["x_range"][1]: bb["x_range"][1] = r["x_range"][1]
        if r["y_range"][0] is not None:
            if r["y_range"][0] < bb["y_range"][0]: bb["y_range"][0] = r["y_range"][0]
            if r["y_range"][1] > bb["y_range"][1]: bb["y_range"][1] = r["y_range"][1]
        for col in SENSOR_COLS:
            sr = r["sensor_ranges"].get(col, [None, None])
            if sr[0] is not None:
                if sr[0] < bb["sensor_ranges"][col][0]: bb["sensor_ranges"][col][0] = sr[0]
                if sr[1] > bb["sensor_ranges"][col][1]: bb["sensor_ranges"][col][1] = sr[1]
        if r["time_first"]:
            if bb["time_first"] is None or r["time_first"] < bb["time_first"]:
                bb["time_first"] = r["time_first"]
        if r["time_last"]:
            if bb["time_last"] is None or r["time_last"] > bb["time_last"]:
                bb["time_last"] = r["time_last"]

        # Per data_type
        bd = by_data_type[d]
        bd["total_files"] += 1
        bd["total_rows"] += rows
        bd["total_filesize_bytes"] += r["filesize_bytes"]
        for col in SENSOR_COLS:
            bd["null_counts"][col] += r["null_counts"].get(col, 0)
        for coord in r.get("coord_pairs_list", []):
            bd["unique_coords"].add(tuple(coord))
        bd["rows_per_file"].append(rows)

        # Per building+data_type+scenario
        key_bds = f"{b}|{d}|{s}"
        bds = by_bds[key_bds]
        bds["total_files"] += 1
        bds["total_rows"] += rows
        for col in SENSOR_COLS:
            bds["null_counts"][col] += r["null_counts"].get(col, 0)
        for coord in r.get("coord_pairs_list", []):
            bds["unique_coords"].add(tuple(coord))
        bds["phones"].add(p)
        bds["users"].add(u)
        bds["rows_per_file"].append(rows)
        if r["x_range"][0] is not None:
            if r["x_range"][0] < bds["x_range"][0]: bds["x_range"][0] = r["x_range"][0]
            if r["x_range"][1] > bds["x_range"][1]: bds["x_range"][1] = r["x_range"][1]
        if r["y_range"][0] is not None:
            if r["y_range"][0] < bds["y_range"][0]: bds["y_range"][0] = r["y_range"][0]
            if r["y_range"][1] > bds["y_range"][1]: bds["y_range"][1] = r["y_range"][1]
        if r["time_first"]:
            if bds["time_first"] is None or r["time_first"] < bds["time_first"]:
                bds["time_first"] = r["time_first"]
        if r["time_last"]:
            if bds["time_last"] is None or r["time_last"] > bds["time_last"]:
                bds["time_last"] = r["time_last"]

        # Per building+data_type+scenario+phone
        key_bdsp = f"{b}|{d}|{s}|{p}"
        bdsp = by_bdsp[key_bdsp]
        bdsp["total_files"] += 1
        bdsp["total_rows"] += rows
        for col in SENSOR_COLS:
            bdsp["null_counts"][col] += r["null_counts"].get(col, 0)
        for coord in r.get("coord_pairs_list", []):
            bdsp["unique_coords"].add(tuple(coord))
        bdsp["users"].add(u)
        bdsp["rows_per_file"].append(rows)

    return grand, by_building, by_data_type, by_bds, by_bdsp


def serialize_aggregation(grand, by_building, by_data_type, by_bds, by_bdsp):
    """Convert aggregation dicts to JSON-serializable format."""
    
    def stats(arr):
        if not arr: return {"min": 0, "max": 0, "mean": 0, "median": 0, "total": 0}
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        return {
            "min": arr_sorted[0],
            "max": arr_sorted[-1],
            "mean": round(sum(arr) / n, 2),
            "median": arr_sorted[n // 2],
            "total": sum(arr),
            "count": n,
        }

    def clean_range(r):
        if r[0] == float('inf'): return [None, None]
        return r

    # Grand
    grand_out = {
        "total_files": grand["total_files"],
        "total_rows": grand["total_rows"],
        "total_filesize_mb": round(grand["total_filesize_bytes"] / 1024 / 1024, 2),
        "total_null_counts": dict(grand["total_null_counts"]),
        "total_unique_coords": len(grand["all_unique_coords"]),
        "global_x_range": clean_range(grand["global_x_range"]),
        "global_y_range": clean_range(grand["global_y_range"]),
        "rows_per_file_stats": stats(grand["rows_per_file"]),
        "time_range": [grand["time_range_global_first"], grand["time_range_global_last"]],
        "global_sensor_ranges": {col: clean_range(r) for col, r in grand["global_sensor_ranges"].items()},
    }

    # Per building
    building_out = {}
    for b, data in by_building.items():
        building_out[b] = {
            "total_files": data["total_files"],
            "total_rows": data["total_rows"],
            "total_filesize_mb": round(data["total_filesize_bytes"] / 1024 / 1024, 2),
            "null_counts": dict(data["null_counts"]),
            "total_nulls_all_cols": sum(data["null_counts"].values()),
            "unique_coords": len(data["unique_coords"]),
            "coord_list": sorted(list(data["unique_coords"])),
            "x_range": clean_range(data["x_range"]),
            "y_range": clean_range(data["y_range"]),
            "rows_per_file_stats": stats(data["rows_per_file"]),
            "data_types": sorted(list(data["data_types"])),
            "scenarios": sorted(list(data["scenarios"])),
            "phones": sorted(list(data["phones"])),
            "users": sorted(list(data["users"])),
            "modes": sorted(list(data["modes"])),
            "sensor_ranges": {col: clean_range(r) for col, r in data["sensor_ranges"].items()},
            "time_range": [data["time_first"], data["time_last"]],
        }

    # Per data type
    dtype_out = {}
    for d, data in by_data_type.items():
        dtype_out[d] = {
            "total_files": data["total_files"],
            "total_rows": data["total_rows"],
            "total_filesize_mb": round(data["total_filesize_bytes"] / 1024 / 1024, 2),
            "null_counts": dict(data["null_counts"]),
            "unique_coords": len(data["unique_coords"]),
            "rows_per_file_stats": stats(data["rows_per_file"]),
        }

    # Per building+data_type+scenario
    bds_out = {}
    for key, data in by_bds.items():
        bds_out[key] = {
            "total_files": data["total_files"],
            "total_rows": data["total_rows"],
            "null_counts": dict(data["null_counts"]),
            "total_nulls_all_cols": sum(data["null_counts"].values()),
            "unique_coords": len(data["unique_coords"]),
            "coord_list": sorted(list(data["unique_coords"])),
            "phones": sorted(list(data["phones"])),
            "users": sorted(list(data["users"])),
            "rows_per_file_stats": stats(data["rows_per_file"]),
            "x_range": clean_range(data["x_range"]),
            "y_range": clean_range(data["y_range"]),
            "time_range": [data["time_first"], data["time_last"]],
        }

    # Per building+data_type+scenario+phone
    bdsp_out = {}
    for key, data in by_bdsp.items():
        bdsp_out[key] = {
            "total_files": data["total_files"],
            "total_rows": data["total_rows"],
            "null_counts": dict(data["null_counts"]),
            "total_nulls_all_cols": sum(data["null_counts"].values()),
            "unique_coords": len(data["unique_coords"]),
            "users": sorted(list(data["users"])),
            "rows_per_file_stats": stats(data["rows_per_file"]),
        }

    return {
        "grand_totals": grand_out,
        "per_building": building_out,
        "per_data_type": dtype_out,
        "per_building_datatype_scenario": bds_out,
        "per_building_datatype_scenario_phone": bdsp_out,
    }


def main():
    print("=" * 80)
    print("SURA EXHAUSTIVE DATASET SCANNER")
    print("=" * 80)

    # 1. Scan Magnetic Field Data
    print("\n[1/3] Scanning Magnetic Field CSVs (Static + Continuous)...")
    mag_results = walk_magnetic_data()
    print(f"  Total magnetic CSV files found: {len(mag_results)}")

    # 2. Aggregate
    print("\n[2/3] Aggregating results...")
    grand, by_building, by_data_type, by_bds, by_bdsp = aggregate_magnetic(mag_results)
    agg = serialize_aggregation(grand, by_building, by_data_type, by_bds, by_bdsp)

    # 3. WiFi
    print("\n[3/3] Scanning WiFi files (binary .xls)...")
    wifi_results = walk_wifi_data()
    wifi_agg = {
        "total_files": len(wifi_results),
        "total_filesize_mb": round(sum(r["filesize_bytes"] for r in wifi_results) / 1024 / 1024, 2),
        "per_building": {},
    }
    wifi_by_building = defaultdict(lambda: {"files": 0, "size_bytes": 0, "scenarios": set(), "phones": set(), "users": set(), "file_list": []})
    for r in wifi_results:
        b = r["building"]
        wb = wifi_by_building[b]
        wb["files"] += 1
        wb["size_bytes"] += r["filesize_bytes"]
        wb["scenarios"].add(r["scenario"])
        wb["phones"].add(r["phone"])
        wb["users"].add(r["user"])
        wb["file_list"].append(r["filename"])

    for b, data in wifi_by_building.items():
        wifi_agg["per_building"][b] = {
            "total_files": data["files"],
            "total_filesize_mb": round(data["size_bytes"] / 1024 / 1024, 2),
            "scenarios": sorted(list(data["scenarios"])),
            "phones": sorted(list(data["phones"])),
            "users": sorted(list(data["users"])),
        }

    agg["wifi"] = wifi_agg

    # Save full JSON
    out_path = Path(r"c:\Users\lenovo\Documents\GitHub\SURA\WorkSpace\dataset_scan_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f"\nFull JSON report saved to: {out_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    g = agg["grand_totals"]
    print(f"\n--- GRAND TOTALS (Magnetic Field) ---")
    print(f"  Total CSV files:        {g['total_files']}")
    print(f"  Total data rows:        {g['total_rows']:,}")
    print(f"  Total file size:        {g['total_filesize_mb']} MB")
    print(f"  Unique (X,Y) coords:    {g['total_unique_coords']}")
    print(f"  Global X range:         {g['global_x_range']}")
    print(f"  Global Y range:         {g['global_y_range']}")
    print(f"  Time range:             {g['time_range'][0]} -> {g['time_range'][1]}")
    print(f"  Rows/file:              min={g['rows_per_file_stats']['min']}, max={g['rows_per_file_stats']['max']}, mean={g['rows_per_file_stats']['mean']}, median={g['rows_per_file_stats']['median']}")

    print(f"\n  Null counts per column:")
    for col, cnt in g['total_null_counts'].items():
        pct = (cnt / g['total_rows'] * 100) if g['total_rows'] > 0 else 0
        print(f"    {col:12s}: {cnt:>8,}  ({pct:.2f}%)")

    print(f"\n  Global sensor ranges:")
    for col, rng in g['global_sensor_ranges'].items():
        if col == "Time": continue
        if rng[0] is not None:
            print(f"    {col:12s}: [{rng[0]:.4f}, {rng[1]:.4f}]")
        else:
            print(f"    {col:12s}: [no data]")

    print(f"\n--- PER DATA TYPE ---")
    for dtype, data in agg["per_data_type"].items():
        print(f"\n  {dtype}:")
        print(f"    Files: {data['total_files']},  Rows: {data['total_rows']:,},  Size: {data['total_filesize_mb']} MB")
        print(f"    Unique coords: {data['unique_coords']}")
        print(f"    Rows/file: min={data['rows_per_file_stats']['min']}, max={data['rows_per_file_stats']['max']}, mean={data['rows_per_file_stats']['mean']}")

    print(f"\n--- PER BUILDING ---")
    for b in sorted(agg["per_building"].keys()):
        data = agg["per_building"][b]
        print(f"\n  {b}:")
        print(f"    Files: {data['total_files']},  Rows: {data['total_rows']:,},  Size: {data['total_filesize_mb']} MB")
        print(f"    Unique coords: {data['unique_coords']},  X: {data['x_range']},  Y: {data['y_range']}")
        print(f"    Data types: {data['data_types']}")
        print(f"    Scenarios: {data['scenarios']}")
        print(f"    Phones: {data['phones']}")
        print(f"    Users: {data['users']}")
        print(f"    Modes: {data['modes']}")
        print(f"    Time range: {data['time_range'][0]} -> {data['time_range'][1]}")
        print(f"    Total nulls (all cols): {data['total_nulls_all_cols']:,}")
        print(f"    Rows/file: min={data['rows_per_file_stats']['min']}, max={data['rows_per_file_stats']['max']}, mean={data['rows_per_file_stats']['mean']}")

    print(f"\n--- PER BUILDING+DATATYPE+SCENARIO ---")
    for key in sorted(agg["per_building_datatype_scenario"].keys()):
        data = agg["per_building_datatype_scenario"][key]
        parts = key.split("|")
        print(f"\n  {parts[0]} / {parts[1]} / {parts[2]}:")
        print(f"    Files: {data['total_files']},  Rows: {data['total_rows']:,}")
        print(f"    Unique coords: {data['unique_coords']},  X: {data['x_range']},  Y: {data['y_range']}")
        print(f"    Phones: {data['phones']},  Users: {data['users']}")
        print(f"    Nulls (all cols): {data['total_nulls_all_cols']:,}")
        print(f"    Time: {data['time_range'][0]} -> {data['time_range'][1]}")

    print(f"\n--- WIFI DATASET ---")
    print(f"  Total files: {wifi_agg['total_files']}")
    print(f"  Total size:  {wifi_agg['total_filesize_mb']} MB")
    for b in sorted(wifi_agg["per_building"].keys()):
        data = wifi_agg["per_building"][b]
        print(f"\n  {b}:")
        print(f"    Files: {data['total_files']},  Size: {data['total_filesize_mb']} MB")
        print(f"    Scenarios: {data['scenarios']}")
        print(f"    Phones: {data['phones']}")
        print(f"    Users: {data['users']}")

    print(f"\n--- PER BUILDING+DATATYPE+SCENARIO+PHONE ---")
    for key in sorted(agg["per_building_datatype_scenario_phone"].keys()):
        data = agg["per_building_datatype_scenario_phone"][key]
        parts = key.split("|")
        print(f"  {parts[0]:20s} | {parts[1]:12s} | {parts[2]:15s} | {parts[3]:6s} | Files: {data['total_files']:>4} | Rows: {data['total_rows']:>8,} | Coords: {data['unique_coords']:>4} | Users: {data['users']} | Nulls: {data['total_nulls_all_cols']:>6,}")

    print("\n" + "=" * 80)
    print("DONE. Full report at:", out_path)
    print("=" * 80)


if __name__ == "__main__":
    main()
