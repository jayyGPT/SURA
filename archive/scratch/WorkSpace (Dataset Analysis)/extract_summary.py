import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'c:\Users\lenovo\Documents\GitHub\SURA\WorkSpace\dataset_scan_report.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

g = d['grand_totals']
print('=== GRAND TOTALS (Magnetic) ===')
print('Files:', g['total_files'])
print('Rows:', f"{g['total_rows']:,}")
print('Size:', g['total_filesize_mb'], 'MB')
print('Unique coords:', g['total_unique_coords'])
print('X range:', g['global_x_range'])
print('Y range:', g['global_y_range'])
print('Rows/file:', g['rows_per_file_stats'])
print('Time start:', g['time_range'][0])
print()

print('=== NULL COUNTS ===')
for col, cnt in g['total_null_counts'].items():
    pct = cnt / g['total_rows'] * 100 if g['total_rows'] > 0 else 0
    print(f'  {col:12s}: {cnt:>8,} ({pct:.2f}%)')
print()

print('=== SENSOR RANGES ===')
for col, rng in g['global_sensor_ranges'].items():
    if col == 'Time':
        continue
    print(f'  {col:12s}: {rng}')
print()

print('=== PER DATA TYPE ===')
for dt, data in d['per_data_type'].items():
    print(f'{dt}: files={data["total_files"]}, rows={data["total_rows"]:,}, size={data["total_filesize_mb"]}MB, coords={data["unique_coords"]}')
    print(f'  rows/file: {data["rows_per_file_stats"]}')
print()

print('=== PER BUILDING ===')
for b in sorted(d['per_building'].keys()):
    bd = d['per_building'][b]
    print(f'\n{b}:')
    print(f'  files={bd["total_files"]}, rows={bd["total_rows"]:,}, size={bd["total_filesize_mb"]}MB')
    print(f'  coords={bd["unique_coords"]}, X={bd["x_range"]}, Y={bd["y_range"]}')
    print(f'  nulls_total={bd["total_nulls_all_cols"]:,}')
    print(f'  data_types={bd["data_types"]}')
    print(f'  scenarios={bd["scenarios"]}')
    print(f'  phones={bd["phones"]}')
    print(f'  users={bd["users"]}')
    print(f'  modes={bd["modes"]}')
    print(f'  rows/file: {bd["rows_per_file_stats"]}')
    t0 = bd['time_range'][0] or 'N/A'
    t1 = bd['time_range'][1]
    t1_str = str(t1)[:30] if t1 else 'N/A'
    print(f'  time: {t0} -> {t1_str}')
    nz = {k: v for k, v in bd['null_counts'].items() if v > 0}
    if nz:
        print(f'  null_cols: {nz}')

print()
print('=== WIFI ===')
w = d['wifi']
print(f'Total files: {w["total_files"]}, size: {w["total_filesize_mb"]}MB')
for b in sorted(w['per_building'].keys()):
    wb = w['per_building'][b]
    print(f'  {b}: files={wb["total_files"]}, size={wb["total_filesize_mb"]}MB, scenarios={wb["scenarios"]}, phones={wb["phones"]}, users={wb["users"]}')

print()
print('=== PER BUILDING+DATATYPE+SCENARIO ===')
for key in sorted(d['per_building_datatype_scenario'].keys()):
    data = d['per_building_datatype_scenario'][key]
    parts = key.split('|')
    print(f'  {parts[0]:20s} | {parts[1]:12s} | {parts[2]:15s} | files={data["total_files"]:>4} | rows={data["total_rows"]:>8,} | coords={data["unique_coords"]:>5} | X={data["x_range"]} | Y={data["y_range"]} | nulls={data["total_nulls_all_cols"]:>6,} | phones={data["phones"]} | users={data["users"]}')

print()
print('=== PER BUILDING+DATATYPE+SCENARIO+PHONE ===')
for key in sorted(d['per_building_datatype_scenario_phone'].keys()):
    data = d['per_building_datatype_scenario_phone'][key]
    parts = key.split('|')
    print(f'  {parts[0]:20s} | {parts[1]:10s} | {parts[2]:15s} | {parts[3]:8s} | files={data["total_files"]:>4} | rows={data["total_rows"]:>8,} | coords={data["unique_coords"]:>5} | users={data["users"]} | nulls={data["total_nulls_all_cols"]:>6,}')
