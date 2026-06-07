import pandas as pd
import numpy as np

phones = ['A8', 'G7', 'S8', 'S9+']
for p in phones:
    df = pd.read_csv(f'../Datasets/Continuous_Fused_{p}.csv')
    print(f'=== {p} ===')
    print(f'  Rows: {len(df)}')
    tx = df['True_X']
    ty = df['True_Y']
    print(f'  Start: ({tx.iloc[0]:.1f}, {ty.iloc[0]:.1f})')
    print(f'  End:   ({tx.iloc[-1]:.1f}, {ty.iloc[-1]:.1f})')
    print(f'  X range: [{tx.min():.1f}, {tx.max():.1f}]')
    print(f'  Y range: [{ty.min():.1f}, {ty.max():.1f}]')
    
    # Detect the number of waypoint turns
    dx = np.diff(tx.values)
    dy = np.diff(ty.values)
    heading = np.arctan2(dy, dx)
    heading_changes = np.abs(np.diff(heading))
    significant_turns = np.sum(heading_changes > 0.1)
    print(f'  Significant heading changes (>0.1 rad): {significant_turns}')
    print()

# Now check: is the path IDENTICAL across phones?
print('=== PATH IDENTITY CHECK ===')
df_a8 = pd.read_csv('../Datasets/Continuous_Fused_A8.csv')
df_s9 = pd.read_csv('../Datasets/Continuous_Fused_S9+.csv')
# Downsample both to compare
ds_rate = 25
a8_path = df_a8[['True_X', 'True_Y']].iloc[::ds_rate].reset_index(drop=True)
s9_path = df_s9[['True_X', 'True_Y']].iloc[::ds_rate].reset_index(drop=True)
min_len = min(len(a8_path), len(s9_path))
a8_path = a8_path.iloc[:min_len]
s9_path = s9_path.iloc[:min_len]
diff = np.abs(a8_path.values - s9_path.values)
print(f'Max absolute diff between A8 and S9+ paths: X={diff[:,0].max():.4f}, Y={diff[:,1].max():.4f}')
print(f'Mean absolute diff: X={diff[:,0].mean():.4f}, Y={diff[:,1].mean():.4f}')
if diff.max() < 1.0:
    print('WARNING: Train and Test are walking the EXACT SAME PATH!')
    print('The model may be memorizing the route, not learning sensor interpretation!')
