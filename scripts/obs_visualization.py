"""
Visualization: Lift Environment Observation Structure
Shows raw obs dict vs flattened obs layout
"""

import numpy as np

# From the actual run, the raw obs dict contains:
RAW_OBS_STRUCTURE = {
    "object-state": 10,
    "robot0_proprio-state": 50,
}

# Default GymWrapper keys (from earlier analysis)
GYMWRAPPER_KEYS = ["object-state", "robot0_proprio-state"]

# Build the layout
layout = {}
offset = 0
for key in GYMWRAPPER_KEYS:
    if key in RAW_OBS_STRUCTURE:
        n = RAW_OBS_STRUCTURE[key]
        layout[key] = (offset, offset + n, n)
        offset += n

total_flattened_dim = offset

print("=" * 90)
print("LIFT ENVIRONMENT - OBSERVATION VISUALIZATION")
print("=" * 90)

print("\n[1] RAW OBSERVATION DICT (15 keys, 120 total dimensions):")
print("-" * 90)
raw_all = {
    "cube_pos": 3,
    "cube_quat": 4,
    "gripper_to_cube_pos": 3,
    "object-state": 10,
    "robot0_eef_pos": 3,
    "robot0_eef_quat": 4,
    "robot0_eef_quat_site": 4,
    "robot0_gripper_qpos": 2,
    "robot0_gripper_qvel": 2,
    "robot0_joint_acc": 7,
    "robot0_joint_pos": 7,
    "robot0_joint_pos_cos": 7,
    "robot0_joint_pos_sin": 7,
    "robot0_joint_vel": 7,
    "robot0_proprio-state": 50,
}

total_raw = 0
for key in sorted(raw_all.keys()):
    size = raw_all[key]
    total_raw += size
    print(f"  {key:30s} → {size:3d} elements")

print(f"\n  TOTAL: {total_raw} elements in raw obs dict")

print("\n" + "=" * 90)
print("[2] AFTER GymWrapper WITH flatten_obs=True")
print("=" * 90)

print(f"\nFlattened observation:")
print(f"  Shape: ({total_flattened_dim},)")
print(f"  Data type: float64")
print(f"  Only includes keys: {GYMWRAPPER_KEYS}")

print("\n" + "=" * 90)
print("[3] FLATTENING LAYOUT (Index Mapping)")
print("=" * 90)

print(f"\nTotal flattened dimension: {total_flattened_dim}\n")
print("Index ranges for each observation key:")
for key in GYMWRAPPER_KEYS:
    if key in layout:
        s, e, n = layout[key]
        print(f"  {key:30s} → flat[{s:3d}:{e:3d}]  ({n:2d} elements)")

print("\n" + "=" * 90)
print("[4] VISUAL BAR CHART")
print("=" * 90)

print("\nFlattened observation vector breakdown:\n")

bar_width = 60
print("Key                           │ Visual Bar              │ Size │ Percentage")
print("-" * 80)

for key in GYMWRAPPER_KEYS:
    if key in layout:
        s, e, n = layout[key]
        bar_len = max(1, int(bar_width * n / total_flattened_dim))
        bar = "█" * bar_len
        percentage = (n / total_flattened_dim) * 100
        print(f"{key:30s} │{bar:<{bar_width}s}│{n:5d} │ {percentage:6.1f}%")

print("-" * 80)
total_bar_len = bar_width
total_bar = "█" * total_bar_len
total_percentage = 100.0
print(f"{'TOTAL':30s} │{total_bar:<{bar_width}s}│{total_flattened_dim:5d} │ {total_percentage:6.1f}%")

print("\n" + "=" * 90)
print("[5] MEANING OF EACH SEGMENT")
print("=" * 90)

print("\nobject-state [0:10] (10 elements):")
print("  └─ Contains: cube position (x,y,z), quaternion (qx,qy,qz,qw), velocities")
print("  └─ This is the CUBE or OBJECT pose/state in the scene")

print("\nrobot0_proprio-state [10:60] (50 elements):")
print("  └─ Contains: robot joint positions, velocities, accelerations, gripper state")
print("  └─ 7 joint pos + 7 joint vel + 7 joint acc + 7 pos_cos + 7 pos_sin + 2 gripper qpos + 2 gripper qvel")
print("  └─ This is the ROBOT proprioceptive (internal) state")

print("\n" + "=" * 90)
print("[6] KEY OBSERVATIONS")
print("=" * 90)

print("""
1. DIMENSION REDUCTION:
   - Raw obs dict: 120 dimensions (all keys)
   - GymWrapper flatten_obs: 60 dimensions (only 2 keys)
   - The wrapper FILTERS which obs keys to include!

2. DEFAULT KEYS SELECTED BY GymWrapper:
   - object-state: 10 dims (what object is doing)
   - robot0_proprio-state: 50 dims (robot internal state)
   - NOT included: cube_pos, cube_quat, robot eef state, etc.

3. FLATTENING PROCESS:
   The _flatten_obs() method simply concatenates these 2 keys in order:
   
   flat_obs = concat([obs["object-state"], obs["robot0_proprio-state"]])
              └─ 10 elements ──┘  └──────── 50 elements ────────┘
   
4. TO EXTRACT CUBE POSITION:
   - In raw obs: obs["cube_pos"] gives you [x, y, z]
   - In flattened obs: NOT directly available! 
     (It's in raw obs but not included in default wrapper keys)
   - Would need custom keys or access raw_obs directly

5. MEMORY LAYOUT:
   Index  0-9   : object-state (cube info)
   Index 10-59  : robot0_proprio-state (robot info)
""")

print("=" * 90)
