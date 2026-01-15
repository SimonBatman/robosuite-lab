import os
os.environ.setdefault("MUJOCO_GL", "wgl")

import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
import json


def visualize_obs_structure(env_name="Lift", robots="Panda"):
    """Visualize raw obs dict and flattened obs."""
    
    # Create raw environment
    print("=" * 80)
    print(f"Creating {env_name} environment with {robots} robot")
    print("=" * 80)
    
    rs_env = suite.make(
        env_name=env_name,
        robots=robots,
        has_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    
    # Get raw obs
    print("\n[1] RAW OBSERVATION (dict structure):")
    print("-" * 80)
    raw_obs = rs_env.reset()
    
    # Print raw obs structure
    for key in sorted(raw_obs.keys()):
        val = np.array(raw_obs[key]).ravel()
        print(f"  {key:30s} | shape: {np.array(raw_obs[key]).shape} | flat_len: {len(val):3d} | values: {val}")
    
    # Compute total dim
    total_raw_dim = sum(len(np.array(raw_obs[k]).ravel()) for k in raw_obs.keys())
    print(f"\nTotal raw obs dimension: {total_raw_dim}")
    
    # Now wrap with GymWrapper and flatten
    rs_env.reset()  # reset before wrapping
    rs_env.close()
    
    # Create wrapped environment with flatten_obs=True
    print("\n" + "=" * 80)
    print("[2] WRAPPED ENVIRONMENT WITH FLATTEN_OBS=TRUE")
    print("=" * 80)
    
    rs_env = suite.make(
        env_name=env_name,
        robots=robots,
        has_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )
    
    gym_env = GymWrapper(rs_env, flatten_obs=True)
    
    # Get flattened obs
    flat_obs, _ = gym_env.reset()
    print(f"\nFlattened observation:")
    print(f"  shape: {flat_obs.shape}")
    print(f"  dtype: {flat_obs.dtype}")
    print(f"  min: {flat_obs.min():.6f}")
    print(f"  max: {flat_obs.max():.6f}")
    print(f"  mean: {flat_obs.mean():.6f}")
    print(f"  std: {flat_obs.std():.6f}")
    print(f"\n  First 20 elements: {flat_obs[:20]}")
    
    # Get the keys layout mapping
    print("\n" + "=" * 80)
    print("[3] FLATTENING LAYOUT (which indices correspond to which obs keys)")
    print("=" * 80)
    
    # Re-get raw obs to understand the layout
    raw_obs2 = gym_env.env.reset()
    offset = 0
    layout = {}
    for key in gym_env.keys:
        if key in raw_obs2:
            v = np.array(raw_obs2[key]).ravel()
            n = len(v)
            layout[key] = (offset, offset + n, n)
            offset += n
    
    print(f"\nTotal flattened obs dimension: {offset}")
    print("\nLayout mapping:")
    for key in gym_env.keys:
        if key in layout:
            s, e, n = layout[key]
            print(f"  {key:30s} | flat_indices [{s:3d}:{e:3d}) | len={n:3d}")
            # Show a sample slice
            if key in raw_obs2:
                sample = np.array(raw_obs2[key]).ravel()
                print(f"    └─ sample values: {sample[:min(3, len(sample))]}" + 
                      (" ..." if len(sample) > 3 else ""))
    
    # Visualization with ASCII art
    print("\n" + "=" * 80)
    print("[4] VISUAL REPRESENTATION OF FLATTENED OBS")
    print("=" * 80)
    
    print(f"\nFlattened vector (total {len(flat_obs)} elements):\n")
    
    bar_width = 50
    cumsum = 0
    for key in gym_env.keys:
        if key in layout:
            s, e, n = layout[key]
            # Calculate bar length proportional to dimension size
            bar_len = max(1, int(bar_width * n / len(flat_obs)))
            bar = "█" * bar_len
            print(f"  {key:30s} [{s:3d}:{e:3d}) | {bar} {n} dims")
            cumsum += n
    
    print(f"\n  Total: {cumsum} dimensions in flattened vector")
    
    gym_env.close()
    
    # ASCII visualization of the full vector
    print("\n" + "=" * 80)
    print("[5] SAMPLE VALUES FROM EACH SEGMENT")
    print("=" * 80)
    
    for key in gym_env.keys:
        if key in layout:
            s, e, n = layout[key]
            segment = flat_obs[s:e]
            print(f"\n  {key}:")
            print(f"    Indices [{s}:{e}] ({n} elements)")
            if n <= 10:
                print(f"    Values: {segment}")
            else:
                print(f"    Values (first 5): {segment[:5]} ... (last 3): {segment[-3:]}")


if __name__ == "__main__":
    visualize_obs_structure()
