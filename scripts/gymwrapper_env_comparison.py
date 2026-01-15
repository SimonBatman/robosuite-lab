"""
GymWrapper Flattening Strategy - Environment Comparison
Shows how different environments result in different flattened obs
"""

import json

print("=" * 120)
print("GymWrapper FLATTENING - ENVIRONMENT COMPARISON")
print("=" * 120)

# Define different environment configurations
environments = {
    "Lift": {
        "description": "Single cube lifting task",
        "use_object_obs": True,
        "use_camera_obs": False,
        "num_robots": 1,
        "object_state_dim": 10,
        "robot_proprio_dim": 50,
    },
    "Wipe": {
        "description": "Wiping task",
        "use_object_obs": False,
        "use_camera_obs": False,
        "num_robots": 1,
        "object_state_dim": 0,
        "robot_proprio_dim": 50,
    },
    "PickPlace": {
        "description": "Pick and place task",
        "use_object_obs": True,
        "use_camera_obs": False,
        "num_robots": 1,
        "object_state_dim": 10,
        "robot_proprio_dim": 50,
    },
    "TwoArmPickPlace": {
        "description": "Two robots picking and placing",
        "use_object_obs": True,
        "use_camera_obs": False,
        "num_robots": 2,
        "object_state_dim": 10,
        "robot_proprio_dim": 50,
    },
    "ImageBased": {
        "description": "Vision-based task with 2 cameras",
        "use_object_obs": False,
        "use_camera_obs": True,
        "num_cameras": 2,
        "camera_dim": 128 * 128 * 3,  # RGB image
        "num_robots": 1,
        "robot_proprio_dim": 50,
    },
}

# Calculate flattened dimensions for each
print("\n" + "=" * 120)
print("COMPARISON TABLE")
print("=" * 120)

print("\n{:<20s} {:<30s} {:<20s} {:<20s} {:<20s}".format(
    "Environment", "Config", "Keys Selected", "Dimensions", "Total Flat Dim"
))
print("-" * 120)

for env_name, config in environments.items():
    description = config["description"]
    
    # Build keys list
    keys = []
    dims_breakdown = []
    total_dim = 0
    
    if config.get("use_object_obs", False):
        keys.append("object-state")
        dim = config.get("object_state_dim", 0)
        dims_breakdown.append(f"object:{dim}")
        total_dim += dim
    
    if config.get("use_camera_obs", False):
        num_cams = config.get("num_cameras", 1)
        for i in range(num_cams):
            cam_name = f"cam{i}"
            keys.append(f"{cam_name}_image")
        cam_dim = config.get("camera_dim", 0) * num_cams
        dims_breakdown.append(f"images:{cam_dim}")
        total_dim += cam_dim
    
    num_robots = config.get("num_robots", 1)
    for i in range(num_robots):
        keys.append(f"robot{i}_proprio-state")
    
    robot_dim = config.get("robot_proprio_dim", 0) * num_robots
    dims_breakdown.append(f"robots:{robot_dim}")
    total_dim += robot_dim
    
    keys_str = ", ".join(keys[:2]) + ("..." if len(keys) > 2 else "")
    dims_str = " + ".join(dims_breakdown)
    
    print(f"{env_name:<20s} {description:<30s} {str(len(keys)):<20s} {dims_str:<20s} {total_dim:<20d}")

print("\n" + "=" * 120)
print("KEY INSIGHTS")
print("=" * 120)

print("""
1. KEY SELECTION is DYNAMIC and depends on:
   ✓ use_object_obs      - Does the task have an object?
   ✓ use_camera_obs      - Are we using vision?
   ✓ num_robots          - Single or multi-arm?
   ✓ num_cameras         - How many cameras?

2. FLATTENING ALGORITHM is ALWAYS THE SAME:
   ob_lst = []
   for key in self.keys:
       ob_lst.append(np.array(obs_dict[key]).flatten())
   return np.concatenate(ob_lst)

3. CONSEQUENCE: Different environments produce different observation dimensions!
   - Lift:              60 dims
   - Wipe:              50 dims (no object)
   - TwoArmPickPlace:  110 dims (2 robots)
   - ImageBased:     3074 dims (images take most space!)

4. ORDER OF CONCATENATION (ALWAYS SAME):
   [object-state] + [images...] + [robot0_proprio-state] + [robot1_proprio-state] + ...

5. PORTABILITY:
   ✓ CODE: GymWrapper works for all robosuite environments
   ✗ OUTPUT: Observation dimensions and meanings vary
""")

print("\n" + "=" * 120)
print("DETAILED LAYOUT EXAMPLES")
print("=" * 120)

print("""
EXAMPLE 1: Lift Environment
─────────────────────────────
Configuration:
  - use_object_obs: True
  - use_camera_obs: False
  - num_robots: 1

Keys selected (in order):
  1. "object-state"         → 10 elements
  2. "robot0_proprio-state" → 50 elements

Flattened observation layout:
  
  flat_obs[0:10]   ← object-state (cube position, orientation, velocity)
  flat_obs[10:60]  ← robot0_proprio-state (joint pos/vel/acc, gripper)
  
Total: 60 dimensions


EXAMPLE 2: TwoArmPickPlace Environment
───────────────────────────────────────
Configuration:
  - use_object_obs: True
  - use_camera_obs: False
  - num_robots: 2

Keys selected (in order):
  1. "object-state"         → 10 elements
  2. "robot0_proprio-state" → 50 elements
  3. "robot1_proprio-state" → 50 elements

Flattened observation layout:
  
  flat_obs[0:10]   ← object-state
  flat_obs[10:60]  ← robot0_proprio-state (left arm)
  flat_obs[60:110] ← robot1_proprio-state (right arm)
  
Total: 110 dimensions


EXAMPLE 3: VisionBased Environment
───────────────────────────────────
Configuration:
  - use_object_obs: False
  - use_camera_obs: True
  - num_cameras: 2
  - num_robots: 1

Keys selected (in order):
  1. "cam0_image"           → 128*128*3 = 49152 elements
  2. "cam1_image"           → 128*128*3 = 49152 elements
  3. "robot0_proprio-state" → 50 elements

Flattened observation layout:
  
  flat_obs[0:49152]         ← camera 0 RGB image
  flat_obs[49152:98304]     ← camera 1 RGB image
  flat_obs[98304:98354]     ← robot0_proprio-state
  
Total: 98354 dimensions


EXAMPLE 4: Wipe Environment (No Object)
────────────────────────────────────────
Configuration:
  - use_object_obs: False (no movable object)
  - use_camera_obs: False
  - num_robots: 1

Keys selected (in order):
  1. "robot0_proprio-state" → 50 elements

Flattened observation layout:
  
  flat_obs[0:50] ← robot0_proprio-state
  
Total: 50 dimensions (smallest!)
""")

print("\n" + "=" * 120)
print("CUSTOMIZATION: How to Override Key Selection")
print("=" * 120)

print("""
By default (keys=None), GymWrapper auto-selects keys based on env flags.

But you CAN customize which keys to include and in what order:

# Example 1: Use ONLY robot state, skip object state
gym_env = GymWrapper(
    rs_env,
    keys=["robot0_proprio-state"],  ← Custom!
    flatten_obs=True
)
# Result: obs shape becomes (50,) instead of (60,)


# Example 2: Reverse the order
gym_env = GymWrapper(
    rs_env,
    keys=["robot0_proprio-state", "object-state"],  ← Reversed order
    flatten_obs=True
)
# Result: obs[0:50] is robot state, obs[50:60] is object state
# (opposite of default)


# Example 3: Include only specific components
gym_env = GymWrapper(
    rs_env,
    keys=["robot0_gripper_qpos"],  ← Only gripper position
    flatten_obs=True
)
# Result: obs shape becomes (2,) - only 2 gripper joints


# Example 4: Multi-robot with custom order
gym_env = GymWrapper(
    rs_env,
    keys=["robot0_proprio-state", "robot1_proprio-state", "object-state"],
    flatten_obs=True
)
# Result: obs structure is [robot0 | robot1 | object] instead of [object | robot0 | robot1]
""")

print("\n" + "=" * 120)
print("✓ SUMMARY: Same Algorithm, Different Results")
print("=" * 120)

print("""
┌────────────────────────┬──────────────────────────────────────────────────────┐
│ Aspect                 │ Answer                                               │
├────────────────────────┼──────────────────────────────────────────────────────┤
│ Flattening algorithm   │ IDENTICAL for all envs: flatten + concatenate       │
│ Key selection logic    │ DIFFERENT: dynamic based on env flags               │
│ Observation dimension  │ DIFFERENT: env-dependent (50-100K+ dims)            │
│ Order consistency      │ FIXED: [objects] + [images] + [robots]             │
│ Customizable?          │ YES: pass custom keys parameter                     │
│ Reversible?            │ HARD: need to know key sizes and order             │
│ Environment agnostic?  │ CODE: Yes | OUTPUT: No                              │
└────────────────────────┴──────────────────────────────────────────────────────┘
""")
