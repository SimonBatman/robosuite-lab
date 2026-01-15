"""
GymWrapper Flattening Strategy Analysis
- How does it work?
- Is it the same for all environments?
"""

print("=" * 100)
print("GymWrapper FLATTENING STRATEGY - DETAILED ANALYSIS")
print("=" * 100)

print("""
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [1] HOW DOES GymWrapper FLATTEN OBS?                                                │
└─────────────────────────────────────────────────────────────────────────────────────┘

GymWrapper has TWO key components:

A) KEY SELECTION (which obs to include)
   └─ Happens in __init__() when keys=None (default):
   
   if keys is None:
       keys = []
       if self.env.use_object_obs:
           keys += ["object-state"]                    # Object/task state
       if self.env.use_camera_obs:
           keys += [f"{cam_name}_image" for cam in self.env.camera_names]  # Images
       for idx in range(len(self.env.robots)):
           keys += ["robot{}_proprio-state".format(idx)]  # Robot state
   
   └─ This is ENVIRONMENT-ADAPTIVE (see below)

B) FLATTENING (concatenate selected keys)
   └─ Happens in _flatten_obs():
   
   def _flatten_obs(self, obs_dict, verbose=False):
       ob_lst = []
       for key in self.keys:
           if key in obs_dict:
               ob_lst.append(np.array(obs_dict[key]).flatten())  # Flatten each key
       return np.concatenate(ob_lst)  # Concatenate all
   
   └─ This is ALWAYS THE SAME for all environments


┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [2] IS IT THE SAME FOR ALL ENVIRONMENTS?                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘

NO! The key selection is DYNAMIC and environment-dependent:

Different robosuite environments have different env flags:
- use_object_obs:  Does the task involve an object? (Lift, PickPlace, etc.)
- use_camera_obs:  Should we include camera observations?
- Number of robots: Single-arm, dual-arm, etc.

Examples:

[Environment: Lift (single cube)]
  use_object_obs = True  ✓ (cube in scene)
  use_camera_obs = False (not using cameras in this config)
  num_robots = 1
  
  → Keys selected: ["object-state", "robot0_proprio-state"]
  → Total dims: 10 + 50 = 60


[Environment: Pick & Place (single cube, pick + place)]
  use_object_obs = True  ✓ (cube in scene)
  use_camera_obs = False (not using cameras in this config)
  num_robots = 1
  
  → Keys selected: ["object-state", "robot0_proprio-state"]
  → Total dims: 10 + 50 = 60
  (Same as Lift!)


[Environment: Manipulation (no object, just reaching)]
  use_object_obs = False  ✗ (no object)
  use_camera_obs = False
  num_robots = 1
  
  → Keys selected: ["robot0_proprio-state"]
  → Total dims: 50
  (Only robot state!)


[Environment: Dual Arm Assembly (two robots, object)]
  use_object_obs = True  ✓
  use_camera_obs = False
  num_robots = 2
  
  → Keys selected: ["object-state", "robot0_proprio-state", "robot1_proprio-state"]
  → Total dims: 10 + 50 + 50 = 110
  (More robot states!)


[Environment: VisionBased Task (with cameras)]
  use_object_obs = False
  use_camera_obs = True  ✓
  num_cameras = 2
  num_robots = 1
  
  → Keys selected: ["front_image", "side_image", "robot0_proprio-state"]
  → Total dims: (H*W*C) + (H*W*C) + 50
  (Much larger due to image data!)


┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [3] FLATTENING ALGORITHM - ALWAYS THE SAME                                          │
└─────────────────────────────────────────────────────────────────────────────────────┘

The actual flattening process is IDENTICAL for all environments:

Step 1: For each key in self.keys:
  ├─ Get the numpy array: obs_dict[key]
  ├─ Flatten it to 1D: np.array(...).flatten()
  └─ Add to list

Step 2: Concatenate all flattened arrays in order
  └─ np.concatenate([arr1, arr2, arr3, ...])

Result: One big 1D numpy array

The ORDER MATTERS:
  flat_obs = [arr1 | arr2 | arr3 | ...]
              └────┘  └────┘  └────┘
           (indices 0-9) (10-59) (60-...)


┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [4] CUSTOMIZATION OPTIONS                                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

You CAN customize the keys! Pass them explicitly:

gym_env = GymWrapper(
    rs_env, 
    keys=["object-state", "robot0_gripper_qpos"],  ← Custom keys!
    flatten_obs=True
)

This allows you to:
- Select ONLY the obs you need
- Change the order of concatenation
- Include/exclude specific state components


┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [5] FLATTEN_OBS=FALSE BEHAVIOR                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

When flatten_obs=False, NO concatenation happens!

Instead:
- observation_space = spaces.Dict() instead of spaces.Box()
- _filter_obs() is called instead of _flatten_obs()
- Returns: {key1: array1, key2: array2, ...}

Useful for:
- Algorithms that can handle dict observations (some models can)
- Preserving semantic separation between different obs types


┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [6] SUMMARY TABLE                                                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┬──────────────────────────────────────┐
│ Aspect              │ Same or Different?                   │
├─────────────────────┼──────────────────────────────────────┤
│ Flattening method   │ SAME - np.flatten() + concatenate    │
│ Key selection       │ DIFFERENT - env-dependent flags      │
│ Dimension reduction │ DIFFERENT - depends on env config    │
│ Final dimension     │ DIFFERENT - env/config dependent     │
│ Order of concat     │ FIXED - object, images, then robots  │
│ Reversibility       │ HARD - need to know key order/sizes  │
└─────────────────────┴──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [7] KEY SELECTION ORDER (ALWAYS THE SAME)                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

When keys=None (default), they're selected in this order:

1. First:  "object-state"        (if use_object_obs=True)
2. Then:   "{cam}_image"          (for each camera, if use_camera_obs=True)
3. Last:   "robot{i}_proprio-state" (for each robot)

For Lift (use_object_obs=True, 1 robot, no cameras):
  flat_obs = ["object-state"] + ["robot0_proprio-state"]
               └──────┬──────┘   └────────┬────────┘
              indices 0-9         indices 10-59


┌─────────────────────────────────────────────────────────────────────────────────────┐
│ [8] PRACTICAL IMPLICATION                                                           │
└─────────────────────────────────────────────────────────────────────────────────────┘

✓ PORTABLE CODE:
  GymWrapper(env, flatten_obs=True) works for ALL robosuite environments
  because it automatically adapts to each environment's structure

✗ NOT PORTABLE OUTPUT:
  The resulting observation dimension and meaning CHANGES per environment
  You can't assume obs[0:3] is always position
  
  Solutions:
  1. Always use env.env.reset() to get raw dict if you need specific keys
  2. Store the env config and know which dims map to what
  3. Use the raw obs dict (flatten_obs=False) when you need semantic clarity


""")

print("=" * 100)
