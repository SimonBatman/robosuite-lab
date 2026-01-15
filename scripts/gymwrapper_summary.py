"""
SUMMARY: GymWrapper Flattening Strategy
Quick Reference Guide
"""

print("""
╔════════════════════════════════════════════════════════════════════════════════╗
║                   GymWrapper FLATTENING STRATEGY SUMMARY                       ║
╚════════════════════════════════════════════════════════════════════════════════╝

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ QUESTION 1: How does GymWrapper flatten observations?                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

STEP 1: SELECT KEYS (in __init__)
  if keys=None (default):
      keys = ["object-state"]  (if use_object_obs=True)
      keys += ["{cam}_image" for each camera]  (if use_camera_obs=True)
      keys += ["robot{i}_proprio-state" for each robot]

STEP 2: FLATTEN EACH KEY (in _flatten_obs())
  for each key in self.keys:
      flatten to 1D: np.array(obs_dict[key]).flatten()

STEP 3: CONCATENATE ALL
  np.concatenate([flattened_key1, flattened_key2, ...])

RESULT: Single 1D numpy array


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ QUESTION 2: Is it the same for all environments?                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

ALGORITHM: ALWAYS THE SAME
  └─ flatten() + concatenate() is universal

KEY SELECTION: DIFFERENT PER ENVIRONMENT
  └─ Depends on: use_object_obs, use_camera_obs, num_robots, etc.

FINAL DIMENSION: ALWAYS DIFFERENT
  └─ Lift: 60 dims
  └─ Wipe: 50 dims (no object)
  └─ TwoArmPickPlace: 110 dims (2 robots)
  └─ VisionBased: 98K+ dims (images!)


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ SUMMARY TABLE: What Stays the Same vs What Changes                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

╔─────────────────────────┬──────────┬────────────────────────────────────────╗
║ Aspect                  │ Same?    │ Details                                ║
╠─────────────────────────╼──────────╪────────────────────────────────────────╣
║ flatten() operation     │ ✓ SAME   │ .flatten() converts any shape to 1D   ║
║ concatenate() operation │ ✓ SAME   │ np.concatenate() always used         ║
║ Key selection logic     │ ✗ DIFF   │ Dynamic: if-based on env flags       ║
║ Key selection order     │ ~ FIXED  │ [object] → [images] → [robots]      ║
║ Final dimension         │ ✗ DIFF   │ 50, 60, 110, 98K+, ...              ║
║ Semantic meaning        │ ✗ DIFF   │ obs[0:10] ≠ same across envs         ║
║ Customizable?           │ ✓ YES    │ Pass custom keys= parameter          ║
╚─────────────────────────┴──────────┴────────────────────────────────────────╝


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ KEY INSIGHTS                                                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

1. PORTABILITY
   ✓ CODE IS PORTABLE: One GymWrapper class works for all robosuite envs
   ✗ OUTPUT NOT PORTABLE: Different envs = different obs shapes & meanings

2. ENVIRONMENT-AWARE BEHAVIOR
   GymWrapper checks environment properties and adapts:
   • Does the env have an object? → include "object-state"
   • Does it use cameras? → include "{camera}_image"
   • How many robots? → include all "robot{i}_proprio-state"

3. ORDER MATTERS
   The order of keys in self.keys determines the index mapping:
   
   Lift (default order):
     obs[0:10]   = object-state
     obs[10:60]  = robot0_proprio-state
   
   Lift (custom reversed order):
     obs[0:50]   = robot0_proprio-state
     obs[50:60]  = object-state

4. DIMENSION EXPLOSION WITH IMAGES
   State-based: 50-200 dims (manageable)
   With images: 98K+ dims (huge!)
   
   Example: 128×128 RGB image = 49,152 dims per camera

5. NO SEMANTIC PRESERVATION
   Once flattened, you lose the semantic structure:
   • Can't tell where joint positions end and velocities begin
   • Can't easily access specific body parts
   • Solution: Use flatten_obs=False for dict obs, or store keys/offsets


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ CODE EXAMPLES                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# DEFAULT BEHAVIOR (auto-select keys)
gym_env = GymWrapper(rs_env, flatten_obs=True)
obs, _ = gym_env.reset()
print(obs.shape)  # (60,) for Lift

# CUSTOM KEYS (select only what you need)
gym_env = GymWrapper(
    rs_env,
    keys=["object-state", "robot0_gripper_qpos"],
    flatten_obs=True
)
obs, _ = gym_env.reset()
print(obs.shape)  # (12,) - only object (10) + gripper (2)

# NO FLATTENING (keep dict structure)
gym_env = GymWrapper(rs_env, flatten_obs=False)
obs, _ = gym_env.reset()
print(type(obs))  # <class 'dict'>
print(obs["object-state"].shape)  # (10,)
print(obs["robot0_proprio-state"].shape)  # (50,)

# RECONSTRUCT ORIGINAL OBSERVATION
raw_obs = gym_env.env.reset()  # Get raw dict from underlying env
# Now you have all original 120 dims and semantic meaning


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ COMPARISON: Lift vs Wipe vs TwoArmPickPlace                                   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

LIFT (Single Arm, Object)
  Config: use_object_obs=True, num_robots=1
  Keys: ["object-state", "robot0_proprio-state"]
  Layout:
    obs[0:10]   ← object-state
    obs[10:60]  ← robot0_proprio-state
  Total: 60 dims

WIPE (Single Arm, No Object)
  Config: use_object_obs=False, num_robots=1
  Keys: ["robot0_proprio-state"]
  Layout:
    obs[0:50]   ← robot0_proprio-state
  Total: 50 dims
  
  ⚠️ INCOMPATIBLE DIM! (60 vs 50)
     Can't reuse Lift-trained policy for Wipe!

TWOARMPICKPLACE (Dual Arm, Object)
  Config: use_object_obs=True, num_robots=2
  Keys: ["object-state", "robot0_proprio-state", "robot1_proprio-state"]
  Layout:
    obs[0:10]   ← object-state
    obs[10:60]  ← robot0_proprio-state
    obs[60:110] ← robot1_proprio-state
  Total: 110 dims
  
  ⚠️ DIFFERENT DIM! (110 vs 60)
     Can't transfer Lift policy directly!


┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ BEST PRACTICES                                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

1. DOCUMENT THE OBS SPACE
   # Store metadata about what each segment means
   obs_metadata = {
       "object_state": (0, 10),
       "robot_proprio": (10, 60),
   }

2. USE FLATTEN_OBS=FALSE FOR CLARITY
   gym_env = GymWrapper(rs_env, flatten_obs=False)
   # Work with dict obs when you need semantic understanding

3. CUSTOM KEYS FOR SPECIFIC COMPONENTS
   # Only use what the policy needs
   gym_env = GymWrapper(
       rs_env,
       keys=["object-state", "robot0_joint_pos"],
       flatten_obs=True
   )

4. STORE KEY INFORMATION
   # After wrapping, save this for later reference
   env_info = {
       "keys": gym_env.keys,
       "obs_dim": len(flat_obs),
       "layout": compute_layout(gym_env)  # Your function
   }

5. HANDLE MULTI-ENVIRONMENT POLICIES
   # If switching between envs, verify dimension compatibility
   if obs.shape != expected_shape:
       raise ValueError(f"Dimension mismatch: got {obs.shape}, expected {expected_shape}")


╔════════════════════════════════════════════════════════════════════════════════╗
║                              KEY TAKEAWAY                                      ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  GymWrapper uses the SAME ALGORITHM for all environments:                      ║
║  "flatten each selected key, then concatenate"                               ║
║                                                                                ║
║  But the KEY SELECTION is ENVIRONMENT-ADAPTIVE, so the final                  ║
║  observation shape and meaning VARIES across environments.                    ║
║                                                                                ║
║  This makes the code PORTABLE but requires careful handling of obs            ║
║  when working with multiple environments.                                     ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
""")
