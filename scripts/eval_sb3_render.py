import os
import numpy as np

# ---- (optional but recommended on Windows) avoid OpenMP crash / oversubscription ----
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rslab.envs.make_sb3_env import SB3EnvConfig, make_sb3_env


MODEL_PATH = "outputs/checkpoints/ppo_lift_panda.zip"
VECNORM_PATH = "outputs/checkpoints/vecnorm_lift.pkl"

# From your obs layout: cube_pos slice [0:3], z is index 2
CUBE_Z_INDEX = 2

# Success criterion: lift height relative to episode start
LIFT_DELTA_SUCCESS = 0.05  # 5 cm

# If success is achieved, end episode early
EARLY_STOP_ON_SUCCESS = True

# Safety cap in case something goes wrong (should be 1000 for Lift)
MAX_STEPS_PER_EP = 1000


def main():
    def _make():
        cfg = SB3EnvConfig(
            env_name="Lift",
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            controller=None,
            flatten_obs=True,
        )
        return make_sb3_env(cfg)

    env = DummyVecEnv([_make])

    env = VecNormalize.load(VECNORM_PATH, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH)

    n_episodes = 10
    success_count = 0
    returns = []
    success_steps = []

    for ep in range(n_episodes):
        obs = env.reset()
        ep_return = 0.0

        init_z = None
        max_z = -np.inf
        success = False
        first_success_step = None

        for t in range(1, MAX_STEPS_PER_EP + 1):
            # obs is (1, obs_dim)
            cube_z = float(obs[0][CUBE_Z_INDEX])
            if init_z is None:
                init_z = cube_z
                max_z = cube_z

            max_z = max(max_z, cube_z)
            lift_delta = max_z - init_z

            if (not success) and (lift_delta > LIFT_DELTA_SUCCESS):
                success = True
                first_success_step = t

                if EARLY_STOP_ON_SUCCESS:
                    # End this episode early (evaluation shortcut)
                    # Note: we still do one more env.step below only if you want the viewer to show the moment.
                    # Here we will just break after logging at end of loop.
                    pass

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_return += float(reward[0])

            # Do NOT call env.render(): robosuite viewer updates when has_renderer=True.

            # natural termination (usually TimeLimit at 1000)
            if done[0]:
                break

            # early stop right after success is detected (faster eval)
            if EARLY_STOP_ON_SUCCESS and success:
                break

        returns.append(ep_return)
        if success:
            success_count += 1
            success_steps.append(first_success_step if first_success_step is not None else t)

        print(
            f"[EVAL] ep={ep:02d} return={ep_return:8.3f} "
            f"steps={t:4d} success={success} "
            f"success_step={first_success_step}"
        )

    success_rate = success_count / n_episodes
    avg_return = float(np.mean(returns)) if returns else float("nan")
    avg_success_steps = float(np.mean(success_steps)) if success_steps else float("nan")

    print("\n[EVAL SUMMARY]")
    print(f"success_rate     = {success_rate:.2%} ({success_count}/{n_episodes})")
    print(f"avg_return       = {avg_return:.3f}")
    print(f"avg_success_step = {avg_success_steps:.1f}  (lower is better)")

    env.close()


if __name__ == "__main__":
    main()
