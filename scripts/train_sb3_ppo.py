import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from rslab.envs.make_sb3_env import SB3EnvConfig, make_sb3_env


def make_env_fn():
    def _thunk():
        cfg = SB3EnvConfig(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            controller="BASIC",
            flatten_obs=True,
        )
        env = make_sb3_env(cfg)
        env = Monitor(env)
        return env
    return _thunk

def main():
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    vec_env = DummyVecEnv([make_env_fn()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log="outputs/logs/tb",
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(total_timesteps=200_000)
    model.save("outputs/checkpoints/ppo_lift_panda.zip")
    vec_env.save("outputs/checkpoints/vecnorm_lift.pkl")


if __name__ == "__main__":
    main()