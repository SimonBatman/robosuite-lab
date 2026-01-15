"""Sanity check rollout script."""
import numpy as np
import robosuite as suite
import os
os.environ["MUJOCO_GL"] = "wgl"

def main():
    env = suite.make(
        env_name="Lift",
        robots="Panda", 
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    obs = env.reset()
    print("obs keys:", list(obs.keys()))
    print("action_dim:", env.action_dim)
    low, high = env.action_spec
    print("action_spec low/high:", low, high)

    ep_ret = 0.0
    for t in range(200):
        action = np.random.uniform(-1, 1, size=env.action_dim)  # Sample random action 
        obs, reward, done, info = env.step(action)
        ep_ret += reward
        if t % 20 == 0:
            print(f"Step {t}, Reward: {reward:.3f}, EpRet: {ep_ret:.3f}, done: {done}")
        env.render()
        if done:
            break

    print(f"Final EpRet: {ep_ret:.3f}")
    env.close()


if __name__ == "__main__":
    main()