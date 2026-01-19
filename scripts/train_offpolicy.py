#!/usr/bin/env python3
"""
scripts/train_offpolicy.py

Train SAC or TD3 on a robosuite environment using SB3.
- GPU selection via --gpu_id (uses torch.cuda.set_device; does NOT alter CUDA_VISIBLE_DEVICES)
- Saves models, vecnormalize stats and eval logs (evaluations.npz).
"""

import os
import json
import argparse
import numpy as np
import torch

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

# Import your env factory (make sure path is correct)
from rslab.envs.make_sb3_env import SB3EnvConfig, make_sb3_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["SAC", "TD3"], default="SAC", help="Algorithm")
    p.add_argument("--env", default="Lift", help="robosuite env name")
    p.add_argument("--robot", default="Panda", help="robot name")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--total-timesteps", type=int, default=int(2e6), help="total timesteps")
    p.add_argument("--num-envs", type=int, default=1, help="num envs for data collection (SubprocVecEnv if >1)")
    p.add_argument("--eval-freq", type=int, default=10000, help="eval frequency (steps)")
    p.add_argument("--n-eval-episodes", type=int, default=5, help="eval episodes")
    p.add_argument("--log-dir", default="outputs/checkpoints", help="where to save runs")
    p.add_argument("--gpu_id", type=int, default=-1,
                   help="GPU id to use (>=0). -1: auto choose cuda if available, else cpu.")
    p.add_argument("--torch-threads", type=int, default=0, help="If >0 set torch.set_num_threads(n)")
    p.add_argument("--num-cpu-cores", type=int, default=0, help="(optional) set OMP_NUM_THREADS/MKL_NUM_THREADS")
    return p.parse_args()


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def choose_device(gpu_id: int):
    """
    Choose device string and set current process device (without modifying CUDA_VISIBLE_DEVICES).
    - gpu_id >= 0 : try to use cuda:gpu_id (physical index)
    - gpu_id == -1: auto choose 'cuda' if available else 'cpu'
    Returns a torch.device object.
    """
    if gpu_id >= 0:
        if not torch.cuda.is_available():
            print("[WARN] CUDA not available. Falling back to CPU.")
            return torch.device("cpu")
        n = torch.cuda.device_count()
        if gpu_id >= n:
            print(f"[WARN] Requested gpu_id={gpu_id} but only {n} devices available. Falling back to CPU.")
            return torch.device("cpu")
        try:
            torch.cuda.set_device(gpu_id)
        except Exception as e:
            print(f"[WARN] torch.cuda.set_device({gpu_id}) failed: {e}. Falling back to CPU.")
            return torch.device("cpu")
        dev = torch.device(f"cuda:{gpu_id}")
        print(f"[INFO] Using GPU device {dev} (physical id {gpu_id}).")
        return dev
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Auto device selection -> {dev}")
        return dev


def setup_torch(seed: int, device: torch.device, torch_threads: int = 0, num_cpu_cores: int = 0):
    # seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # allow cudnn autotuner when input sizes fixed
        torch.backends.cudnn.benchmark = True
    # set CPU threads if requested
    if torch_threads and torch_threads > 0:
        torch.set_num_threads(torch_threads)
    # optionally set OMP/MKL threads
    if num_cpu_cores and num_cpu_cores > 0:
        os.environ["OMP_NUM_THREADS"] = str(num_cpu_cores)
        os.environ["MKL_NUM_THREADS"] = str(num_cpu_cores)


def make_env_fn(seed, env_name, robots, flatten_obs=True):
    def _thunk():
        cfg = SB3EnvConfig(
            env_name=env_name,
            robots=robots,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            controller=None,
            flatten_obs=flatten_obs,
        )
        env = make_sb3_env(cfg)
        try:
            env.seed(seed)
        except Exception:
            # some wrappers may not have seed()
            pass
        return Monitor(env)
    return _thunk


def get_algo_class(name: str):
    if name == "SAC":
        return SAC
    if name == "TD3":
        return TD3
    raise ValueError(name)


def main():
    args = parse_args()

    # Device selection & torch setup
    device = choose_device(args.gpu_id)
    setup_torch(seed=args.seed, device=device, torch_threads=args.torch_threads, num_cpu_cores=args.num_cpu_cores)

    # reproducibility helper for SB3 and numpy
    set_random_seed(args.seed)
    np.random.seed(args.seed)

    # prepare directories
    run_name = f"{args.algo.lower()}_{args.env.lower()}_{args.robot.lower()}_seed{args.seed}"
    run_dir = os.path.join(args.log_dir, run_name)
    ensure_dir(run_dir)

    # Build training env(s)
    if args.num_envs > 1:
        env_fns = [make_env_fn(args.seed + i, args.env, args.robot) for i in range(args.num_envs)]
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv([make_env_fn(args.seed, args.env, args.robot)])

    # VecNormalize for training (obs normalized; reward not norm'ed for clearer eval)
    vec_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Eval env (single)
    eval_env = DummyVecEnv([make_env_fn(args.seed + 1000, args.env, args.robot)])
    eval_vec = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Algorithm selection and kwargs
    Algo = get_algo_class(args.algo)
    common_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tb"),
        seed=args.seed,
        device=str(device),  # pass device to SB3 (string accepted)
    )

    if args.algo == "SAC":
        algo_kwargs = dict(
            **common_kwargs,
            learning_rate=3e-4,
            buffer_size=int(1e6),
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    else:  # TD3
        algo_kwargs = dict(
            **common_kwargs,
            buffer_size=int(1e6),
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            policy_delay=2,
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    model = Algo(**algo_kwargs)

    # Setup EvalCallback
    eval_callback = EvalCallback(
        eval_env=eval_vec,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    print(f"[INFO] Starting training: {run_name} on device {device}")
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    # Save final model and vecnormalize (training) stats
    model.save(os.path.join(run_dir, "final_model"))
    vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    # Save metadata
    meta = {
        "algo": args.algo,
        "env": args.env,
        "robot": args.robot,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "device": str(device),
        "num_envs": args.num_envs
    }
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] Training finished. Results in:", run_dir)


if __name__ == "__main__":
    main()
