import os
from dataclasses import dataclass
from typing import Optional

import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper


@dataclass
class SB3EnvConfig:
    env_name: str = "Lift"
    robots: str = "Panda"

    # training speed defaults (state-based)
    has_renderer: bool = False
    has_offscreen_renderer: bool = False
    use_camera_obs: bool = False

    reward_shaping: bool = True
    control_freq: int = 20

    # composite controller name (e.g. "BASIC") or path to a controller json (optional)
    controller: Optional[str] = "BASIC"

    # GymWrapper
    flatten_obs: bool = True


def make_sb3_env(cfg: SB3EnvConfig):
    # Win11 / VSCode consistency (you already validated this)
    os.environ.setdefault("MUJOCO_GL", "wgl")

    rs_env = suite.make(
        env_name=cfg.env_name,
        robots=cfg.robots,
        has_renderer=cfg.has_renderer,
        has_offscreen_renderer=cfg.has_offscreen_renderer,
        use_camera_obs=cfg.use_camera_obs,
        reward_shaping=cfg.reward_shaping,
        control_freq=cfg.control_freq,
        controller_configs=None,
    )

    # Robosuite -> Gym/Gymnasium style env
    gym_env = GymWrapper(rs_env, flatten_obs=cfg.flatten_obs)
    return gym_env
