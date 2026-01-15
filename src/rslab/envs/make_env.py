"""Environment factory for creating RoboSuite environments."""
import os
from dataclasses import dataclass
from typing import Optional

import robosuite as suite

@dataclass
class EnvConfig:
    env_name: str = "Lift"
    robot_name: str = "Panda"

    # rendering
    render: bool = True
    offscreen: bool = False

    # observations
    use_camera_obs: bool = False

    # misc
    reward_shaping: bool = True
    control_freq: int = 20

    # controller (optional)
    controller: Optional[str] = None

def make_env(cfg: EnvConfig):
    # Make rendering consistent across terminals / VSCode on Win11
    os.environ.setdefault("MUJOCO_GL", "wgl")

    env = suite.make(
        env_name=cfg.env_name,
        robots=cfg.robots,
        has_renderer=cfg.render,
        has_offscreen_renderer=cfg.offscreen_render,
        use_camera_obs=cfg.use_camera_obs,
        reward_shaping=cfg.reward_shaping,
        control_freq=cfg.control_freq,
        controller_configs=cfg.controller,  # robosuite accepts None / config depending on version
    )
    return env