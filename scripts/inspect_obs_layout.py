import os
os.environ.setdefault("MUJOCO_GL", "wgl")

import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper


def main():
    rs_env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # IMPORTANT: get raw dict obs BEFORE flattening
    raw_obs = rs_env.reset()

    # build layout
    keys = sorted(raw_obs.keys())
    offset = 0
    layout = {}
    for k in keys:
        v = np.array(raw_obs[k]).ravel()
        n = v.size
        layout[k] = (offset, offset + n, n)
        offset += n

    print("Total obs dim:", offset)
    print("---- layout (key: [start:end) len) ----")
    for k in keys:
        s, e, n = layout[k]
        print(f"{k:30s}: [{s}:{e}) len={n}")

    # highlight cube_pos if exists
    for cand in ["cube_pos", "object_pos", "object-state", "object_state"]:
        if cand in layout:
            s, e, _ = layout[cand]
            print(f"\nFOUND {cand}: slice [{s}:{e})")

    rs_env.close()


if __name__ == "__main__":
    main()
