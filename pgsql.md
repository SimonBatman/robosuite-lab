robosuite-lab/
├─ README.md
├─ pyproject.toml
├─ requirements/
│  ├─ base.txt
│  ├─ dev.txt
│  ├─ train.txt
│  └─ server.txt
├─ configs/
│  ├─ env/
│  │  ├─ lift.yaml
│  │  ├─ door.yaml
│  │  └─ peg_in_hole.yaml
│  ├─ controller/
│  │  ├─ panda_osc_pose_precise.json
│  │  ├─ panda_osc_pose_fast.json
│  │  └─ notes.md
│  ├─ algo/
│  │  ├─ bc.yaml
│  │  ├─ ppo.yaml
│  │  └─ sac.yaml
│  └─ runtime/
│     ├─ local.yaml
│     └─ server_headless.yaml
├─ src/
│  └─ rslab/
│     ├─ __init__.py
│     ├─ envs/
│     │  ├─ make_env.py
│     │  ├─ wrappers/
│     │  │  ├─ flatten_obs.py
│     │  │  ├─ action_clip.py
│     │  │  ├─ frame_stack.py
│     │  │  └─ time_limit.py
│     │  ├─ custom_tasks/
│     │  │  ├─ __init__.py
│     │  │  └─ my_task_template.py
│     │  └─ assets/
│     │     └─ (可选：自定义 xml/mesh/texture)
│     ├─ controllers/
│     │  ├─ load_controller.py
│     │  └─ presets.py
│     ├─ datasets/
│     │  ├─ hdf5_io.py
│     │  └─ demos/
│     │     └─ (采集的示教数据)
│     ├─ algo/
│     │  ├─ __init__.py
│     │  ├─ bc/
│     │  │  ├─ train.py
│     │  │  └─ model.py
│     │  ├─ rl/
│     │  │  ├─ ppo.py
│     │  │  └─ sac.py
│     │  └─ common/
│     │     ├─ logger.py
│     │     ├─ replay_buffer.py
│     │     └─ eval.py
│     ├─ runners/
│     │  ├─ rollout.py
│     │  ├─ train.py
│     │  ├─ eval.py
│     │  └─ collect_demos.py
│     ├─ utils/
│     │  ├─ seeding.py
│     │  ├─ video.py
│     │  ├─ timing.py
│     │  └─ paths.py
│     └─ cli.py
├─ scripts/
│  ├─ sanity_rollout.py
│  ├─ teleop_keyboard.py
│  ├─ list_envs_and_robots.py
│  ├─ render_from_policy.py
│  ├─ export_video.py
│  └─ server_smoke_test.py
├─ experiments/
│  ├─ 000_sanity/
│  ├─ 010_lift_bc/
│  └─ 020_lift_rl/
├─ outputs/
│  ├─ logs/
│  ├─ checkpoints/
│  └─ videos/
├─ tests/
│  ├─ test_env_make.py
│  ├─ test_flatten_obs.py
│  └─ test_determinism.py
├─ docs/
│  ├─ troubleshooting_win.md
│  ├─ headless_server.md
│  └─ controller_tuning.md
└─ .gitignore
