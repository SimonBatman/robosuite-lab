# robosuite-lab

Personal robosuite learning lab scaffold — tools to train/evaluate manipulation tasks (Panda + Lift baseline).

## One-line
Conda env → pip install deps → train PPO → eval with rendering.

---

## Quickstart (Windows / conda)

### 1. Clone & enter
```bash
git clone https://github.com/SimonBatman/robosuite-lab.git
cd robosuite-lab
```

### 2. Create conda env (recommended)

```bash
conda create -n robosuite python=3.10 -y
conda activate robosuite
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you plan to train on GPU, install the CUDA-matched `torch` first, then install the rest:
>
> 1) Follow the PyTorch install guide for your CUDA version to get the correct pip command  
> 2) Run: `pip install -r requirements.txt`

### 4. Run quick sanity rollout (visual)

```bash
python scripts/sanity_rollout.py
```

### 5. Train PPO baseline (state obs)

```bash
python scripts/train_sb3_ppo.py
```

### 6. Evaluate trained model (rendered window)

```bash
python scripts/eval_sb3_render.py
```

* `eval` will automatically load `outputs/checkpoints/ppo_lift_panda.zip` and the corresponding `vecnorm_lift.pkl` if present.
* If an `obs_layout` JSON exists for the environment, `eval` will use it to extract metrics (success rate / success step).

---

## Files & structure

* `src/rslab/` — package code (env factory, wrappers, runners)
* `scripts/` — convenience scripts: `sanity_rollout.py`, `train_sb3_ppo.py`, `eval_sb3_render.py`, `inspect_and_dump_layout.py`
* `outputs/` — logs, checkpoints, videos (ignored by git)

---

## obs layout & reproducibility

We record the flattening layout used by the `GymWrapper` into `outputs/checkpoints/obs_layout_{env}_{robot}.json`.
This ensures eval scripts and metrics always use the same indexing as training, avoiding hard-coded indices.

Run:
```bash
python scripts/inspect_and_dump_layout.py
```

to generate the layout JSON.

---

## Common troubleshooting

* If you hit `OMP Error #15`: start script with:

```bash
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python scripts/train_sb3_ppo.py
```

* If `numpy`/`torch` import errors occur, re-install stable packages with conda-forge:

```bash
pip uninstall -y numpy scipy
conda install -y -c conda-forge "numpy<2" scipy
pip install -r requirements.txt
```

---

## How I recommend using the repo

* Keep `outputs/` local (do not commit)
* Use separate branch per feature, open PRs for review
* For experiments, tag runs: `git tag v0.1-baseline-lift` and push the tag
