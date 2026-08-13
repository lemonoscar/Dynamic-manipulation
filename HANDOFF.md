# ConveyorVLA handoff — 2026-08-13

## Canonical state

- Repository: `https://github.com/lemonoscar/Dynamic-manipulation.git`
- Branch: `benchmark/arm-vla-3dgs-v3`
- Remote host: SSH alias `4xH20` (`VM-0-3-ubuntu`)
- Remote work root: `/diff/wallx_workspace/dzb`
- Remote repository: `/diff/wallx_workspace/dzb/ConveyorVLA`
- Isaac environment:
  `/diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804/envs/conveyor_py311`
- LeRobot environment:
  `/diff/wallx_workspace/dzb/.conda-envs/conveyorvla-al0-lerobot044`

The repository has one implementation tree: `assets/`, `configs/`, `docs/`,
`scripts/`, `src/` and `tests/`. Historical V1/V2/V3 names remain only where
they identify a protocol or data schema; they are not parallel source trees.

## Clean remote layout

The destructive cleanup requested on 2026-08-13 completed successfully. The
work root now has exactly eight top-level entries:

```text
.conda-envs/
ConveyorVLA/
assets/
datasets/
dynamic-isaaclab-5.1-20260804/
models/
results/
workspace-manifest/
```

`workspace-manifest/` contains the pre-cleanup inventory, exact 200-path
top-level deletion list, nested/internal deletion lists, retained paths,
validation reports and cleanup exit codes. Package/download caches, duplicate
repositories, old Git bundles, superseded experiments and launch fragments
were removed. Offline Isaac recovery materials (`kit-portable`, local wheels,
the IsaacLab bundle and source) were deliberately retained.

No process outside the work root and no GPU workload owned by another user was
modified.

## Retained data and models

- Sidecar assets: `assets/conveyorvla-v3` (68 files; 2,563,083,283 bytes in the
  transfer manifest).
- Training set: `datasets/conveyorvla-al0-grasp-v1` (LeRobot v3, 392 episodes,
  29,155 frames, 5 Hz query rate, four H.264 video features).
- Base models: `models/base` (Qwen3-VL-4B-Instruct, ABot-M0-Robocasa and the
  config-listed VGGT-1B weight).
- Trained output: `models/conveyorvla-al0` (action safetensors, exact config,
  state statistics, training report and log).
- Final training report: 10,000 steps, final loss `0.0011057633673772216`,
  final gradient norm `0.08196507394313812`, `ok=true`.
- Latest raw evidence: `results/joint-smoke-r23`.
- Retained demo: `results/demos/conveyorvla_joint_r22_success.mp4`.

The saved training report contains the old dataset location in
`sources[].lerobot_root` as immutable provenance. The active dataset path is the
canonical path above. Likewise, use
`workspace-manifest/current_successful_episode_roots.txt` instead of the
historical pointer inside r23.

## Validation evidence

- Remote Python environment and robot asset check: passed.
- Repository test suite: `328 passed`.
- Model artifact size audit: passed for all registered base artifacts and the
  677,736,536-byte trained action model.
- LeRobot runtime: `lerobot==0.4.4`, PyAV 15.1.0, PyArrow 25.0.0.
- Four LeRobot video features: H.264, 224×224, first-frame PyAV decode passed.
- r23 schema validation: 1 run, 1 episode, 1,223 steps, 1,223 object records,
  1,833 PNG frames.
- r23 quality audit: clean, task success, `training_eligible=true`.
- r23 camera gate: passed; head and wrist are policy cameras, overview remains
  observer-only.

The full model and sidecar SHA-256 results are recorded under
`workspace-manifest/validation/` and the expected model hashes are in
`workspace-manifest/core_model_hashes.sha256`.

## Last successful task and next work

r23 completed the full teacher sequence: navigation to the conveyor, continuous
target following, dynamic grasp, vertical lift, compact-arm retract, base
backoff, left turn, loaded navigation, parked placement and gripper release.
It lasted 24.46 s and produced 611 frames per camera. The inner collector and
all data gates succeeded; the historical outer `exit_code=2` is a launcher
status-propagation bug, not an invalid episode.

The next engineering gate is not more cleanup. It is to register and validate
three additional training-grade rigid objects, freeze a second belt speed, run
GPU 2/3 pilots for all eight object-speed cells, and only then start the
384-success-episode collection matrix. Teacher success must not be reported as
VLA closed-loop success.
