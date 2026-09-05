# Manipulation-Navi / liangzhuNeW_500 training-readiness audit

Date: 2026-09-04 (Asia/Shanghai)

## Decision

The current `Manipulation_Navi_v1` model and training configuration pass the
architecture, initialization, schedule, optimizer, loss, routing, and runtime
contract checks. After the user confirmed that no complete high-rate logs are
available, the breaking data/model/runtime contract was explicitly versioned
at 5 Hz. All 500 `OscarXu/liangzhuNeW_500` episodes were materialized without
interpolation and the immutable release passed its post-publication audit.

## Source and model identity

- Branch: `Manipulation_Navi_v1`
- Audited commit: `134cf61fc13b0eca5c9c985812dece556c2ac1ca`
- Action architecture: `ABotM0LastHiddenDualDiT`
- VLM feature source: the final Qwen hidden-state sequence, width 2560
- Action experts: separate NAV and Mani ABot-M0-compatible DiTs, 16 blocks,
  even blocks cross-attend to the VLM sequence and odd blocks perform action
  self-attention
- Initialization mode: `abot_m0_pretrain_strict_domain_transfer`
- Source model: `amap_cvlab/ABot-M0-Pretrain`
- Checkpoint SHA256:
  `94478682b5c9eecf6f02179ba67ae47ea41257ca059bea6dd20e161716f5e16b`

The local Qwen3-VL directory supplies the model skeleton and processor. It is
not the training initialization source. The trainer strictly transfers the
Qwen tensors and both ABot action trunks from the pinned ABot-M0 checkpoint,
then reinitializes only task/domain boundary parameters and new special-token
rows. A previous initialization report recorded 714 loaded Qwen tensors and
240/242 loaded NAV/Mani action keys.

## Configuration audit

The config validator now fails closed on the critical VLM, router, action,
loss, stage, optimizer, initialization, sampling, route-transition, and
runtime fields. The formal schedule is taken from the audited config rather
than duplicated defaults. A fresh run also records the tracked source patch,
its SHA256, and dirty-worktree status for reproducibility.

Validation evidence:

- Full test suite: 548/548 passed (the only in-sandbox failures were two tests that
  require localhost sockets; both passed outside the socket-restricted sandbox)
- `git diff --check`: passed
- ABot checkpoint independent SHA256: matched the pinned value
- Available launch topology at audit time: GPU 2 and GPU 3, NVIDIA H20,
  approximately 97.4 GB free per GPU

## Dataset identity and integrity

- Dataset: `OscarXu/liangzhuNeW_500`
- ModelScope revision:
  `6806fadf2e8e125ca871f576676b60e7db1605dc`
- Local root: `artifacts/datasets/modelscope-liangzhuNeW_500`
- Archives: 20, with 25 episodes per archive
- Episode rows: 500 unique members
- Archive payload size: 25,026,600,960 bytes
- Incomplete downloads: 0
- Archive SHA256: all 20 independently matched the ModelScope file manifest

The full 500-episode scan found no packaging or reference-integrity problems:

- 500/500 outer success flags are true
- 500/500 outer collection-quality gates pass
- 500/500 inner manifests are readable
- 104,454/104,454 sampled actions have dimension 11
- every adjacent sampled step is 10 control ticks / 0.2 seconds
- 0 missing referenced front/wrist images

The published training release is:

- Root: `${DATA_RELEASES_ROOT}/liangzhuNeW_500-joint-trajectory-5hz-v1`
- Schema: `conveyorvla-joint-trajectory-5hz-v1`
- Profile: `conveyorvla-liangzhunew500-sampled-control-5hz-v1`
- Episodes: 500; train/val/test: 400/50/50, episode-disjoint
- Derived rows: 96,856; train/val/test: 77,213/9,587/10,056
- Unique copied images: 197,712
- Manifest SHA256:
  `aa5f0f3dbef18985e3f25c6d98d8ea573f48f398ac2d153bd7857e27b862789d`
- Normalizer: train-only,
  `joint-trajectory-normalizer:44ee9d21bfd916587c23161e`
- Post-publication audit: passed with zero problems

## Explicit 5 Hz contract resolution

Every inner manifest reports:

- control loop: 50 Hz
- saved dataset: 5 Hz
- capture stride: 10 control ticks
- `vla_training_action_available=false`
- `vla_training_eligible=false`
- reason: `lerobot_export_not_training_ready`

Across all saved `frames.jsonl` and `samples.jsonl` members:

- episodes containing `q_command_applied`: 0/500
- episodes containing `gripper_command_applied`: 0/500

The original 25 Hz Mani contract could not be supported because the snapshot
does not contain ten post-saturation controller commands at 0.04-second
stride. The dataset adapter therefore does not reconstruct them and never
opens the CuRobo plan JSON. The replacement contract uses only the saved 5 Hz
rows:

- head image, wrist image, measured q/dq and action are joined by exact
  `frame_index`, `simulation_step`, and timestamp;
- state is read from `frames.jsonl.observation`, not the one-step-later
  `post_step_observation`;
- Mani target provenance is `sampled_control_target_5hz`;
- NAV target provenance is `sampled_future_base_pose_5hz`;
- both action domains now use 0.20-second points; Mani `[10,7]` covers 2.0 s;
- runtime executes every Mani point for ten 50 Hz control ticks;
- gripper metres are averaged and clipped to the declared `[0,0.04]` source
  joint range before mapping to `[0,1]`;
- unavailable route-physical progress remains masked and its loss weight is
  zero; elapsed time and row index are still forbidden fallbacks.

Each successful episode contributes four ordered routes. Exactly four query
rows per episode are dropped: the last row of each route has no later target
inside that route. Short real prefixes near a route/success boundary are
terminal-held to ten targets; no missing point is interpolated. Planning and
verification observations are assigned to their adjacent executable route so
that boundary pairs remain consecutive in the saved 5 Hz stream instead of
pretending that seconds of skipped observations do not exist.

The old inner `vla_training_action_available=false` declaration is preserved
as a source fact: it describes the unavailable legacy LeRobot export. It is
not silently rewritten. Eligibility is provided by this new, separately
versioned 5 Hz joint-trajectory schema and its strict adapter/audit.

## Formal training launch

The complete two-data-equivalent-epoch run was launched without `--overfit`
and remains active:

- Run root:
  `${TRAINING_RUNS_ROOT}/conveyorvla-abot-m0-liangzhunew500-5hz-formal-20260905-r1`
- Persistent session: `conveyorvla_lz500_5hz_formal_r1`
- Device: GPU 1; world size 1, micro-batch 2, gradient accumulation 32,
  effective global batch 64
- Eligible train rows: 77,213; epoch steps: 1,207; Stage A ends at step 302;
  formal terminal step: 2,414; checkpoint interval: 250
- At stable-start acceptance: `status=running`, step 12, 12 consecutive finite
  optimizer events, latest loss `18.480487823486328`, gradient norm
  `24.47745132446289`, 32 NAV + 32 Mani rows and four boundary pairs
- Latest optimizer-step time: 12.67 s; process alive; no OOM, NaN, data-worker
  exit, or throughput collapse

The ABot initialization report created by this run records 714 transferred
Qwen tensors, 240 transferred NAV action tensors, and 242 transferred Mani
action tensors. Only special-token and domain input/output boundary parameters
were reinitialized. The goal is closed at stable launch; this does not stop or
declare completion of the ongoing 2,414-step training process.
