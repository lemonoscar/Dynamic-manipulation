# ConveyorVLA handoff — 2026-08-13

## Current state

- Repository: `https://github.com/lemonoscar/Dynamic-manipulation.git`
- Branch: `benchmark/arm-vla-3dgs-v3`
- Baseline commit before repository cleanup:
  `6b285e85bfe72b19707236bf12372c529573ae8a`
- Remote host: SSH alias `4xH20` (`VM-0-3-ubuntu`)
- Remote work root: `/diff/wallx_workspace/dzb`
- Remote repository: `/diff/wallx_workspace/dzb/ConveyorVLA`
- Runtime environment: `/diff/wallx_workspace/dzb/dynamic-isaaclab-5.1-20260804/envs/conveyor_py311`
- Sidecar assets: `/diff/wallx_workspace/dzb/assets/conveyorvla-v3`

No ConveyorVLA collection or training process started by Codex is currently
running. The last collection tmux session has exited.

## Last completed smoke run

Run directory:
`/diff/wallx_workspace/dzb/results/joint-smoke-r23`

- 1/1 successful episode; `training_eligible_count=1`
- Duration: 24.46 s; 611 camera frames
- Target: `cola`; belt speed: 0.01 m/s
- Task outcome: `success`; correct-sort rate: 1.0
- The target exists and moves from episode initialization.
- The post-grasp order is lift, compact-arm retract, base backoff, left turn,
  bin navigation, controlled placement, gripper release, then base motion.
- Raw, M0, M0-mobile, DynamicVLA and ConveyorVLA temporal exports were written.

The inner collector returned 0 and the episode is training-eligible, while the
outer run directory contains `exit_code=2`. Treat that wrapper-code mismatch as
an operations bug if the launcher is reused; it does not invalidate the episode.

## Work intentionally paused

The next planned step was to audit the r23 temporal gates and encode a new
three-camera MP4. It was paused before any new job was launched because the
remote workspace cleanup was requested.

The workspace cleanup retains the 392-episode LeRobot v3 baseline at
`/diff/wallx_workspace/dzb/datasets/conveyorvla-al0-grasp-v1`, base weights at
`/diff/wallx_workspace/dzb/models/base`, and the trained action head plus its
exact config/report/statistics at `/diff/wallx_workspace/dzb/models/conveyorvla-al0`.

## Cleanup contract

Before deleting remote data, preserve:

1. the canonical repository and its clean Git history;
2. the current sidecar asset bundle and its SHA-256 manifest;
3. the minimum working Isaac/Conda runtime;
4. one validated raw episode plus its LeRobot/exported form;
5. the best usable checkpoint and the exact training/model/data configs;
6. a machine-readable inventory and deletion manifest.

Do not touch processes or files outside `/diff/wallx_workspace/dzb`, and do not
delete any item whose ownership or replacement is uncertain.
