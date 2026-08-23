# Step-500 K10 Seed-147 Arm Stage Evaluation (2026-08-23)

> 后续根因复核说明：本文记录的是当时的诊断证据，不再代表当前 production 执行语义。
> 该 run 跳过不可规划的 target0，执行了 target1；旧数据的 gripper channel 又是测得开度，
> 而不是专家命令。两者已由 `8970dea` 修正。参见
> [操作时序与夹爪监督修正](waypoint_v2_manipulation_sequence_and_gripper_correction_20260823.md)。

## Scope and outcome

This evaluation asks whether the step-500 Waypoint v2 checkpoint can complete the
intermediate milestone `walk -> model-owned route transition -> base hold -> arm
extension/gripper close`. It does not claim grasp success or full-task success.

The milestone passed on seed 147. Qwen changed from `NAV_TO_SOURCE` to `PICK` at
0.4650 m with `P(PICK)=0.9746`; the manipulation command held base velocity at
exactly zero, cuRobo selected a collision-free model target, the TCP translated
0.22335 m, the six arm joints moved 2.02588 rad in L2, and both gripper joints
closed from about 0.0273 rad to about 0.0001 rad.

The episode still ended with `manipulation_chunk_timeout`: the final maximum
joint error was 0.04359 rad versus the unchanged 0.03 rad completion tolerance.
This is therefore a stage success, not a complete pick or task success.

## Root-cause chain

1. The original local continuity gate rejected the first predicted PICK target
   before cuRobo. The current-to-target translation was 0.26072 m versus the
   0.15 m limit; the maximum wrapped per-axis rotation was 179.62 degrees versus
   35 degrees. The target itself was finite and inside the configured workspace.
2. A simulation-only diagnostic disabled only the translation/rotation step
   limits. Workspace, gripper range, collision, IK, plan error, per-cycle joint
   rate, and base-hold checks remained active. The first predicted target then
   reached cuRobo but had no feasible direct-pose plan.
3. Replaying the checkpoint's own target index 1 with the same joints and
   collision scene produced a valid 41-point plan in 4.56 seconds. This proved
   that the ARM chunk was not wholly unusable and exposed the runtime's fixed
   index-0 selection as the next blocker.
4. The diagnostic selector was changed to try model targets in predicted order
   and execute the first target accepted by cuRobo/IK. It never uses task truth,
   changes the Qwen route, or synthesizes an external target.
5. The first successful closed-loop retry still moved only 0.05356 rad in joint
   L2 because the controller required measured convergence at every interpolated
   path point. One joint settled at approximately 0.0315 rad error against a
   0.0300 rad intermediate threshold, permanently blocking path advancement.
6. The controller was aligned with the reference arm-vla trajectory semantics:
   play every pre-validated intermediate cuRobo point in order, while retaining
   measured convergence for the final point. The existing 0.15 rad per-cycle
   joint-command check remains fail-closed.

## Final seed-147 evidence

| Measure | Result |
|---|---:|
| Checkpoint | `step_000500@7ec8424cc7d1` |
| Runtime commit | `d1419b6be60199537451d6eba90bca526cbb421b` |
| Effective NAV prefix cap | 10 |
| Route trace | `NAV_TO_SOURCE x4 -> PICK` |
| PICK distance | 0.4650 m |
| `P(PICK)` | 0.9746 |
| Manipulation base command | `[0.0, 0.0, 0.0]` |
| Selected model target | index 1 |
| Successful cuRobo path | 61 points, collision-free |
| cuRobo target position error | `5.96e-8` m |
| cuRobo target orientation error | `0.0` rad |
| Arm joint L2 motion | 2.02588 rad |
| TCP translation | 0.22335 m |
| Gripper joint 7 | 0.02730 -> 0.00012 rad |
| Gripper joint 8 | 0.02729 -> 0.00013 rad |
| Final base velocity | six measured components all zero |
| Video | 23.08 s, 25 FPS, 577 frames, overview/head/wrist |
| Terminal reason | `manipulation_chunk_timeout` |

## Code and verification

- `7bd6461`: simulation-only ARM target step-limit diagnostic with explicit trace.
- `921291f`: first-plannable selection over model-predicted ARM targets only.
- `d1419b6`: ordered cuRobo path playback with final measured convergence.
- 39 contract, runtime, planner-adapter, and controller tests passed locally.

The evidence videos, query frames, trace, launchers, and run manifest remain
machine-private artifacts and are intentionally excluded from Git.
