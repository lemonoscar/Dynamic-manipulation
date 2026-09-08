# Independent evidence audit: step 1700

Audit date: 2026-09-08. Read-only review of completed open-loop artifacts and physical attempts 03/04. No GPU work, model changes, remote writes, or additional attempts were performed by this reviewer. Attempts 05/06 are outside this snapshot.

## Identity and open-loop coverage

Both complete reports bind checkpoint step 1700, SHA256 `474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0`. Validation and test protocol objects are identical. The prediction and transition file hashes independently recomputed from local copies match the reports. Validation has 1022 action queries / 3066 sampled blocks / 66 transition queries; test has 1094 / 3282 / 66. Seeds are 17, 29, 43. Both action and transition coverage flags are true.

Actions are conditional on the known active task and causal prior task memory; this is not autonomous route selection. Transition generation uses RGB, instruction, and input task context. Offline event times are used only for scoring strata. Validity masks are nonempty contiguous prefixes; errors use only those valid prefixes. The complete sampled prediction block remains subject to the independent saturation gate, including its unlabeled tail. This distinction is intentional.

### Saturation gate recomputation

| Split | Mani blocks | Position events | Rate events | Gripper events | Total events / (blocks × 70) | Gate ≤0.5% |
|---|---:|---:|---:|---:|---:|---|
| validation | 1281 | 2611 | 62 | 5596 | 9.221590% | failed |
| test | 1263 | 2948 | 79 | 5273 | 9.388078% | failed |

These counts reproduce the reported sample-mean rates exactly. Position/rate/gripper clipping events are not deduplicated; the metric preserves the frozen historical definition. Geometry or physical task success cannot be inferred from these open-loop errors.

### Gripper weighting and PLACE failure

The short results report uses the mean of each query/seed block's valid-prefix accuracy. Short partial prefixes therefore receive the same block weight as full prefixes. This is not a mask error; it differs from pooling all valid points. Family-equal statistics are separately available in the JSON report.

| Split / route | Block-equal accuracy | Pooled valid-point accuracy | Correct / valid points | Family-equal accuracy |
|---|---:|---:|---:|---:|
| validation PICK | 92.582011% | 91.959366% | 5341 / 5808 | 91.718818% |
| validation PLACE | 47.250471% | 39.174810% | 1804 / 4605 | 47.245098% |
| test PICK | 94.730956% | 93.572650% | 5474 / 5850 | 94.746285% |
| test PLACE | 45.492662% | 37.649880% | 1570 / 4170 | 45.490132% |

All 2600 incorrect test PLACE points are target-closed / predicted-open at the frozen 0.5 fraction threshold. There are 190 correct closed points and 1380 correct open points, with zero target-open / predicted-closed errors. Thus the low PLACE accuracy is specifically premature/excessive opening in this conditional dataset, not a balanced random error. It remains an open-loop finding, not proof of observed physical drops.

Transition JSON is valid for 66/66 queries in each split. Validation gets 34/66 correct and test 41/66; test confusion is CONTINUE→CONTINUE 23, CONTINUE→ADVANCE 10, ADVANCE→CONTINUE 15, ADVANCE→ADVANCE 18. These labels cover neither FINISH nor REPAIR_SUFFIX.

## N physical pair: attempts 03 and 04

Both are new IsaacSim5.1 state-initialized fixed PICK diagnostics at source family `liangzhu_seed_16100019`, source query tick 180. They are not Sim6 solver restores, full episodes, or ordinary autonomous policy runs. No grasp assist is enabled, while manipulation base/support locks remain declared. Strict contact is unknown; neither run verifies pick, carry, release, or geometry hold. Full-task and strict-full success remain null.

Both traces contain 91 consecutive 50 Hz physical control steps: 20 visual-history wait steps and 71 actual model-driven `joint_trajectory_pick` applications. The summary field `queue_applied_points=71` counts control applications, not 71 independent 5 Hz predictions. Both runs advance 1.82 seconds of simulated time with inference pausing physics.

### First safety trigger, checked by joint name

The articulation ordering is interleaved with locomotion joints. This audit located `arm_joint2` using `state_after.joint_names`, not an assumed array slot.

| Branch | First violation | Joint | Measured speed | Frozen bound | Saturation |
|---|---|---|---:|---:|---:|
| RTC off | tick 91 / 1.82 s | arm_joint2 | 3.407499074935913 rad/s | 3.0 rad/s | 48 / 280 = 17.142857% |
| RTC on | tick 91 / 1.82 s | arm_joint2 | 5.0486063957214355 rad/s | 3.0 rad/s | 46 / 280 = 16.428571% |

No earlier arm-joint velocity exceeded 3.0 rad/s in either trace. The off branch's maximum across the first 90 steps was 1.1943968534469604 rad/s. Each final violating state is immediately followed by `safety_stop` and no subsequent control applications. The target interval rate cap is not a guarantee on 50 Hz measured motor velocity. Both deployment and saturation gates fail. RTC's somewhat lower clipping-event count does not compensate for its larger measured-speed violation in this pair.

### Queries and timing

RTC off makes four actual HTTP/model calls at simulated times 0.4, 0.8, 1.2, 1.6 s. All responses state `rtc_applied=false`. Client round trips are 0.723805, 0.181959, 0.178758, 0.183656 s; service durations are 0.680796, 0.170620, 0.167440, 0.170920 s. The first is a cold call and should be reported separately from the three warm calls.

RTC on has four request/response log entries but only three new HTTP/model calls: its first result explicitly adopts the shared off-branch proposal. The first local elapsed value 0.005547 s is file adoption, not inference, and the response's 0.680796 s service field is inherited from the original off call. Exclude both from on-branch fresh inference statistics. The three genuine VJP calls have client round trips 0.264291, 0.220209, 0.218740 s and service durations 0.250138, 0.208597, 0.207005 s. Their requests contain the actual prior trajectory/weights and their responses state `rtc_applied=true`.

These timings use client-local or server-local elapsed measurements; none requires subtracting different machines' monotonic clocks. Simulation is paused during inference, so they cannot establish sustained real-time deployment.

### Shared prediction and visual mismatch

The first physical action proposals are exactly equal. All physical state snapshots and applied commands through control tick 40 are equal. The first applied arm-command/state difference occurs at tick 41, after the shared first two 5 Hz points. Shared-first-plan SHA256 is `0341c99059d60e61b6202e423d2ed6507b07c397f01b83507e867dc6f48a0f50`.

The real RGB inputs are not byte-identical. CPU JPEG decoding shows small but nonzero differences, also present at query 1 before the trajectories diverge:

| Query | Camera/history | MAE, 0–255 | Maximum difference | PSNR dB |
|---|---|---:|---:|---:|
| 0 | front previous | 0.08938 | 15 | 54.8103 |
| 0 | front current | 0.07877 | 11 | 55.3792 |
| 0 | wrist previous | 0.10434 | 13 | 54.6521 |
| 0 | wrist current | 0.09476 | 13 | 55.2752 |
| 1 | front previous | 0.07952 | 12 | 55.5492 |
| 1 | front current | 0.08098 | 12 | 55.7019 |
| 1 | wrist previous | 0.10034 | 13 | 55.0566 |
| 1 | wrist current | 0.09781 | 15 | 55.0439 |

These are numerically small rendering/encoding discrepancies consistent with nondeterministic visual output, but this audit does not establish their cause or their model effect. The comparison does not isolate RTC perfectly: genuine later model calls see slightly different RGB. No image-equality threshold was invented after seeing results and no rerun was requested.

## Reporting defects and limits

The legacy outer wrapper reports `KeyError: pure_physics_success` while aggregating an already completed diagnostic. Its resulting `startup_failure.json` is misleading for attempts 03/04: complete model requests, physical traces, and inner failed summaries exist. Record this as a post-run aggregation failure, not a failure to initialize physics. A zero subprocess exit status likewise does not override inner `status=failed` and the observed safety stop.

The summary's video includes missing-overview warnings. Video file existence is not proof that all expected panels were recorded. Primary action/physics evidence is the trace and captured policy RGB.

The object pose and velocity fields remain unchanged throughout these short N trajectories; no object transfer is observed. This audit does not infer a new assistance mechanism from that alone. No contact certificate, autonomous carrying/placement feedback, full NAV-to-PLACE traversal, or strict task success is established.

Local audit evidence is under `independent_audit/`, including both closed traces/summaries and `N_rgb_pair_comparison.json`. Remote evidence root is `/diff/wallx_workspace/dzb/integration_runs/full_step1700_evaluation_20260908_v1`.
