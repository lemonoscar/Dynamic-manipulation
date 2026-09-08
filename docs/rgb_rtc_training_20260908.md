# RGB-only RTC and qualified action training — 2026-09-08

The user paused depth collection and depth model inputs, and authorized completing RTC then training on existing qualified data. Historical RGB-D artifacts and old checkpoints are retained. This release trains action experts only; unresolved planner labels are excluded.

## Implementation and contracts

- `collection-to-staged-v2` has an explicit RGB-only path. It does not call depth decoding/calibration. Full native 50 Hz control records remain the label/audit source; queries are selected every 20 native ticks (0.4 s). Four RGB frames use front/wrist at query−0.2 s and query, joined by source identities.
- New weights use `causal_command_5hz` (first target at query, 10 points, 0.2 s intervals). Legacy checkpoints retain `legacy_future_5hz` and their old normalizers. NAV labels retain their measured-reference identity.
- RGB task/live encoding is shared between frozen condition caching and `StagedRGBBackend`; the backend uses the same normalizer and `RollingRuntime` request/queue boundary. Qwen remains frozen. Both DiT experts are initialized strictly from the existing checkpoint, then adapted with a new train-only normalizer.
- `StagedExperts.sample` now supports the existing inference VJP implementation. RTC-off is unchanged. This run has `training_rtc=false`; it does not silently substitute clean-prefix training semantics. Offline RTC contexts bind the checkpoint, episode, active task/epoch, normalizer, query, source action, and actual apply times.
- Safety rejection latches a separate stop. Later ticks, late responses, new requests and episode finalization cannot resume the ordinary hold path. Normal waiting retains its existing trusted-target hold behavior.
- Candidate inference rejects a record/checkpoint time-profile mismatch, unknown domains, untrained NAV prefixes and unbound prefix files.

## Existing data and eligibility

The existing RGB collection has 117 attempts; 116 closed episodes passed source raw auditing (N90/B14/R12, 77 successes and 39 failures). The interrupted final episode is excluded. Source raw acceptance is not model eligibility: the importer separately quarantines illegal action windows, interventions, invalid command provenance, out-of-range joint7 targets and task/application boundaries. Legitimate windows from failed episodes remain eligible. The source teacher uses declared physical assistance; training data are not evidence of unassisted policy success.

A sparse RGB transfer preserves all required raw metadata and only images needed at the frozen query cadence. It contains 41,216 source files, 10,663,416,552 uncompressed bytes; archive SHA256 `ff59a2f9cd068e6867de1ee195f0da8fd6c9a268429b12cb9c96bdd493e4f069`. Depth files and the separate evaluator-truth file are excluded. Source metadata remain unchanged; the importer allowlist prevents embedded truth from entering model conditions.

Family splits are frozen before derivation: 90 train / 13 validation / 13 test, stratified N/B/R, deterministically ordered by SHA256 of `rgb-rtc-20260908-v1:` plus family ID. Normalization fits all eligible train rows only. Test rows are not cached or used for checkpoint selection. The old 500-episode model data with the incompatible gripper command reduction are not reused as training labels.

## Bounded training and operational evidence

The approved GPU allocation remains H20 physical GPU2/3. This release uses at most one cache worker per GPU, at most 7,200 seconds cache wall time, followed by one GPU training run: seed 20260908, at most 1,000 optimizer steps, effective batch 16, learning rate 2e-5, cumulative training wall limit 7,200 seconds. Cache/step timing is measured before extrapolating. No automatic retry, new collection, multi-seed sweep or shared Qwen fine-tuning is launched.

Training rejects synthetic/smoke releases, missing routes/splits, foreign caches, depth tensors for RGB-only, and mismatched initialization. It evaluates a fixed validation-family/route sample with fixed noise, saves best weights and periodic atomic optimizer/RNG state, rejects nonfinite loss/gradients, and supports identity-bound recovery into a new output directory with the remaining budget. Validation flow loss is not task success.

Exact frozen HEAD, full dataset eligibility counts, hashes, GPU UUIDs, process identities, logs and launch/stop/resume records are external under `/diff/wallx_workspace/dzb/integration_runs/rgb_rtc_qualified_20260908_v1/`. Model weights and data are outside Git. Master records the final launch decision there after real data verification and RTC checks; this document alone is not evidence that training has started.

## Evidence boundaries

Existing real-RGB VJP numerical tests and legacy off regression are reusable; previous physical RTC traces did not demonstrate successful grasping or deployment. New code-level safety/codec/backend tests and real-condition numerical checks validate implementation, not complete task competence. Macro planner adaptation, unassisted full transfer, real-time network behavior and strict contact still need their own evidence after training. No depth benefit is claimed or tested in this run.
