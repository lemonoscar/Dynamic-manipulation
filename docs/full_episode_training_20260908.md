# RGB full-model episode training, 2026-09-08

The user authorized completing the 1,000-step RGB action run and starting more complete full-parameter episode training to address subtask transitions. New full-model GPU work and validation share an eight-hour wall limit on H20 physical GPU2/3. Depth and new collection remain disabled.

The previous bounded run finished 1,000 steps with exit 0. Its best validation DiT weights and exact train-only normalizer initialize this release; the matching old formal step_002414 provides Qwen weights and tokenizer identity. No ABot special token reset occurs. Newly updated Qwen receives a new checkpoint identity, so old frozen condition caches cannot be used.

## Transition and data corrections

Task entry requirements and ongoing invariants are distinguished. PLACE requires carrying on entry, but releasing the object does not cancel the stable-placement wait. Carrying remains an ongoing NAV_TO_TARGET requirement. Ordinary waits retain trustworthy effective gripper targets. Runtime action requests expose a public plan summary consistent with training: historical completed tasks, active task, remaining suffix and explicitly unknown current facts. Teacher phase and evaluator truth are excluded from transition prompts.

The existing RGB 116-family source remains read-only. Family splits are unchanged (90/13/13). The derived full-episode view has 9,564 action queries, including 2,352 restored valid prefixes. A scored 5 Hz target needs its entire 0.2-second native application interval, with valid same-epoch actuator provenance. Crossing, unknown, intervened or out-of-range tails are explicitly masked; they are not terminal holds. Independent parity checks preserve all 7,212 previous full action windows at action atol 1e-12, including their image and command identities. The normalizer bytes are unchanged from the qualified preceding train pool; padding is never fitted as data.

580 paired transition examples (290 CONTINUE/290 ADVANCE) use assisted-teacher events only as offline labels. Both sides retain the same pre-transition memory, and real capture wall times determine event ordering. They train proposals, not feedback certificates. All 77 complete successful families have four action routes represented; legal windows from failures remain. FINISH has zero eligible examples: success events occur after the final recorded camera control. No label is invented, and repair/FINISH competence is not claimed.

## Full-parameter recipe

Qwen and both complete DiT action experts are unfrozen. FP32 master parameters/AdamW use BF16 autocast and existing Qwen gradient checkpointing. One process places Qwen on GPU2 and action experts on GPU3; tied input/output embeddings are optimized exactly once. Action loss retains the differentiable real RGB graph; transition assistant-token CE also trains the language output. No LoRA or detached Qwen cache is used.

Each pass shuffles family/episode order, then visits each episode's queries chronologically without replacement. Effective batch is 8 via microbatch 1. The ceiling is three full data passes within the remaining approved wall time, with LR 1e-5 actions, 2e-6 Qwen language/embeddings, 5e-7 vision; transition CE weight 0.2. This is windowed training over complete episode query pools with teacher-forced memory, not backpropagation through the entire recorded video or through a robot simulator. Fixed validation categories exclude test families. Best/last/final checkpoints remain in a new external run directory.

A separate two-step real-data preflight checks all optimizer groups, actual parameter updates, memory and step timing before the Master releases production training. Preflight weights are discarded as initialization. No automatic retry or recovery is scheduled. The total wall budget is enforced by a shared server-side deadline and stage timeouts. Full checkpoint reload converts the architecture to FP32 before loading, preserving small learned updates.

## Evidence and limitations

Runtime records, exact launch SHA/config/manifest, preflight/production PIDs and qualification decisions live outside Git in `/diff/wallx_workspace/dzb/integration_runs/rgb_full_episode_20260908_v1`. This design document alone does not establish a successful launch. CPU tests, source parity, real GPU preflight, checkpoint reload and physical capability are reported separately.

The first full test sweep exposed one tiny legacy CUDA buffer test to default GPU0; it exited and the complete suite was rerun with CUDA hidden. No other process was signaled. Future GPU stage launchers explicitly validate only the authorized GPU2/3 UUIDs and absence of occupants.

The repository still requires an independent deployed feedback estimator. A predicted ADVANCE is not Evidence and cannot bypass TaskMemory's causal completion requirements. Autonomous full transfer, safe real-time execution, strict contact, FINISH and suffix-repair ability remain separate future evaluations.
