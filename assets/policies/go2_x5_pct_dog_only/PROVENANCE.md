# Go2-X5 PCT DogOnly policy provenance

This directory versions the contract and provenance of ConveyorBench's
TorchScript locomotion actor. The weight file is an external artifact and is
not distributed in the current Git tree. Obtain an authorized copy separately,
verify its identity below, and place it at `policy.pt` in this directory when
using the legacy default loader. The file is ignored by Git.

The contract/observation tests run without weights. The installed-artifact hash
test is explicitly skipped when the file is absent and fails if a present file
has the wrong identity. Runtime loading continues to require the real file and
its correct SHA-256; no placeholder actor is substituted.

## Artifact

- Local file: `policy.pt`
- Format: TorchScript feed-forward actor
- Size: 1,209,746 bytes
- SHA256:
  `f02e6467472e90671a28d97cd6dc02ed7fdeb59d2ece18e082f254314558d383`
- Exported checkpoint: `model_26000.pt`
- Checkpoint SHA256:
  `b46813c2daf17354c6344766816528cb04200f0fb996b53d5f3eb12302b49d51`

The exported actor was compared against the training checkpoints. Its actor
weights exactly match `model_26000.pt`; they do not match `model_26249.pt`.

## Source record

- Training repository: `Go2-X5-lab`
- Audited commit: `4c6213f524f38088309b523cc79e167d98c92c1e`
- Original relative artifact:
  `logs/rsl_rl/go2_x5_dog_only_rough/2026-05-12_08-16-24/exported/policy.pt`
- Reference integration repository: `arm-vla-grasp-sim`
- Audited commit: `b0f4f39ddf7ce2a94ad5c174e48da0ec31f6534a`
- Reference profile: `checkpoints/go2_x5/pct_multifloor`

`contract.json` records the checkpoint-matched observation, action, timing,
joint-order, default-pose, actuator, and command contracts.

## Direct flat height-scan approximation

The model was trained with a 187-value RayCaster height scan. A read-only
trace of the checkpoint-matched ManagerBased environment on a flat reset
measured the entire policy slice `[66:253]` at approximately `-0.20`
(`min=-0.20000023`, `max=-0.19999987`, `mean=-0.20000005`).

The local Direct adapter therefore uses a constant `-0.20` only when callers
omit `height_scan`. This is a flat-ground approximation, not a replacement for
a live RayCaster on non-flat terrain. CPU Direct comparisons showed that zero
and `+0.40` scans did not sustain locomotion, while `-0.20` reproduced the
expected `vx=0.20 m/s` response.

## License boundary

Relevant RobotLab source files inspected during the audit carry
`SPDX-License-Identifier: Apache-2.0`. Some inherited Isaac Lab configuration
files carry BSD-3-Clause notices. Neither audited repository contained a root
license file, and the policy weight has no embedded or adjacent
weight-specific license declaration.

Consequently, this record does not assert a redistribution license for the
weight. Review the upstream ownership and licensing status before distributing
the binary outside this project.
