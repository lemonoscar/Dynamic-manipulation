# Go2-X5 asset snapshot

This directory is a self-contained local snapshot used by ConveyorBench.

- The robot source was audited against the local `arm-vla-grasp-sim` snapshot
  at commit `b0f4f39ddf7ce2a94ad5c174e48da0ec31f6534a`. Its source URDF SHA-256 is
  `d52f9690ec3828692e1bcafaea14f08ae0b790126bc55c3619288d508cb1e23e`.
- All 18 upstream Go2-X5 mesh files are byte-identical. ConveyorBench adds only
  `meshes/X5/link6_truncated.stl` for the validated wrist collision model.
- `go2_x5.urdf` and `meshes/` are kept together so every mesh reference is
  relative to this directory.
- Machine-specific conversion files were intentionally excluded.
- The upstream dog body, joints and `front_camera` frame are retained. The X5
  mount inertials, conservative collision proxies and joint limits are the
  benchmark's already validated manipulation model and are intentionally not
  overwritten by the older upstream arm dynamics.
- A massless `grasp_tcp_link` was added at 0.125 m along `arm_link6`, at the
  center of the usable parallel contact-pad region. The original FinRay mesh
  tip extends to approximately 0.15757 m.
- Runtime camera calibration is vendored in `scene_v1.py`: the head camera is
  parented to `base`, and the wrist camera to `arm_link6`, using the exact ROS
  optical-frame transforms from the audited reference runtime. No sibling
  repository is imported at runtime.
- Runtime code resolves this directory relative to the ConveyorBench project;
  it does not import robot configuration code from sibling repositories.
