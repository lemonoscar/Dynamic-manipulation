# Go2-X5 asset snapshot

This directory is a self-contained local snapshot used by ConveyorBench.

- The robot snapshot is byte-for-byte aligned with the local
  `arm-vla-grasp-sim` `pct_scene` snapshot at commit
  `c7fe62c7f9f8dcda89fef9c7e363594d7e486375`. Its source URDF SHA-256 is
  `d52f9690ec3828692e1bcafaea14f08ae0b790126bc55c3619288d508cb1e23e`.
- The active fixed-base and mobile runtimes both spawn that canonical URDF.
  The five retained USD layers are also exact PCT files, but remain inactive
  compatibility artifacts and cannot replace the canonical runtime URDF.
- All 18 upstream Go2-X5 mesh files are byte-identical; no benchmark-only mesh
  remains in the aligned robot asset.
- `go2_x5.urdf` and `meshes/` are kept together so every mesh reference is
  relative to this directory.
- CuRobo planning files were intentionally excluded because ConveyorBench does
  not load them; every robot file that is retained here is an exact PCT file.
- The upstream dog body, X5 mount, inertials, joints, collision geometry,
  limits, `front_camera` frame, and 0.15757 m `grasp_tcp_link` are retained
  without benchmark-side edits.
- The runtime TCP is the PCT FinRay tip frame at
  `arm_link6 + (0.15757, 0, 0) m`. The original `arm_link7` and `arm_link8`
  collision meshes remain active and receive PCT's `convexDecomposition`,
  0.002 m contact offset, and zero rest offset before the first physics reset.
- Runtime camera calibration is vendored in `scene_v1.py`: the head camera is
  parented to `base`, and the wrist camera to `arm_link6`, using the exact PCT
  ROS optical-frame transforms and D436 640x480 OpenCV intrinsics. No sibling
  repository is imported at runtime.
- Runtime code resolves this directory relative to the ConveyorBench project;
  it does not import robot configuration code from sibling repositories.
