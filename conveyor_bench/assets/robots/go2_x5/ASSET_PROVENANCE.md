# Go2-X5 asset snapshot

This directory is a self-contained local snapshot used by ConveyorBench.

- `go2_x5.urdf` and `meshes/` are kept together so every mesh reference is
  relative to this directory.
- Machine-specific conversion files were intentionally excluded.
- A massless `grasp_tcp_link` was added at 0.125 m along `arm_link6`, at the
  center of the usable parallel contact-pad region. The original FinRay mesh
  tip extends to approximately 0.15757 m.
- Runtime code resolves this directory relative to the ConveyorBench project;
  it does not import robot configuration code from sibling repositories.
