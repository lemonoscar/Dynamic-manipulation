# Conveyor station V3 3DGS assets

本目录是 `configs/v3_3dgs.json` 约定的本地资产落点。设计分支不提交虚构或占位的
Gaussian 数据；在真实采集、训练和标定完成前，V3 必须保持
`collection_ready=false`。

放量前必须在本目录物化以下文件，且不得使用指向仓库外的软链接：

- `scene_static_gaussians.ply`：静态工位 Gaussian splat；
- `calibration_sim_from_gs.json`：`S_sim_from_gs`、控制点与误差报告；
- `capture_masks/`：与每张训练图像同名、排除机器人、皮带、零件、盒子和人员的
  逐帧 mask；
- `photometric_calibration.json`：固定曝光、白平衡和 Isaac 灯光匹配记录；
- `ASSET_MANIFEST.json`：文件 SHA-256、顶点数、来源、许可证与训练配置。

完整分层、相机和门禁定义见
[BENCHMARK_V3_3DGS_SPEC.md](../../../BENCHMARK_V3_3DGS_SPEC.md)。
