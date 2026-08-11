# Conveyor station V3 NuRec asset contract

本目录只保存可审计的代码侧说明，不保存大体积场景数据。Liangzhu NuRec、碰撞 USD
和 object USD 由 SSH 传到服务器的 sidecar 目录，永远不进入 Git。

运行时资产根由 `CONVEYOR_BENCH_V3_ASSET_ROOT` 指定，且必须包含
`TRANSFER_MANIFEST.sha256`、`liangzhu/` 和 `objects/`。完整契约、坐标变换、门禁与
采集阶段见 [BENCHMARK_V3_3DGS_SPEC.md](../../../BENCHMARK_V3_3DGS_SPEC.md)。

代码入口：

- `scripts/validate_v3_asset_bundle.py`：路径、软链接、成员和 SHA-256 校验；
- `src/conveyor_bench/v3/assets.py`：生成原生 NuRec/碰撞组合层；
- `src/conveyor_bench/isaac/scene_v3.py`：V1 动态层与 Liangzhu 静态层组合；
- `scripts/probe_v1_scene.py --scene-profile v3_nurec`：三相机场景 smoke；
- `scripts/run_benchmark_v3.py`：通过所有门禁后的单回合/正式采集入口。
