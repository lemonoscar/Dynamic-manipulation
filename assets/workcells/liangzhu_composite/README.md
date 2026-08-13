# Liangzhu NuRec sidecar contract

本目录只保存可审计的代码侧说明，不保存大体积场景数据。Liangzhu NuRec、碰撞 USD
和 object USD 由 SSH 传到服务器的 sidecar 目录，永远不进入 Git。

运行时资产根由 `CONVEYOR_BENCH_ASSET_ROOT` 指定，且必须包含
`TRANSFER_MANIFEST.sha256`、`liangzhu/` 和 `objects/`。完整契约、坐标变换、门禁与
采集阶段见 [Benchmark 规范](../../../docs/benchmark.md)。

代码入口：

- `scripts/validate_assets.py`：路径、软链接、成员和 SHA-256 校验；
- `src/conveyor_bench/sidecar/assets.py`：生成原生 NuRec/碰撞组合层；
- `src/conveyor_bench/isaac/scene.py`：动态工位与 Liangzhu 静态层组合；
- `scripts/probe_scene.py --asset-root <path>`：三相机场景 smoke；
- `scripts/run_benchmark.py`：通过门禁后的仿真入口。
