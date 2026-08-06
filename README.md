# Dynamic Manipulation

Dynamic Manipulation 是面向 Go2-X5 移动操作机器人的动态传送带抓取与分拣项目。
当前核心实现为 **ConveyorBench**：它提供 Isaac Sim / Isaac Lab 场景、机器人与
工位资产、动态抓取任务协议、轨迹记录、数据校验，以及面向 DynamicVLA 和
ConveyorVLA 的离线导出工具。当前策略基线的正式名称是 **ConveyorVLA AL0**；
仓库中的 `m0_*` 仅保留为既有数据、检查点和在线协议的兼容标识。

## 项目内容

- V0：单目标、固定机身的动态传送带抓取基线。
- V1：多物体动态分拣任务，支持 `fixed_base` 消融模式与
  `whole_body_policy` 移动操作模式。
- V1 静态诊断：传送带速度严格为零的单物体抓取—携带—投放任务，冻结
  `3 train / 1 val / 1 test` 五个场景，用于先验证 AL0 的基础操作能力；不计入
  动态 benchmark 分数。
- V2：双目标连续分拣和强制持物移动的远端投放任务，提供 2 个场景、7 个允许
  组合、严格事件/位移校验及 AL0/DynamicVLA 上下文投影。
- 统一的 400 Hz 物理、50 Hz 控制和 25 Hz 相机/模型时钟。
- Go2-X5 的本地 USD、URDF、mesh 与移动策略资产。
- episode 原子写入、严格校验、质量审计和相机时变门禁。
- DynamicVLA 与 ConveyorVLA AL0 两种离线数据视图；后者继续读取 legacy
  `m0_mobile_v1` profile。
- AL0 50 Hz 因果动作块导出，以及不依赖外部仓库的最小 AML 训练烟测。
- AL0 temporal v1：head/wrist 双帧运动观测、20×10 的 25 Hz 独立未来目标、
  generation-aware 流式合并和低速单物体成功配额采集器。
- AL0 在线服务、阶段门禁与 Go2-X5/Isaac 闭环入口，完整记录模型身份、
  service/AL0 控制边界和请求时延。

> 本仓库包含基准框架和本地资产，不包含训练完成的 VLA 模型。Isaac Sim、
> Isaac Lab、PyTorch、NumPy 和 OpenCV 等运行环境需要在宿主机上准备。

> 许可证边界：whole-body 运行所需的本地 `policy.pt` 尚无已确认的权重专属
> 再分发许可证。它可用于当前项目的本地研究与验收，但在公开分发该二进制前，
> 必须取得授权或替换为许可明确的权重；审计记录见
> `conveyor_bench/assets/policies/go2_x5_pct_dog_only/PROVENANCE.md`。

## 目录结构

```text
Dynamic/
├── conveyor_bench/
│   ├── assets/                 # 机器人、工位、物体、分拣盒与策略资产
│   ├── configs/                # V0/V1 冻结配置与 V2 suite 快照
│   ├── scripts/                # 预检、仿真、采集、校验、审计与导出入口
│   ├── src/conveyor_bench/     # 协议、控制、记录和 Isaac 运行时实现
│   ├── tests/                  # 无需启动 Isaac Sim 的单元测试
│   ├── BENCHMARK_V1_SPEC.md    # V1 冻结规范
│   ├── BENCHMARK_V2_SPEC.md    # V2 场景、任务与评价规范
│   ├── COLLECTION_V2_GUIDE.md  # V2 采集、校验和导出手册
│   └── README.md               # 完整使用与采集说明
```

详细的任务定义、物理门禁、数据格式和完整采集命令请阅读
[ConveyorBench 使用说明](conveyor_bench/README.md)。V1 的冻结协议见
[BENCHMARK_V1_SPEC.md](conveyor_bench/BENCHMARK_V1_SPEC.md)，采集与验收流程见
[COLLECTION_GUIDE.md](conveyor_bench/COLLECTION_GUIDE.md)。V2 规范与采集入口见
[BENCHMARK_V2_SPEC.md](conveyor_bench/BENCHMARK_V2_SPEC.md) 和
[COLLECTION_V2_GUIDE.md](conveyor_bench/COLLECTION_V2_GUIDE.md)。

## 环境准备

推荐使用已经安装 Isaac Sim、Isaac Lab、PyTorch 和 OpenCV 的 Python 3.11
环境：

```bash
cd conveyor_bench
conda activate env_isaaclab
python -m pip install -e .
python scripts/check_environment.py
```

协议、记录器和大部分校验逻辑可以脱离 Isaac Sim 测试：

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -p no:cacheprovider
```

## V2 最小预检与运行示例

不启动 Isaac 即可先解析双目标连续任务：

```bash
cd conveyor_bench
python scripts/run_benchmark_v2.py \
  --scene transverse_near_sort_v2 \
  --task-family continuous_multi_target \
  --robot-mode fixed_base \
  --seed 0 \
  --dry-run-task
```

冻结源码候选 `0a2fd7c…` 已完成 near fixed-base 双目标连续分拣，以及 remote
whole-body 蓝/黄双向投放；连续持物位移分别为 `0.735903 m` 和 `0.778166 m`。
本机 RTX 4060 上的 near/remote 三相机正例也已依次通过 strict validator、
temporal camera gate 与 DynamicVLA/AL0 双导出。尚未覆盖的语言条件、
near whole-body、其余物体/速度/seed 矩阵仍需按 V2 手册小规模验收，不能直接
外推为大规模采集结论。

面向下一阶段训练的 AL0 profile 已按“观测后预测未来动作”重做因果
对齐；其数据契约、导出和 AML 烟测命令见
[V1 采集手册](conveyor_bench/COLLECTION_GUIDE.md)。
在线部署、闭环命令和 2026-08-03 的实测失败边界见
[ConveyorVLA AL0 在线闭环与验收](conveyor_bench/CONVEYORVLA_AL0_GUIDE.md)。
DynamicVLA 时序机制的代码级分析与下一代动态抓取方案见
[ConveyorVLA AL1 设计](conveyor_bench/CONVEYORVLA_AL1_DESIGN.md)。
当前正式执行的数据、流式控制与低速采集合同见
[ConveyorVLA AL0 执行方案](conveyor_bench/CONVEYORVLA_AL0_EXECUTION_PLAN.md)。

## V1 最小运行示例

下面的命令采集一条固定机身、单目标、带三相机数据的 V1 episode：

```bash
cd conveyor_bench
python scripts/run_benchmark_v1.py \
  --robot-mode fixed_base \
  --episodes 1 \
  --seed 0 \
  --split train \
  --task-family single_target \
  --belt-speed 0.06 \
  --max-duration 20 \
  --active-objects 1 \
  --target-asset part_red_block \
  --destination sort_bin_blue \
  --output-dir outputs/gate/v1_fixed \
  --enable_cameras \
  --save-camera-frames \
  --require-all-success \
  --headless \
  --device cpu
```

校验生成的数据：

```bash
python scripts/validate_v1_dataset.py outputs/gate/v1_fixed
```

运行产生的 episode、视频、导出数据和缓存默认写入 `conveyor_bench/outputs/`，
该目录不会提交到 Git。

## 当前状态

V1 框架已经覆盖任务配置、动态场景、固定机身与全身模式、三相机记录、严格数据
校验和模型视图导出。新增静态诊断的 5/5 oracle episode 均通过 strict validator
与 temporal camera gate；其中 3 条 train episode 已导出 1,428 条 AL0
记录，val/test 会由导出器明确隔离。AL0-M1 checkpoint 在线测试证明静态闭夹、双侧持有和抬升
primitive 存在，但无辅助回合仍在底盘靠近阶段失败，辅助隔离回合也会提前开爪，
因此旧 checkpoint 仍不能产生 policy-only 成功轨迹；当前正式采集使用严格门禁的
oracle teacher，为新的时序 AL0 训练准备成功示教，不能把两者混为一谈。
完整证据见
[ConveyorVLA AL0 在线闭环与验收](conveyor_bench/CONVEYORVLA_AL0_GUIDE.md)。
