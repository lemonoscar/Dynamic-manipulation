# ConveyorVLA

ConveyorVLA 是面向 Go2-X5 移动操作机器人的视觉语言动作模型与仿真评估框架。
当前 `Manipulation_Navi_v1` 分支使用 **ABot-M0 初始化的双 DiT joint-trajectory 模型**，
覆盖导航到源位置、抓取、运输、放置四阶段。

**2026-09-05 状态**：正式训练完成 `2414/2414` steps；独立 validation 和 test 开环已完成。
Test 的 episode 等权路由准确率为 **98.33%**，NAV 平均误差 **4.19 cm**，
六关节 MAE **0.0466 rad**。动作裁剪事件率 **10.90%**，未通过既有 **≤0.5%** 门槛。
冻结迁移闭环已结束 300 次尝试，其中 178 次未产生模型请求；旧合约未记录完整成功。
这些结果包含启动、规划、控制与评价问题，不能直接解释为模型失败率，尚未证明完整搬运能力。
[结果、95% 区间与证据身份](docs/experiments/formal_5hz_20260905.md)。

执行接口诊断已加入：无模型采样命令回放、同刻/提前一拍对照、validation 源/模型目标成对 PCT 探测，
以及 G/A/B/C 位姿记录和 DWA 失败分类。12 次隔离抓取回放已完成，均观察到抬升但原抓取判据未通过；
500 个源导航目标中 78 个也因栅格量化超过端点门槛。
这些结果不等于完整策略成绩，见[执行一致性验证](docs/execution_consistency_validation_20260905.md)。

**2026-09-06 接口诊断**：评价与抓取辅助已拆分；源绝对命令经过部署解码/限制后的配对回放已完成。
最大裁剪样本追溯到 `plan_pick` 的 reset 前缓存目标，包含该前缀的回放失败；仅从 `exec_pick` 开始的四个示范均出现持续几何抬升。
固定最终模型的标准站位独立进程对照已完成：持续几何夹持为 10 点组 2/3、2 点组 0/3；
这是小样本条件诊断，严格接触成绩仍未知，动作裁剪门槛均未通过。
连续 PCT 末段仍是待真实几何和控制验证的候选，门槛没有放宽。
见[新开发与实验文档](docs/execution_interfaces_v2_20260906.md)，其中区分接触未知、几何代理、执行完成和模型成绩。

## 当前模型与执行合同

```text
指令 + head/wrist[t-0.20s, t]
               │
Qwen Pass 1：四阶段 route + subtask（两个新观测确认）
               │
Qwen Pass 2：同一观测 + 模型自己的回答前缀
               │ last_hidden_state
        ┌──────┴──────┐
      NAV DiT       Mani DiT ← 13D q/dq/gripper
      [10,3]        [10,7]
      @0.20s        @0.20s
        │             │
第10点→PCT→DWA   关节/夹爪目标逐点执行
                  每点保持10个50Hz tick
```

- Qwen 读取 RGB 与语言；13 维可测状态只进入 Mani 专家。评估真值不输入模型。
- 两个专家不共享参数，实际使用 4 次 flow-matching 采样迭代。
- 操作块覆盖 2 秒；5 Hz 目标点率不等于 5 Hz 模型重规划。
- DONE、prefix 选择、Mani IK/cuRobo 与 on-policy correction 在本合同中禁用。
- 当前闭环为源 Sim6 任务在 Isaac Sim 5.1 的迁移测试；源抓取辅助和无抓取辅助两组分别报告，
  两组都保留操作底座/支撑锁定，推理期间暂停仿真。它们不是纯物理或实时部署验收。

生效配置：[manipulation_navi_v1.json](configs/manipulation_navi_v1.json)。
旧 Waypoint v1/v2 与 [SPCGVLA 草案](docs/SPCGVLA/README.md) 是不同合同，不能混用
它们的动作时钟、路由、checkpoint 或启动命令。

## 开发与检查

Python ≥3.10。从仓库根目录运行基础安装与不依赖模型权重的检查：

```bash
python -m pip install -e . pytest
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -p no:cacheprovider tests/test_formal_evaluation.py tests/test_formal_physics.py tests/test_formal_closed_loop.py tests/test_locomotion_contract.py
```

模型训练和推理需要 `conveyorvla` 可选依赖、单独准备的预训练权重与数据；Isaac 使用独立环境。
检查点和数据路径绑定不可随意移动。环境、测试分层、复现命令和已知限制见
[开发指南](docs/joint_trajectory_development_guide.md)与
[正式评估操作说明](docs/formal_joint_trajectory_evaluation.md)。

## 仓库与工件

| 路径 | 内容 |
|---|---|
| `src/conveyor_bench/` | 模型、数据、运行时、评估及仿真适配器 |
| `scripts/` | 数据构建、训练、开环、服务、闭环及独立感知诊断入口 |
| `configs/` | 可版本化配置和合同 |
| `tests/` | 单元、合同、集成与可选工件检查 |
| `docs/` | 当前开发/评估文档、实验摘要和明确标识的历史材料 |
| `assets/` | 已版本化的机器人资源、合同与来源记录；不包含模型权重 |
| `artifacts/` | 被忽略的本机依赖、权重、数据源、场景和运行环境 |

正式训练器和数据 materializer 要求发布数据与训练输出位于 Git 工作树之外，例如同级的
`data_releases/` 和 `training_runs/`。**模型权重、数据集、日志、视频、缓存和私有配置均不提交**；
Git 只保存代码、配置、文档及小型可审计摘要。已有 locomotion `policy.pt` 的本地安装和
许可证边界见 [PROVENANCE](assets/policies/go2_x5_pct_dog_only/PROVENANCE.md)。

## 文档入口

- [开发指南与修改边界](docs/joint_trajectory_development_guide.md)
- [训练与全量评估结果](docs/experiments/formal_5hz_20260905.md)
- [架构问题、证据与后续诊断](docs/joint_trajectory_architecture_analysis_20260905.md)
- [正式开环与迁移闭环操作](docs/formal_joint_trajectory_evaluation.md)
- [文档索引与历史合同](docs/README.md)
- [当前状态](docs/status.md)
- [贡献与提交规则](CONTRIBUTING.md)

当前仓库发布的是研究代码；数值能力与兼容性以明确的模型合同、源码身份和实验摘要为准。
问题反馈请附分支/commit、配置、复现命令和脱敏错误摘要，不上传权重、完整观测或凭据。
