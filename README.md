# Dynamic Manipulation

Dynamic Manipulation 是面向 Go2-X5 移动操作机器人的动态传送带抓取与分拣项目。
当前核心实现为 **ConveyorBench**：它提供 Isaac Sim / Isaac Lab 场景、机器人与
工位资产、动态抓取任务协议、轨迹记录、数据校验，以及面向 DynamicVLA 和 M0 的
离线导出工具。

## 项目内容

- V0：单目标、固定机身的动态传送带抓取基线。
- V1：多物体动态分拣任务，支持 `fixed_base` 消融模式与
  `whole_body_policy` 移动操作模式。
- 统一的 400 Hz 物理、50 Hz 控制和 25 Hz 相机/模型时钟。
- Go2-X5 的本地 USD、URDF、mesh 与移动策略资产。
- episode 原子写入、严格校验、质量审计和相机时变门禁。
- DynamicVLA 与 M0 两种离线数据视图。

> 本仓库包含基准框架和本地资产，不包含训练完成的 VLA 模型。Isaac Sim、
> Isaac Lab、PyTorch、NumPy 和 OpenCV 等运行环境需要在宿主机上准备。

## 目录结构

```text
Dynamic/
├── conveyor_bench/
│   ├── assets/                 # 机器人、工位、物体、分拣盒与策略资产
│   ├── configs/                # V0/V1 冻结配置
│   ├── scripts/                # 预检、仿真、采集、校验、审计与导出入口
│   ├── src/conveyor_bench/     # 协议、控制、记录和 Isaac 运行时实现
│   ├── tests/                  # 无需启动 Isaac Sim 的单元测试
│   ├── BENCHMARK_V1_SPEC.md    # V1 冻结规范
│   └── README.md               # 完整使用与采集说明
```

详细的任务定义、物理门禁、数据格式和完整采集命令请阅读
[ConveyorBench 使用说明](conveyor_bench/README.md)。V1 的冻结协议见
[BENCHMARK_V1_SPEC.md](conveyor_bench/BENCHMARK_V1_SPEC.md)，采集与验收流程见
[COLLECTION_GUIDE.md](conveyor_bench/COLLECTION_GUIDE.md)。

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
校验和模型视图导出。正式扩大数据采集前，请依照采集手册逐项完成环境、物理、
移动策略和相机门禁。
