# Step1700 测试日志总索引

本索引归档已有测试；新3个种子的测试尚未启动。权重始终为step1700（不是1700轮epoch），SHA256 474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0。没有重启训练，也没有depth输入。

## 已完成记录

|测试|实际数量/时长|结果|执行源码|
|---|---|---|---|
|训练seed16100000已录完整episode模型视图|96动作query、6交接query|交接3/6；PICK首两点夹爪100%，PLACE全前缀宏平均46.67%|e03cb0b（外置评测脚本身份见原报告）|
|物理尝试1|42ticks、2query、0.84s|实测关节位置容差触发，未进PICK|92dfddc|
|物理尝试2|60ticks、3query、1.20s|PCT原0.10m snap门触发，未进PICK|691a144|
|物理尝试3|161ticks、8query、3.22s|PCT门放宽0.50m后通过，转向/移动切换处guarded零速度终止|772d5a1|
|仅60秒时钟结束|3000连续ticks、150query、60.00s|计时正常结束；实际未抓起可乐，未进PLACE|920ccc5|

60秒测试：21.2s模型声明进入PICK，31.8s声明进入NAV_TO_TARGET。PICK最近TCP–物体仍21.83cm，该时刻夹爪全开；物体最大位移0.300mm，峰值抬升0。完整搬运失败，严格接触unknown。运行中的关节、导航、零速度门仅记录；原统计门仍保留独立失败身份。

## 日志与报告

- [机器可读索引及哈希](test_log_index.json)：各次开始/结束时间、PID、运行目录、确切源码、停止原因、原始trace SHA256。
- [关键事件逐行记录](logs/key_events.jsonl)：原始模型响应的时刻/任务选择、阶段切换、计时启动和异常事件；省略图像base64载荷。
- 每次原始attempt、command、summary直接归档在logs/对应运行目录；未改写其历史内容。
- [训练种子拟合报告](../FIT_ANALYSIS.md)、[总集成测试报告](../MASTER_TRAIN_SEED_REPORT.md)。
- [第3次停止点审核](../retry3/DWA_FAILURE_AUDIT.md)、[60秒运行报告](../timer60/TIMER60_REPORT.md)、[60秒独立审核](../timer60/TIMER60_AUDIT.md)。
- [此前标准开环及RTC物理诊断](../../evaluation_step1700_20260908/MASTER_EVALUATION_REPORT.md)。该早期报告涉及的旧物体成功/失败判断，须结合下述后续修正阅读。

## 证据修正与边界

尝试1及更早未确认physics tensor句柄/唤醒状态的轨迹，物体相关评分降为unknown。原summary原样保留，不能据其false值宣称已验证物体抓取失败。尝试2起已核验真实动态tensor与唤醒；60秒测试3000步证据覆盖完整。

尝试2的第3个PCT端点snap=0.116181m为同源码/地图CPU重建，非原日志直接值；尝试3的0.116419m为直接记录，不冒充逐位配对。尝试3零速度来源在当时缺少raw DWA记录，不能凭默认容差猜测；60秒模式已补debug并继续运行。

物理内部summary与进程退出码分别保留；前三次进程退出码为0但内部failed。仅60秒计时完成不等于完整任务成功；模型ADVANCE声明也不是独立完成证据。

## 录像与原始数据

60秒录像：1280×720、5Hz、300帧、60.000秒。日志另有9次overview缺失记录，不能宣称所有视角逐帧无缺失。视频原文件和原始含图像trace留在运行目录，不把大体积录像/模型/数据载荷混入文档提交。

远端视频：/diff/wallx_workspace/dzb/integration_runs/train_seed_fit_20260908_timer60_v1/physical_timer60/runtime/episode_000000/overview_videos/episode_000001_composite.mp4

本地视频：/home/lemon/research/Issac/doc/train_seed_fit_20260908/timer60/results/physical_timer60/runtime/episode_000000/overview_videos/episode_000001_composite.mp4

原始包发送/接收SHA256均为0f42ce7e6a0bc48b5e256800843e03e22b0635ee85dc5cb546e3411e839fe030。GPU2/3已在上轮结束后释放；本索引提交只修改docs。
