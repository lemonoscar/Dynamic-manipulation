# 训练集 best-of demo 候选审计

核查时间：2026-09-08T16:15:48.267446+00:00；远端：4xH20 / VM-0-3-ubuntu。只读 CPU；本审计没有新增物理尝试。

## 选择结果与边界

从 90 个 train family 中找到 66 个四路完整 action-view family；排除已测/在运行 16100000、16100003、16100004、16100006，以及缺少教师 place/episode 成功事件的 7 个 family，剩 55 个。下面固定前 24 个供 Master 在新增 24 次 / 6 小时 / 每次 60 仿真秒的共享预算内取舍。

先要求四路有效动作、train-only normalizer 家族成员、源 raw 合同通过、教师导航/抓起/稳定放置/完整 episode 成功事件、t80（1.6 秒）唯一合法 NAV_TO_SOURCE query，以及所有引用 RGB 文件存在。再按 t80 本体 XY 位置到源任务 pick base goal 的距离升序，平局按教师两段 NAV 总时间、seed 升序排序。所有候选的初始四张 RGB 已真实解码并计算哈希；后续图像仅检查存在，未声称全部解码。

**这是训练集挑选较容易实例的 best-of 演示搜索，不是泛化测试，也未证明模型能完成。** 原教师成功包含声明的 verified_contact_physx_fixed_joint 持物辅助；严格接触成功仍为 unknown。候选选择可以离线看 teacher 成功事件，普通 planner 不可读取这些未来事件或 evaluator truth；物理初始化保持 t80 NAV，不切到 PICK。

## 训练已消费证据

固定 trainer 的 episode_order 函数从其 AST 原样提取，未重新实现调度；train 共 7448 条 action + 448 条 transition = 7896 条，每批 8 条，共 987 步/epoch。训练日志 step987 的 cursor=7896、epoch_rows=7896，随后 epoch_complete 记录 rows_seen_once=7896。结合冻结文件与 seed20260908，可重建全部候选在 step1700 之前至少完成一次优化消费；表中是对应第一轮 optimizer 步区间。**日志没有逐样本/逐 family 标识，因此这是可重建消费证据，不是逐样本直接日志见证。**

release manifest SHA256：`4e5aaff4a65878e873add2d153f447f6e639bbc9537498e66ca98cd2a0856391`；train normalizer SHA256：`a58d70a2ddf77e4750a7bc5986f32d865fb8cf77330b1617f94d982581c92591`；trainer SHA256：`acee2ebd21868927cd0b84916a0ce3cc91b2d67323d233fe7f19748e5a53662d`。

第1700步权重既有验证身份：`474d03ff22da939f763574bf434c75226a2a2533ae984ced80e25e34a1df7fa0`。本次只读筛选校验 release/normalizer 所有 manifest 文件哈希与训练事件；未重新加载大权重。训练事件 SHA256：`72ae93148567c6c396de8ce5bafad0d36f8c8b912beb9701f2ec0afaa68c88c5`。

## 冻结候选表

|优先级|实际 seed|类型|t80→pick 距离 m|原 NAV 总秒|四路 action 数¹|transition 数|epoch1 优化步|
|---:|---:|:---:|---:|---:|:---:|---:|:---:|
|1|16100076|N|0.327897|19.40|4/19/44/18|6|194–205|
|2|16100249|R|0.410137|22.02|7/23/48/18|6|934–947|
|3|16100015|N|0.525452|20.32|8/19/42/17|6|182–194|
|4|16100068|N|0.585453|18.76|5/19/41/17|6|95–106|
|5|16100243|R|0.592035|22.14|7/22/47/17|6|457–469|
|6|16100069|N|0.725794|24.06|15/19/44/17|6|822–835|
|7|16100078|N|0.752961|18.24|7/20/38/17|6|648–659|
|8|16100044|N|0.764792|19.40|7/19/41/18|6|28–39|
|9|16100034|N|0.765644|22.32|7/20/49/18|6|796–808|
|10|16100021|N|0.783592|21.16|8/19/43/17|6|546–558|
|11|16100037|N|0.797185|22.50|7/19/48/17|6|975–987|
|12|16100022|N|0.886658|25.24|15/20/48/17|6|380–393|
|13|16100089|N|0.934262|19.76|8/20/41/18|6|68–79|
|14|16100083|N|0.945320|21.60|8/20/45/17|6|419–431|
|15|16100030|N|0.948571|27.58|12/20/57/18|6|808–822|
|16|16100070|N|0.969496|21.42|9/19/45/18|6|124–136|
|17|16100187|B|0.981477|25.12|8/19/55/17|6|698–711|
|18|16100010|N|0.987530|23.90|9/18/50/18|6|712–724|
|19|16100087|N|1.031307|22.82|8/19/49/18|6|724–737|
|20|16100026|N|1.042810|21.78|8/18/45/17|6|489–501|
|21|16100081|N|1.050498|22.78|18/20/39/17|6|861–874|
|22|16100079|N|1.055645|21.52|9/19/45/17|6|297–309|
|23|16100071|N|1.060172|23.74|9/19/50/18|6|509–522|
|24|16100020|N|1.065173|23.16|8/19/49/17|6|686–698|

¹ NAV_TO_SOURCE / PICK / NAV_TO_TARGET / PLACE。排序不使用模型输出或先验闭环结果。

## 每个候选的源身份

task.json 实际文件 SHA256 和 manifest 的 resolved_task_sha256 是分别保留的两种身份，不混称同一个 hash。源 episode_start 事件 seed 已再次与 task.randomization.seed 逐一核对。

### 1. seed 16100076 / liangzhu_seed_16100076

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000016`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000016`
- task.json SHA256：`5ab1561a636d4be51135624fbfc15af04922824b5e22c39d6dbdc5fa3bb1601b`
- manifest resolved task SHA256：`51888df9044ce99aa7ffb562274af782cf4439d72d4adedce8dfa00216d8def7`
- 教师抓起高度 0.213648 m；稳定放置确认 True；原 episode 结束 35.80 s。

### 2. seed 16100249 / liangzhu_seed_16100249

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/01_R/part_000/episode_000003`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/01_R/part_000/episode_000003`
- task.json SHA256：`475d2ca2fc07e0d7be95bb544e5f834893f8e2ffbe621d1333a6ef2d235c048d`
- manifest resolved task SHA256：`9eef87cb384651946dac40ca730c8b5922ca1bb256c8e30c124ef5e7d94833ff`
- 教师抓起高度 0.212230 m；稳定放置确认 True；原 episode 结束 41.08 s。

### 3. seed 16100015 / liangzhu_seed_16100015

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/00_N/part_000/episode_000015`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/00_N/part_000/episode_000015`
- task.json SHA256：`e1244b8171d33c1f5e5428e4ac7b1cd23548cef4d6b1656656e0e3c9cbb93af0`
- manifest resolved task SHA256：`7e9a9cee015b36d94fd20af5c2fb9d6afe29c88acf45d05f44632e4eb7d33549`
- 教师抓起高度 0.211808 m；稳定放置确认 True；原 episode 结束 36.98 s。

### 4. seed 16100068 / liangzhu_seed_16100068

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000008`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000008`
- task.json SHA256：`135419a48edb7b8f96e7564eae29f9bb1dc1379cb885cace66c17f4cc2f728dd`
- manifest resolved task SHA256：`31f12f2594805ae79d9d25f42801fbfdd1adf6af9ab43c1e294bbfe5300b0362`
- 教师抓起高度 0.211888 m；稳定放置确认 True；原 episode 结束 35.22 s。

### 5. seed 16100243 / liangzhu_seed_16100243

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/00_R/part_000/episode_000003`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/00_R/part_000/episode_000003`
- task.json SHA256：`77b239b73a2b538f303a835b1065dc42ed7a0beead7a77dc9988cc63a1d82778`
- manifest resolved task SHA256：`a9ccd0f784c89b9a5e9665745fe100997a67d6b423208725934d40c5da8b2c18`
- 教师抓起高度 0.218618 m；稳定放置确认 True；原 episode 结束 41.26 s。

### 6. seed 16100069 / liangzhu_seed_16100069

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000009`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000009`
- task.json SHA256：`0072af2249a8b7687ab1ebf3955fdec311fb87b6c359ff9cfc54e8bd37022d79`
- manifest resolved task SHA256：`2c939a11994c9f567b135d497c27a5bc10abb31c2419f7418ebe8de76976d79f`
- 教师抓起高度 0.230883 m；稳定放置确认 True；原 episode 结束 40.56 s。

### 7. seed 16100078 / liangzhu_seed_16100078

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000018`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000018`
- task.json SHA256：`2b1dec7ff42433f112bbacf77c3919469b058f1b348f40520cf583b4b1140492`
- manifest resolved task SHA256：`789ee1f335f07284b0ed79cb3023594878a5c17e0c7b949ee4cbab3b234accbc`
- 教师抓起高度 0.221445 m；稳定放置确认 True；原 episode 结束 34.80 s。

### 8. seed 16100044 / liangzhu_seed_16100044

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/01_N/part_000/episode_000014`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/01_N/part_000/episode_000014`
- task.json SHA256：`a58207b33e62870ddfcf2a75d5afa134f13b4d5e397f191bb4968684940e2294`
- manifest resolved task SHA256：`68dbac4bbd87dfd6560ff250078749c05b02908f57e6cc0d88efd276a63676b9`
- 教师抓起高度 0.208480 m；稳定放置确认 True；原 episode 结束 35.82 s。

### 9. seed 16100034 / liangzhu_seed_16100034

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/01_N/part_000/episode_000004`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/01_N/part_000/episode_000004`
- task.json SHA256：`8611062202394df2e3ccedce33227e15d1bae38649b3b6f1c6cd1c8f2e37ca6e`
- manifest resolved task SHA256：`27fdb4f1f88555b8e50289712366b5e9054ea4e81879be98d46faec0d95d71ca`
- 教师抓起高度 0.217458 m；稳定放置确认 True；原 episode 结束 38.68 s。

### 10. seed 16100021 / liangzhu_seed_16100021

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/00_N/part_000/episode_000021`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/00_N/part_000/episode_000021`
- task.json SHA256：`74fc811f72f05d92a0a279ceceb4db283b4ec11149759079459cc40025569c70`
- manifest resolved task SHA256：`2ee47fca6ef28088986edb3a3b7e89a57d2eafd8e1c4f243357b3086efab57dc`
- 教师抓起高度 0.231807 m；稳定放置确认 True；原 episode 结束 37.72 s。

### 11. seed 16100037 / liangzhu_seed_16100037

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/01_N/part_000/episode_000007`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/01_N/part_000/episode_000007`
- task.json SHA256：`676159c1b2650af5322870b63f58fcbdcd57efbfad20fad04f6e2f5c0d3d143d`
- manifest resolved task SHA256：`75e2962fcd1a384c21a28329fba8035829f4f2fa7ce06b83c45392db815c9a08`
- 教师抓起高度 0.207320 m；稳定放置确认 True；原 episode 结束 38.98 s。

### 12. seed 16100022 / liangzhu_seed_16100022

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/00_N/part_000/episode_000022`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/00_N/part_000/episode_000022`
- task.json SHA256：`d07c3065d5027dd819b2271e4c1f55383e008358968b4d72a75fbd0a6cd0f790`
- manifest resolved task SHA256：`72c5fd2b98ed0548389b0120f8e6978bf156d36b1afc67a4d7047784db4ef30a`
- 教师抓起高度 0.220762 m；稳定放置确认 True；原 episode 结束 41.74 s。

### 13. seed 16100089 / liangzhu_seed_16100089

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000029`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000029`
- task.json SHA256：`539f8ba1b749aa36811e4607f49005c328d985c180c8c0a4bb1133ba95ee9307`
- manifest resolved task SHA256：`ec1bee6bb51aa142b326facd746870b2edcc88eaa70187a1f28ba9b9177a4512`
- 教师抓起高度 0.205049 m；稳定放置确认 True；原 episode 结束 36.26 s。

### 14. seed 16100083 / liangzhu_seed_16100083

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000023`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000023`
- task.json SHA256：`fdbfa584a7b0de99ec4e9bedd353aff3632979ed1bee25b38930bac2b8506523`
- manifest resolved task SHA256：`f4e0587b389337a9966f78163234a6447b1cf43690358ef3ce5b9d39866c056c`
- 教师抓起高度 0.215849 m；稳定放置确认 True；原 episode 结束 38.08 s。

### 15. seed 16100030 / liangzhu_seed_16100030

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/01_N/part_000/episode_000000`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/01_N/part_000/episode_000000`
- task.json SHA256：`7064f3452c083bfa9799f9c3eeb3fb39a5c028b74913fe801445fec027d07878`
- manifest resolved task SHA256：`ec154af43baf18e5a777ea58276d6f18cf4790bd1a21b198496e28e358646943`
- 教师抓起高度 0.223814 m；稳定放置确认 True；原 episode 结束 44.24 s。

### 16. seed 16100070 / liangzhu_seed_16100070

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000010`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000010`
- task.json SHA256：`e54f38a55ae099fb4d5219c1cd561892999c7dc5da4e830385bf9bd16d45cf9c`
- manifest resolved task SHA256：`74db9afaf14c71c3baa4161613d7fe03ae5375169ba9d14cf849d94f8da46875`
- 教师抓起高度 0.216578 m；稳定放置确认 True；原 episode 结束 38.02 s。

### 17. seed 16100187 / liangzhu_seed_16100187

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/01_B/part_000/episode_000001`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/01_B/part_000/episode_000001`
- task.json SHA256：`a5f740a568a7b2018994b90e615f53b0dc4ab0c52616825a5d351477f2e3f589`
- manifest resolved task SHA256：`f4f32c0d6244d1f0df292fc31e81ea4a580549e270f0a4f04bb94f65fbd8a122`
- 教师抓起高度 0.228077 m；稳定放置确认 True；原 episode 结束 41.74 s。

### 18. seed 16100010 / liangzhu_seed_16100010

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/00_N/part_000/episode_000010`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/00_N/part_000/episode_000010`
- task.json SHA256：`bca5f210cf126130941ced32220a5fa06489408d91197e34c6cc4d6f5194f0c4`
- manifest resolved task SHA256：`8ef6d5028cc8690c3c0a4b6db94b89253df5464d4d2503ce99ea96da57b8f5f2`
- 教师抓起高度 0.230932 m；稳定放置确认 True；原 episode 结束 40.38 s。

### 19. seed 16100087 / liangzhu_seed_16100087

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000027`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000027`
- task.json SHA256：`d96aa4f81e161c924f65d269449bbd2d2432c9b76701efc062ffb7234580c1e2`
- manifest resolved task SHA256：`8f7028a4c9520f2b2e20d5546d01535ab1f69b114d6ac7680257fa6ad8f9cb52`
- 教师抓起高度 0.225724 m；稳定放置确认 True；原 episode 结束 39.42 s。

### 20. seed 16100026 / liangzhu_seed_16100026

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/00_N/part_000/episode_000026`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/00_N/part_000/episode_000026`
- task.json SHA256：`8c7ee9f75b4d340caabd8da0396a8892ad539ae7b7d92d5e06baad7208015fa2`
- manifest resolved task SHA256：`ec3fa57046266ec9020a10ba1e21491251266d196c6a5db8b36ce9e034b64683`
- 教师抓起高度 0.204463 m；稳定放置确认 True；原 episode 结束 38.12 s。

### 21. seed 16100081 / liangzhu_seed_16100081

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000021`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000021`
- task.json SHA256：`9ee12d9fe3b864aeea7dc274a57c90b8bb150ee1bb71de33be08311b8a330a85`
- manifest resolved task SHA256：`9237b790cfa4a6bc702598d3c8f7ed9eeb2f6ef16d52e9bcdbdc3c53391cef93`
- 教师抓起高度 0.224758 m；稳定放置确认 True；原 episode 结束 39.32 s。

### 22. seed 16100079 / liangzhu_seed_16100079

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000019`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000019`
- task.json SHA256：`813039c768aeee9d78e4076adee6565e834a7fbe1d33febf4acadb61fac36a2d`
- manifest resolved task SHA256：`ef16ad937710561337b282e810487c40153abd2608abdbd0d23285e950797255`
- 教师抓起高度 0.215040 m；稳定放置确认 True；原 episode 结束 38.08 s。

### 23. seed 16100071 / liangzhu_seed_16100071

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/02_N/part_000/episode_000011`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/02_N/part_000/episode_000011`
- task.json SHA256：`aa8ff2fcc714148e73ca22ca6230f77e8a8c6a8fe60ca18afec2c5ee2ce0cf41`
- manifest resolved task SHA256：`fc90919abb4f2dd39eea17c1b60e70b51afa4aef2fcf5fe58fdadef0543626b2`
- 教师抓起高度 0.221331 m；稳定放置确认 True；原 episode 结束 40.42 s。

### 24. seed 16100020 / liangzhu_seed_16100020

- H20 episode：`/diff/wallx_workspace/dzb/data_releases/rgb_rtc_sparse_20260908_v1/episodes/00_N/part_000/episode_000020`
- 原采集 episode：`/hdd1/dzb_xhq/arm-vla-grasp-sim-collection-release/outputs/collection_v2_release_20260907/00_N/part_000/episode_000020`
- task.json SHA256：`4c1fdbc4dacaf8469fa0880a3bc43cc79291a13e4a95012d8b794345eb1f6add`
- manifest resolved task SHA256：`c42687e9c01f114476507940e08a43b2a8149dbc5371371eb57e6571e8e1d035`
- 教师抓起高度 0.220574 m；稳定放置确认 True；原 episode 结束 39.78 s。

完整候选/排除原因、观察引用、初始 RGB 哈希、事件时间和其他源文件哈希见同目录 candidate_audit.json。README 外没有修改运行代码；是否启动及资源安排由 Master 决定。
