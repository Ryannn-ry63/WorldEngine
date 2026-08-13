# DiffusionDrive selector GRPO rollout v1

## 目的

这一步不是继续修改 V3 方法，而是把训练样本从 human-log observation 换成 epoch-100
DiffusionDrive 自己闭环 rollout 到达的 observation。每个闭环帧仍由冻结 generator
产生完整 20 条候选；SimEngine 在该真实闭环状态一次性模拟并计算 20 个 PDM reward；
V3 selector 再用完全相同的 exact-group GRPO 学习组内选择。

固定不变量：generator/perception 全冻结、20 条完整 action set、PDM 是唯一 reward、
无 IL、无 8192 词表、无 gate、reward components 不进入 selector 输入。

## 数据链

1. AlgEngine 从实际部署 forward 原子导出 20 条候选、candidate feature、route BEV、
   status/ego/agent context、reference/current logits。
2. SimEngine 读取和当前 action 同步的 sidecar，打包 reference + 20 candidates，在同一
   PDM scorer pass 中计算逐候选 reward。
3. 每帧写一个 schema-v2 自包含 record；merge 只在该目录存在时处理，不影响其他算法。
4. 9 个 rollout noise seed 使用相同 scene；按 scene hash 固定为 85/9/6 的
   train/development/certification，禁止 scene 泄漏。
5. cache 为 schema-v3，直接从 Git 中 config 与固定 baseline checkpoint 重建 selector
   contract，不依赖旧 V3 cache。
6. 固定 V3 参数 `T=1, lr=1e-4, KL=0, epoch=16`，只跑 3 个 optimizer seed；
   development 要求 worst-seed gain 非负，NC/DAC/TTC 各自下降不超过 0.005。
7. 冻结选择后只使用一次 seed 6/7/8 certification，最后运行三 seed 四块正式评测。

## H100 执行顺序

所有命令从 WorldEngine 根目录运行，每个任务使用固定 8 张 H100。先将
`DATA_TYPE` 与 `ASSET_NAME` 设为机器上实际存在并与目标训练分布一致的值；当前脚本
会在 preflight 阶段检查路径，不会启动 rollout。

### 1. 只做环境预检

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh preflight 0 DATA_TYPE ASSET_NAME r1
```

### 2. 一个场景 smoke，再做 8 场景 pilot

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh smoke 0 DATA_TYPE ASSET_NAME r1
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh pilot 0 DATA_TYPE ASSET_NAME r1pilot
```

只有 audit 同时确认 candidate/action parity、20 rewards、checkpoint SHA、noise namespace
和 code SHA 后，才能开始完整 collection。

### 3. 完整 9-seed rollout

同一 `RUN_ID`、`DATA_TYPE`、`ASSET_NAME` 提交 seed 0 到 8；最多 9 个任务，符合
同时不超过 10 个 H100 任务的约束。

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh collect 0 DATA_TYPE ASSET_NAME r1full
# 将上行 seed 依次替换为 1,2,3,4,5,6,7,8
```

### 4. 冻结 split 并建立 cache

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_prepare_cache.sh DATA_TYPE r1full
```

### 5. development 与一次性 certification

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_development_h100.sh
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_certify_h100.sh
```

### 6. 学长同口径四块正式评测

三个任务可并行，每个任务依次运行 Openloop-navtest、Openloop-failures、CL-NR、CL-R。

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_formal_seed_h100.sh 0
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_formal_seed_h100.sh 1
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_formal_seed_h100.sh 2
```

## 版本门

在 H100 smoke 之前，此分支是“代码预检版”，不是实验 release。smoke PASS 后创建预检
tag；certification + formal 完成后，按 `GRPO_SELECTOR_RELEASES.md` 的 V3 格式归档
三 seed 权重、selector-only state、结果、manifests、code bundle 与 SHA256。
