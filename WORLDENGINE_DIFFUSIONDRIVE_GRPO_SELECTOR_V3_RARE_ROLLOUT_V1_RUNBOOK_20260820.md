# DiffusionDrive V3 Rare-Rollout V1 Runbook（2026-08-20）

## 1. 这次实验到底在做什么

这是学长 `RLFT rare rollout` 在 DiffusionDrive 动态 20 候选上的等价实现。它不是在已经训练好的 V3 checkpoint 上继续训练，也不是用 rollout 数据完全替换原数据。

- behavior policy、训练初始化和 KL reference 都使用同一个不可变 epoch-100 DiffusionDrive base checkpoint：
  `/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/experiments/diffusiondrive/e2e_diffusiondrive/epoch_100.pth`
- SHA256：`1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514`
- rollout 时挂载 V3 scene selector 架构，但最后一层严格全零，所以部署动作与 epoch-100 base 完全一致；不会加载之前 V3/rare-original 的 selector 权重。
- 训练时重新初始化一个严格零残差的 V3 selector，只训练 54 个 scene-selector tensors；963 个 base tensors 保持逐位不变。
- 训练分布严格为 50% common + 50% hard；hard = 原始 real rare + 通过学长 v1 筛选的在线 synthetic frames。

因此主因果对照是：

```text
rare_rollout_v1 - rare_original_frozen
```

两者使用相同 base、V3 架构、超参数、304,272 examples、4,800 optimizer steps 和 paired eval seeds；唯一核心变化是 hard 半边是否加入在线 synthetic rollout。

## 2. 对齐学长的具体依据

学长配置：

```text
/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/
projects/AlgEngine/configs/hydramdp/e2e_hydramdp_50pct_rlft_rare_rollout.py
```

学长数据集实现：

```text
/inspire/hdd/project/roboticsystem2/wangcaojun-240208020180/repo-wcj/WorldEngine/
projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_synthetic.py
```

对齐的语义：

- `NavSimOpenSceneE2EFineTuneSynthetic`
- `customized_filter="v1"`
- `include_real_failures=True`
- `normal_ratio=1`
- 每个 hard row 配一个同 log common row，整体 50/50。

学长 v1 synthetic filter 的等价条件为：

```text
max(candidate_score) >= 0.9
AND
(
  deployed_score <= 2/3
  OR
  (deployed_NC == 1 AND deployed_DAC == 1 AND deployed_EP < 0.2)
)
```

DiffusionDrive 不使用 HydraMDP 的固定 8192 trajectory vocabulary，因此这里只复用数据语义，不直接实例化该 dataset class。SimEngine 在真实闭环状态下对 DiffusionDrive 当步动态生成的 20 条候选做 PDM 评分，并保存可被 V3 selector 直接训练的模型无关记录。

## 3. 数据与审计契约

### 3.1 起始 rare 集

- 来源：全量 navtrain 103,288 tokens。
- epoch-100 DiffusionDrive 在 noise seeds 0/1/2 上做 official PDM scoring。
- 任一 seed 出现 collision、offroad，或安全样本 EP 严格低于该 seed 的 1st percentile，即进入 rare union。
- 最终 6,271 个 rare tokens。
- 每个 rare token 已有一个同 log、严格 common 的确定性配对 token。

既有审计：

```text
experiments/grpo_sources/diffusiondrive_selector_rare_original_v1/
rare_data/rare_data_audit.json
```

### 3.2 在线采集

- 将 6,271 个 rare token 转换为三个确定性、互斥、完整的 SimEngine scenario shards。
- 三个 lane 可提交到三个独立的 8×H100 实例并行采集。
- 每个 rare scene 产生 8 个闭环训练帧，每帧保存：
  - 20 candidate trajectories；
  - candidate feature / route BEV / status / ego / agents context；
  - epoch-100 reference logits 与当前 logits；
  - 20 个修正后 pairwise official PDM rewards 和 6 个 components；
  - deployed action/candidate parity；
  - raw SimEngine observation 路径；
  - checkpoint/config/code SHA provenance。
- raw SimEngine synthetic observations 和模型无关 candidate/reward records 都保留，因此如果最后决定做 V2 selector，可以复用 rollout，不必再次跑 6,271 个场景。

### 3.3 修正后的 PDM progress

每个 candidate 单独与同一个 PDM reference 比较：

```text
normalized_progress = candidate_raw_progress /
                      max(reference_raw_progress, candidate_raw_progress)
```

只有在 raw progress 都低于 official threshold 时才按官方规则置 1；之后再应用 candidate 自己的 multiplicative gate。不会复用旧的“先在 20 candidates 内归一化”的错误顺序。

### 3.4 训练混合

hard pool：

```text
6,271 real rare rows
+ senior-v1-filtered synthetic rollout rows
```

每个 synthetic row 继承其 origin rare token 的同-log paired-common token。训练每一块严格抽取一半 hard positions、一半对应 common positions。日志会额外生成 85/9/6 的 deterministic log-disjoint 标签，供审计和未来调参使用；本次固定超参数正式训练使用全部数据，与学长 rare-rollout 的训练语义一致。

正式训练固定为：

```text
temperature = 1.0
learning_rate = 1e-4
KL weight = 1e-3
epochs = 16
examples/cache/epoch = 6,339
three real fixed-noise caches = seeds 0/1/2
total examples = 304,272
optimizer steps = 4,800
train replicas = seeds 0/1/2
```

若 filtered synthetic + real rare 的 hard pool 大于每个 cache 保证可覆盖的 50,704 次 hard draws，程序会 fail closed，而不是悄悄漏掉一部分数据或增加计算量（全局三 cache 合计仍严格为 152,136 hard examples）。

## 4. 版本管理

- rare-original 正式锚点：branch `diffusiondrive-selector-grpo-v3-rare-original-v1`，commit `08a11c7`，formal tag `diffusiondrive-selector-grpo-v3-rare-original-v1-formal-20260820`。
- 当前新分支：`diffusiondrive-selector-grpo-v3-rare-rollout-v1`。
- 当前 V2 progress-fix 作业使用的主 worktree 不应在运行中切分支。本实现是在独立 worktree 完成的。
- 一卡 smoke 通过后打 smoke-ready tag；三个正式结果通过后再打 formal tag。
- Git 只保存代码、文档、manifest 和 SHA；大体量 scenario/rollout/cache/checkpoint artifacts 留在 `experiments/`。

## 5. 运行顺序

所有命令都在：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
```

执行正式任务前，等当前 V2 作业结束，再确认已切到并拉取：

```bash
git checkout diffusiondrive-selector-grpo-v3-rare-rollout-v1
git pull --ff-only
```

### Step A：本地 CPU 准备三个 scenario lanes

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_prepare_scenarios_local.sh
```

输出：

```text
experiments/grpo_sources/diffusiondrive_selector_rare_rollout_v1/scenarios/
  scenario_shard_00_of_03.pkl
  scenario_shard_01_of_03.pkl
  scenario_shard_02_of_03.pkl
  scenario_shards_manifest.json
  rare_rollout_scenario_audit.json
```

converter 逐 scenario 保留 chunk，意外中断后执行同一命令会验证并复用完整 chunks。

如果已经开好单卡 H100，希望把 Step A 和下一步 smoke 串行放在同一个任务里，
可直接执行：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_prepare_smoke_1h100.sh
```

该包装器只会在 scenario 准备与审计成功后启动 smoke；任一步失败都会立即退出。

### Step B：正式长任务前的一卡 H100 全链路 smoke

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_1h100_smoke.sh
```

它会验证：一个真实 rare scene 在线采集、动态 20-candidate PDM、v1 filter、混合 batch、H100 backward/CPU grad clip/AdamW、fresh zero V3 materialization，以及只有 54 个 selector tensors 被新增。只有该命令最终打印 `PASS one-H100 DiffusionDrive V3 rare-rollout end-to-end smoke`，才提交下面三个正式长任务。

### Step C：三个 8×H100 collect lanes 并行提交

如果 Step A+B 的单卡串行任务尚未结束，但希望提前提交排队，分别在三个独立
8×H100 任务中运行以下等待版本：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh collect-wait 0
```

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh collect-wait 1
```

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh collect-wait 2
```

三个任务等待同一个 `prepare_smoke_status.txt`。当前 commit 的 smoke 为 PASS 后，
三条 lane 同时开始；若当前 commit 的 smoke 为 FAIL，则全部立即失败退出。默认最多
等待 12 小时、每 60 秒检查一次。三个 lane 不互相串行。

若 smoke 已经 PASS，也可分别直接运行：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh collect-lane 0
```

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh collect-lane 1
```

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh collect-lane 2
```

同一 lane 中断后直接提交完全相同的命令。`completed_scenarios` 是持久 resume ledger；脚本只清理进程级旧 completion flag，不删除已完成 scenario、raw observation 或 reward record。最终 audit 允许“先失败、后续成功”的历史，但要求每个可模拟 scene 最终成功且严格有 steps 4..11 共 8 条记录。仅当 scenario shard 内的 `log_length < 20` 时允许排除；每个排除项必须在 collection audit 中记录 scene id、实际长度和证据哈希。

如果三条 lane 已完成模拟，但旧版审计因短场景仍被计入输入覆盖而在
`premerge_audit` 退出，不要重新进行 8×H100 rollout。使用 CPU-only 恢复入口：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh recover-collection
```

该命令复用已有 split records，依次完成三条 lane 的新版 premerge audit、merge
和 final audit。正式数据合同要求全部 rare token 被严格分解为“已有完整 rollout”
或“有长度证据的不可模拟短场景”，不允许未知缺失。

按此前 412 scenes 约 9 小时估算，每个约 2,090-scene lane 约 40–50 小时；三个 lane 并行时墙钟仍约 40–50 小时。

### Step D：本地 CPU 聚合、v1 过滤和 50/50 mixture

三个 lane 均 PASS 后运行：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_prepare_local.sh
```

输出：

```text
experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/data/
  manifest.json
  hard_pool.jsonl
  synthetic_cache.pt
```

### Step E：一个 8×H100 任务完成训练与全部评测

若希望把 Step D 的 CPU 聚合和 Step E 合并成一次提交，三个 collection
lane 全部 PASS 后直接运行：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh finalize
```

`finalize` 会先执行 Step D；聚合、过滤或审计失败时会立即退出，不会进入训练。
下面的 `finish` 命令只用于已经单独完成 Step D 的情况。

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh finish
```

同一 allocation 内：

1. GPUs 0/1/2 并行训练 seeds 0/1/2；
2. 逐 seed 运行 OpenLoop-navtest、OpenLoop-rare、CL-NonReactive、CL-Reactive；
3. 汇总 paired-seed mean/std 和 scenario-level deltas；
4. 主比较写到：

```text
experiments/diffusiondrive/grpo_selector_v3_rare_rollout_v1/formal/
rare_rollout_comparison.md
```

finish 预计约 8–10 小时，主要由三个串行四块正式评测决定。

若有三个独立的 8×H100 allocation，推荐把三个 seed 并行提交以缩短墙钟时间：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh finish-seed 0
```

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh finish-seed 1
```

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh finish-seed 2
```

每个任务只写自己的 `models/seedN`、正式评测目录和状态文件，三者可安全并行。
每个 seed 的 selector 训练只使用本 allocation 的 GPU 0；四块正式评测使用全部 8 卡。
三个任务全部 PASS 后，在本地任意实例运行 CPU-only 汇总：

```bash
./run_diffusiondrive_grpo_selector_v3_rare_rollout_8h100.sh summarize
```

三个 seed 并行时总计算量不变，预计墙钟接近单个 seed 的训练加四块评测；不要同时再启动
旧的 `finish`，否则会争用相同的 immutable seed 输出目录。

## 6. 本阶段明确不做的内容

- 不加载 rare-tuned/V3 checkpoint 继续训练。
- 不让训练好的 V3 充当 behavior policy。
- 不加入 Behaviour World Model augmentation；BWM 是下一阶段。
- 不用 synthetic 完全替换 real data。
- 不因一次分数不好临时改变 v1 filter、训练量或 eval seed。

这些限制保证论文归因仍是“同一个 base policy 上的后训练数据/方法变化”，而不是预训练架构或额外算力变化。
