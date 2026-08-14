# WorldEngine DiffusionDrive base-model rollout v1 handoff

日期：2026-08-13

## 冻结决定

- 分支：`diffusiondrive-selector-grpo-rollout-v1`
- base policy：immutable DiffusionDrive `epoch_100.pth`
- base checkpoint SHA256：
  `1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514`
- `DATA_TYPE=navtrain_50pct_collision`
- `ASSET_NAME=navtrain`
- single-GPU preflight `RUN_ID=r1preflight1g`
- single-GPU smoke `RUN_ID=r1single1g`
- single-GPU pilot `RUN_ID=r1pilot1g`
- smoke `RUN_ID=r1`
- pilot `RUN_ID=r1pilot`
- full collection `RUN_ID=r1full`

这一阶段只把 selector 的训练 observation 从 human log 换为 base model 的 closed-loop
rollout state。V3 selector 架构、20-action set、PDM-only reward、exact-group GRPO 目标和
冻结的 generator/perception 均不改变。rollout source 使用 exact-zero residual selector，
因此部署动作与 epoch-100 base policy 保持一致，不加载 V3 训练后的 selector。

## 当前状态

- V3 已通过独立 branch/tag/release 冻结，不会被 rollout 覆盖。
- rollout v1 CPU/V3 contract tests：`22 passed`。
- baseline SHA256 已核验。
- 单 H100 preflight、smoke、pilot 已 PASS；9 个 full collection seed 已完成并重新审计 PASS。
- 本地已有两个无关 `.orig` 未跟踪文件，后续操作不要删除或覆盖。

### 单 H100 首次 smoke 记录

- attempt：`20260813T122929Z`
- H100 `sm_90`、MMCV、gsplat preflight：PASS。
- smoke：FAIL，不是性能结果；warm-up 第 3 帧在第一份 planner sidecar 可生成前就请求
  sidecar，属于 rollout glue 的 off-by-one。
- 修复：第 3 帧只补齐四帧历史，从第 4 帧（第一条 planner action 已发布）开始动态
  candidate scoring；新增 runner-report pre-merge fail-closed audit。
- 失败目录保留，不参与后续 cache；一键脚本的 timestamp RUN_ID 会自动避开它。

### 单 H100 smoke/pilot PASS 记录

- attempt：`20260813T123748Z`
- smoke：1 scene、8 reward records，audit PASS。
- pilot：最多输入 8 scenes；dense-reward short-scene filter 合法跳过 1 个，最终 7 个
  runner reports 全部 succeeded、0 failed。
- pilot records：56，7 个场景各 8 帧（step 4--11）。
- pilot 最大 deployed/candidate parity error：`2.2888183579539145e-06`。
- pilot mean selected/oracle reward：`0.6054327094 / 0.7974017539`，mean oracle
  headroom：`0.1919690445`。
- checkpoint、noise namespace、resolved config 和 code provenance 一致，最终 rollout
  audit：PASS。
- runner gate 已修正为 pilot 接受 7--8 个有效场景；仍要求所有 report succeeded。
- full collection 前必须先冻结/提交当前修复，使 rollout record 中的 Git code SHA 对应
  干净工作树；本次 smoke/pilot 属开发门验证，不作为正式训练数据。

### 8-H100 full collection PASS 记录（2026-08-14）

- seed `0--8` 的 WorldEngine rollout execution 均完成，merge 均成功；正式数据不需要重跑。
- collection 数据代码提交：`dfbd348af6fc45773f4286e06b24acbf2cd77d0f`。
- 每个 seed：412 scenes、3296 reward records、每个场景固定 8 帧、0 duplicate。
- 9 个 seed 的 scene/step 文件集合完全相同；每个 seed 的 checkpoint、源配置和代码
  provenance 唯一且一致。
- 最大 deployed/candidate parity error 均为 `2.2888183579539145e-06`。
- seed `0--8` 的 mean selected reward 分别为：`0.570983`、`0.576702`、
  `0.567137`、`0.567231`、`0.567934`、`0.538768`、`0.574846`、`0.576979`、
  `0.568807`。
- 9 份 `rollout_audit.json` 均为 `PASS`。

原始 8-H100 任务在仿真、runner audit 和 merge 成功后，全部仅在最后的 provenance audit
报 `config/code provenance drifted within rollout`。根因是旧审计器错误地要求一个 seed 的
所有记录共享同一个 `resolved_config_sha256`；实际 8 个 planner worker 的解析配置包含
各自 `split_0--split_7` 输出路径，因此应当有 8 个不同哈希。全量检查确认每个 worker
内部哈希稳定，源配置 SHA、checkpoint SHA 和 code SHA 全局一致。这是审计器 false
negative，不是 rollout 数据失败。

修复后的门控同时检查：全局唯一源配置 SHA、全局唯一 code SHA、恰好 8 个 worker，
以及每个 worker 内唯一 resolved-config SHA；真实漂移仍会 fail closed。下游 cache builder
也改用同一 provenance 合约，避免在 prepare-cache 阶段重复误报。CPU regression：
`15 passed`；Python/Bash syntax：PASS。修复冻结 tag：
`diffusiondrive-selector-grpo-rollout-v1-audit-fixed-20260814`。

## GPU worker 约束

所有命令从 WorldEngine 根目录运行。每条 rollout 命令是一个独立的 8-H100 任务；只有
一个 8-H100 节点时按顺序运行。若集群可同时分配多个 8-H100 节点，full collection
的 seed 0--8 可分别提交，但必须使用相同的 `DATA_TYPE`、`ASSET_NAME`、`RUN_ID` 和代码
commit。

## 单 H100 执行顺序

单卡入口保持与 8 卡入口完全相同的数据和模型合约，只把 WorldEngine、planner 和 merge
收缩为一个 `split_0`。单卡耗时会显著增加。为避免 immutable output 与未来 8 卡任务
冲突，单卡 smoke/pilot 使用独立 RUN_ID。

推荐直接执行仓库根目录的一键门控脚本；脚本内部使用固定的绝对 Python 环境，无需手动
执行 `conda activate`：

```bash
bash /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/run_diffusiondrive_rollout_v1_1gpu.sh
```

该文件顺序运行 preflight、smoke、pilot，任何一步失败都会立即停止，并打印
`wrapper_log` 和底层 `persistent_log` 路径。每次执行会生成带 UTC timestamp 的独立
RUN_ID，因此失败后重试不会覆盖或碰撞旧的 immutable output。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine

./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_1gpu_h100.sh \
  preflight 0 navtrain_50pct_collision navtrain r1preflight1g

./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_1gpu_h100.sh \
  smoke 0 navtrain_50pct_collision navtrain r1single1g

./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_1gpu_h100.sh \
  pilot 0 navtrain_50pct_collision navtrain r1pilot1g
```

前三步必须顺序执行；可以在同一个单 H100 allocation 中运行整段。任一步失败即停止。

## 8-H100 执行顺序

单卡只用于开发 smoke/pilot 和故障定位。正式 rollout collection 的每个 seed、后续
development、certification 与 formal evaluation 均使用固定 8 张 H100。单卡 smoke/
pilot records 不进入正式 cache。

正式 collection 每个 8-H100 任务只运行一个 seed。任务入口会强制校验冻结 tag、干净
tracked worktree、8 张可见 H100 和 immutable 输出目录：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
bash run_diffusiondrive_rollout_v1_collect_8h100.sh 0
```

将末尾 seed 分别替换为 `1,2,3,4,5,6,7,8`，共提交 9 个独立 8-H100 任务；有多个
8-H100 节点时可以并行。不要在正式任务前重新运行单卡 smoke/pilot。

以下是底层展开命令，仅用于排查入口：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine

./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh \
  preflight 0 navtrain_50pct_collision navtrain r1

./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh \
  smoke 0 navtrain_50pct_collision navtrain r1

./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh \
  pilot 0 navtrain_50pct_collision navtrain r1pilot
```

确认 smoke 和 pilot 的 `rollout_audit.json` 均为 `PASS` 后，执行 9 个 full rollout seed：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine

for seed in 0 1 2 3 4 5 6 7 8; do
  ./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_h100.sh \
    collect "${seed}" navtrain_50pct_collision navtrain r1full || exit 1
done
```

9 个 seed 全部完成后冻结 split 并建立 cache：

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_prepare_cache.sh \
  navtrain_50pct_collision r1full
```

之后依次运行 development 和一次性 certification：

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_development_h100.sh
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_certify_h100.sh
```

最后运行三个正式 replica；单个 8-H100 节点顺序执行：

```bash
for seed in 0 1 2; do
  ./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_rollout_v1_formal_seed_h100.sh \
    "${seed}" || exit 1
done
```

## 停止门

- preflight 不为 `PASS` 时不得启动 smoke。
- smoke 或 pilot audit 不为 `PASS` 时不得启动 full collection。
- full collection 的 9 个 seed 必须具有相同 scene/step coverage；split 脚本会拒绝漂移。
- development gate 失败时不得查看或使用 certification 结果调参。
- certification 只运行一次；不能基于 certification 反向重新选择参数。
