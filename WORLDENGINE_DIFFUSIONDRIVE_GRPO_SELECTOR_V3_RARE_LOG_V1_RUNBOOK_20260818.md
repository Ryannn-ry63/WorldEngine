# DiffusionDrive GRPO Selector V3 Rare-Log V1 执行说明

日期：2026-08-18

> **版本状态：pre-progress-fix，仅用于代码留档。**
>
> 2026-08-18 已确认 `diffusiondrive_online_pdm_reward.py::pairwise_official_scores`
> 的 progress 归一化顺序与 official PDM scorer 不一致。本版本生成的 rare 日志挖掘
> 结果仍可审计，但 candidate reward context cache、selector checkpoint 和正式评测结果
> 不得作为修复后结论，也不得在修复版中复用。修复后必须从 context cache 阶段重新生成。

## 1. 这次修正了什么

原来的 human-log V3 实际对应学长表格中的 **RLFT common（原数据）**：从 epoch-100
checkpoint 开始，在原始 navtrain observation 上用 RLFT/GRPO 后训练 selector。它不是
“rollout 数据替换人类数据”。

本次新增的 rare-log-v1 仍然使用原始 navtrain observation，不使用 WorldEngine
closed-loop rollout observation。变化只在于训练数据选择：

- 在 full navtrain 上部署 epoch-100 DiffusionDrive；
- 用官方 NAVSIM PDM 找出 collision、off-road 和低 progress 样本；
- 使用所有被选中的 rare token，不做比例下采样；
- 每个 rare token 配一个同一 log、三个噪声种子均非 rare 的 strict-common token；
- 保持原 V3 的模型、20 候选、PDM reward、exact-group GRPO 和 selector-only 更新。

因此，它要回答的是：在 DiffusionDrive V3 架构不变时，把 common 原数据改成
rare-focused 原数据，能否复现学长结果中 **RLFT rare（原数据）** 明显优于
**RLFT common（原数据）** 的趋势。

rare-log-v1 是本仓库的明确、可审计定义，不宣称和学长 HydraMDP 配置中的 rare_log
逐行相同。学长代码中的训练 head、损失和数据生成机制不同，最终讨论时应把
“rare 原数据”和“按 rare log 整段取样”继续区分。

## 2. 与学长实验的对应关系

| 实验 | observation 来源 | 数据选择 | 当前状态 |
|---|---|---|---|
| 学长 RLFT common（原数据） | 原训练数据 | common | 对应原 human-log V3 |
| 学长 RLFT rare（原数据） | 原训练数据 | rare | 本次 rare-log-v1 的目标对照 |
| 学长 supervised on rare logs | 原训练数据 | rare logs | 损失不同，不是本次实验 |
| 学长 RLFT rare rollout | WorldEngine synthetic | recoverable rare frame | 对应后续 rollout 路线，不是本次实验 |
| DiffusionDrive rollout-v1 | WorldEngine collision rollout | 全部保留帧 | 已完成的独立消融，不参与本次训练数据 |

学长表格里 common RLFT 的 common/closed-loop 退化、rare RLFT 的 rare PDM 增益说明
“RLFT 本身”不是充分条件，数据选择才是当前最重要的变量。本次保持 V3 训练预算不变，
就是为了隔离这个变量。

## 3. Formal 数据合同

### 3.1 全量范围

- 物理 metric cache：115,434 token；
- full navtrain：103,288 token、1,192 logs；
- navtest：12,146 token；
- full navtrain 与 navtest 必须零重叠；
- baseline SHA256：
  1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514。

脚本只新建 metadata index view，不复制或修改原 metric-cache payload。

### 3.2 三噪声 rare 定义

epoch-100 baseline 在相同 103,288 token 上使用三个 token-wise 固定 namespace：

- selector_context_train_seed0
- selector_context_train_seed1
- selector_context_train_seed2

每个 seed 使用官方 NAVSIM scorer。单 seed 的 rare 是以下集合的并集：

1. no_at_fault_collisions == 0；
2. drivable_area_compliance == 0；
3. NC=1、DAC=1、EP>0 的样本中，EP 位于该 seed 最低 1%（含阈值）。

三个 seed 中任意一次为 rare，即进入 formal rare 全量集合。三个 seed 从未为 rare 的
token 才能作为 strict common。

### 3.3 同 log pairing

每个 rare token 确定性匹配一个同 log strict-common token：

- rare 不下采样；
- 同一 rare token 只出现一行；
- common 足够时不复用，不足时循环复用；
- 某个 rare log 没有 strict-common token 时直接失败；
- rare/common 不允许重叠。

这使训练集既聚焦难例，又保留局部环境相近的 common anchor。

## 4. 训练合同

训练算法与原 V3 相同：

- scene-conditioned residual set selector；
- 20 个动态候选；
- candidate feature、轨迹几何、route BEV、ego/agent/status context；
- exact-group GRPO；
- temperature=1，learning rate=1e-4，KL=0；
- generator、perception 和 baseline 963 tensors 全冻结；
- 只新增/更新 scene selector 54 tensors；
- optimizer seeds 0、1、2。

为公平比较，预算严格等于原 V3：

- 3 个 fixed-noise cache；
- 每个 cache 每 epoch 6,339 examples；
- 16 epochs；
- 304,272 examples；
- rare/common 全局各 152,136；
- batch size 64；
- 4,800 optimizer steps；
- 不做 development 超参搜索或 early stopping，固定取 epoch 16。

训练器要求每个 noise cache 在复用前覆盖所有 rare/common pair；如果 full rare 数量超过
固定 V3 预算能覆盖的范围，会 fail closed，而不是静默下采样。

## 5. 单 H100 工程 smoke

在单卡 H100 实例中直接运行：

~~~bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_rare_log_1h100_smoke.sh
~~~

它从 full navtrain 中最密集的一个 log 确定性选择 64 token，实际走完：

~~~text
3× baseline inference
→ 3× official PDM score
→ rare/common mining
→ 3× context cache
→ 2-epoch selector training
→ checkpoint materialization/audit
→ trained checkpoint inference
~~~

它会自动 source 已验证的 AlgEngine 环境并检查唯一可见 GPU 是 H100 sm_90。不需要手动
conda activate。smoke 只证明工程链闭合，不能作为论文分数。

输出位置：

~~~text
experiments/diffusiondrive/grpo_selector_v3_rare_log_v1/local_smoke/<run_id>/
~~~

如需复用同一次失败任务的中间结果，可先设置固定 ID：

~~~bash
export RARE_LOG_SMOKE_ID=smoke_manual_01
./run_diffusiondrive_grpo_selector_v3_rare_log_1h100_smoke.sh
~~~

## 6. 8×H100 正式全流程

在 8×H100 分布式实例的命令框中直接运行：

~~~bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_rare_log_8h100.sh all
~~~

这是推荐入口。一次排队会顺序完成：

~~~text
preflight/index
→ full-navtrain mining seeds 0/1/2
→ official PDM score seeds 0/1/2
→ rare/common 全量集合
→ context cache seeds 0/1/2
→ optimizer replicas 0/1/2（卡 0/1/2 并行）
→ checkpoint materialization/audit
→ formal four-block eval seeds 0/1/2
→ rare vs common V3 vs epoch-100 paired 汇总
~~~

各昂贵阶段都有固定路径、SHA256 和覆盖审计。任务重启时，已有产物只有通过当前 provenance
验证才会复用；不通过会报错，不会把旧结果重新标记成新结果。

也可以从已经完成的阶段继续：

~~~bash
./run_diffusiondrive_grpo_selector_v3_rare_log_8h100.sh mine
./run_diffusiondrive_grpo_selector_v3_rare_log_8h100.sh cache
./run_diffusiondrive_grpo_selector_v3_rare_log_8h100.sh train
./run_diffusiondrive_grpo_selector_v3_rare_log_8h100.sh eval
~~~

cache/train/eval 不会补跑缺失的前置昂贵阶段，而是明确报缺失文件；正常情况下优先使用
all。

## 7. 输出位置

数据与审计：

~~~text
experiments/grpo_sources/diffusiondrive_selector_rare_log_v1/
├── metric_cache_navtrain_full/index_audit.json
├── mining/seed{0,1,2}/
└── rare_data/
    ├── rare_tokens.yaml
    ├── common_tokens.yaml
    ├── rare_common_union.yaml
    ├── pairs.jsonl
    └── rare_data_audit.json
~~~

训练和正式结果：

~~~text
experiments/diffusiondrive/grpo_selector_v3_rare_log_v1/
├── cache/train_seed{0,1,2}/
├── replicas/seed{0,1,2}/
├── formal/formal_eval/
└── formal/
    ├── rare_vs_common_vs_base.json
    └── rare_vs_common_vs_base.md
~~~

正式评测仍是原来的四块：OpenLoop navtest、OpenLoop navtest failures、ClosedLoop
NonReactive、ClosedLoop Reactive。

汇总器对 rare-log V3、原 common V3、epoch-100 paired reference 使用相同 seed 0/1/2，
同时输出均值、population std、同 seed summary delta 和逐 scenario paired delta。

## 8. 仍需与学长确认的研究定义

工程实现已经固定，但汇报时建议明确询问：

1. 学长表格里的 RLFT rare（原数据）是 token/frame 级 rare，还是整段 rare log；
2. rare_log 配置是否把 rare log 内 common frame 全部加入；
3. 学长 rare 的 collision/off-road/low-EP 阈值和本次三噪声 union 是否应进一步对齐；
4. 学长 common/rare 实验是否严格等训练 step、real/synthetic 比例和 loss 权重；
5. 如果本次 rare 原数据有效，下一步应先做 rare-log 粒度消融，还是再加入 recoverable
   WorldEngine rollout frame。

不要把本次 full-navtrain rare-log-v1 和已完成的 collision-only rollout-v1 混在一个结果名
中；前者测试原数据的 rare selection，后者测试 closed-loop synthetic observation。
