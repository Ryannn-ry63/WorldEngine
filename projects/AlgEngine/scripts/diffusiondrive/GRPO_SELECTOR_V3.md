# DiffusionDrive GRPO Selector V3

## 目标与不变量

V3 只解决已经确认的瓶颈：20 条动态候选中存在更好轨迹，但原 final feature + MLP
不能稳定选到。它不改变最初与学长确定的 selector-only GRPO 主线：

- 冻结 perception、DiT、轨迹回归、anchor 和原 selector；
- `pi_old == pi_ref`，都来自 SHA256 固定的 epoch-100 baseline；
- action set 仍是同一次 forward 动态生成的完整 20 条轨迹；
- reward 仍只是真实 NAVSIM-v1 PDM score；
- loss 是 complete-action exact group GRPO + 可选小 KL；
- 没有 imitation、pairwise/ranking loss、gate 或新的 reward；
- reward/component/GT 不进入 selector 输入。

V3 新增的只是可训练的场景条件 residual selector。输入包括：最终 candidate
feature、最终轨迹几何、沿最终轨迹采样的冻结 BEV、冻结 status/ego/agents context。
它输出 `reference_logits + delta_logits`，最后一层零初始化，因此训练前与 baseline
严格相同。

## 不可变数据划分

- train：6339 tokens / 5133 scenes；
- development：671 tokens / 544 scenes，只用于调参和选 checkpoint；
- certification：447 tokens / 359 scenes，只允许在选择冻结后使用一次；
- 三者按完整 scene 隔离，且与 navtest 无 token overlap。

划分审计：

`experiments/grpo_sources/diffusiondrive_selector_navtrain_split_v3/split_audit_v3.json`

## H100 执行顺序

从 WorldEngine 根目录运行。每个任务固定使用 8 张 H100。

### 1. 生成 schema-v2 context cache（3 个任务可同时提交）

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_cache_bundle_h100.sh 0
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_cache_bundle_h100.sh 1
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_cache_bundle_h100.sh 2
```

每个 bundle 串行生成一个 train seed、一个 development seed 和一个 certification
seed。只有三个 bundle 全部 PASS 后才能继续。

### 2. 表示输入消融（1 个任务）

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_ablation_h100.sh
```

这个步骤只诊断 `feature / +geometry / +route BEV / +scene context` 的贡献，不选择
正式模型。结果：

`experiments/diffusiondrive/grpo_selector_v3/ablation/report.json`

### 3. 完整 action-set GRPO 的 V3 sweep（1 个任务）

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_sweep_h100.sh
```

8 张 GPU 各跑一个 full-context 超参数。只能根据 development 的三个未见噪声
seed 选择，certification 不会被读取。结果：

`experiments/diffusiondrive/grpo_selector_v3/sweep/selection.json`

### 4. 一次性 certification 与 checkpoint 物化（1 个任务）

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_certify_h100.sh
```

结果：

- `experiments/diffusiondrive/grpo_selector_v3/certification/report.json`
- `experiments/diffusiondrive/grpo_selector_v3/certification/selected_checkpoint.pth`
- `experiments/diffusiondrive/grpo_selector_v3/certification/checkpoint_manifest.json`

### 5. 学长同口径四块正式评测（3 个任务可同时提交）

```bash
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_formal_seed_h100.sh 0
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_formal_seed_h100.sh 1
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v3_formal_seed_h100.sh 2
```

每个 seed 都调用现有正式入口，依次完成：

1. Openloop-navtest；
2. Openloop-navtest_failures；
3. Closedloop-navtest_failures NR；
4. Closedloop-navtest_failures R。

seed 1/2 会用完全冻结的超参数重新训练 replica，不会再次选模型。

## 正式结果

正式设置由 development split 选定为 `T=1, lr=1e-4, KL=0, epoch=16`。下表为
三个独立训练/评测 seed 的结果；每个 seed 均完成 12147 条 OpenLoop-navtest、289
条 OpenLoop-failures、289 条 CL-NR 和 289 条 CL-R，各闭环均为 288 个 simulation
成功、0 失败。

| Seed | OpenLoop-navtest PDM | OpenLoop-failures PDM | CL-NonReactive PDM | CL-Reactive PDM | Success Rate |
|---|---:|---:|---:|---:|---:|
| 0 | 0.871823 | 0.642350 | 0.691032 | 0.679906 | 0.802768 |
| 1 | 0.873217 | 0.629046 | 0.694225 | 0.700079 | 0.823529 |
| 2 | 0.866382 | 0.595778 | 0.704007 | 0.693254 | 0.799308 |
| **均值** | **0.870474** | **0.622391** | **0.696421** | **0.691079** | **0.808535** |

相对同协议 epoch-100 DiffusionDrive reference，三 seed 均值提升为：

- OpenLoop-navtest PDM：`+0.013337`；
- OpenLoop-failures PDM：`+0.034358`；
- CL-NonReactive PDM：`+0.062493`；
- CL-Reactive PDM：`+0.059787`；
- Success Rate：`+0.040369`；
- Reactive EP：`+0.085619`。

独立 certification 的 top-1 reward gain 均值为 `+0.010043`，noise seed 6/7/8
分别为 `+0.010128/+0.008682/+0.011320`。scene-bootstrap 95% CI 为
`[-0.002173, +0.022933]`：均值为正，但区间下界尚未严格大于零。

需要同时报告的代价是 OpenLoop-navtest ADE/FDE 为 `0.893085/2.179479`，高于
reference 的 `0.638488/1.574205`。这是 selector 从拟合 human log 转向选择高 PDM
轨迹后出现的目标差异，并非 generator 或 perception 被更新；二者在 V3 中始终冻结。

## 进入后续 rollout / 难例的条件

先看 certification 与三 seed 四块评测，再决定是否扩展数据。无论结果好坏，后续
rollout 和 rare data 只允许更换 schema-v2 cache 的数据来源，不再改变 action set、
reward、GRPO objective 或 selector 输入定义。这样归因保持为同一个通用方法，而不是
针对 human-log/navtrain 修补分数。
