# DiffusionDrive GRPO Selector V2 progress-fix v1 runbook

## 实验目的

这是 V2 的 PDM progress normalization 修复复现实验。它保持原 DiffusionDrive
架构不变，只训练原 `plan_cls_branch` 的 10 个 tensor，是比 V3 更干净的
“冻结 base model、仅做 RL 后训练”归因证据。

本实验不重新跑 perception 或 frozen diffusion generator。它复用历史 V2 的
固定候选轨迹、候选特征和 logits，只用修复后的 reward 对这些候选重评分，然后
完整重做 27-grid 选择和三 seed 正式评测。

## 固定输入

- V2 代码锚点：`05442c1`；
- epoch-100 baseline SHA256：
  `1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514`；
- train cache：6339 tokens，noise seed 0；
- calibration cache：1118 tokens × noise seeds 0/1/2；
- train/calibration scene-disjoint；
- 20 个候选/样本；
- corrected reward contract：
  `navsim_pairwise_raw_progress_then_candidate_gate_v1`。

corrected cache 会写到：

```text
experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1/cache/
```

历史 cache 只读，不覆盖。

## 本地 1×H100 smoke

先在一张 H100 上运行完整小链路：公式单测、MMCV、真实候选
CPU/CUDA/official parity、4 份小 cache 重评分、一次 AdamW 训练、checkpoint
materialize 和 frozen-tensor audit。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v2_progress_fix_1h100_smoke.sh
```

成功标志：

```text
PASS one-H100 V2 progress normalization end-to-end smoke
```

默认 parity 是 16 tokens / 320 candidates，cache smoke 每份 2 tokens。只想快速
排查 shell/环境时可临时缩小 parity：

```bash
V2_PROGRESS_FIX_SMOKE_PARITY_TOKENS=2 \
./run_diffusiondrive_grpo_selector_v2_progress_fix_1h100_smoke.sh
```

## 8×H100 提交方式

### 1. 合并任务：prepare + seed0

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh prepare-seed0
```

一个排队任务内依次完成：

1. 回归测试和真实候选 parity；
2. train seed0、calibration seed0/1/2 的 reward-only 重评分；
3. 原 V2 27-grid：`T={1,4,8}`、`lr={1e-4,3e-4,1e-3}`、
   `KL={0,1e-4,1e-3}`，200 epochs；
4. 原 scene-held-out calibration gate；
5. seed0 checkpoint materialize、10-tensor audit、四块正式评测。

若 calibration gate 没有候选通过，程序不会暗中改规则；它会把
`gate_status=CALIBRATION_GATE_FAIL` 和
`selection_mode=predeclared_old_hparams_fallback` 写入 manifest，并使用实验开始前
声明的旧 V2 超参 `T=1, lr=1e-3, KL=1e-3, epoch=32`。

### 2. 可以一次提交三个任务后睡觉

推荐同时提交下面三个 8×H100 任务：

```bash
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh prepare-seed0
```

```bash
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh formal-wait 1
```

```bash
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh formal-wait 2
```

两个 waiter 会检查 `corrected_sweep` receipt、git commit、reward SHA 和 selection
manifest；selection 尚未生成时每 60 秒轮询，前序任务明确失败时立即退出。selection
通过后 seed1/seed2 会自动开始，并与 seed0 的正式评测并行，不需要等 seed0 评测
结束。默认等待上限 12 小时，可通过
`V2_PROGRESS_FIX_WAIT_TIMEOUT_SECONDS` 修改。

注意：waiter 等待期间不会伪造 GPU 计算量。如果平台会主动杀死低 GPU 利用率的
已调度任务，最稳妥的选择是只提交一个无空转的串行 overnight 任务：

```bash
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh overnight
```

它会在同一个任务里依次完成 prepare/seed0、seed1、seed2 和最终汇总，耗时更长，
但不依赖跨任务等待。

如果 selection 已经生成，也仍可直接并行提交：

```bash
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh formal 1
```

```bash
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh formal 2
```

`formal 1/2` 是立即运行模式；selection 不存在时会快速报错。

每个 seed 都评测：

- OpenLoop navtest：12147；
- OpenLoop navtest_failures / rare：289；
- CL-NonReactive：289；
- CL-Reactive：289；
- simulation failure 必须为 0。

### 3. 三个 seed 完成后，本地 CPU 汇总

```bash
./run_diffusiondrive_grpo_selector_v2_progress_fix_8h100.sh summarize
```

输出：

```text
experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1/formal/progress_fix_comparison.json
experiments/diffusiondrive/grpo_selector_v2_progress_fix_v1/formal/progress_fix_comparison.md
```

汇总同时对比 corrected V2、历史 pre-fix V2 和 epoch-100 baseline，并记录
mean/std、paired delta、最高 CL mean PDMS seed。`PASS` 只表示 provenance、覆盖率
和 frozen-tensor audit 通过，不代表分数必须上升。

## rare rollout 的边界

本轮只修 V2 common/original-data 后训练证据。rare rollout 的昂贵候选收集可以
继续保持 trainer-neutral，保存 V2/V3 都能消费的 superset context；但最终 rare
训练使用 V2 原 selector 还是 V3 scene-conditioned residual head，等与学长确认论文
归因口径后再定，不在本轮用 navtest 分数反向选择方法定义。
