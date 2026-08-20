# DiffusionDrive GRPO Selector V3 Rare Original V1 执行说明

日期：2026-08-19

## 1. 实验问题

本实验复现学长 result.md 中同一个研究问题：

> 从已训练 checkpoint 出发，保持 RLFT/GRPO 方法不变，只把原始训练数据的
> common 分布换成 rare-focused 分布，rare 指标是否相对 common 对照改善？

它不是 rollout 数据训练，也不是把一整段 log 都标成 rare。训练 observation 仍来自
原始 NAVSIM navtrain：

- rare token：epoch-100 DiffusionDrive 在原 navtrain 上经 official PDM 判定的难例；
- common token：与 rare token 同 log、且三个 mining seeds 都非 rare 的 token；
- rare-original：rare token 与 same-log common anchor 以 1:1 采样；
- paired-common：只采样同一批 common token，训练 examples 和 optimizer steps
  与 rare-original 完全相同。

学长 HydraMDP 的 rare_log 配置实际也是 rare token 与同 log normal token 配对，不是
整段 log。DiffusionDrive 不复制 HydraMDP 网络或 imitation-loss 开关，只对齐研究任务、
rare 挖掘规则和最终四列比较。

## 2. 版本边界

代码基线是已验证的 PDM progress normalization 修复：

- 基线 commit：7eddfb8；
- baseline checkpoint SHA256：
  1c450bad0cf62ab9110a8101d2ff6c96984541bd975ddea598ddb2add086a514；
- CPU、CUDA 与 official PDM parity 已通过；
- selector-only 更新，baseline 963 tensors 必须保持不变。

旧 rare-log-v1 分支、旧 candidate context cache、旧 selector checkpoint 和旧正式分数
都属于 pre-progress-fix，不被本流程复用。物理 NAVSIM metric-cache payload 只读；新流程
在独立 rare_original_v1 根目录建立 index 和 provenance。

## 3. 全量 rare 数据合同

范围：

- full navtrain：103,288 token、1,192 logs；
- navtest：12,146 token；
- full navtrain 与 navtest 零 token overlap；
- 物理 metric cache：115,434 token。

epoch-100 baseline 在 full navtrain 上使用 fixed noise seeds 0、1、2。每个 seed 都执行
完整模型 inference 和 official NAVSIM PDM score。单 seed 的 rare 是：

1. no_at_fault_collisions == 0；
2. drivable_area_compliance == 0；
3. 在 NC=1、DAC=1、EP>0 的样本中，ego_progress 严格低于该 seed 第 1 percentile。

三个 seeds 中任意一次 rare 就进入主实验 rare union；从未 rare 的 token 才能作为
strict common。每行保存 rare_votes（1、2 或 3）和逐 seed failure modes，最终报告同时
给出 vote strata。主实验保留 union 是为了覆盖安全类长尾；votes>=2 可在主实验后作为
独立消融，不能事后替换主定义。

每个 rare token 确定性配一个 same-log strict-common token。rare 不下采样；common
不足时只在同 log 内循环复用。没有同 log strict-common、rare/common overlap、token
覆盖或 SHA 漂移都会 fail closed。

## 4. 实验矩阵

| 行 | 数据 | 作用 |
|---|---|---|
| epoch100 | 无后训练 | baseline |
| common_v3_progress_fix | 既有 corrected V3 common 原数据 | 连续性对照 |
| paired_common | rare 对应的同 log common，common-only | 主等算力因果对照 |
| rare_frozen | rare/common 1:1，冻结超参 | 主实验 |
| rare_tuned | rare/common 1:1，开发/认证后全量重训 | 补充上界 |

rare_frozen 与 paired_common 的正式合同相同：

- temperature=1，learning rate=1e-4，KL=1e-3；
- 3 个 fixed-noise train caches；
- 每 cache 每 epoch 6,339 examples；
- 16 epochs，batch size 64；
- 总计 304,272 examples、4,800 optimizer steps；
- optimizer seeds 0、1、2。

两者唯一主变量是采样分布：rare_frozen 为 rare/common 1:1，paired_common 为
common-only。因此 rare_frozen - paired_common，特别是 OP-PDMS（rare），是最干净的
主证据。common_v3_progress_fix 用于与既有 V3 连续比较。

## 5. tuned rare 的数据隔离

全部 pairs 按 log_name 确定性切成目标 80% train、10% development、10%
certification。切分按 pair 数做确定性平衡，但 log 绝不跨 split。每个 split 单独生成
pairs、rare/common filters、union filter 和 SHA audit。

固定 56 candidates：

- 8 组 temperature、learning rate、KL；
- epochs 为 1、2、4、8、16、32、64；
- train noise seeds 0、1、2；
- development noise seeds 3、4、5；
- certification noise seeds 6、7、8；
- development 只在 rare token 上排序；
- certification 只消费一次并报告 scene bootstrap 95% CI；
- formal eval 不参与选参。

若没有候选通过 development 正增益 gate，工程流程会记录
development_gate_passed=false，并选择最佳 stability fallback；这种 tuned formal
结果只能作描述性补充。frozen rare 与 paired-common 主对照仍然有效。

选中的 tuned 超参最终在全部 rare pairs 上重新训练 optimizer seeds 0、1、2。

## 6. corrected reward provenance

context-cache manifest 记录以下 SHA：

- context extractor 和 diagnostic helper；
- diffusiondrive_online_pdm_reward.py；
- online planning head、scene selector、NavFormer detector。

训练 report 还记录 trainer、公共 exact-group loss 和 corrected reward SHA。缓存、
checkpoint 或代码任一漂移都会拒绝复用。checkpoint audit 必须证明 baseline tensors
不变，仅 scene selector tensors 更新。

## 7. 单 H100 smoke

直接运行：

~~~bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_rare_original_1h100_smoke.sh
~~~

无需手动 conda activate。脚本会 source 固定环境并检查 H100 sm_90，走完：

~~~text
3× baseline inference
→ 3× official PDM score
→ rare/common mining
→ 3× corrected context cache
→ KL=1e-3 rare-balanced two-epoch training
→ checkpoint materialize/audit
→ trained checkpoint inference
~~~

smoke 只验证工程闭环，不是论文分数。输出位于：

~~~text
experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/local_smoke/<run_id>/
~~~

## 8. 一次排队的 8×H100 正式命令

直接提交：

~~~bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_rare_original_8h100.sh all
~~~

必须直接执行文件或显式使用 bash，不要使用平台默认 sh；脚本依赖 Bash pipefail、arrays
和 process substitution。

一个作业内顺序完成：

~~~text
preflight + full-navtrain index
→ mining inference/official PDM seeds 0/1/2
→ rare union + same-log paired common
→ log-disjoint train/development/certification split
→ full formal caches seeds 0/1/2
→ tuning caches train 0/1/2, development 3/4/5, certification 6/7/8
→ 8-GPU parallel 56-candidate rare sweep
→ development selection + one-shot certification
→ rare_frozen / paired_common / rare_tuned 各 3 个 optimizer replicas
→ 三种方法各 3 个 four-block formal evaluations
→ 五行 result.md 格式汇总
~~~

昂贵产物会先做覆盖、角色、seed、filter SHA、代码 SHA 和 checkpoint SHA 校验；通过
才复用。可按阶段续跑：

~~~bash
./run_diffusiondrive_grpo_selector_v3_rare_original_8h100.sh mine
./run_diffusiondrive_grpo_selector_v3_rare_original_8h100.sh cache
./run_diffusiondrive_grpo_selector_v3_rare_original_8h100.sh sweep
./run_diffusiondrive_grpo_selector_v3_rare_original_8h100.sh train
./run_diffusiondrive_grpo_selector_v3_rare_original_8h100.sh eval
./run_diffusiondrive_grpo_selector_v3_rare_original_8h100.sh summarize
~~~

正常提交优先使用 all，避免重复排队。

## 9. 输出

~~~text
experiments/grpo_sources/diffusiondrive_selector_rare_original_v1/
├── metric_cache_navtrain_full/
├── mining/seed{0,1,2}/
├── rare_data/
└── tuning_split/{train,development,certification}/

experiments/diffusiondrive/grpo_selector_v3_rare_original_v1/
├── cache/{full,tuning}/
├── sweep/selection.json
├── certification/report.json
├── models/{paired_common,rare_frozen,rare_tuned}/seed{0,1,2}/
└── formal/
    ├── formal_eval/
    ├── rare_original_comparison.json
    └── rare_original_comparison.md
~~~

rare_original_comparison.md 主表与学长 result.md 对齐：

- OP-PDMS（navtest）=openloop_navtest.score；
- OP-PDMS（rare）=openloop_failures.score；
- CL - Valid Rate=reactive 中 NC=1 且 DAC=1 的 episode 比例；
- CL - PDMS=closedloop_reactive.score。

JSON 额外保留 ADE/FDE、CL-NR、CL-R、逐 seed、population std、scenario-level paired
wins/ties/losses、rare vote strata 和输入 SHA。

## 10. 判定原则

这项实验验证通性，不以单次分数必须全面上涨为正确性标准。方法性有效需要：

1. official mining、严格 percentile 和 token/log 合同成立；
2. progress reward 与 official PDM parity 成立；
3. train/development/certification 无 log 泄漏；
4. frozen rare 与 paired-common 等 examples、steps 和 eval seeds；
5. 仅 selector tensors 变化；
6. 五行三 seed formal 结果及 SHA 完整。

主要问题是 rare-focused 分布是否相对 paired common 改善 rare 指标，以及是否存在
common/closed-loop trade-off。正负结果都应保留；只有合同或 provenance 失败才重跑。

## 11. 2026-08-20 正式结果封存

正式五行结果为 3 个 paired evaluation seeds 的均值，数值按百分制展示：

| Model | OP-PDMS（navtest） | OP-PDMS（rare） | CL - Valid Rate | CL - PDMS |
|---|---:|---:|---:|---:|
| epoch100 | 85.71 | 58.80 | 76.82 | 63.13 |
| common_v3_progress_fix | 87.07 | 60.43 | 80.05 | 68.31 |
| paired_common | 86.15 | 57.88 | 76.24 | 65.48 |
| rare_frozen | 86.13 | 69.63 | 92.27 | 73.31 |
| rare_tuned | 87.27 | 68.14 | 89.97 | 75.73 |

主因果结论使用等算力的 `rare_frozen - paired_common`。`rare_tuned` 使用相同
rare/common 1:1 数据，但从 16 epochs 增加到 64 epochs，是 secondary tuned result，
不能替换等算力主对照。

`rare_tuned` 三个 seed 的 Reactive CL-PDMS 分别为：

- seed0：0.7677418685121105；
- seed1：0.7452733910034604；
- seed2：0.7590137370242214；
- 三 seed mean：0.7573429988465974。

当前最佳单 seed 是 seed0（76.77）。checkpoint SHA256 为
`bd35f0293c878c6cddb985ac0b8aaae91d4f7f8e0bf2c3bf70ba49ba329c02a3`。
正式 provenance：

- code commit：`3939f02154ed9f801bdcc1bba0190c3520bcbb11`；
- `formal/rare_original_comparison.json` SHA256：
  `f467d0fcd66d55ca525c9c07bdb131112175b1bb661f47b56dcc53dd9bec0056`；
- `sweep/selection.json` SHA256：
  `64f69b34727551db4f0188643d8a32230a7262006a2f0369843fc0d97acd70b3`；
- `certification/report.json` SHA256：
  `cdd0a0acd77597b8208c0a64fb3128b465189f69e032e7d88eff10ed5b4919eb`。

上述大型实验产物保留在 experiments 下，不提交 Git；本节保存足以定位和校验它们的
路径、SHA 和结果。后续 CL-PDMS 定向调参必须从本版本另开分支，不能覆盖本实验。
