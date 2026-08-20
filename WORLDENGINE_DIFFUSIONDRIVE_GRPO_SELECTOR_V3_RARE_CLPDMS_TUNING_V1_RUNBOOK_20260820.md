# DiffusionDrive GRPO Selector V3 Rare CL-PDMS Tuning V1

日期：2026-08-20

## 1. 版本与实验边界

本分支从 tag
`diffusiondrive-selector-grpo-v3-rare-original-v1-formal-20260820` 创建。
rare-original v1 的主因果实验已经封存，不能被本实验覆盖。

本实验只把 `rare_tuned` 作为 incumbent，在相同 rare/common 1:1 数据、相同
corrected PDM reward 和 selector-only 更新约束下，小范围调整 learning rate、KL 与
训练 epoch。目标是三 seed mean Reactive CL-PDMS，而不是重新定义 rare 数据。

主论文结论仍使用等算力 `rare_frozen vs paired_common`；本实验属于 secondary tuned
result。

## 2. 闭环数据隔离

源数据是 navtest_failures 的 288 个场景、89 个 logs。固定 split seed 为
`20260820`，按 log_name 的 SHA256 顺序选择最接近 20% 场景数的前缀：

- CL-dev：58 scenes、22 logs；
- CL-confirm：230 scenes、67 logs；
- token 和 log 均零重叠；
- 两部分 union 必须严格还原源 288 scenes。

候选只能读取 CL-dev。选定模型后，才允许执行一次三 seed full formal；最终报告会从
full Reactive CSV 中单独提取未参与选参的 CL-confirm 指标。

## 3. 固定候选与晋级规则

所有候选固定 temperature=1、batch size=64、每 cache epoch 6,339 examples：

| Name | LR | KL | Epoch |
|---|---:|---:|---:|
| early32 | 1e-4 | 1e-3 | 32 |
| early48 | 1e-4 | 1e-3 | 48 |
| lowlr48 | 5e-5 | 1e-3 | 48 |
| lowlr64 | 5e-5 | 1e-3 | 64 |
| anchored64 | 1e-4 | 3e-3 | 64 |
| lowlr_anchored64 | 5e-5 | 3e-3 | 64 |

`rare_tuned` epoch64 和 `rare_frozen` epoch16 是只读控制。

流程先用 optimizer/eval seed0 对六个 challenger 做 CL-dev screening，再对排名前二
补齐 seeds1/2。最终按三 seed mean Reactive CL-PDMS 排名。challenger 只有同时满足：

1. 相对 rare_tuned 平均提升至少 0.005；
2. 至少 2/3 paired seeds 不下降；
3. 每个 seed 的 58 个场景全部完成；

才能替换 incumbent。否则正式选择仍为 rare_tuned。

## 4. 提交方式

先在单 H100 实例执行工程 smoke：

~~~bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_rare_clpdms_tuning_1h100_smoke.sh
~~~

smoke 会验证 H100/MMCV/gsplat、真实场景 pickle 覆盖、1-split Ray、Reactive inference、
CSV 合并和指标审计，不需要手动 activate conda。

然后提交一个 8×H100 tune 任务：

~~~bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_rare_clpdms_tuning_8h100.sh tune
~~~

生成 PASS selection 后，同时提交三个正式 lane：

~~~bash
./run_diffusiondrive_grpo_selector_v3_rare_clpdms_tuning_8h100.sh formal 0
./run_diffusiondrive_grpo_selector_v3_rare_clpdms_tuning_8h100.sh formal 1
./run_diffusiondrive_grpo_selector_v3_rare_clpdms_tuning_8h100.sh formal 2
~~~

三个 lane 完成后，本地 CPU 或任意已有实例运行：

~~~bash
./run_diffusiondrive_grpo_selector_v3_rare_clpdms_tuning_8h100.sh summarize
~~~

`summarize` 不检查 GPU 数，也不会重新选参。

## 5. 输出与判定

主要输出：

~~~text
experiments/diffusiondrive/grpo_selector_v3_rare_clpdms_tuning_v1/
├── closed_loop_split/split_audit.json
├── models/<candidate>/seed{0,1,2}/
├── development_metrics/<candidate>/seed{0,1,2}.json
├── selection/{seed0_screen.json,selection.json}
└── formal/{formal_eval/,clpdms_tuning_summary.json,clpdms_tuning_summary.md}
~~~

正式报告以三 seed mean 为主，best seed 只作填表补充。若 CL-confirm 或 full mean 没有
超过 incumbent 75.73，必须保留 negative result，并继续使用原 rare_tuned 作为当前最佳，
不能根据单个 seed 事后换模型。
