# P1 seed2 — preserved version, 2026-09-13

代码分支：`research/selector-p1-seed2-snapshot-v1`。
版本标签：`selector-p1-seed2-v1-20260913`。
实际评测代码提交：`8f1df15`。本次整理只增加文档、结果和临时文件忽略规则，
不改变冻结评测代码的哈希。

## 完整评测结果

Run: `p1_seed2_full_20260913_v1`。12/12 正式条件、4/4 bridge 条件完成。
四卡正式流水线约 6 小时 24 分钟，含初次 preflight 共计 25.73093 GPU-hours。
以下 PDM 使用 0–1 尺度。

| 条件 | V3 rare_tuned seed0 | P1 seed2 | 差值 |
|---|---:|---:|---:|
| rare288 CL-NonReactive | 0.776398 | 0.795172 | +0.018775 |
| rare288 CL-Reactive | 0.767742 | 0.802759 | +0.035017 |
| navtest12146 开环 | 0.875426 | 0.868246 | -0.007181 |
| rare288 开环 | 0.680170 | 0.684461 | +0.004291 |
| common64 CL-NonReactive | 0.853193 | 0.875733 | +0.022540 |
| common64 CL-Reactive | 0.876981 | 0.917442 | +0.040461 |

成功定义为单场景 `NC == 1 and DAC == 1`。rare NR 成功数为 269 → 265，
rare R 为 266 → 266；common NR 为 62 → 61，common R 为 63 → 63。
rare NR 挽回 6 个失败、新增 10 个失败；rare R 两者均为 5。
闭环行驶进度改善，同时部分可行驶区域合规指标与标准开环指标下降。

[results.json](results.json) 包含全部均值、ADE/FDE、安全分量以及
full288/development58/remainder230/common64 的配对差值和日志簇置信区间。
[provenance.json](provenance.json) 包含模型与原始报告的 SHA256 和本地归档位置。

## 评测系统

核心闭环对应学长的 `run_ray_distributed_testing.sh ... navtest_failures NR/R`：

- 同一份 288 场景文件，SHA256：
  `27203d52454f2538e6a70e95136b85ee8df35822f7707dc0091b4943ca7ad41d`。
- 同一 SimEngine、LogPlayController 和 metric manager。
- NR 使用轨迹回放，R 使用 IDM；常规 reward manager 关闭。
- 新入口支持四卡、固定噪声 `formal_navtest_seed0`、实际动作审计及预算管理。
- V3 的完整 rare NR/R 逐场景指标已复现历史结果。

common64 来自 navtrain 训练池，是固定抽样的部署检查集；不能替代标准 navtest，
也不能把 0.917442 当作标准 NAVSIM benchmark 成绩。

## P1 方法与代码导览

感知、生成器和参考 selector 冻结，更新 V3 的 scene selector。
P1 在既有 dense GRPO、成对偏好损失和 V3 KL 保持项基础上，增加按回报差加权的
winner negative log-probability，使用完整 20 候选 softmax。P1 使用固定反馈数据，
不使用 P2/P3 的额外候选查询。

本次导出 P1 seed2 第二阶段 step500，替换 54 个 selector 张量，
另外 963 个张量逐位不变。原生完整模型推理不需要奖励、Q 或 CFPI 路由。

| 功能 | 入口 |
|---|---|
| 四卡评测启动 | [launcher](../../run_diffusiondrive_selector_p1_snapshot_4h100.sh) |
| 阶段调度、预算、恢复 | [run_selector_snapshot.py](../../projects/AlgEngine/scripts/diffusiondrive/run_selector_snapshot.py) |
| 原生模型配置 | [snapshot config](../../projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_selector_p1_snapshot.py) |
| 模型导出、输入冻结 | [selector_snapshot_data.py](../../projects/AlgEngine/scripts/diffusiondrive/selector_snapshot_data.py) |
| 动作观察器 | [selector_snapshot_manager.py](../../projects/SimEngine/worldengine/manager/selector_snapshot_manager.py) |
| 闭环审计、bridge | [selector_snapshot_audit.py](../../projects/AlgEngine/scripts/diffusiondrive/selector_snapshot_audit.py) |
| 开环与辅助任务 | [selector_snapshot_tasks.py](../../projects/AlgEngine/scripts/diffusiondrive/selector_snapshot_tasks.py) |
| 指标、身份校验 | [selector_snapshot_common.py](../../projects/AlgEngine/scripts/diffusiondrive/selector_snapshot_common.py) |
| 最终报告 | [report_selector_snapshot.py](../../projects/AlgEngine/scripts/diffusiondrive/report_selector_snapshot.py) |
| P1 偏好损失 | [selector_decision_common.py](../../projects/AlgEngine/scripts/diffusiondrive/selector_decision_common.py) |
| P1 训练实现 | [train_selector_decision.py](../../projects/AlgEngine/scripts/diffusiondrive/train_selector_decision.py) |
| 训练来源和四臂对照 | [decision-feedback protocol](../../research/decision_feedback/README.md) |
| 完整运行命令 | [snapshot 操作说明](../../research/p1_snapshot/README.md) |

## 保存与复现

Git 保存代码、文档和轻量结果。完整模型、optimizer/RNG 恢复状态、缓存、
逐帧轨迹与数据资产保留在原工作区，定位与哈希见 provenance.json。
仅 clone 仓库不足以启动评测；新机器需要另行准备环境、数据、扩展和源 run 产物。

在现有工作区查看已完成报告：

```bash
bash run_diffusiondrive_selector_p1_snapshot_4h100.sh report p1_seed2_full_20260913_v1 --gpus 4 --gpu-hours 192
```

重评测使用新的 run ID，按操作说明依次运行 audit、preflight、all。
原 run 的输入、模型、契约、报告保持冻结。本发布页位于 docs 目录，
不影响运行契约覆盖的代码及 research/p1_snapshot 文件。

## 验证与解释边界

已有 130 项软件测试通过；CPU 真缓存导出分数差和 argmax 差异均为 0。
CUDA preflight/parity、四组实际 bridge、12 个正式条件均通过工程审计。
最终报告已与 12 个原始指标文件核对，归档输出哈希验证通过。

这是 development 上选出的 seed2 阶段快照，不代表三训练 seed 均值。
rare288 中 58 个场景已用于开发，另外 230 个也有历史暴露；不能称为盲测。
原 P3 的 STOP_EFFECT_FAIL 结论保留，论文应说明实际数据使用与模型选择过程。
