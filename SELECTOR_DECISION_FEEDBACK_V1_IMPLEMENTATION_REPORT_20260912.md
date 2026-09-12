# Decision-feedback v1 实现交接（2026-09-12）

## 当前状态

已按批准的诊断优先、四组对照方案完成实现。历史 feedback-v2 实现保存为
`aab4752`；历史实验文件和 `STOP_NO_MECHANISM_SIGNAL` 结论未修改。
新分支为 `research/selector-decision-feedback-v1`，独立工作目录为：

```text
/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-decision-feedback-v1
```

实现完成不代表方法有效，也不代表已达到论文要求。本轮未启动正式 GPU 训练或闭环仿真。
GPU 环境、真实干预一致性、P0 完整训练复现和研究假设，仍须由后续实验验证。

## 已实现的研究边界

首先检验：稀疏候选偏好监督与全 20 候选 argmax 决策之间的错配，是否确实导致
可改进的闭环错误。不把“选择了未评估候选”直接解释成坏决策。

- 对现有 64 个训练状态中 17 个未评估 argmax（NR 9 / R 8）补充真实闭环回报。
- 固定 16 对有信息候选，检查 shared 与 T 后续策略下的偏好稳定性。
- 切换后续策略的干预使用对应的 hybrid reference；先验证实际访问动作 sentinel。
- 诊断需满足至少 3 个独立日志上的支配错误、至少 2 个 R 决策，且 16 对中严格反转不超过 4 对。
  未通过则停止，不自动换方法、放宽阈值或进入训练。
- 诊断标签与训练标签隔离；训练只使用训练集状态，不把开发集失败案例变成训练标签。

通过诊断才进入固定的 3 seed、4 arm 实验：P0 原损失；P1 加 gap 加权 winner NLL；
P2 加匹配数量的随机候选查询；P3 查询当前未评估的全局 argmax。
第 100/300 步先冻结 P2/P3 两份查询计划，再获取任一组的新回报。
冻结生成器、候选集和推理架构；仅训练 selector，无推理时 reward/Q 依赖。

P0 seed0 最终必须复现原 T 的参数、分数和 argmax，否则按工程错误停止。
每组保留独立缓存、优化器与 RNG 状态；完成全部固定条件后统一报告，不选最好 seed。
NR/R 为共同主指标，同时报告 rare 改善和 common 保留。未授权 confirmation，入口也无 confirm 命令。

完整协议、阈值与命令见 [research/decision_feedback/README.md](research/decision_feedback/README.md)。

## 软件验证

- 新协议单元/集成测试 19 项通过；既有 feedback 40、continuous deployment 27、rare 25 项通过，共 111 项。
- 覆盖分段训练断点恢复、优化器/RNG 一致性、旧 pair loss 梯度一致性、跨组标签隔离、
  查询冻结顺序、诊断失败停止、报告重复生成和小批量 worker 分配。
- Shell 语法、Python 语法及 `git diff --check` 通过。
- NR/R 配置静态继承通过；不等同于完整 GPU/plugin 运行检查。
- 当前代码真实缓存 CPU 审计：`software_audit_20260912_v3`，状态 PASS。
  同一 run ID 重复审计也通过，锁定产物无漂移。
  963 个非 selector 张量保持不变；新目标在真实训练缓存上前向/反向通过，梯度有限且非零。
  optimizer steps = 0，rollout = false，GPU ledger = 0。
- 审计代码清单 SHA256：`dfe1c653bf7fa2386a046c380453df9908e4fe9818c70e3f3e9d999e57c2c073`。

审计证据位于
`experiments/diffusiondrive/selector_decision_feedback_v1/runs/software_audit_20260912_v3/`，
包括 `audit_report.json`、`decision_preflight_cpu.json`、`baseline_tensor_audit.json`、
`budget_forecast.json` 和 `decision_ledger.json`。v1/v2 是保留的早期软件审计，不是正式实验。

## 四卡运行

38 条历史耗时样本给出的预算估计：诊断 7.55、候选查询至多约 39.15、训练预留 8、
评测约 81.63 GPUh，总计约 136.33 GPUh，即四卡约 34.08 小时。
诊断本身约 1.89 小时；未通过会提前停止。新流程实际耗时尚未测量，此估计不是完成保证。

默认上限仍是 128 GPUh，可能触发预算停止。建议显式选择 192 GPUh，为四卡最多 48 小时的
计费上限留出余量；程序不会自动扩预算或删减组别。各阶段也有独立预算上限。

进入新工作目录后，逐条执行；前两条成功后再启动第三条。正式实验使用新的 run ID，
不要复用 `software_audit_*`。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-decision-feedback-v1

bash run_diffusiondrive_selector_decision_feedback.sh audit decision_feedback_20260912_192h_v1 --gpus 4 --gpu-hours 192

bash run_diffusiondrive_selector_decision_feedback.sh preflight decision_feedback_20260912_192h_v1 --gpus 4 --gpu-hours 192

nohup bash run_diffusiondrive_selector_decision_feedback.sh all decision_feedback_20260912_192h_v1 --gpus 4 --gpu-hours 192 >> decision_feedback_20260912_192h_v1.log 2>&1 &
```

audit 可在预留 GPU 前运行。preflight 和 all 必须在已分配四卡的实例上运行，保留调度器提供的
`CUDA_VISIBLE_DEVICES`。all 自动衔接诊断与通过后的 pilot，不会进入独立确认实验。

查看日志：

```bash
tail -n 60 -f decision_feedback_20260912_192h_v1.log
```

中断后可用相同命令恢复；必须保持 run ID、预算、代码、历史输入和 GPU 数一致。
若发生审计或一致性错误，应先检查日志，不应修改锁定文件或绕过门槛。
