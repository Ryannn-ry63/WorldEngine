# CFPI 首轮实现交接 — 2026-09-05

## 当前结论

首轮代码与 CPU 验证完成；真实闭环采集、方法训练和效果评估尚未开始。
研究方向仍是待证伪假设，不能根据实现完成宣布创新成立或效果超过 V3。

工作树：`experiments/worktrees/WorldEngine-selector-cfpi-v1`（相对原 WorldEngine）。
工作分支：`research/frozen-selector-cfpi-v1`。
继承基线提交为 `d90aa71`，本文件与首轮新方法代码在后续独立提交中。
协议见 [PROTOCOL.md](PROTOCOL.md)，启动与恢复见 [RUNBOOK.md](RUNBOOK.md)。

## 研究决定保持不变

- ACCEPT：冻结生成器、感知和 base selector，继续研究 V3 的纯 selector 进化。
- AMEND：检验自然重规划下的闭环决策纠错与原有能力保留，而非泛称迁移 GRPO。
- REJECT：把 IPCT/basin 作为默认正确设计，或声称 GRPO 在扩散模型上数学失效。

本轮分类权重属于已知成本敏感学习思想，不以换损失公式本身作为论文贡献。
首先判断完整实际分支反馈能否被可见输入学到，再判断分类更新是否真的优于
相同标签、相同优化预算的 GRPO 与 Q 回归。若只有 Q 回归有效，就接受标签方向，
不接受新损失贡献。首轮 CV 不能代替连续闭环部署和独立测试。

## 已实现的边界

train-only source 1024 → baseline A/B → Q-blind 分层冻结 64 targets →
8 个原策略 sentinel → pilot64 / repeat8 各 20 分支 → 工程与信息 gate →
6 种方法 × 2 LR × 4 log-disjoint folds × 3 seeds → 筛选报告。

所有分支仅在指定决策替换一次动作，然后恢复冻结 V3；不 latch 整条轨迹。
保留原始 20 候选与原控制器。重复试验是固定条件重复性检查，不是未来不确定性估计。
推理输入白名单不含奖励、Q 标签、事件状态或日志 ID。Q 回归部署使用 `direct_q`；
其余使用原 base logits + residual。导出完整 checkpoint 时检查非 selector 张量不变。

运行器冻结协议、源码、噪声、checkpoint、数据和拓扑的 provenance；
限制 144 GPUh、管理本次进程组、审计恢复产物、每 25 步保存优化器和 RNG 状态。
汇总报告重新校验 cache/gate、配置身份、各折预测归属和 checkpoint 哈希。
未提供 expand/dev/test 入口；任何结果都不自动扩量或推广。

## 实际验证证据

2026-09-05 使用 AlgEngine 的 torch 2.0.1+cu118 环境：

- CFPI、已有 V4 cache / CCV / V3 cache / sidecar 回归：47 passed，51.89 秒。
- 完整规划插件环境的 V3 与 scene selector 测试：11 passed，87.07 秒。
- 合计 58 项通过。警告为现有依赖的弃用提示。
- CE 训练主动在第 81 步中断，从第 75 步恢复；50/150/500 检查点的参数和预测
  与不中断训练逐张量完全一致。Q 回归的 500 步训练、导出、加载也通过。
- 合成数据覆盖 freeze → 41 treatments → 分支审计 → 两份 cache → gate →
  完整 144 job 报告接口；这些是软件测试，不是真实驾驶效果。
- 真实规划头验证 `direct_q` 不叠加 base logits，优化后所有冻结参数保持不变。
- 原 V3 selector 文件通过实际 SHA256 和 strict state loading：54 tensors，1,364,213 参数。
- 新采集配置通过实际插件导入及 MMCV Config.pretty_text 检查。
- 新 CFPI 管理器和 BaseEnv 在 SimEngine 解释器中导入成功，未实例化模拟器。
- `bash -n` 和 `git diff --check` 通过。

真实模拟器候选 parity、H100 扩展实算、1024 源池实际物化和实际闭环效果仍未验证。
训练 CPU 单测使用小型合成表示，不替代真实 256-dim V3 的 GPU 训练验证。

## 环境阻塞的准确描述

旧沙箱检查记录曾显示驱动不可达，不能据此推断宿主机完全没有 GPU。
审批后只读检查显示当前为 1 张 NVIDIA GeForce RTX 4090。
正式预检于 2026-09-05 12:39 UTC 拒绝启动，明确报错：
`Expected 8 visible H100 GPUs, found 1: ['NVIDIA GeForce RTX 4090']`。

诊断 run：`cfpi_hardware_audit_20260905_v2`。
证据相对新工作树位于：
`experiments/diffusiondrive/selector_cfpi_v1/runs/cfpi_hardware_audit_20260905_v2/`，
查看 `preflight.json`、`logs/cuda_preflight.log` 与 `decision_ledger.json`。
没有启动任何真实采集、训练或 Hopper kernel preflight。
此次诊断 run 不复用为正式实验；交接文档更新也改变代码库存哈希。

下一步需要用户提供实际 H100 节点/会话，或明确批准另行制定非 H100 硬件配置。
在 H100 节点使用 RUNBOOK 中的新 run ID，先 preflight，再 first_phase。
不要删除原 contract 绕过检查，也不要未经研究决定继续扩量或消费 development/test。

## 旧工作保全

原 V3、rapg/V4 工作树均未修改。再次逐文件哈希校验通过：V3 的 36 个文件、
rapg/V4 的 244 个文件及各自 HEAD 与快照一致。
成功快照：`experiments/research_snapshots/cfpi_20260905T080752Z`（相对原 WorldEngine），
包含 manifest、两份工作树 overlay 和 verified Git bundle；大型原 artifacts 原地保留。
此前不完整快照单独保留为 `cfpi_20260905T080709Z_incomplete`，不作为恢复依据。
