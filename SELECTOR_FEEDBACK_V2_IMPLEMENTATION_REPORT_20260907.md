# Selector Feedback V2 实现交接（2026-09-07）

## 四卡更新（同日，用户当前仅有四卡）

当前实现同时支持 --gpus 4 / 8，新增中性名称入口
`run_diffusiondrive_selector_feedback.sh`，建议四卡新 run ID：
`rare_feedback_20260907_4gpu_v1`，各阶段统一 --gpus 4 --gpu-hours 128。
新命令以 [README](research/feedback/README.md) 为准，下文八卡命令与代码 SHA 为历史记录。

- 场景、种子、训练步数、NR/R、S/U/T、8 个决策时刻与 sentinel 数量完全保留。
- 仿真 GPU 可见性、规划器绑定、运行契约、主进程和子进程计费均随卡数变化。
- Ray 返回的物理 ID／UUID 在本协议中显式映射为逻辑 split_0..N-1；旧协议行为不变。
- 四卡 collection watchdog 为 12 小时，预算和阶段上限仍独立强制执行。
- 实测样本带 GPU 数量，按 GPUh 归一到原历史预测基准，避免把四卡双倍时长当成双倍 GPUh。
- 128 GPUh 对应四卡累计 32 小时；当前约 19.6 小时预测是八卡历史外推，尚非四卡实测。
- 卡数／预算不可在同名运行中改写，完成条件与账本按原设计恢复，不清零。

最终代码清单 SHA256：`a283d0de723cefcafe31fa577c3fe250422b2d93fcde0badc1f35245b69ddf52`。
40 项 feedback（含 11 项资源调度测试）+ 25 项 rare + 27 项 continuous deployment，
合计 92 项通过；shell 语法、入口 --help 与 git diff --check 通过。
真实四卡 GPU preflight、闭环与效果评测尚未运行；由用户按 README 顺序启动。
本次未操作其他占卡进程，也未覆盖旧软件审计／正式实验产物。

## 预算授权更新（同日，用户确认“实验预算不要紧”之后）

当前入口已支持显式正有限值 --gpu-hours，默认仍为 64。
推荐本次新 run 使用 128 GPUh：feedback48 / train8 / eval60 / buffer12。
只扩大资源容限，不增加训练步数、场景、种子或反馈轮数，不放宽效果门槛。
同一 run 各阶段必须传相同预算；旧运行契约不改写、不重置。

更新后代码清单 SHA256：
`8935f0f9333ce7a4d25309a5ce7a0ed143a3a22882cb0b84aa4e53ba345e5807`。
新增预算测试 3 项后，29 项 feedback + 25 项 rare + 27 项 continuous deployment
= 81 项全部通过；audit/preflight/all/report 的 128 GPUh CLI 模拟入口检查通过。
未启动正式训练或闭环，预算更新后的真实缓存审计由下面的新正式 run 的 audit 步骤执行。

启动命令已更新至 [README](research/feedback/README.md)，建议新 run ID：
`rare_feedback_20260907_128h_v1`。
下文保留预算扩展前的验证和预测，原“当前代码哈希”和“64 GPUh 不宜开跑”结论
属于更新前记录；不要把旧软件审计 SHA 当成此次修改后的源码 SHA。

## 结论与边界

已在独立工作树实现经批准的两轮反馈实验，并完成当前代码的软件与真实缓存 CPU 检查。
尚未启动本方法的正式 GPU 训练、真实闭环采集或性能评测；软件 PASS 不代表方法有效或已经具备论文结果。

- 分支：`research/frozen-selector-feedback-v2`，基于 rare-v1 的 `0c237e2`。
- 工作树：`experiments/worktrees/WorldEngine-selector-feedback-v2`（相对 canonical WorldEngine）。
- 实现保存在工作树中，尚未创建新提交。旧 rare-v1 和 CFPI 工作树及已有实验产物未修改。
- 方法说明与入口：[research/feedback/README.md](research/feedback/README.md)。
- 当前实现代码清单 SHA256：`788d4ea9bf615155f4fb1bfcb1e633e4c81ce1f691dc96b532e0bc37efe43505`。
- 本交接文件位于代码清单之外；不改变已经通过的运行契约。

## 已实现的实验问题

检验策略变化后，针对未解决／新退化难例更新闭环反馈，能否比静态反馈 S、
均匀刷新反馈 U 更有效地修复难例，同时保留 V3 的成功率并接近 Gate-V3 的 PDM。
目标方法为 T；不是对新 loss 或顶会新颖性的预先确认。

固定感知、生成器、原始 20 个候选、V3 selector 架构、LogPlayController 和 0.5s 重规划。
NR/R 为共同主要指标；部署不消费奖励、未来信息或训练用强制动作接口。

- 训练源固定 128 场景／89 日志，每日志最多 2 场景，配额与排除规则不放宽。
- 第一轮共享反馈与初步策略；第二轮比较 S/U/T，统一学习目标、每臂匹配查询预算。
- 实际访问动作 sentinel 先于干预验证；仅强制一个动作，随后恢复冻结策略续行。
- 保存真实策略分数与显式干预动作，不伪造 argmax；失败轨迹禁止采违规后的无效状态。
- 两轮各 500 更新，继承模型、优化器和随机状态；额外 T 种子共享 seed0 采集的反馈。
- screen 只有 T 达到预设双模式门槛才进入额外种子；不得临时改推表现更好的控制组。
- 报告完整种子、救回／破坏场景、按日志聚类的配对区间；证据不足明确输出 INSUFFICIENT_EVIDENCE。
- 自动阶段推进、累计预算、断点恢复和代码／模型／数据哈希核对已接入。
- 当前开发评测集有历史暴露，不称为盲测；confirmation230 未获授权，未纳入运行。

## 最终验证证据

最新软件审计目录：
`experiments/diffusiondrive/selector_feedback_v2/runs/feedback_software_audit_20260907_v3`。

审计完成时间：2026-09-07 09:15:03 UTC。当前文件清单与运行契约、CPU preflight 的代码哈希完全一致。

| 检查 | 结果 |
| --- | --- |
| 新增单元／集成测试 | 26 项通过 |
| 旧 rare 回归测试 | 25 项通过 |
| 旧 continuous deployment 回归测试 | 27 项通过 |
| 真实基线非 selector 张量核对 | 963 个未变，PASS |
| V3 初始化一致性 | PASS |
| 8 个真实上下文的 selector bank 分数误差 | 最大绝对误差 1.52587890625e-05，argmax 一致 |
| 真实过滤 dense 缓存反向传播 | 梯度有限，范数 2.3446321522320757 |
| 极低概率配对梯度 | 有限，PASS |
| dense 数据排除检查 | 8851 逻辑记录／592 日志，排除集重叠为 0 |
| NR/R × train/development/common 配置解析 | 6 种组合通过（配置级检查，非 GPU 插件／闭环检查） |
| shell 语法与 git diff --check | PASS |

集成测试覆盖真实路由接口到 sidecar 审计的自然动作／强制动作语义、500 步玩具 CPU 训练、
中断续跑逐位一致和共享父优化器条件。另有纯内存完整报告 smoke，验证 screen 放行与最终
区间不足时停止过度宣称。真实缓存审计本身 optimizer_steps=0、training_performed=false、
rollout_performed=false；预算账本 gpu_hours_used=0，active=null。

开发期软件审计 v1/v2 保留，不冒充当前代码验证或正式实验。正式运行须用新 run ID，
先 audit 再 preflight；冻结正式契约后不得直接修改代码继续同一 run。

## 开跑前预算风险：尚不能保证 64 GPUh 完整运行

历史耗时与保守 NR 系数给出的预测：

| 阶段 | GPUh |
| --- | ---: |
| 两轮反馈与访问轨迹采集 | 38.9083 |
| screen 评测 | 15.7442 |
| final 新增评测 | 19.7803 |
| 训练预算 | 4.0000 |
| 合计 | 78.4329 |

预测尚不包含全部 CPU 开销，不是实测；NR 使用 1.2 倍保守系数。
当前上限仍为 64 GPUh（feedback24 / train4 / eval30 / shared buffer6）。
反馈预测甚至超过 feedback24 加全部 buffer6，因此不能假定反馈阶段一定能走完。
运行器会在下一条件预计无法容纳时停止，不自动扩预算、缩减对照、删种子或清零账本。

建议下一步先审查采集启动／服务开销是否可以在不改变实验协议和配对真实性的条件下压缩，
或由用户另行批准预算与分配调整；在此之前不建议直接启动整轮 `all`。
之后在 8 卡实例执行正式 preflight，再让 sentinel 和 prefix 审计决定是否准许真实干预继续。
GPU 插件、NR/R 实际闭环复现、运行时预算以及方法收益，均仍待正式运行验证。
