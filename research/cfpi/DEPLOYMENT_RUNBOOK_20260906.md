# CFPI 连续部署 A/B：执行说明

日期：2026-09-06。分支：`research/frozen-selector-cfpi-v1`。

实现已完成，正式 A/B 实验尚未启动。当前开发节点为 1×RTX 4090；
以下命令必须在 8×H100 实例上运行。代码不会在非 H100 节点上静默降级执行。
旧 pilot/CV 的 checkpoint、contract、gate、report 和原 winner 均未修改。

## 1. 已冻结的研究范围

| 方法 | LR | steps | 线上评分 |
|---|---:|---:|---|
| q_grpo_t1 | 1e-4 | 500 | 原分类器 + 学习残差 |
| q_grpo_t5 | 3e-5 | 500 | 原分类器 + 学习残差 |
| local_grpo_t1 | 1e-4 | 500 | 原分类器 + 学习残差 |
| q_mse | 1e-4 | 50 | 直接 Q 预测，不加原分数 |

每种方法固定 seeds 0/1/2。不增加 loss，不重新搜索超参。
T1-Q 与 local-T1 是匹配配置的反馈比较；T1/T5 同时改变 LR 和训练温度，
不能把差异单独归因于温度。线上均为 argmax。

A：原 pilot64、原四折路由，目标前 scalar V3，目标及此后持续使用新 selector。
新反馈训练按 origin log 折外；这不表示初始化 V3 从未见过该日志。
共 768 条主体 rollout，之前增加原 repeat8 场景的 scalar 同模型路由 sentinel。
旧 20 个分支的 CSV 仅用于配对 one-shot 参考，不重新采集分支。

A 的推进门槛在新 outcome 前锁定：均值 PDM gain >=0.005、至少 2/3 seed 为正，
均值 SR/NC/DAC 的下降分别不超过 0.01，且完整工程审计通过。
没有方法通过时停止，不 refit、不进行 B、不扩 train256。

B：A 通过后，四种配置各三个 seed 在全部 pilot64 上重新拟合；
在另 128 个场景从第一个决策起部署，另跑 scalar V3 seed0 和固定
gate-conditioned V3 seed0，两种 baseline 先运行。共 1,792 条 rollout。
自然筛选仅按源场景 ID 和 family：rare/common 各 64，严格全局每 log 一个；
排除全部 pilot logs 和已知历史 V3 development/certification logs，
不按成败、违规时刻、reward 或 Q 抽样。名单在 A 开始前即冻结。
这是有固定 family 配额的 training-pool screening，不是总体分布代表性样本，
也不是独立 development/test。历史曝光审计仍不穷尽。

## 2. 在 8 卡节点运行

首先进入现有研究 worktree，不能在 canonical WorldEngine 根目录运行旧入口：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-cfpi-v1

CFPI_PILOT_RUN="$PWD/experiments/diffusiondrive/selector_cfpi_v1/runs/cfpi_pilot_20260905_r2"
CFPI_CV_RUN="$PWD/experiments/diffusiondrive/selector_cfpi_v1/runs/cfpi_pilot_20260905_r2_cv1"
CFPI_DEPLOY_RUN=cfpi_continuous_20260906_v1

bash run_diffusiondrive_selector_cfpi_deployment_8h100.sh \
  preflight "$CFPI_DEPLOY_RUN" \
  --pilot-run "$CFPI_PILOT_RUN" --cv-run "$CFPI_CV_RUN" \
  --gpus 8 --gpu-hours 64
```

`preflight` 校验 H100/torch/CUDA、MMCV/gsplat、基线非 selector 张量一致，
冻结输入/名单/模型路由，并在 CUDA 上重算全部 768 个已有折外目标动作。
此 stage 不启动场景 rollout，也不 refit。`COMPLETE` 后再运行：

```bash
bash run_diffusiondrive_selector_cfpi_deployment_8h100.sh \
  phase_a "$CFPI_DEPLOY_RUN" \
  --pilot-run "$CFPI_PILOT_RUN" --cv-run "$CFPI_CV_RUN" \
  --gpus 8 --gpu-hours 64
```

顺序为预检 → scalar sentinel8（含闭环结果复现）→ 12 个学习策略条件 → A 报告。
每一个条件先通过审计才继续。A 结束后停止在阶段边界；不会自动进入 B。
查看输出目录中的 `phase_a_report.json`，仅当 `phase_b_authorized=true`：

```bash
bash run_diffusiondrive_selector_cfpi_deployment_8h100.sh \
  phase_b "$CFPI_DEPLOY_RUN" \
  --pilot-run "$CFPI_PILOT_RUN" --cv-run "$CFPI_CV_RUN" \
  --gpus 8 --gpu-hours 64
```

B 会重新验证 A 的完整结果，然后 refit、部署、报告；没有后续扩张入口。
如果 A 未通过或结果不完整，即使命令被提前调用，也不会训练或采集 B。

只重新审计/汇总已有结果，不启动 GPU 作业：

```bash
bash run_diffusiondrive_selector_cfpi_deployment_8h100.sh \
  report "$CFPI_DEPLOY_RUN" \
  --pilot-run "$CFPI_PILOT_RUN" --cv-run "$CFPI_CV_RUN" \
  --gpus 8 --gpu-hours 64
```

`report` 在预算耗尽后仍可使用；未完成阶段只写 progress，不产生通过报告。
AlgEngine/SimEngine 的 Python、扩展和资源路径沿用旧 launcher 的环境构造；
若实例布局不同，仍可使用 `DIFFUSIONDRIVE_ALGENGINE_ENV_OVERRIDE` 和
`DIFFUSIONDRIVE_SIMENGINE_PYTHON`，不要靠更换 torch/CUDA 版本绕过预检。

## 3. 预算、停止和续跑

同一 run 的 A <=24 GPUh，B <=40 GPUh，总计 <=64 GPUh。
每次调用都传 `--gpu-hours 64`；这不是给各阶段再各分配 64 GPUh。
runner 对已启动 stage 的整个 8 卡预留时间计费，包括准备和小批量 refit；
预留了轮询/终止进程的缓冲，在达到预算前停止。账本不替代实例实际计费记录。

只管理自己创建的进程组，不执行全局 `ray stop`、`pkill` 或 GPU 抢占。
不正常退出会保守计入未记录的时间；发现旧会话仍有进程时拒绝并发重启。
不要删除账本、改旧合同或换 run ID 清零已经消耗的预算。

重跑相同命令即续跑：

- 有 `collection_audit.json` 的完整条件：核验全部 artifact，不重复 rollout。
- 未审计完成的条件：输出移入本条件的 `attempt_archive/<timestamp>/`，
  保留 `recovery_map.json`，然后从该条件的第一个场景重新运行。
  这是**条件级续跑**，不是保留某个未完成条件中的部分场景。
  原文件监视器会重读旧 frames，因此不直接沿用旧场景级 resume。
- refit：每 25 steps 保存模型、优化器和 RNG，支持精确继续。
- 输入或代码哈希变化：拒绝复用合同。不能自动开一个新 run 重置预算；
  应先复核修复内容、已用预算与继承方案。

归档不删除实验输出；检查旧归档时用 recovery map 解析 payload 中原来的路径。
最终审计只接受当前 attempt 的成功报告，不能拿旧 attempt 的成功记录补齐覆盖。

## 4. 实现和审计边界

`sim_test.py` 只有 opt-in 部署 hook。完整冻结 V3 每个状态前向一次；
小 selector bank 使用同一批可见候选/场景特征重新评分，不运行第二个生成器。
场景 ID 和 fold 只用于路由，不作为网络输入；reward/Q/违规信息不进入 selector。
prefix/目标的原始记录及 OOF 预测只进入独立断言，不被用于替代模型决策。

SimEngine 新 manager 只验证已经收到的动作，不计算候选 reward，也不替换动作。
每条记录保留 sidecar/plan SHA、模型身份、评分模式、路由、实际/预期全局轨迹，
同时核对 prefix、目标动作和所有场景/决策/最终指标覆盖。

执行窗口仍为 decision 4–11，即 state 3–10 开始的八次动作。
历史规划器通常还会在最后观测上发布 decision 12，但该动作不被模拟器执行。
新实现保留这个生命周期，单独审计 `terminal_unexecuted_publications`，
不能把它加入驾驶动作或指标的分母。

## 5. 输出位置和阅读顺序

所有新正式输出在：

```text
experiments/diffusiondrive/selector_cfpi_deployment_v1/runs/<run_id>/
  run_contract.json / decision_ledger.json
  deployment_inputs.json / natural128_membership.json
  baseline_tensor_audit.json / cached_routing_audit_cuda.json
  phase_a_report.json / phase_a_paired_scenes.csv
  refit/<policy>/report.json
  phase_b_report.json / phase_b_paired_scenes.csv
  collections/<phase_policy>/
    routing.json / deployment_collection.json
    collection_audit.json / continuous_metrics.csv
    logs/ / split_*/ / attempt_archive/
```

报告包含 PDM/SR/NC/DAC/EP/TTC/comfort/direction、挽回/破坏场景、三个 seed、
family 等分层、等日志权重的配对 bootstrap，以及 A 的 continuous-minus-one-shot。
B 同时列出相对 scalar 和 gate-conditioned 的结果。筛选通过不等于论文认证，
区间跨零也不自动等于方法无效。B 完成后必须研究复核，不能自动扩大 train256。

## 6. 已完成的软件核验与尚未完成的部分

- 27 项新增 CPU 测试通过：路由、接管时刻、场景切换、两种评分语义、输入隔离、
  原 Torch / SimEngine NumPy 插值一致性、真实 manager 方法的合成动作测试、
  完整审计/末尾未执行输出、缺失/篡改拒绝、可恢复归档、预算与旧进程识别、
  合成 20 参数优化器的中断/不中断权重逐位一致、ID 抽样和报告数值。
- 原有回归 35 项通过：34 项使用正式 AlgEngine 依赖路径；1 项场景限制测试
  在 SimEngine Python 中运行（AlgEngine 环境不含 simulator-only 的 psutil）。
- 实际新 config 解析/模型来源检查，以及实际 SimEngine manager 导入通过。
- 指定的 48 个 CV checkpoint 哈希及来源核验通过。
- 真实 pilot 缓存的 768 个新路由目标预测、64 个 scalar 预测在 CPU 上全部复现；
  scalar 重算最大误差 `4.482269287109375e-05`，小于独立重算容差 `1e-4`。
- 两个真实 baseline full checkpoint 的 963 个非 selector 张量逐一相等。
- 自然抽样验证了 128 场景、128 logs、64/64 family、排除日志无重叠；
  已在临时目录成功准备 A 的 13 个条件（含 sentinel）。

真实缓存和场景的反序列化仅在核对本地既有实验 manifest/SHA 后进行。
软件检查产物保存在临时目录，不冒充正式 A/B run。

尚未完成：H100 CUDA 路由 parity、真实 sentinel、A/B 连续闭环、真实全 pilot refit。
CPU 检查不替代这些验证；若 H100 sentinel/prefix/target 不一致，必须停下来诊断，
不能调宽容差、删场景或放弃强对照以取得通过结果。
