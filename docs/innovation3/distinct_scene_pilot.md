# 不同场景 H100 DDP 工程验收

本阶段将已有同场景 8-rank pilot 扩展到每 rank 一个不同原始日志的场景。它不是正式训练；不改变 reward 权重、H1 协议、冻结 generator/V3 或已有 H100 结果。

## 已准备

- `innovation3_runtime.py --scene-manifest`：仅用于 `ddp-online-throughput-probe`；启动前检查配置、场景源、精确资产、单场景 pickle 的 SHA256，以及 R / 8-step 协议。
- 每 rank 保存独立 `.rankN.settings.json`，显式指定 `visual_scene_id`；worker ready 回执必须匹配该场景。聚合再次校验场景、资产/源摘要和 seed namespace。
- 训练 seed、动作采样 seed、candidate seed、scene seed 分开记录。动作 seed 为 training seed + rank；candidate/scene seed 由清单明确指定。
- 校验初始 V3/residual 一致；每步校验 residual、optimizer、policy version 和 attempts 一致。只有全局无信号才跳过；记录 local signal 与 global update 的区别。
- 所有 rank 都完成时，即使某 rank 的学习证据不足，也写 `INCOMPLETE_DDP_ONLINE_LEARNING_EVIDENCE`，退出码 2；不会强制更新或冒报通过。

## 数据清单与范围

`registry/distinct_scene_pilot_20260923_v1/scene_manifest.json` 固定 8 个场景，2-rank 使用前两项、8-rank 使用全部项。每项来自不同的原始日志（去掉 clip 的 `_start_end` 后缀），不是只要求切片名不同。

已用 known C excluded logs、C/D source contract 的额外日志以及 navtest/rare288 图片清单中的日志做保守排除。此前按切片排除的 82 个候选缩为 27 个场景、20 个原始日志；另有 1 个场景不足 22 个记录帧，不能用于当前 13 warmup + 8 decisions 探针，故初步池为 26 场景、20 日志。

工程场景按预先固定的 SHA256 排序选择，未查看 reward/性能。`development_pool.json` 保存排除原因及 pilot 暴露日志。这仍不是完整历史认证排除审计，也不是 80/10/10 正式划分；`formal_ready=false`、`exclusion_audit_complete=false`。后续冻结时必须考虑本 pilot 已暴露的日志。

## H100 运行

需要 8 张空闲 H100。使用新 basename，禁止覆盖已有报告：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine-selector-v3-online/code
bash run_innovation3_distinct_scene_h100.sh all v1
```

脚本依次运行环境预检、2-rank、8-rank。只有 2-rank 报告状态、不同场景数量、manifest SHA256 和当前源文件 SHA256 全部符合，才运行 8-rank。任一失败或学习证据不足即停止；保留 `.rank*.json`、`.rank*.worker.log`、`.rank*.settings.json` 和一次性 checkpoint。

如果只先分配两张卡：

```bash
bash run_innovation3_distinct_scene_h100.sh 2 v1
```

2-rank 通过后，在同一代码/清单下使用八张卡继续：

```bash
bash run_innovation3_distinct_scene_h100.sh 8 v1
```

若任何输出已存在或修改了源代码，不要覆盖旧报告，改用 `v2` 等新 tag，从两卡阶段重新开始。

结果：`registry/ddp_distinct_h100_2rank_v1.json` 和 `registry/ddp_distinct_h100_8rank_v1.json`。预期 `PASS_DDP_REAL_ONLINE_THROUGHPUT_PILOT`、`distinct_scene_verified=true`、`unique_scene_count=2/8`、`formal_ready=false`、`used_for_formal_training=false`。

`INCOMPLETE_DDP_ONLINE_LEARNING_EVIDENCE` 表示至少一个 rank 未达到既定“更新改变 logits 且后续动作使用更新策略”门槛，不能按成功处理，也不能看结果后改 reward 权重或静默替换场景。应先分析记录的 local signal、global update、版本、场景/渲染失败信息，再预登记下一次工程实验。

## 仍待完成

本地 CPU 测试只能验证控制流、清单守卫和 Gloo 更新一致性，不代替 H100 的 NCCL、真实渲染、真实生成轨迹与反馈验收。不同场景 H100 验收通过后，继续完整排除审计、development H1/H4/其他条件与正式 manifest 冻结，再进入 seeds 0/1/2；当前没有启动正式训练。

## 本地验证记录

2026-09-23：114 项 CPU 回归通过（selector_v2_v3、innovation3、innovation3_visual；未重跑未修改的 headless dynamics 专用环境测试）。其中包含真实双进程 Gloo mixed-signal 更新、全局无信号跳过、参数/optimizer/version parity、错场景回执拒绝、场景清单篡改与同原始日志泄漏拒绝，以及原视觉输入管线回归。测试日志和本阶段源码归档在 `registry/distinct_scene_pilot_20260923_v1/`。H100 运行尚未执行。
