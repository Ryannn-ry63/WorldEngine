# WorldEngine Generation-GRPO 迁移与验证结果

日期：2026-07-16
目标仓库：`/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine`
分支：`diffusiondrive-pr`
起始 HEAD：`63ed6ce`

## 1. 当前结论

本次已经把 DiffusionDrive 中验证过的 generation-only GRPO 设计迁入正式 WorldEngine，并保留原有 `DiffusionPlanningHead` 和 `e2e_diffusiondrive.py` 不变。新增实现采用独立 head、独立 config、独立 rollout dataset 和独立单轮编排脚本。

当前完成度是：

- 代码迁移、接口接线和配置构建已完成。
- 精确 diffusion action replay、clipped GRPO、固定 reference KL、8→40 轨迹插值已通过单元测试。
- rollout record、同状态动态 PDM 候选评分、版本化 reward/manifest 数据闭环已接通。
- 使用真实 `epoch_8.pth` 的 head 构建和 reference 严格加载已通过。
- 合成 manifest 和 dataset smoke 已通过。
- 尚未实际启动 8 worker WorldEngine rollout、8-update GPU smoke、128-update 训练或 paired WE evaluation，因此本文不报告新的 WE PDMS。

也就是说，现在可以进入运行阶段，但不能把静态与合成测试描述成已经获得 WorldEngine 性能提升。

## 2. 最终采用的设计

数据流如下：

`当前 policy rollout → 保存精确随机 diffusion trace → 输出全部 20 条 40 点候选 → SimEngine 在同一状态动态计算 PDM reward → 按 policy version 生成 manifest → 离线重放同一 action → generation-only GRPO update`

关键约束：

- 不加入 imitation loss；训练目标只有 generation policy loss 和 reference KL。
- 感知、BEV、query 和其他非生成模块冻结，只训练 `diff_decoder.*`。
- reference decoder 从固定 checkpoint 严格加载并永久冻结。
- 后续 policy checkpoint 中即使包含 `reference_decoder.*`，加载时也会被忽略，避免跨轮覆盖固定 reference。
- 候选选择仍使用冻结 reference classifier，不让 reward 训练直接漂移分类选择器。
- rollout 和训练按 policy version 隔离；record、reward 和 manifest 版本不一致时直接报错。
- 每个训练清单只允许一个 policy version。
- 每轮训练更新数为 `min(manifest.num_entries, GRPO_MAX_UPDATES)`，默认上限 128，避免小清单循环重复。
- 推荐正式轮次为 8 workers × 每 worker 2 个成功场景，再做最多 128 次更新。
- 推荐 smoke 为 8 workers × 每 worker 1 个成功场景，最多 8 次更新。

固定生成参数：

| 参数 | 值 |
|---|---:|
| diffusion roll timesteps | `(8, 0)` |
| scheduler inference steps | `125` |
| DDIM eta | `1.0` |
| final action std | `0.05` |
| PPO/GRPO clip ratio | `0.2` |
| reference KL weight | `0.1` |
| optimizer | AdamW |
| learning rate | `1e-6` |
| batch size | 1 |
| default max updates | 128 |

## 3. 新增文件

- `projects/AlgEngine/configs/navformer/e2e_diffusiondrive_grpo.py`
  - 独立 GRPO 配置。
  - `GRPO_REFERENCE_CKPT` 与 `GRPO_POLICY_CKPT` 分离。
  - 自动按 manifest 样本数限制单轮更新数。

- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_utils.py`
  - Gaussian log-prob、equal-std KL、随机 DDIM transition/replay。
  - group-relative advantage 和 clipped generation objective。
  - 带 heading wrap 的 8 点 2 Hz 到 40 点 10 Hz 插值。

- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_planning_head.py`
  - 新的 `DiffusionGRPOPlanningHead`，继承原 `DiffusionPlanningHead`。
  - eval 时收集精确 stochastic trace。
  - train 时重放存储 action，不重新采样。
  - 固定 reference decoder 严格加载和 state_dict 覆盖保护。

- `projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_grpo_rollout.py`
  - 读取单版本 rollout manifest。
  - 将 trace、reward、valid mask 和 selected index 注入训练 batch。
  - 不读取固定 8192 PDM cache。

- `projects/AlgEngine/scripts/build_grpo_rollout_manifest.py`
  - 合并 8 worker rollout record、`grpo_rewards/` sidecar 和 OpenScene metadata。
  - 校验 policy version、state key、selected index、候选数和 trace 形状。
  - 把 sensor 路径改为绝对路径并生成 consolidated annotation。

- `projects/AlgEngine/tests/test_diffusion_grpo_utils.py`
  - generation-GRPO 数学与插值回归测试。

- `projects/SimEngine/scripts/run_worldengine_grpo_round.sh`
  - 单轮 `rollout → manifest → GRPO update` 编排入口。

## 4. 修改的既有文件

- `projects/AlgEngine/closed_loop/sim_test.py`
  - 原子写入全部候选 sidecar。
  - rollout record 增加四个 diffusion trace 字段和 policy version。

- `projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py`
  - 训练路径接收 GRPO tensors。
  - rollout 路径导出全部候选和精确 trace。

- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/__init__.py`
  - 注册新 head。

- `projects/AlgEngine/mmdet3d_plugin/datasets/__init__.py`
  - 注册新 dataset。

- `projects/SimEngine/worldengine/manager/dense_reward_manager.py`
  - 新增任意 `[N, 40, 3]` 局部轨迹的动态 PDM 评分。
  - GRPO candidate-only 模式跳过原固定 8192 候选保存流程。

- `projects/SimEngine/worldengine/components/agents/client/navformer_client.py`
  - 等待当前状态候选，调用 DenseRewardManager 评分，再原子写 reward sidecar。
  - 校验 policy version，保存 selected index 和 state key。

- `projects/SimEngine/scripts/run_ray_distributed_rollout.sh`
  - 传入 GRPO 候选/reward 路径、policy version 和每 worker 场景上限。

- `projects/SimEngine/worldengine/configs/default_runner.yaml`
  - 增加默认关闭的 GRPO 配置项，不改变原 baseline 默认行为。

- `projects/SimEngine/worldengine/envs/base_env.py`
  - 支持按成功场景数提前结束 worker；默认 0 表示保持原行为。

## 5. 本次验证结果

| 验证 | 结果 |
|---|---|
| Python compile：所有新增文件及被修改核心文件 | 通过 |
| Shell syntax：两个 rollout/round 脚本 | 通过 |
| GRPO 数学 pytest | `5 passed in 1.45s` |
| 合成 rollout→reward→manifest | 通过，`MANIFEST_SMOKE_OK` |
| 合成 GRPO dataset 初始化与 trace/reward 读取 | 通过，1 个样本，final action `(20, 8, 3)` |
| 真实 `epoch_8.pth` 构建新 head | 通过 |
| 可训练参数审计 | 84 个 parameter tensors，全部属于 `diff_decoder.*` |
| reference decoder requires_grad 审计 | 全部为 false |
| schedule 审计 | `(8, 0)`，scheduler steps `125` |
| manifest 单样本时更新上限 | `128 → 1`，通过 |
| policy state_dict 覆盖 reference 防护 | 通过，`FIXED_REFERENCE_LOAD_GUARD_OK` |

真实构建使用的 checkpoint：

`/inspire/hdd/global_user/wangcaojun-240208020180/nry/Worldengine-Diffusion/experiments/navformer/e2e_diffusiondrive/epoch_8.pth`

未执行项：

- 真实 SimEngine 动态 PDM 候选评分运行。
- 8 worker rollout smoke。
- 8-update GPU forward/backward 和 gradient audit。
- 128-update 训练。
- WorldEngine paired baseline/GRPO evaluation。

## 6. 迁移依据：DiffusionDrive paired full navtest

以下是 2026-07-15 在 DiffusionDrive 上完成的严格配对完整 navtest，不是本次 WorldEngine 新结果。

两边均为相同 12,146 个 token、相同 metric cache、相同 deterministic diffusion noise、`[8,0]` schedule 和 125 scheduler steps；均为 `12,146/12,146` 成功、0 失败。

| 指标 | Base | D-128 | 差值 |
|---|---:|---:|---:|
| PDMS / selected score | 0.8491642572 | 0.8498702853 | **+0.0007060281** |
| no-at-fault collisions | 0.9819693726 | 0.9822163675 | +0.0002469949 |
| drivable area compliance | 0.9331467150 | 0.9338876997 | +0.0007409847 |
| ego progress | 0.7858142537 | 0.7866524946 | +0.0008382410 |
| time to collision within bound | 0.9468137658 | 0.9467314342 | -0.0000823316 |
| comfort | 0.9995883418 | 0.9995883418 | 0.0000000000 |
| driving direction compliance | 0.9786349415 | 0.9787172732 | +0.0000823316 |

Paired 统计：

- mean difference：`+0.0007060281`
- median difference：`+0.0000002082`
- bootstrap：10,000 samples，seed `20260715`
- 95% CI：`[+0.0000438298, +0.0014030718]`
- wins / ties / losses：`6100 / 4655 / 1391`
- base score `< 0.5` 的 1,060 个 token：`+0.0132967`
- bottom decile 的 1,387 个 token：`+0.0107664`
- base score 为 0 的 994 个 token：D-128 平均 `0.0133384`

结论仍然是小幅正提升，且主要来自困难/失败场景；它低于 provisional `+0.005` engineering gate，不能描述成大幅提升。

Checkpoints：

- Base：[/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2026.04.14.03.49.58/lightning_logs/version_0/checkpoints/eval_model)
- D-128：[/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_stage2_D_gen_128/2026.07.15.05.10.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/grpo_stage2_D_gen_128/2026.07.15.05.10.31/lightning_logs/version_0/checkpoints/grpo-00-128.ckpt)

逐 token 和配对分析文件：

- [Base full navtest CSV](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_base_20260715/2026.07.15.09.04.23/2026.07.15.09.24.58.csv)
- [D-128 full navtest CSV](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_d128_20260715/2026.07.15.09.26.47/2026.07.15.09.47.18.csv)
- [summary.json](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_comparison_20260715/summary.json)
- [paired_per_token.csv](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_full_navtest_comparison_20260715/paired_per_token.csv)
- [base 2-token smoke CSV](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_navtest_smoke_base/2026.07.15.09.03.40/2026.07.15.09.03.52.csv)
- [D-128 2-token smoke CSV](/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/paired_navtest_smoke_d128/2026.07.15.09.14.26/2026.07.15.09.14.42.csv)

原始结论文档：

[GRPO_FULL_NAVTEST_RESULTS_20260715.md](/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/GRPO_FULL_NAVTEST_RESULTS_20260715.md)

## 7. 运行方式

单轮入口：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine

bash projects/SimEngine/scripts/run_worldengine_grpo_round.sh \
  /absolute/path/to/current_policy.pth \
  /absolute/path/to/fixed_reference.pth \
  DATA_TYPE \
  round0 \
  ASSET_NAME \
  8 \
  1
```

上面是 smoke：最多 8 次更新、每 worker 1 个成功场景。

正式单轮把最后两个参数改为 `128 2`。下一轮必须：

- 第一个 checkpoint 改为上一轮产出的 policy checkpoint。
- 第二个 fixed reference checkpoint 保持不变。
- policy version 改为新的唯一值，例如 `round1`。

输出位置：

- rollout：`experiments/closed_loop_exps/e2e_diffusiondrive_grpo_<version>/<DATA_TYPE>_NR/`
- manifest：`experiments/grpo_rounds/<version>/rollout_manifest.pkl`
- consolidated annotations：`experiments/grpo_rounds/<version>/rollout_manifest.annotations.pkl`
- training：`experiments/grpo_rounds/<version>/training/`

## 8. Git 与工作区说明

最终核对时：

- branch：`diffusiondrive-pr`
- HEAD：`63ed6ce`
- 相对 `origin/diffusiondrive-pr`：ahead 1、behind 1
- 本次未 pull、未切分支、未提交。
- 正式仓库在本次开始前已有大量未提交修改，包括 UniAD modules、sampler、encoder、train script、executor、`.gitignore` 和 `meeting_note.md`。
- 本次没有格式化、覆盖或清理这些既有修改。
- `git diff --check` 看到的 5 处 trailing whitespace 均位于既有 UniAD 修改中，不属于本次 GRPO 文件。

下一步应先跑真实 8-update smoke；只有确认动态 reward sidecar 数量、训练 loss、ratio、clip fraction、KL 和 `diff_decoder.*` 梯度正常后，再启动 128-update 轮次和 paired WE evaluation。
