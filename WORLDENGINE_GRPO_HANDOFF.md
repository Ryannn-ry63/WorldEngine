# WorldEngine GRPO 迁移交接摘要

日期：2026-07-16
分支：`diffusiondrive-pr`

## 已完成

本次将 DiffusionDrive 中已经验证的 generation-only GRPO 方案迁入正式 WorldEngine，同时保留原有 `DiffusionPlanningHead` 和 baseline 配置。

完成内容：

- 新增独立 `DiffusionGRPOPlanningHead` 和 `e2e_diffusiondrive_grpo.py`。
- 接入 stochastic DDIM trace、exact action replay、group-relative advantage、clipped GRPO objective 和 reference KL。
- 固定 reference decoder 从 checkpoint 严格加载，后续 policy checkpoint 不能覆盖它。
- 冻结 perception、BEV、query 等模块，只训练 `diff_decoder.*`。
- 将 20 个候选轨迹插值到 PDM 所需的 40 点轨迹。
- SimEngine 支持同一状态下对任意候选集动态计算 PDM score。
- rollout record 保存完整 diffusion trace、候选轨迹、selected index 和 policy version。
- 新增版本化 reward sidecar、manifest builder 和 GRPO rollout dataset。
- 新增单轮编排脚本：`rollout → reward → manifest → GRPO update`。
- 默认推荐 8 workers，每 worker 2 个成功场景，单轮最多 128 次更新；smoke 使用 1 个场景和最多 8 次更新。

## 主要文件

新增：

- `projects/AlgEngine/configs/diffusiondrive/e2e_diffusiondrive_grpo.py`
- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_utils.py`
- `projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_planning_head.py`
- `projects/AlgEngine/mmdet3d_plugin/datasets/navsim_openscene_grpo_rollout.py`
- `projects/AlgEngine/scripts/build_grpo_rollout_manifest.py`
- `projects/AlgEngine/tests/test_diffusion_grpo_utils.py`
- `projects/SimEngine/scripts/run_worldengine_grpo_round.sh`

相关既有文件已完成注册和接口接线，完整文件清单见 [WORLDENGINE_GRPO_IMPLEMENTATION_RESULTS_20260716.md](WORLDENGINE_GRPO_IMPLEMENTATION_RESULTS_20260716.md)。

## 验证结果

- GRPO 数学单元测试：`5 passed`。
- Python compile 和 shell syntax 检查通过。
- 合成 rollout/reward/manifest 闭环通过。
- 合成 GRPO dataset 读取通过，候选 final action 形状为 `(20, 8, 3)`。
- 使用真实 `epoch_8.pth` 构建 head 通过。
- 可训练参数共 84 个 parameter tensors，全部属于 `diff_decoder.*`。
- schedule 为 `(8, 0)`，scheduler inference steps 为 `125`。
- 固定 reference state_dict 覆盖保护通过。

## DiffusionDrive 已有 paired navtest 依据

这不是新的 WorldEngine 分数，只是迁移依据：

- Base PDMS：`0.8491642572`
- D-128 PDMS：`0.8498702853`
- paired 差值：`+0.0007060281`
- bootstrap 95% CI：`[+0.0000438298, +0.0014030718]`
- 完整评测：12,146/12,146 token 成功。

完整历史结果和 CSV/JSON 路径见 [GRPO_FULL_NAVTEST_RESULTS_20260715.md](/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive/GRPO_FULL_NAVTEST_RESULTS_20260715.md)。

## 下一步

先运行 8-worker、每 worker 1 个场景、最多 8 updates 的 smoke：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
bash projects/SimEngine/scripts/run_worldengine_grpo_round.sh \
  /absolute/path/to/current_policy.pth \
  /absolute/path/to/fixed_reference.pth \
  DATA_TYPE round0 ASSET_NAME 8 1
```

确认以下项目正常后，再运行 `128 2` 正式轮次：

- reward sidecar 数量和 state key 一一对应；
- policy version、selected index 和候选数校验通过；
- GRPO loss、KL、ratio、clip fraction 为有限值；
- 只有 `diff_decoder.*` 有梯度；
- 输出 checkpoint 能作为下一轮 policy，而 fixed reference 保持不变。

当前尚未执行真实 WorldEngine rollout、GPU smoke、128-update 训练和 paired WE evaluation，因此还没有新的 WorldEngine PDMS 结论。
