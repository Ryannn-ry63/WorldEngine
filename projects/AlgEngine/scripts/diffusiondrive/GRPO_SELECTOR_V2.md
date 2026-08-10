# DiffusionDrive GRPO Selector V2

## 这版只改变什么

V2 只替换 selector 的 policy loss。候选生成、DiT、轨迹 regression、感知、
在线 NAVSIM PDM reward、冻结 reference selector 以及部署时 `argmax` 都和 V1
相同；仍然不使用 IL、GT-nearest、8192 轨迹词表或 generator loss。

V1 对冻结 reference 使用 PPO `0.8--1.2` clip。reference 分布很尖时，较好
候选即使已经生成，概率也很难跨过当前 top-1。V2 利用“20 条候选就是完整
action set”这一事实，直接计算完整期望：

```text
A_i = (reward_i - group_mean) / group_std
pi_i = softmax(selector_logit_i / temperature)
L_policy = - mean_group sum_i pi_i * A_i
L_total = L_policy + beta * KL(pi || pi_ref)
```

它等价于不做固定 clip 的精确 importance expectation：

```text
sum_i pi_old_i * (pi_i / pi_old_i) * A_i = sum_i pi_i * A_i
```

这不是额外监督 loss，也没有把 oracle index 当标签；reward 和 V1 一样来自同一
forward 中 20 条轨迹的 PDM score。

## 实现位置

- 正式 head：`mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py`
- V2 config：`configs/diffusiondrive/e2e_diffusiondrive_grpo_selector_v2.py`
- 单个缓存 trial：`scripts/diffusiondrive/train_grpo_selector_v2_cached.py`
- scene-held-out gate：`scripts/diffusiondrive/select_grpo_selector_v2.py`
- 8xH100 bundle：`scripts/diffusiondrive/run_grpo_selector_v2_sweep_h100.sh`

V1 仍是默认 `clipped_reference_grpo`，因此旧 checkpoint/config 可复现。只有 V2
config 会显式选择 `exact_group_grpo`。

## 首轮实验

复用已验证的 frozen-candidate cache，不重新运行感知、DiT 或 PDM 提取：

- train：6339 个 navtrain token；
- calibration：scene-disjoint 1118 个 token，3 个 paired diffusion noise seed；
- temperature：`1, 4, 8`；
- learning rate：`1e-4, 3e-4, 1e-3`；
- KL beta：`0, 1e-4, 1e-3`；
- 200 epochs；保存 `1,2,4,8,16,32,64,128,200`。

H100 命令（一个任务占 8 张卡，内部最多同时运行 8 个轻量 trial）：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v2_sweep_h100.sh
```

输出：

```text
experiments/diffusiondrive/grpo_selector_v2/selection/selection.json
experiments/diffusiondrive/grpo_selector_v2/selection/selected_checkpoint.pth
```

## Calibration gate

只有同时满足以下条件才物化可进入正式 navtest 的 checkpoint：

- mean top-1 PDM gain `>= +0.003`；
- scene-bootstrap 95% CI 下界 `> 0`；
- 至少 2/3 noise seed 为正；
- 最差 seed `>= -0.001`；
- selector 与 reference 的 argmax disagreement `> 2%`。

若不通过，任务仍技术成功，但 `selection.json` 状态为
`CALIBRATION_GATE_FAIL`，不会伪造或启动正式闭环结果。

## Calibration result and rollback anchor

2026-08-10 的 scene-disjoint calibration gate 已 PASS：

- selected：`T=1, lr=1e-3, KL=1e-3, epoch=32, train seed=0`；
- mean top-1 PDM gain：`+0.010607`；
- scene-bootstrap 95% CI：`[+0.004397, +0.017078]`；
- noise seed 0/1/2：`+0.012574 / +0.008087 / +0.011161`；
- selection disagreement：`30.92%`；
- selected checkpoint SHA256：`1717fadcb52b33a3e4055cd4cf5256aa8e9f743df478bd5f57daf19febf17181`。

该 selection checkpoint 和 `selection.json` 是 V2 的固定回退点。

## 后续决策

Calibration gate 已通过后，固定 `T=1, lr=1e-3, KL=1e-3, epoch=32`，不再用
navtest 选择超参。seed 0 使用 selection checkpoint；seed 1/2 从同一 baseline、
同一 train cache 独立打乱训练，并固定取 epoch 32。

## Formal three-seed four-block evaluation

三个任务可同时提交，每个任务固定使用 8 张 H100：

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v2_formal_seed_h100.sh 0
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v2_formal_seed_h100.sh 1
./projects/AlgEngine/scripts/diffusiondrive/run_grpo_selector_v2_formal_seed_h100.sh 2
```

每个 seed 一次跑完 OpenLoop-navtest、OpenLoop-navtest_failures、ClosedLoop-NR、
ClosedLoop-R。已有 V1 reference 只在 baseline SHA、eval seed 与四块计数一致时复用；
V2 current 始终重新运行。seed 1/2 在同一任务中先完成短 replica 训练和只改 10 个
selector tensor 的 checkpoint 审计。

三个任务都 PASS 后执行 CPU 汇总：

```bash
python projects/AlgEngine/scripts/diffusiondrive/finalize_grpo_selector_v2_formal.py \
  --v2-root experiments/diffusiondrive/grpo_selector_v2 \
  --reference-root experiments/diffusiondrive/grpo_selector_formal \
  --output experiments/diffusiondrive/grpo_selector_v2/formal/final/formal_three_seed_result.json
```

Finalizer 的 `PASS` 只表示 checkpoint、数据计数和 paired 结果完整，不表示性能显著。
无论 V2 的正式增益大小，selection checkpoint 与正式结果都作为 immutable anchor；
后续 V2.1 必须从新分支和新 artifact 目录开始，不能覆盖 V2。
