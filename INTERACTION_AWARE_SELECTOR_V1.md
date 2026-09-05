# Interaction-Aware Residual Selection V1

## 论文主张

固定 DiffusionDrive 的感知、生成器、20 个候选和原始 selector，仅改进候选选择：生成器已经能覆盖较好的轨迹，但 V3 主要依赖候选 latent、沿轨迹 BEV 和隐式 BEV query，缺少“某一条候选会怎样与具体动态目标交互”的显式对应关系。V1 用冻结 tracker 的当前状态构造候选—目标关系，再学习对原始 selector logits 的零初始化 residual。

这不是 AutoVLA 的 fast/slow 迁移，也不训练工况分类器；AutoVLA 只说明“不同工况需要不同推理能力”是合理现象。本文归因保持为 selector representation bottleneck。

## 不变项

- 候选数固定为 20，候选生成、噪声视图、PDM reward 与原 V3 相同。
- epoch-100 generator、BEV encoder、tracker 与原 selector 全部冻结。
- tracker 只做 inference，不做 GT matching；不增加标注和数据种类。
- 每个目标只使用当前帧 `x,y,length,width,yaw,vx,vy,confidence,class`。
- 目标按 frozen track score 过滤（阈值 0.35），取 top-30，带显式 mask。
- 未来位置只用确定性 constant velocity，时间点为 0.5–4.0 秒。
- reward 和六个 PDM components 绝不作为 selector 输入。
- 训练保持 exact scalar full-action GRPO：LR `1e-4`、KL `1e-3`、batch `64`、16 epochs、每 cache 每 epoch 6339 examples、rare/common 50/50。

## 三个因果臂

- A0：重新训练的 matched V3 control。使用相同 schema-v4 cache、相同候选、样本顺序、seed、步数与 loss，但模型不读取 frozen tracks。
- A1：A0 输入 + 候选坐标系中的动态目标关系；共享编码器与无序 set attention 聚合目标。
- A2：A1 + directed candidate-relation set block。只有这一处区别，用来判断候选间关系推理是否提供额外收益。

所有 residual 输出层从严格的全零开始，因此 step 0 与 frozen epoch-100 selector 完全一致。

## 执行顺序与不可跳过的门

### Stage I：只使用 train split 的表征探针

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-rapg-v1
./run_diffusiondrive_selector_interaction_probe_8hopper.sh
```

脚本生成 train noise seeds 0/1/2 的 schema-v4 cache，然后按 log 做五折 pairwise PDM probe。只有同时满足以下条件才授权 A0/A1/A2：

- mean AUC gain ≥ 0.01；
- paired log bootstrap 95% CI lower > 0；
- 任一 fold 的下降不超过 0.01。

输出为 `experiments/diffusiondrive/interaction_selector_v1/probe/<RUN_ID>/probe_gate.json`。

若失败：停止本方法，不调数据比例、不加 verifier、不继续改 interaction loss；下一条且仅一条路线是使用同样 frozen outputs 的显式多帧 history representation probe。

### Stage II：固定 A0/A1/A2 development 实验

仅当 Stage I 输出 `AUTHORIZE_A0_A1_A2` 时运行：

```bash
./run_diffusiondrive_selector_interaction_development_8hopper.sh \
  experiments/diffusiondrive/interaction_selector_v1/probe/<RUN_ID>/probe_gate.json
```

脚本生成 development seeds 3/4/5 cache，在 GPU 0/1/2 并行训练三个臂，并自动审计每臂 `304272` examples、`4800` optimizer steps 和完全一致的 rare/common 数量。

候选臂必须同时满足：

- common/real-rare 等权 PDM 比 A0 高至少 0.005；
- common PDM 不比 A0 低 0.005；
- real-rare PDM 不低于 A0；
- common degraded fraction 不高于 A0；
- common 与 real-rare 的 pairwise ordering AUC 增量 scene-bootstrap lower 均 > 0；
- 两个 strata 的 noise-view agreement 均不比 A0 低 0.02。

若 A2 通过且等权 PDM 比 A1 再高至少 0.002，选择 A2；否则若 A1 通过，选择 A1；否则停止并保留 V3。输出为 `<development RUN_ID>/development_gate.json`。

若 train-only probe 通过但 A1/A2 都未通过：保留已经锁定的 interaction representation，只允许做一次 listwise/top-1 objective 对照；不再改表示、比例或加入 CPV 类旁路。

### Stage III：锁定 winner 后做 full-data 三训练 seed

```bash
./run_diffusiondrive_selector_interaction_formal_train_8hopper.sh \
  experiments/diffusiondrive/interaction_selector_v1/development/<RUN_ID>/development_gate.json
```

脚本重建 full rare-original 50/50 schema-v4 caches，使用锁定架构独立训练 seeds 0/1/2，并 materialize 三个可部署 checkpoint。输出 `formal_train_manifest.json`。

### Stage IV：三个正式评测任务

每个 seed 独占一个 8-GPU 任务；可以提交到三台机器，也可在一台机器串行：

```bash
./run_diffusiondrive_selector_interaction_formal_eval_8hopper.sh <FORMAL_TRAIN_MANIFEST> 0
./run_diffusiondrive_selector_interaction_formal_eval_8hopper.sh <FORMAL_TRAIN_MANIFEST> 1
./run_diffusiondrive_selector_interaction_formal_eval_8hopper.sh <FORMAL_TRAIN_MANIFEST> 2
```

每个任务复用既有四块正式评测：navtest open-loop、failure open-loop、non-reactive closed-loop、reactive closed-loop。直到 development gate 锁定 winner 前，不消费 certification 或正式测试结果。

## 结果解释边界

- Probe 失败：显式 current-state interaction 没有足够新增信息，优先验证多帧状态历史，而不是继续堆 selector。
- Probe 通过、GRPO 失败：信息存在但当前 group objective 没把它转成 top-1；只做一次锁定表示上的 listwise/top-1 对照。
- A1 通过、A2 无额外收益：论文使用更简洁的 A1，结论是显式 candidate-agent grounding 有效，而非更深关系建模有效。
- A2 以 ≥0.002 的 margin 通过：可把 candidate-agent grounding 与 candidate-set relational reasoning 都作为方法组成。
- 三 seed closed-loop 不稳定：不宣称整体突破；报告表征/离线收益，把 V3 保持为部署主结果。
- 三 seed closed-loop 稳定提升：主结果升级为 interaction-aware selector；之后才允许检查与 CQR 的兼容性，且 CQR 只能作为正交扩展，不回写本轮方法选择。
