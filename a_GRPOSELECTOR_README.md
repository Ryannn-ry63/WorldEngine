# DiffusionDrive Selector-GRPO 代码阅读指南

本文只解释当前 WorldEngine 版本中 DiffusionGRPOOnlineSelectorPlanningHead 的 GRPO 路径，重点说明它如何复用原始 DiffusionDrive、如何得到 reward，以及如何形成 GRPO loss。

## 1. 一次训练 batch 的调用链
源码： [NavFormer.forward_train()](projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py#L662)；
[GRPO planning head](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L93)。

    NavFormer.forward_train()
        |
        |-- planning_head.forward(..., sample_tokens=...)
        |
        |-- DiffusionGRPOOnlineSelectorPlanningHead.forward()
        |       |
        |       +-- super().forward(...)
        |               |
        |               +-- self._forward_train(...)
        |
        +-- planning_head.loss(plan_results, ...)
                |
                |-- _extract_candidate_rewards()
                |-- _group_relative_advantages()
                |-- PPO/GRPO clipped surrogate
                +-- KL/reference diagnostics

GRPO head 调用原始 DiffusionPlanningHead.forward。原始 forward 根据 self.training 调用 self._forward_train；由于 Python 动态分派，实际执行的是 GRPO 子类自己的 _forward_train。

因此 GRPO 复用了原始 DiffusionDrive 的输入处理、BEV、query 和 decoder，但替换了训练路径和 loss。

## 2. _forward_train 的四步
源码：[_forward_train()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L435)。

核心函数大约从第 435 行开始：

    def _forward_train(self, bev_feature, ego_query, agents_query, status_token):
        candidates_8, candidate_feature = self._generate_frozen_candidates(
            bev_feature, ego_query, agents_query, status_token
        )
        current_logits, reference_logits = self._selector_outputs(
            candidate_feature
        )
        reward_payload = self.online_reward(
            candidates_8, self._active_sample_tokens
        )
        return self._build_result(
            candidates_8,
            current_logits,
            reference_logits,
            reward_payload=reward_payload,
        )

它做四件事：

1. 冻结的 DiffusionDrive 生成候选轨迹；
2. current selector 和 reference selector 对候选打分；
3. 对同一批候选计算真实 PDM reward；
4. 把轨迹、logits、reward 打包给 loss。

这里 GRPO 不负责生成轨迹。轨迹由原始 DiffusionDrive 生成，selector 只决定候选集合中选哪一条。

## 3. _generate_frozen_candidates：生成候选
源码：[_generate_frozen_candidates()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L332)。

函数有 torch.no_grad 装饰，因此候选生成过程不建立反向传播图。

    plan_anchor
        ↓ normalize
    初始 diffusion 状态
        ↓ 加噪
    DDIM timestep 反向循环
        ↓
    diff_decoder(..., return_traj_features=True)
        ↓
    最终候选轨迹和候选 feature

设 batch size 为 B，候选数为 K=20，规划长度为 T=8：

    candidates_8:      [B, 20, 8, 3]
    candidate_feature: [B, 20, D]

核心步骤是：

    plan_anchor = self.plan_anchor.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    image = self.norm_odo_xy(plan_anchor)
    noise = self._candidate_noise(image)
    image = self.diffusion_scheduler.add_noise(image, noise, truncation)

    for timestep in roll_timesteps:
        noisy_traj_points = self.denorm_odo_xy(
            torch.clamp(image, min=-1, max=1)
        )
        traj_feature = self.plan_anchor_encoder(...)
        time_embed = self.time_mlp(timesteps)
        poses_reg_list, _, feature_list = self.diff_decoder(
            ..., return_traj_features=True
        )
        final_reg = poses_reg_list[-1]
        final_feature = feature_list[self.selector_layer]
        image = self.diffusion_scheduler.step(...).prev_sample

    return final_reg.detach(), final_feature.detach()

diff_decoder 仍然是原始 decoder。每层大致是：

    BEV cross-attention
    → agent cross-attention
    → ego cross-attention
    → FFN 和 time modulation
    → task_decoder
          ├── plan_reg_branch
          └── plan_cls_branch

detach 表示 GRPO loss 不会更新 generator。

## 4. candidate_feature 是什么，为什么保留
源码：候选 feature 在[_generate_frozen_candidates()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L371)生成，并在[最终 feature 提取](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L383)保留。

candidate_feature 不是轨迹坐标，也不是 PDM reward。它是每条候选轨迹经过场景信息交互后的隐藏表示：

    候选轨迹初始表示
    + BEV 场景信息
    + 周围 agent 信息
    + ego 状态
    + diffusion timestep
    → candidate_feature

原始 decoder 内部本来会做：

    poses_reg, poses_cls = self.task_decoder(traj_feature)

traj_feature 会同时进入 plan_cls_branch 和 plan_reg_branch。

GRPO 保存 feature 的两个主要原因：

1. current selector 和 reference selector 必须使用完全相同的输入；
2. 不需要重新生成轨迹，也不需要另设计一个基于坐标的 selector。

之后对同一个 feature 调用：

    current_logits = self._current_selector()(candidate_feature)
    reference_logits = self.reference_selector(candidate_feature)

因此两个策略只有参数不同，输入表示完全相同。

## 5. 从 _selector_outputs() 到 online reward：一次完整调用链

本节对应源码：

- [_selector_outputs()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L395)
- [OnlineDiffusionDrivePDMReward.forward()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py#L275)

可以把这一段理解成：

    生成 20 条实际轨迹 → 两个 selector 打分 → PDM simulator 评价 → loss 合并分数和 reward

### 5.1 从 NavFormer 到 GRPO _forward_train

在 [NavFormer.forward_train()](projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py#L662) 中，GRPO head 声明 requires_online_candidate_rewards=True，因此 NavFormer 会把每个样本的 sample token 传给 planning head：

    planning_kwargs["sample_tokens"] = [meta[self.queue_length - 1]["sample_idx"] for meta in img_metas]

随后调用 planning_head.forward()。GRPO head 暂存 token 后调用原始 DiffusionPlanningHead.forward()：

    self._active_sample_tokens = sample_tokens
    return super().forward(bev_embed.detach(), ...)

原始 forward 完成 BEV、status token 和 query 准备后执行：

    if self.training:
        return self._forward_train(...)

由于 Python 动态分派，这里的 _forward_train() 实际是 GRPO 子类版本：[GRPO _forward_train()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L435)。

### 5.2 _forward_train() 先得到候选和 feature

GRPO 训练函数依次执行：

    candidates_8, candidate_feature = self._generate_frozen_candidates(...)
    current_logits, reference_logits = self._selector_outputs(candidate_feature)
    reward_payload = self.online_reward(candidates_8, self._active_sample_tokens)

对于 batch size 为 B：

    candidates_8      : [B, 20, 8, 3]
    candidate_feature : [B, 20, D]

candidate_feature 来自最后一次 decoder 调用：

    poses_reg_list, _, feature_list = self.diff_decoder(..., return_traj_features=True)
    final_reg = poses_reg_list[-1]
    final_feature = feature_list[self.selector_layer]

默认 selector_layer=-1，所以使用最后一个 decoder layer 的 feature。

### 5.3 _current_selector() 实际指向什么

_current_selector() 的实现是：

    return self.diff_decoder.layers[self.selector_layer].task_decoder.plan_cls_branch

所以 selector 不是额外的大模型，而是复用 DiffusionDrive decoder 中的 plan_cls_branch。

[plan_cls_branch 定义](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py#L273)

    self.plan_cls_branch = nn.Sequential(..., nn.Linear(embed_dims, 1))

如果 D=256：

    candidate_feature [B, 20, 256] → plan_cls_branch → [B, 20, 1] → squeeze(-1) → current_logits [B, 20]

第 i 个 logit 对应 candidate_i；这些值还不是概率，只是未归一化分数。

### 5.4 reference selector 从哪里来

初始化时，GRPO head 会复制一份当前 selector：

    self.reference_selector = copy.deepcopy(self._current_selector())

然后从官方 DiffusionDrive checkpoint 加载对应的 plan_cls_branch 参数：[初始化](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L138)，[checkpoint 加载](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L196)。

随后：

    current_logits = self._current_selector()(candidate_feature).squeeze(-1)
    with torch.no_grad():
        reference_logits = self.reference_selector(candidate_feature).squeeze(-1)

两者输入完全相同，区别只有参数：current 可训练，reference 冻结。

### 5.5 online_reward() 如何得到 20 个 reward

调用：

    reward_payload = self.online_reward(candidates_8, self._active_sample_tokens)

[OnlineDiffusionDrivePDMReward.forward()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py#L275) 先检查输入为 [B, 20, 8, 3]，然后每个样本调用 [_score_one()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusiondrive_online_pdm_reward.py#L253)：

    metric_cache = self._get_metric_cache(token)
    states = self._trajectory_states(candidates, metric_cache)
    simulated_states = self._simulate(...)
    self._scorer.score_proposals(...)
    return pairwise_official_scores(self._scorer)

实际流程：20 条候选 → 40-step 状态 → simulator → PDMScorer → 20 个 reward。

最终返回：

    score       : [B, 20]
    components  : [B, 20, 6]
    valid_mask  : [B, 20]

score 是候选质量，不依赖 selector 当前选中哪条；20 条候选都会被评估。

### 5.6 logits、概率和 reward 是三种不同的东西

    candidate_feature [B, 20, D]
        ↓ plan_cls_branch
    logits [B, 20]
        ↓ loss() 中的 log_softmax
    log_probability [B, 20]
        ↓ exp()
    probability [B, 20]

另一条独立路径是：

    candidates_8 [B, 20, 8, 3] → PDM simulator → candidate_rewards [B, 20]

因此 logits 是 selector 打分，probability 是选择分布，reward 是 PDM 驾驶质量评价。

### 5.7 为什么当前路径不使用 8192，也不使用 GT-nearest reward

GRPO loss 优先读取 result["candidate_rewards"]，并在 [_extract_candidate_rewards()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L526) 中要求：

    candidate_rewards.shape == selector_logits.shape == [B, 20]

旧的 8192 词表 reward 是 [B, 8192]，会被直接拒绝。NavFormer 仍传 gt_pdm_score 只是为了兼容框架；正式 online 路径已经把 candidate_rewards 放进 result，因此不会使用旧 GT reward。

原始 DiffusionDrive 是随机加噪后，用 poses_reg_list / poses_cls_list 和 GT 比较，形成 GT-nearest 分类与回归损失：[原始 _forward_train()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py#L775)

当前 GRPO 则是：

    固定 generator 生成 20 条候选
        ↓ selector 打分
        ↓ PDM 评价
        ↓ reward 形成 advantage
        ↓ 只更新 selector

## 6. result 如何进入 loss
源码：[_build_result()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L402)；[NavFormer 中的 planning_head.loss 调用](projects/AlgEngine/mmdet3d_plugin/navformer/detectors/navformer.py#L752)。

_build_result 先用 current selector 的 argmax：

    selected_indices = current_logits.argmax(dim=-1)
    selected_8 = torch.gather(candidates_8, ...)

它还保存 selector_logits、reference_selector_logits、candidate_trajectories_8、candidate_rewards 和 reward components。NavFormer 随后调用 planning_head.loss(plan_results, ...)。

当前 GRPO 的 loss 保留 sdc_planning 等参数只是为了兼容框架，正式 selector loss 不使用 GT 轨迹做 imitation。

## 7. loss 如何形成 GRPO
源码：[loss()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L589)；[advantage 计算](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L562)。

### 7.1 advantage
源码：[_extract_candidate_rewards()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L526)；[_group_relative_advantages()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L562)。

    rewards, valid = self._extract_candidate_rewards(...)
    advantages, policy_mask, active_group, reward_mean, reward_std = self._group_relative_advantages(rewards, valid)

对于一个样本的 20 个 reward，计算：

    A_i = (r_i - mean(r)) / std(r)

高于组内平均 reward 的候选 advantage 为正，低于平均值的候选 advantage 为负。没有有效差异的 group 不参与 policy 更新。

### 7.2 logits、log probability 和 probability
源码：概率归一化和 log probability 计算位于[loss()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L589)内部。

    current_log_prob = F.log_softmax(current_masked, dim=-1)
    reference_log_prob = F.log_softmax(reference_masked, dim=-1)

数学上：

    p_i = softmax(logit)_i
    log_prob_i = log(p_i)

例如 p_i=0.2 时，log_prob_i=log(0.2)≈-1.609。因此 log_prob 是概率的对数，不是概率本身；log_prob.exp 才恢复成概率分布。

### 7.3 PPO ratio 和 clipping
源码：ratio、clipping 和 KL 项位于[loss()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L589)内部。

    log_ratio = current_log_prob - reference_log_prob
    ratio = torch.exp(log_ratio)

所以：

    ratio_i = pi_current(i | s) / pi_reference(i | s)

当前实现令 pi_old=pi_ref，没有单独维护另一个 old network。

    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(
        ratio * advantages,
        clipped_ratio * advantages
    )

默认 clip_epsilon=0.2，限制 current selector 相对 reference selector 的更新幅度。

## 8. 为什么 20 条候选是完整 action set
源码：候选数量和完整 action set 在[_generate_frozen_candidates()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L332)与[loss()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_grpo_online_planning_head.py#L589)中使用。

把 selector 的 action 定义为“选择第几个候选”：

    状态 s：当前场景
    动作 0：candidate_0
    动作 1：candidate_1
    ...
    动作 19：candidate_19

因此当前场景的 action set 是：

    A(s) = {candidate_0, candidate_1, ..., candidate_19}

这 20 条候选已经全部由 diffusion generator 生成，而且每条 reward 都已计算。因此这里不是 selector 随机采样 20 个 action，而是枚举当前场景下的 20 个可选 action。

普通 sampled GRPO 可能只看到一个或少数 sampled action；当前实现可以直接对 20 个候选的 reward 全部计算 advantage 和 surrogate。

代码用 reference policy 的概率做权重：

    policy_weight = reference_probability * policy_mask
    policy_weight = policy_weight / policy_weight.sum(dim=-1, keepdim=True)
    policy_objective_per_group = (policy_weight * surrogate).sum(dim=-1)

表示对完整的 20-action 集合求期望。这里不是简单平均，而是使用 reference policy 的概率作为权重。

当前训练更接近单帧 contextual bandit：

    state：当前 navtrain 场景
    action：20 条候选轨迹中的一个索引
    reward：该候选轨迹的 PDM score
    policy：selector 对 20 个索引的概率分布

## 9. 与原始 DiffusionDrive 的差异
对照源码：[原始 DiffusionPlanningHead](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py#L496)，[原始 forward()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py#L709)，[原始 _forward_train()](projects/AlgEngine/mmdet3d_plugin/navformer/dense_heads/diffusion_planning_head.py#L775)。

原始路径：

    anchor 加噪
    → decoder
    → GT-nearest 分类 loss + GT 回归 loss
    → 同时更新 generator 和 selector

当前 GRPO 路径：

    冻结 generator
    → 按 inference DDIM schedule 生成 20 条候选
    → 保存候选 feature
    → current/reference selector 打分
    → 逐条计算 PDM reward
    → group-relative advantage
    → PPO ratio + clipping + KL
    → 只更新最后 selector

一句话总结：

    原始 DiffusionDrive：学习生成接近 GT 的轨迹
    当前 selector-GRPO：学习从已经生成的轨迹中选择 PDM 更高的一条

后续修改时，可以按四个边界定位：

    _generate_frozen_candidates() 负责提出候选
    _selector_outputs()             负责给候选打分
    online_reward()                 负责评价候选
    loss()                          负责根据评价更新 selector

只要先确定修改属于哪一个环节，就不容易把 selector-only 实验误改成 generator GRPO。
