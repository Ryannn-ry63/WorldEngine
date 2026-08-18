# DiffusionDrive Selector GRPO V3 Progress Fix V1 运行手册

日期：2026-08-18

## 目标与版本边界

本实验回到 rollout 之前的原始 V3，在原训练数据上重新生成修复后的 PDM reward cache，
重新完成 ablation、开发集调参、一次 certification 和 seed 0/1/2 正式评测。

- 原始 V3 基点：\`c63a23b\`
- 修复分支：\`diffusiondrive-selector-grpo-v3-progress-normalization-fix\`
- 修复实验目录：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1\`
- 原始 V3 目录 \`experiments/diffusiondrive/grpo_selector_v3\` 只读保留，用于最终对比。
- rare-log rollout 已保存到分支
  \`diffusiondrive-selector-grpo-v3-rare-log-v1\`，标签为
  \`diffusiondrive-selector-grpo-v3-rare-log-v1-pre-progress-fix-20260818\`。

修复前生成的 V3 reward cache 不可复用。新 manifest 记录 reward、planning head、
scene selector、NavFormer、extractor 和 helper 的 SHA256；任何缺失或漂移都会触发
cache 重建。

## 单 H100 验证

单卡实例必须只暴露一张 H100。脚本会自动加载项目指定的 algengine 环境，不需要手工
\`conda activate\`。

如果实例上仍运行 \`occupy_GPU.py\`，先在它所在终端停止占卡程序，避免真实模型 parity
因显存不足失败。然后执行：

\`\`\`bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_progress_fix_1h100_smoke.sh
\`\`\`

它依次检查 MMCV sm90、CPU 公式回归，并用真实模型生成候选，逐项比较：

1. 正式训练路径的 CUDA \`TorchSimulator\` reward；
2. 相同候选的 CPU \`PDMSimulator\` reward；
3. 20 次独立 NAVSIM official \`pdm_score\`。

只有实际样本中出现 multiplicative gate，且三条路径的 score/components 都在
\`1e-5\` 内一致时才 PASS。

## 正式 8×H100 一体化运行

分布式任务必须恰好暴露 8 张 H100。只提交一个任务：

\`\`\`bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine
./run_diffusiondrive_grpo_selector_v3_progress_fix_8h100.sh all
\`\`\`

\`all\` 会在同一个排队任务中串联：

1. CPU 单测、8 卡 MMCV preflight、真实候选 CPU/CUDA/official parity；
2. train/development/certification 三组 cache bundle；
3. 四种表征 ablation、八组开发集 sweep、一次 certification；
4. seed 0/1/2 的四块正式评测；
5. epoch-100、V3 修复前、V3 修复后的自动对比表。

阶段性重提可使用：

\`\`\`bash
./run_diffusiondrive_grpo_selector_v3_progress_fix_8h100.sh preflight
./run_diffusiondrive_grpo_selector_v3_progress_fix_8h100.sh cache
./run_diffusiondrive_grpo_selector_v3_progress_fix_8h100.sh tune
./run_diffusiondrive_grpo_selector_v3_progress_fix_8h100.sh eval
\`\`\`

完成阶段会写入与 Git commit 绑定的 receipt；同一 commit 重提时可以安全复用。只要
tracked 实验代码未提交，正式入口会在消耗 GPU 前拒绝运行。

## 结果位置

- 总状态：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/integrated/status.txt\`
- 持久日志：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/integrated/logs/\`
- parity：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/preflight/parity_report.json\`
- 修复后 cache：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/cache/\`
- certification checkpoint：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/certification/selected_checkpoint.pth\`
- 三组最终 JSON 对比：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/formal/progress_fix_comparison.json\`
- 三组最终 Markdown 对比：
  \`experiments/diffusiondrive/grpo_selector_v3_progress_fix_v1/formal/progress_fix_comparison.md\`

旧的 rare-log 8×H100 排队任务若尚未开始，应在平台页面手工取消；不要让它继续生成
基于修复前 reward 的正式结果。
