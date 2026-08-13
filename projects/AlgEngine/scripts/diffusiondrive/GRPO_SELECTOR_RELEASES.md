# DiffusionDrive selector 版本与恢复清单

## V3：已冻结的 human-log 阶段成果

- 分支：`diffusiondrive-selector-grpo-v3-final`
- tag：`diffusiondrive-selector-grpo-v3-20260812`
- commit：`c63a23b7a59b5dfe90f8f4b020decaa8d1df6df0`
- 本地完整 release：
  `experiments/diffusiondrive/releases/grpo_selector_v3_20260812`
- 方法、实验与正式四块结果见同目录脚本文件 `GRPO_SELECTOR_V3.md`。

V3 不会被 rollout 代码覆盖。rollout 从这个 commit 另开分支，后续任何失败都可以
直接切回 V3。

### 权重与 bundle SHA256

| 文件 | SHA256 |
|---|---|
| `code/worldengine_v3.bundle` | `56eb3dcc057cd60eb398224b00272a90e184a0067fa892dbbed3b764b0a7dd3b` |
| `seed0/checkpoint.pth` | `d35ba31606b3237cd9624c0631e8ee7f16b75468870e41a29a86061baa459b8c` |
| `seed0/epoch_16_scene_selector.pt` | `2d3c2f1a0cadb76cfffe776f9854ef716fa72e361b8d91644107332060ef30c6` |
| `seed1/checkpoint.pth` | `d805b6dbf396d9bde9444876f069610ca77eac4ef1f5c28a04d8516a5ddcea6b` |
| `seed1/epoch_16_scene_selector.pt` | `efd65327826bba903147856fe2ea46e4614118ae6186b2c75cc437552c963aa2` |
| `seed2/checkpoint.pth` | `9713a5339cdad2f8198ee28c81f694cc38331fc964f3da1711ed2f6ba5791814` |
| `seed2/epoch_16_scene_selector.pt` | `74b4f45596a9fc85936e8341e5477dab6d53cdea273b1f7077758cd56ab3937c` |

### 恢复方式

仓库仍存在时：

```bash
git switch diffusiondrive-selector-grpo-v3-final
git verify-tag diffusiondrive-selector-grpo-v3-20260812
```

只剩 release bundle 时：

```bash
git clone experiments/diffusiondrive/releases/grpo_selector_v3_20260812/code/worldengine_v3.bundle WorldEngine-v3
cd WorldEngine-v3
git switch diffusiondrive-selector-grpo-v3-final
```

完整 checkpoint 不放进普通 Git 历史；Git 保存代码、tag、恢复文档，release 目录保存
大权重、结果和 SHA。向 GitHub 发布后仍应使用 `git ls-remote origin` 核验分支和 tag，
不能仅凭本地 `git push` 命令返回前的状态判断成功。

## rollout v1：独立开发版本

- 开发分支：`diffusiondrive-selector-grpo-rollout-v1`
- 基线：上述 V3 clean commit；
- rollout 只改变训练状态分布，不改变 V3 的 selector 架构、20-action set、PDM reward
  或 exact-group GRPO objective；
- H100 smoke 通过后才创建 rollout v1 release tag；
- certification 与三 seed 正式四块评测完成后，另存 checkpoint、selector-only state、
  manifest、summary、代码 bundle 和 SHA256。
