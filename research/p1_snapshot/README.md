# P1 seed2 完整评测快照（四卡）

这轮不训练、不调参：冻结 development 上选出的 P1 seed2 step500，与原始
V3 rare_tuned seed0 同时重跑完整评测。保留既有 P3 `STOP_EFFECT_FAIL` 结论。
这是 observed-best 阶段快照，不是三 seed 主结果，也不是独立盲测确认。

## 固定范围

- selector SHA256：`331ac4459cef19febff865080e3015f955cae0d2ebbd04bed58e207b2b8ebc6b`。
- 导出原生完整模型，仅替换 54 个 scene_selector 张量；963 个其他张量保持不变。
- 推理不使用 CFPI 路由、奖励或 Q。闭环观察器只记录与核对动作。
- bridge：从已使用的 development58 按固定哈希选择 8 场景；两模型 × NR/R，
  使用旧 `selector_cfpi_v1_noise0`，与旧部署逐帧核对候选、选择、动作和指标。
- 正式噪声固定为 `formal_navtest_seed0`。两模型各跑 6 个条件：
  open-navtest12146、open-rare288、full-NR288、full-R288、common-NR64、common-R64。
- 排除 CSV 的 average/overall_average 行；真实分母不是 12147 或 289。
- 报告分开列 full288、development58、remainder230 和 common64。
  remainder230 也有历史暴露，不称作全新盲测。log-cluster 置信区间条件于固定模型和噪声，
  不能代替训练 seed 方差。保留原有三 seed 结果引用。

## 一步步运行

使用新的 worktree，不要在旧 decision-feedback 目录运行。以下是独立的正式 run，
不要用开发期间的 `software_audit_*` ID。代码或预算改变后必须使用新 ID。

```bash
cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-p1-seed2-snapshot-v1

bash run_diffusiondrive_selector_p1_snapshot_4h100.sh audit p1_seed2_full_20260913_v1 --gpus 4 --gpu-hours 192
```

audit 只用 CPU：复制模型和恢复状态、冻结代码/输入、核对真实缓存导出一致性。
看到 audit COMPLETE 后，在四卡实例上运行：

```bash
bash run_diffusiondrive_selector_p1_snapshot_4h100.sh preflight p1_seed2_full_20260913_v1 --gpus 4 --gpu-hours 192
```

preflight 检查 CUDA、扩展、配置和 CUDA 导出一致性，不运行正式场景。
通过后后台执行完整流水线：

```bash
nohup bash run_diffusiondrive_selector_p1_snapshot_4h100.sh all p1_seed2_full_20260913_v1 --gpus 4 --gpu-hours 192 > p1_seed2_full_20260913_v1.log 2>&1 &
```

`all` 自动执行 preflight → bridge → 12 条正式评测 → report，无须半夜追加命令。
bridge 是工程一致性检查，不通过即停止；正式结果即使下降也照常保存，不按效果提前筛选。
默认 192 GPU-hours 是独立上限，按占用四卡记账，约对应最多 48 小时累计运行，
不是预计耗时。GPU 实例空闲时的整机收费不在脚本计账范围内。

```bash
tail -n 40 p1_seed2_full_20260913_v1.log
```

中断后先确认旧进程已退出，再用相同代码、ID、参数恢复（日志用追加）：

```bash
nohup bash run_diffusiondrive_selector_p1_snapshot_4h100.sh all p1_seed2_full_20260913_v1 --gpus 4 --gpu-hours 192 >> p1_seed2_full_20260913_v1.log 2>&1 &
```

已审计完成的条件校验后复用；不完整条件保留旧 attempt，再重跑该条件，
不是逐场景续跑。累计预算不清零。遇到 hash、bridge 或预算失败不要删 ledger 强行继续。

可单独执行 `bridge` 或 `eval`；`eval` 必须已有 PASS bridge。查询/重建报告不启动 GPU：

```bash
bash run_diffusiondrive_selector_p1_snapshot_4h100.sh report p1_seed2_full_20260913_v1 --gpus 4 --gpu-hours 192
```

如实例 SimEngine Python 不在默认 `/root/miniconda3/envs/simengine/bin/python`，
运行前设置 `DIFFUSIONDRIVE_SIMENGINE_PYTHON` 为该实例的实际解释器；其他环境沿用历史流程。

## 输出与解释边界

根目录为 `experiments/diffusiondrive/selector_p1_snapshot_v1/runs/p1_seed2_full_20260913_v1/`。

- `audit_report.json`、`bridge_report.json`：软件/工程一致性，不表示方法有效。
- `summary.json`：完整指标、paired log-bootstrap 区间、rescued/broken、安全分量、历史协议比较。
- `summary.md`、`summary.tsv`：概览及与历史表对应的 29 列结果；`paired_scenes.csv` 为逐场景配对。
- `archive/`：源 selector、optimizer/RNG 恢复状态、源报告、代码/配置、命令与 release manifest。
- `models/`：两份完整推理 checkpoint。大数据集使用锁定路径与 SHA256 引用，不全部复制。
- `ledger.json`：本轮独立累计计账；未跑齐时仅生成 INCOMPLETE progress，不宣称完成。

正式结果齐备后先封存这一版，再讨论下一轮创新与多 seed 验证。单个 seed2 的开发集优势
不能提前推出全量提升，也不能直接充当顶会论文的最终主结论。

## 软件测试

```bash
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python -m unittest discover -s research/p1_snapshot/tests -v
```
