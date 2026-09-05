# CFPI 首轮运行与恢复

工作分支：research/frozen-selector-cfpi-v1。
根路径：/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/worktrees/WorldEngine-selector-cfpi-v1

## 保存与隔离

继承代码提交 d90aa71 与后续新方法提交分离。
V3 根目录、rapg/V4 工作目录原样保留。
恢复快照：
/inspire/hdd/global_user/wangcaojun-240208020180/nry/WorldEngine/experiments/research_snapshots/cfpi_20260905T080752Z

manifest.json 保存校验信息，v3_root.tar.gz 包含 36 个未跟踪文件，
rapg_v4.tar.gz 包含 244 个已修改/未跟踪文件及 staged/unstaged patches；
all_refs.bundle 保存 Git 历史。解包恢复必须到新目录，不覆盖原目录。
大型已有 checkpoint/cache/results 原地保留；采集和训练另校验实际使用 artifact。

## 8 H100 使用

先进入上述工作目录。2026-09-05 审批后硬件只读检查可见 1 张 RTX 4090，
不是本协议要求的 H100 分配；沙箱内旧检查还出现过驱动不可达。
必须在实际分配的 H100 计算节点执行，不把沙箱报错直接解释为宿主机没有 GPU。
AlgEngine 默认环境：
/inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine
SimEngine 默认解释器：
/root/miniconda3/envs/simengine/bin/python
节点路径不同时设置 DIFFUSIONDRIVE_SIMENGINE_PYTHON，不替换全局环境。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_diffusiondrive_selector_cfpi_8h100.sh preflight cfpi_pilot_20260905_r2 --gpus 8 --gpu-hours 143.8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_diffusiondrive_selector_cfpi_8h100.sh first_phase cfpi_pilot_20260905_r2 --gpus 8 --gpu-hours 143.8
```

first_phase 只执行 source → train A/B → freeze → sentinel → pilot64/repeat8 →
gate → CV → report；不提供 expand/dev/test/all 入口。
信息 gate 不通过时跳过训练，输出停止报告。
可独立调用 source、baseline_train、freeze、sentinel、pilot、pilot_audit、train_cv、report。
同一 run ID 可恢复已审计 collection 与每 25 步保存的训练状态。
源码/协议/噪声/数据/拓扑改变时必须新 run ID，不能删除旧 contract 绕过检查。
预算默认 --gpu-hours 144。原 run 在 source 阶段停止，账本已记 0.164984 GPUh；
source 配额实现修复后改用 r2，并将剩余预算保守取为 143.8 GPUh，不重置总预算。
达到预算不会自动续费或扩量，停止时保留进度。

输出位于 experiments/diffusiondrive/selector_cfpi_v1/runs/RUN_ID/：
run_contract.json、decision_ledger.json、preflight.json、source、targets、manifests、
collections、cache、cv、pilot_gate.json、pilot_report.json、PILOT_REPORT.md。
每 30 秒有阶段心跳，错误日志在 logs/ 或 collections/ID/logs/。
缺失最终 metrics CSV 或完整审计不能当作成功，simulation_completed.flag 不是验收依据。

## CPU 检查

```bash
PYTHONPATH=projects/AlgEngine/scripts/diffusiondrive \
  /inspire/hdd/global_user/wangcaojun-240208020180/miniconda3/envs/algengine/bin/python \
  -m pytest -q projects/AlgEngine/tests/test_selector_cfpi.py
```

CPU 单测不替代 Hopper 扩展实算、实际候选 parity 或完整模拟器运行。
Q 回归 selector 导出必须使用 direct_q score mode；普通分类/GRPO 使用 residual。
materialize_selector_cfpi.py 导出完整 checkpoint 并核验非 selector 参数未变，
同时写出必需的 deployment_cfg_options；不得把 Q checkpoint 交给未设置该模式的配置。

本轮结束只产生方法筛选报告。后续闭环部署、扩量或发展新方法需要新的研究决定。

首轮实现、58 项 CPU 测试和真实硬件阻塞证据见
[IMPLEMENTATION_HANDOFF_20260905.md](IMPLEMENTATION_HANDOFF_20260905.md)。
首次 H100 启动与 source 修复说明见 [SOURCE_ALLOCATION_FIX_20260905.md](SOURCE_ALLOCATION_FIX_20260905.md)。
