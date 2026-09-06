# CFPI CV 重算一致性修复 — 2026-09-06

冻结数据源 run：`cfpi_pilot_20260905_r2`。采集、全部审计与信息 gate 已完成：
pilot64 有 64 个完整 20-candidate surface，repeat8 固定条件最大误差为 0，
46/64 场景的 oracle-policy gap >0.02。development/test 未使用，未授权扩量。

第一批 CV 在任何优化 step 之前停止。原因是训练脚本把 H100 完整规划头缓存的
V3 logits，与独立加载 selector 的 float32 重算结果按 1e-5 比较。
离线复核 1,280 个值：max_abs=4.3392181396484375e-05，p95=1.1444091796875e-05，
p99=1.71661376953125e-05，超过 1e-5 的值为 78；64/64 argmax 完全一致。
这排除了 checkpoint 身份错误和选择决策漂移，但说明跨执行路径使用采集数组门槛过严。

修复将两类门槛分开：同路径采集数组仍为 1e-5；完整规划头缓存与离线 selector
重算为 1e-4，并继续要求每个场景 argmax 完全一致。训练 report 保存实际最大误差、
门槛和 mismatch 数。任何一次超过 1e-4 或 argmax 变化仍 fail closed。
门槛只由 logits 数值复核决定，没有按 Q outcome 或方法效果调节。

不修改 r2 的 run_contract 或缓存。使用新 run `cfpi_pilot_20260905_r2_cv1`，
其 contract 锁定新分析代码，并逐文件记录 r2 contract、ledger、pilot/repeat cache、
audit 与 gate 的绝对路径和 SHA256。新 run 只提供 train CV + report，不调用 source、
baseline、sentinel、pilot、development 或 test。

r2 已使用 71.1496896362 GPUh，上限 143.8 GPUh，因此 continuation 上限取
72.6 GPUh，保守小于剩余 72.6503103638 GPUh。此前失败任务未产生训练 checkpoint。
