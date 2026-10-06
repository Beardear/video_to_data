# Track 1 baseline

## 干了什么

- 接通输入准备 → CARI4D 重建 → 官方格式导出与打包，配置 30 段初始人/物提示。
- 补上输入校验、源码/输入版本隔离、失败重试和断点续跑，便于队友分段运行。

## 怎么干的

- 复用 SAM2 分割、MoGe 深度、SAM3D 物体网格和 CARI4D 重建；结果经官方 MHR 拟合与 packer 输出 `submission.parquet`。
- 按 [prepare](track1_prepare.md) → [batch](track1_batch.md) → [pack](track1_pack.md) 运行。前两步默认只生成计划，加 `--execute` 才执行；先用 `--episodes 16` 验证一段。
- 团队固定业务 commit、镜像 digest、权重与配置。业务改动合入 `main`，镜像维护留在 `docker-image`。

## 还有哪些局限

- 140 项 CPU 测试已通过；全 30 段打包测试使用合成结果。新流程尚未完成真实 GPU 端到端验证，目前原机房 H200 缺货。
- 30 段掩码、网格和尺度仍需生成并人工验收；尚无全量真实重建或 Kaggle 分数。
- 沿用官方转换，baseline 的 `report` 策略记录并接受拟合超限误差，仍拒绝非法数据；转换误差对成绩的影响尚未评估。
