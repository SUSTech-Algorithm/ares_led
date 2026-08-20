# Odin1 detector 配置占位

Odin1 detector 目前只是占位，`calibrated: false`。后续流程：

1. 用 Odin1 驱动采集并标注训练图片到 `camera_data/odin1/`；
2. 根据 Odin1 图片覆盖本目录下的 `detector.yaml`，并把 `calibrated` 改为 `true`；
3. 如 Odin1 输入不是 V4L2 摄像头，还需要为 R2 节点增加 ROS image topic 输入源。

当前不要复制 USB RGB 的 `detector.yaml` 到这里冒充 Odin1 参数。
