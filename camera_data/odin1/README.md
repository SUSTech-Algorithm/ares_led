# Odin1 摄像头数据

这里预留 Odin1 的训练数据目录。Odin1 驱动位于：

```text
~/Desktop/odin_ros2_ws
```

后续接入 Odin1 时，应把 Odin1 采集到的标注图片放到本目录的对应类别中，并覆盖
`r2_ws/src/rgb_camera_receiver/config/cameras/odin1/detector.yaml`。标定完成后，
该 detector 的 `calibrated` 应为 `true`。

不要把 USB RGB 摄像头的数据或 detector 参数复制到 Odin1 profile 中直接使用。
