# 摄像头图片采集

本目录有两个独立脚本：

- `capture_usb_rgb.py`：普通 USB RGB 摄像头，直接读取 `/dev/video*`。
- `capture_odin1.py`：Odin1，订阅 ROS 2 图像话题，默认使用 `/odin1/image/compressed`。

两者都默认每 2 秒保存一张未经绘制的 JPG，同时显示实时预览。按 `q`、`Esc` 或
`Ctrl+C` 停止。

## USB RGB 摄像头

```bash
cd ~/ARES_LED/camera_capture
/usr/bin/python3 ./capture_usb_rgb.py
```

默认保存到：

```text
~/ARES_LED/camera_data/usb_rgb/raw
```

常用参数：

```bash
/usr/bin/python3 ./capture_usb_rgb.py \
  --device auto \
  --output ~/ARES_LED/camera_data/usb_rgb/raw \
  --interval 2 \
  --width 1280 --height 720 --fps 30
```

自动选择会排除设备名称中含有 `Integrated`、`Internal` 或 `Built-in` 的笔记本集成摄像头。
如果自动选择错误，先用 `v4l2-ctl --list-devices` 查看，再用 `--device /dev/videoX` 指定。

## Odin1

先启动 Odin1 driver。不要设置 `ROS_DOMAIN_ID`：

```bash
unset ROS_DOMAIN_ID
source /opt/ros/jazzy/setup.bash
source ~/TreeAction/install/setup.bash
ros2 launch odin_ros_driver odin1_ros2.launch.py
```

如果不需要 RViz，只杀 RViz，不要杀 `host_sdk_sample`：

```bash
pkill -f rviz2
```

另开终端采集 Odin1 图片：

```bash
cd ~/ARES_LED/camera_capture
unset ROS_DOMAIN_ID
source /opt/ros/jazzy/setup.bash
source ~/TreeAction/install/setup.bash

./capture_odin1.py
```

默认保存到：

```text
~/ARES_LED/camera_data/odin1/raw
```

常用参数：

```bash
./capture_odin1.py \
  --topic /odin1/image/compressed \
  --output ~/ARES_LED/camera_data/odin1/raw \
  --interval 2
```

如果远程桌面显示卡，只保存不预览：

```bash
./capture_odin1.py --no-preview
```

如果确实要订阅未压缩图像：

```bash
./capture_odin1.py --topic /odin1/image --raw
```
