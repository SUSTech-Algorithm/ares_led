# R1/R2 LED 光学通信

本仓库用 LED 灯带替代原 ArUco 屏幕通信。R1 负责显示两段颜色命令；R2 负责识别
RED/GREEN/BLUE/CYAN/PURPLE 单色灯带候选，并将两段候选按 shared 协议解码为命令 ID。

## 目录

```text
shared/src/   公共 RGB 协议
r1_ws/src/    R1 编码发送节点；默认通过 USB 串口发送 WLED JSON
r2_ws/src/    R2 摄像头实时识别和离线数据集评测
camera_data/  R2 训练图片；按摄像头 profile 隔离
```

`shared/src/rgb_comm_protocol/config/rgb_protocol.yaml` 只保存命令与符号序列。
R1 的物理 RGB 值位于 `r1_ws/src/rgb_led_sender/config/colors.yaml`；R2 的相机观测
颜色模型位于 `r2_ws/src/rgb_camera_receiver/config/cameras/<profile>/detector.yaml`，
两者不得共享数值。

当前 profile：

| profile | 状态 | 数据目录 | detector |
|---|---|---|---|
| `usb_rgb` | 当前 USB 摄像头 | `camera_data/usb_rgb` | 已有 |
| `odin1` | 预留，驱动在 `~/Desktop/odin_ros2_ws` | `camera_data/odin1` | 占位，未标定 |

当前 R1 发送节点订阅 `/aruco_comm/tx_id`，通过 WLED 控制器的 USB CDC 串口直接发送
JSON 状态帧。

普通克隆即可获得构建所需的全部源码：

```bash
git clone <本仓库地址>
```

## R1/R2 两段五色通信协议

R1 只点亮灯带前两段，后面的灯带全部置灭。每段只使用
`RED/GREEN/BLUE/CYAN/PURPLE` 五种高饱和颜色；同一命令的两段颜色必须不同。
协议允许正反等价，因此 R2 解码时 `RED,GREEN` 和 `GREEN,RED` 视为同一命令。

| 符号 | 颜色 | RGB |
|---|---|---:|
| RED | 红 | R1 本地配置 |
| GREEN | 绿 | R1 本地配置 |
| BLUE | 蓝 | R1 本地配置 |
| CYAN | 青 | R1 本地配置 |
| PURPLE | 紫 | R1 本地配置 |

| 命令 ID | 两段颜色 |
|---:|:---:|
| 0 | RED, GREEN |
| 1 | RED, BLUE |
| 2 | RED, CYAN |
| 3 | RED, PURPLE |
| 4 | GREEN, BLUE |
| 5 | GREEN, CYAN |
| 6 | GREEN, PURPLE |
| 7 | BLUE, CYAN |
| 8 | BLUE, PURPLE |

命令 `0` 是普通通信命令，不再表示清空。R2 会多帧确认后发布命令 ID。

## R1 构建与启动

环境为 Ubuntu 22.04 + ROS 2 Humble。Conda 会污染 ROS Python，构建前清理：

```bash
cd ~/Desktop/LED/r1_ws
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
source /opt/ros/humble/setup.bash
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
colcon build --base-paths src ../shared/src \
  --packages-select rgb_comm_protocol rgb_led_sender \
  --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch rgb_led_sender r1_rgb_comm.launch.py
```

启动前确认 WLED 控制器 USB 已连接并枚举为 CDC 串口：

```bash
ls -l /dev/serial/by-id/* /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

应能看到类似：

```text
/dev/serial/by-id/usb-WEMOS.CC_LOLIN-S2-MINI_0-if00 -> ../../ttyACM0
```

注意不要选到 DAPLink 烧录器串口；`auto` 会尽量优先选择 WLED/LOLIN/WEMOS 设备：

```text
usb-ARM_DAPLink_CMSIS-DAP_... -> ../../ttyACM0
```

默认 `r1_ws/src/rgb_led_sender/config/sender.yaml` 使用：

```yaml
transport: serial
serial_device: auto
```

如自动选择错误，可把 `serial_device` 改为上面的
`/dev/serial/by-id/usb-WEMOS.CC_LOLIN-S2-MINI_0-if00` 固定路径。

WLED 灯带连接：24V 电源正极接灯带 `+`，电源负极接灯带 `GND`，WLED `GND` 与灯带
`GND` 共地，WLED `DO` 接灯带 `DIN`。不要把 24V 接到 WLED 板的 5V。
发送命令后，前两段会一直保持对应颜色，直到收到下一条命令；后面的段会被发送端置灭。

发送测试命令：

```bash
ros2 topic pub --once /aruco_comm/tx_id std_msgs/msg/Int32 '{data: 1}'
```

发送另一条命令：

```bash
ros2 topic pub --once /aruco_comm/tx_id std_msgs/msg/Int32 '{data: 8}'
```

也可以绕过 ROS，直接测试 WLED 串口。建议在同一个终端保持串口长连接：

```bash
PORT=/dev/serial/by-id/usb-WEMOS.CC_LOLIN-S2-MINI_0-if00
stty -F "$PORT" 115200 raw -echo -hupcl
exec 3<>"$PORT"
sleep 1
```

整条测试红色：

```bash
printf '{"on":true,"bri":40,"seg":[{"id":0,"start":0,"stop":11,"col":[[255,0,0]],"fx":0}]}\n' >&3
```

关闭：

```bash
printf '{"on":false}\n' >&3
```

## R2 构建与启动

```bash
cd ~/Desktop/LED/r2_ws
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONPATH
source /opt/ros/humble/setup.bash
export PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
colcon build --base-paths src ../shared/src \
  --packages-select rgb_comm_protocol rgb_camera_receiver \
  --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ros2 launch rgb_camera_receiver r2_led_vision.launch.py
```

`camera_device: auto` 会忽略名称包含 Integrated/Chicony 的内置摄像头，并选择首个
能输出彩色图像的外接摄像头。默认 profile 是 `usb_rgb`。

如需显式指定 profile：

```bash
ros2 launch rgb_camera_receiver r2_led_vision.launch.py camera_profile:=usb_rgb
```

Odin1 的 profile 已预留，但当前 R2 节点还没有接入 Odin1 ROS image topic 输入源；
不要直接用 `camera_profile:=odin1` 期待它读取 Odin1 画面。每台相机的运行参数在
`config/cameras/<profile>/receiver.yaml`，R2 专用颜色及几何模型在
`config/cameras/<profile>/detector.yaml`；这些参数与 R1 输出 RGB 完全独立。

离线处理全部标注数据并生成逐图结果：

```bash
ros2 run rgb_camera_receiver evaluate_led_dataset \
  --camera-profile usb_rgb
```

结果目录包含每张图片的全部候选框、颜色、置信度、排名和分差，以及 `results.csv`、
`results.json`。退出码非零表示至少一张图片未满足验收条件。

USB 摄像头重新标定：

```bash
ros2 run rgb_camera_receiver calibrate_led_colors --camera-profile usb_rgb
```

`camera_data/usb_rgb/unused/YELLOW` 中的黄色样本当前不参与默认标定和评估。

## 测试

```bash
cd ~/Desktop/LED
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m pytest \
  shared/src/rgb_comm_protocol/test \
  r1_ws/src/rgb_led_sender/test \
  r2_ws/src/rgb_camera_receiver/test
```

当前硬性回归集为单色图像数据集：五种颜色必须全部识别正确，NONE 必须全部零候选。
R2 另有两段协议解码单元测试。
