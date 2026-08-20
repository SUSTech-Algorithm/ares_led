import glob
import os
import select
import termios
from typing import Optional

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Int32

from rgb_comm_protocol import FixedColorProtocol

from .mapping import build_wled_state_json


_BAUD_RATES = {
    9600: termios.B9600,
    19200: termios.B19200,
    38400: termios.B38400,
    57600: termios.B57600,
    115200: termios.B115200,
    230400: termios.B230400,
    460800: termios.B460800,
    921600: termios.B921600,
}


class LineSerial:
    def __init__(self, device: str, baudrate: int, timeout_sec: float) -> None:
        self.device = device
        self.baudrate = baudrate
        self.timeout_sec = max(timeout_sec, 0.0)
        self.fd: Optional[int] = None

    def open(self) -> None:
        if self.fd is not None:
            return
        if self.baudrate not in _BAUD_RATES:
            raise ValueError(f'unsupported serial baudrate: {self.baudrate}')
        fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(fd)
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[3] = 0
            attrs[4] = _BAUD_RATES[self.baudrate]
            attrs[5] = _BAUD_RATES[self.baudrate]
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except Exception:
            os.close(fd)
            raise
        self.fd = fd

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def write_line(self, line: str) -> str:
        self.open()
        assert self.fd is not None
        os.write(self.fd, line.encode('ascii'))
        if self.timeout_sec <= 0.0:
            return ''

        ready, _, _ = select.select([self.fd], [], [], self.timeout_sec)
        if not ready:
            return ''
        data = os.read(self.fd, 256)
        return data.decode('ascii', errors='replace').strip()


def _find_serial_device(configured: str) -> Optional[str]:
    if configured and configured != 'auto':
        return configured
    patterns = [
        '/dev/serial/by-id/*',
        '/dev/ttyACM*',
        '/dev/ttyUSB*',
    ]
    preferred_keywords = ('wled', 'lolin', 'wemos', 'esp32', 'usb-dev')
    rejected_keywords = ('daplink', 'cmsis-dap')
    candidates = []
    for pattern in patterns:
        for device in sorted(glob.glob(pattern)):
            lower = device.lower()
            if any(keyword in lower for keyword in rejected_keywords):
                continue
            candidates.append(device)
    for device in candidates:
        lower = device.lower()
        if any(keyword in lower for keyword in preferred_keywords):
            return device
    if candidates:
        return candidates[0]
    return None


class RgbLedSender(Node):
    def __init__(self) -> None:
        super().__init__('rgb_led_sender')
        topic = self.declare_parameter('input_topic', '/aruco_comm/tx_id').value
        self.transport = str(self.declare_parameter('transport', 'serial').value).lower()
        self.groups = [int(v) for v in self.declare_parameter('groups', [0, 1]).value]
        self.brightness = float(self.declare_parameter('brightness', 0.2).value)
        self.pixel_count = int(self.declare_parameter('pixel_count', 11).value)
        retry_period = float(self.declare_parameter('retry_period_sec', 0.5).value)

        if len(self.groups) != 2:
            raise ValueError('groups must contain exactly two LED group indexes')
        if any(group < 0 for group in self.groups):
            raise ValueError('groups must contain only non-negative LED group indexes')
        if len(set(self.groups)) != len(self.groups):
            raise ValueError('groups must contain two distinct LED group indexes')
        if self.pixel_count < 2:
            raise ValueError('pixel_count must be at least 2')

        default_colors = os.path.join(
            get_package_share_directory('rgb_led_sender'), 'config', 'colors.yaml')
        colors_path = str(self.declare_parameter(
            'colors_config', default_colors).value) or default_colors
        self.protocol = FixedColorProtocol(colors_path=colors_path)
        self.pending_id: Optional[int] = None
        self.active_command: Optional[int] = None

        self.serial: Optional[LineSerial] = None
        self.serial_device_config = ''
        self.serial_baudrate = 115200
        self.serial_timeout_sec = 0.05

        if self.transport != 'serial':
            raise ValueError("transport must be 'serial'")
        self._init_serial_transport()

        self.create_subscription(Int32, topic, self._on_command, 10)
        self.create_timer(max(retry_period, 0.05), self._dispatch)
        self.get_logger().info(
            f'RGB LED sender ready: topic={topic}, transport={self.transport}, '
            f'groups={self.groups}, pixel_count={self.pixel_count}')

    def _init_serial_transport(self) -> None:
        self.serial_device_config = str(
            self.declare_parameter('serial_device', 'auto').value)
        self.serial_baudrate = int(
            self.declare_parameter('serial_baudrate', 115200).value)
        self.serial_timeout_sec = float(
            self.declare_parameter('serial_timeout_sec', 0.05).value)

    def _on_command(self, message: Int32) -> None:
        command = int(message.data)
        if self.protocol.encode_symbols(command) is None:
            self.get_logger().warning(f'command ID {command} is not in RGB protocol; ignored')
            return
        if command == self.pending_id or command == self.active_command:
            return
        self.pending_id = command
        self._dispatch()

    def _dispatch(self) -> None:
        self._dispatch_serial()

    def _dispatch_serial(self) -> None:
        if self.pending_id is None:
            return
        device = _find_serial_device(self.serial_device_config)
        if device is None:
            self.get_logger().warning('WLED serial device not found; retrying')
            return
        if self.serial is None or self.serial.device != device:
            if self.serial is not None:
                self.serial.close()
            self.serial = LineSerial(device, self.serial_baudrate, self.serial_timeout_sec)

        command = self.pending_id
        code = self.protocol.encode_symbols(command)
        rgb = self.protocol.encode_rgb(command)
        if code is None or rgb is None:
            self.pending_id = None
            return
        payload = build_wled_state_json(rgb, self.brightness, self.pixel_count)
        try:
            response = self.serial.write_line(f'{payload}\n')
        except OSError as exc:
            self.get_logger().warning(f'failed to write WLED serial {device}: {exc}; retrying')
            if self.serial is not None:
                self.serial.close()
            return

        self.pending_id = None
        self.active_command = command
        if response:
            self.get_logger().info(f'sent command {command}: {code}; WLED replied: {response}')
        else:
            self.get_logger().info(f'sent command {command}: {code}')

    def destroy_node(self) -> bool:
        if self.serial is not None:
            self.serial.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RgbLedSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
