import glob
import os
import subprocess
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

from rgb_comm_protocol import FixedColorProtocol

from .classifier import annotate, detect_candidates, load_config, select_winner
from .protocol_decoder import (
    annotate_protocol,
    decode_protocol_candidates,
    load_pairing_config,
    select_protocol_winner,
)
from .profiles import (
    DEFAULT_CAMERA_PROFILE,
    detector_config_path,
    require_calibrated_detector,
    validate_camera_profile,
)


class LedStripReceiver(Node):
    """与协议无关的灯带检测器实时摄像头前端。"""

    def __init__(self) -> None:
        super().__init__('rgb_camera_receiver')
        profile = validate_camera_profile(str(self.declare_parameter(
            'camera_profile', DEFAULT_CAMERA_PROFILE).value))
        requested = str(self.declare_parameter('camera_device', 'auto').value)
        width = int(self.declare_parameter('frame_width', 1280).value)
        height = int(self.declare_parameter('frame_height', 720).value)
        fps = max(float(self.declare_parameter('scan_rate_hz', 30.0).value), 1.0)
        self.preview = bool(self.declare_parameter('show_preview', True).value) and bool(
            os.environ.get('DISPLAY'))
        self.preview_scale = max(float(
            self.declare_parameter('preview_scale', 0.5).value), 0.05)
        self.processing_scale = min(1.0, max(float(
            self.declare_parameter('processing_scale', 1.0).value), 0.1))
        self.save_positive_images = bool(
            self.declare_parameter('save_positive_images', True).value)
        capture_dir = str(self.declare_parameter(
            'positive_capture_dir',
            '~/Desktop/LED/camera_capture_positive').value)
        self.positive_capture_dir = Path(capture_dir).expanduser()
        self.positive_save_interval = max(float(self.declare_parameter(
            'positive_save_interval_sec', 1.0).value), 0.0)
        self.positive_active = False
        self.last_positive_save_time: Optional[float] = None
        if self.save_positive_images:
            self.positive_capture_dir.mkdir(parents=True, exist_ok=True)
        reject = str(self.declare_parameter(
            'camera_reject_keywords', 'Integrated,Chicony').value)
        self.reject_keywords = tuple(
            item.strip().lower() for item in reject.split(',') if item.strip())
        controls = str(self.declare_parameter('v4l2_controls', '').value)
        default_detector = str(detector_config_path(profile))
        configured_detector = str(self.declare_parameter(
            'detector_config', default_detector).value).strip()
        config_path = str(require_calibrated_detector(
            Path(configured_detector or default_detector).expanduser(),
            profile))
        self.config = load_config(config_path)
        protocol_config = str(self.declare_parameter(
            'protocol_config', '').value)
        self.protocol = FixedColorProtocol(
            config_path=protocol_config if protocol_config else None)
        self.pairing_config = load_pairing_config(config_path)
        self.protocol_margin = max(float(self.declare_parameter(
            'protocol_winner_margin', 1.2).value), 1.0)
        output_topic = str(self.declare_parameter(
            'output_topic', '/aruco_comm/rx_id').value)
        self.publisher = self.create_publisher(Int32, output_topic, 10)
        self.confirmation_window = max(int(self.declare_parameter(
            'confirmation_window', 7).value), 1)
        self.confirmation_required = max(int(self.declare_parameter(
            'confirmation_required', 5).value), 1)
        self.confirmation_required = min(
            self.confirmation_required, self.confirmation_window)
        self.off_unlock_sec = max(float(self.declare_parameter(
            'off_unlock_sec', 0.35).value), 0.0)
        self.history: Deque[int] = deque(maxlen=self.confirmation_window)
        self.locked = False
        self.last_protocol_seen_time: Optional[float] = None
        self.capture, self.device = self._open_camera(requested, width, height, fps)
        self._apply_controls(self.device, controls)
        self.last_label = ''
        self.create_timer(1.0 / fps, self._scan)
        self.get_logger().info(
            f'R2 LED vision ready: profile={profile}, camera={self.device}, '
            f'config={config_path}, '
            f'colors={[item.name for item in self.config.colors]}, '
            f'output={output_topic}, confirm={self.confirmation_required}/'
            f'{self.confirmation_window}')

    def _scan(self) -> None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.get_logger().warning('camera frame unavailable')
            return
        work = frame
        scale = self.processing_scale
        if scale < 1.0:
            work = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        candidates = detect_candidates(work, self.config)
        if scale < 1.0:
            inverse = 1.0 / scale
            candidates = [item.scaled(inverse) for item in candidates]
        single_winner = select_winner(candidates, self.config)
        protocol_candidates = decode_protocol_candidates(
            candidates, self.protocol, self.pairing_config)
        protocol_winner = select_protocol_winner(
            protocol_candidates, self.protocol_margin)
        confirmed_count = self._update_protocol_state(protocol_winner)
        self._save_positive_frame(frame, single_winner)
        if protocol_winner is None:
            label = f'{"LOCK" if self.locked else "WAIT"}:NONE'
        else:
            label = (
                f'{"LOCK" if self.locked else "WAIT"}:'
                f'{protocol_winner.command_id}:{confirmed_count}')
        if label != self.last_label:
            if protocol_winner is None:
                self.get_logger().info(
                    f'command=NONE strip_candidates={len(candidates)} '
                    f'protocol_candidates={len(protocol_candidates)} '
                    f'state={"LOCK" if self.locked else "WAIT"}')
            else:
                different = next(
                    (item for item in protocol_candidates[1:]
                     if item.command_id != protocol_winner.command_id),
                    None)
                second = different.score if different is not None else 0.0
                margin = (
                    protocol_winner.score / max(second, 1e-9)
                    if second else float('inf'))
                self.get_logger().info(
                    f'command={protocol_winner.command_id} '
                    f'symbols={protocol_winner.symbols} '
                    f'confidence={protocol_winner.confidence:.3f} '
                    f'score={protocol_winner.score:.3f} margin={margin:.3f} '
                    f'confirm={confirmed_count}/{self.confirmation_required} '
                    f'state={"LOCK" if self.locked else "WAIT"}')
            self.last_label = label
        if self.preview:
            rendered = annotate(frame, candidates, single_winner)
            rendered = annotate_protocol(
                rendered,
                protocol_candidates,
                protocol_winner,
                'LOCK' if self.locked else 'WAIT',
                confirmed_count,
                self.confirmation_required,
            )
            if self.preview_scale != 1.0:
                rendered = cv2.resize(
                    rendered, None, fx=self.preview_scale, fy=self.preview_scale,
                    interpolation=cv2.INTER_AREA)
            cv2.imshow('R2 LED strip detector', rendered)
            cv2.waitKey(1)

    def _update_protocol_state(self, winner) -> int:
        now = time.monotonic()
        if winner is None:
            self.history.clear()
            if (
                    self.locked and self.last_protocol_seen_time is not None and
                    now - self.last_protocol_seen_time >= self.off_unlock_sec):
                self.locked = False
                self.get_logger().info('protocol lock released')
            return 0

        self.last_protocol_seen_time = now
        if self.locked:
            return 0

        self.history.append(winner.command_id)
        counts = Counter(self.history)
        confirmed_count = counts[winner.command_id]
        if confirmed_count >= self.confirmation_required:
            message = Int32()
            message.data = int(winner.command_id)
            self.publisher.publish(message)
            self.locked = True
            self.history.clear()
            self.get_logger().info(
                f'published confirmed command {winner.command_id}: '
                f'{winner.symbols}')
        return confirmed_count

    def _save_positive_frame(self, frame, winner) -> None:
        if winner is None:
            self.positive_active = False
            return
        if not self.save_positive_images:
            return
        now = time.monotonic()
        should_save = (
            not self.positive_active or
            self.last_positive_save_time is None or
            now - self.last_positive_save_time >= self.positive_save_interval
        )
        self.positive_active = True
        if not should_save:
            return
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = (
            f'{winner.color.lower()}_{timestamp}_'
            f'conf{winner.confidence:.3f}_score{winner.score:.3f}.jpg'
        )
        path = self.positive_capture_dir / filename
        if cv2.imwrite(str(path), frame):
            self.last_positive_save_time = now
            self.get_logger().info(f'saved positive frame: {path}')
        else:
            self.get_logger().error(f'failed to save positive frame: {path}')

    def _open_camera(self, requested: str, width: int, height: int, fps: float):
        failures: List[str] = []
        for device in self._camera_candidates(requested):
            capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not capture.isOpened():
                failures.append(device)
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = capture.read()
            if ok and frame is not None and frame.ndim == 3 and frame.shape[2] == 3:
                return capture, device
            failures.append(device)
            capture.release()
        raise RuntimeError(
            f'cannot open camera {requested}; attempted={failures}')

    def _camera_candidates(self, requested: str) -> List[str]:
        if requested != 'auto':
            return [requested]
        devices = sorted(glob.glob('/dev/v4l/by-id/*')) + sorted(glob.glob('/dev/video*'))
        result = []
        seen = set()
        for device in devices:
            resolved = os.path.realpath(device)
            base = os.path.basename(resolved)
            name_path = Path('/sys/class/video4linux') / base / 'name'
            name = name_path.read_text(errors='replace').strip() if name_path.exists() else ''
            if any(keyword in name.lower() for keyword in self.reject_keywords):
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            result.append(device)
        return result

    def _apply_controls(self, device: str, controls: str) -> None:
        if not controls.strip():
            return
        executable = '/usr/bin/v4l2-ctl' if Path('/usr/bin/v4l2-ctl').exists() else None
        if executable is None:
            self.get_logger().warning('v4l2-ctl not installed; camera controls skipped')
            return
        for assignment in controls.split(','):
            assignment = assignment.strip()
            if not assignment:
                continue
            result = subprocess.run(
                [executable, '--device', device, '--set-ctrl', assignment],
                text=True, capture_output=True, check=False)
            if result.returncode != 0:
                message = (result.stderr or result.stdout).strip()
                self.get_logger().warning(
                    f'camera control {assignment} failed: {message}')

    def destroy_node(self):
        self.capture.release()
        if self.preview:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node: Optional[LedStripReceiver] = None
    try:
        node = LedStripReceiver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
