#!/usr/bin/env python3
"""Odin1 图像采集：订阅 ROS 压缩图像，实时预览并保存 JPG。"""

import argparse
from datetime import datetime
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image


def qos_profile(reliable: bool) -> QoSProfile:
    return QoSProfile(
        reliability=(
            ReliabilityPolicy.RELIABLE if reliable
            else ReliabilityPolicy.BEST_EFFORT),
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
    )


def decode_compressed(message: CompressedImage):
    data = np.frombuffer(message.data, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('failed to decode compressed image')
    return frame


def decode_raw(message: Image):
    height = int(message.height)
    width = int(message.width)
    encoding = message.encoding.lower()
    data = np.frombuffer(message.data, dtype=np.uint8)

    if encoding in ('bgr8', 'rgb8'):
        frame = data.reshape((height, width, 3))
        if encoding == 'rgb8':
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame.copy()

    if encoding in ('mono8', '8uc1'):
        frame = data.reshape((height, width))
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    raise ValueError(f'unsupported raw image encoding: {message.encoding}')


class Odin1Capture(Node):
    def __init__(self, topic: str, raw: bool, reliable: bool):
        super().__init__('capture_odin1')
        self._lock = threading.Lock()
        self._frame = None
        self._frame_index = 0
        self._last_error = ''

        if raw:
            self.create_subscription(
                Image, topic, self._on_raw, qos_profile(reliable))
        else:
            self.create_subscription(
                CompressedImage, topic, self._on_compressed,
                qos_profile(reliable))

    def _store_frame(self, frame):
        with self._lock:
            self._frame = frame
            self._frame_index += 1
            self._last_error = ''

    def _store_error(self, error: Exception):
        with self._lock:
            self._last_error = str(error)

    def _on_compressed(self, message: CompressedImage):
        try:
            self._store_frame(decode_compressed(message))
        except Exception as error:  # noqa: BLE001
            self._store_error(error)

    def _on_raw(self, message: Image):
        try:
            self._store_frame(decode_raw(message))
        except Exception as error:  # noqa: BLE001
            self._store_error(error)

    def latest(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            return frame, self._frame_index, self._last_error


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preview Odin1 ROS image topic and save JPG images.')
    parser.add_argument(
        '--topic', default='/odin1/image/compressed',
        help='Odin1 image topic, default: /odin1/image/compressed')
    parser.add_argument(
        '--raw', action='store_true',
        help='subscribe sensor_msgs/Image instead of CompressedImage')
    parser.add_argument(
        '--output', default='~/ARES_LED/camera_data/odin1/raw',
        help='directory in which JPG images are saved')
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='save interval in seconds, default: 2')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--reliable', action='store_true',
        help='use reliable QoS instead of best-effort')
    parser.add_argument(
        '--no-preview', action='store_true',
        help='save images without opening an OpenCV preview window')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.interval <= 0:
        raise SystemExit('--interval must be greater than zero')
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit('--jpeg-quality must be in 1..100')

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Odin1Capture(args.topic, args.raw, args.reliable)
    message_type = 'sensor_msgs/Image' if args.raw else 'sensor_msgs/CompressedImage'
    print(f'topic: {args.topic} ({message_type})')
    print(f'output: {output_dir}')
    print(f'saving every {args.interval:g} seconds; press q/Esc/Ctrl+C to stop')

    if not args.no_preview:
        cv2.namedWindow('Odin1 capture', cv2.WINDOW_NORMAL)

    saved_count = 0
    last_saved_index = 0
    next_save = time.monotonic()
    last_status = 0.0

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            frame, frame_index, last_error = node.latest()
            now = time.monotonic()

            if frame is None:
                if now - last_status >= 1.0:
                    suffix = f'; last error: {last_error}' if last_error else ''
                    print(f'waiting for frames on {args.topic}{suffix}')
                    last_status = now
                continue

            if now >= next_save and frame_index != last_saved_index:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                path = output_dir / f'odin1_{stamp}_{saved_count:06d}.jpg'
                saved = cv2.imwrite(
                    str(path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not saved:
                    raise RuntimeError(f'failed to save {path}')
                saved_count += 1
                last_saved_index = frame_index
                next_save = now + args.interval
                print(f'[{saved_count}] saved {path}')

            if not args.no_preview:
                preview = frame.copy()
                remaining = max(0.0, next_save - time.monotonic())
                cv2.putText(
                    preview,
                    f'Odin1  saved: {saved_count}  next: {remaining:.1f}s  q/Esc: quit',
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
                cv2.imshow('Odin1 capture', preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if not args.no_preview:
            cv2.destroyAllWindows()
        print(f'stopped; saved {saved_count} images in {output_dir}')


if __name__ == '__main__':
    main()
