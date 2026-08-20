#!/usr/bin/env python3
"""USB RGB 摄像头采集：实时预览，并按固定间隔保存 JPG 原图。"""

import argparse
from datetime import datetime
import glob
import os
from pathlib import Path
import time

import cv2


INTEGRATED_CAMERA_KEYWORDS = (
    'integrated',
    'internal camera',
    'built-in',
    'builtin',
)


def camera_name(device: str) -> str:
    video_name = os.path.basename(os.path.realpath(device))
    name_path = Path('/sys/class/video4linux') / video_name / 'name'
    try:
        return name_path.read_text(encoding='utf-8').strip()
    except OSError:
        return ''


def is_integrated_camera(device: str) -> bool:
    name = camera_name(device).lower()
    return any(keyword in name for keyword in INTEGRATED_CAMERA_KEYWORDS)


def camera_candidates(device: str):
    if device.lower() != 'auto':
        if is_integrated_camera(device):
            raise RuntimeError(
                f'refusing integrated camera {device} ({camera_name(device)})')
        return [device]

    paths = sorted(glob.glob('/dev/v4l/by-id/*-video-index0'))
    paths += sorted(glob.glob('/dev/v4l/by-id/*'))
    paths += sorted(glob.glob('/dev/video*'))
    result = []
    real_devices = set()
    for path in paths:
        real = os.path.realpath(path)
        if is_integrated_camera(path):
            continue
        if path in result or real in real_devices:
            continue
        result.append(path)
        real_devices.add(real)
    return result


def open_camera(device: str, width: int, height: int, fps: float):
    failed = []
    for candidate in camera_candidates(device):
        source = int(candidate) if candidate.isdigit() else candidate
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            failed.append(candidate)
            continue

        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture, candidate
        capture.release()
        failed.append(candidate)

    raise RuntimeError(
        f'cannot open USB RGB camera {device}; '
        f'failed candidates: {failed or ["none"]}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preview the USB RGB camera and save raw JPG images.')
    parser.add_argument(
        '--device', default='auto',
        help='auto, /dev/video0, or a /dev/v4l/by-id path')
    parser.add_argument(
        '--output', default='~/ARES_LED/camera_data/usb_rgb/raw',
        help='directory in which JPG images are saved')
    parser.add_argument(
        '--interval', type=float, default=2.0,
        help='save interval in seconds, default: 2')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--jpeg-quality', type=int, default=95)
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
    capture, selected_device = open_camera(
        args.device, args.width, args.height, args.fps)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    print(f'camera: {selected_device} ({camera_name(selected_device)})')
    print(f'format: {actual_width}x{actual_height} @ {actual_fps:.1f} Hz')
    print(f'output: {output_dir}')
    print(f'saving every {args.interval:g} seconds; press q/Esc/Ctrl+C to stop')

    if not args.no_preview:
        cv2.namedWindow('USB RGB capture', cv2.WINDOW_NORMAL)

    count = 0
    next_save = time.monotonic()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print('warning: failed to read a camera frame')
                time.sleep(0.05)
                continue

            now = time.monotonic()
            if now >= next_save:
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                path = output_dir / f'usb_rgb_{stamp}_{count:06d}.jpg'
                saved = cv2.imwrite(
                    str(path), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not saved:
                    raise RuntimeError(f'failed to save {path}')
                count += 1
                next_save = now + args.interval
                print(f'[{count}] saved {path}')

            if not args.no_preview:
                preview = frame.copy()
                remaining = max(0.0, next_save - time.monotonic())
                cv2.putText(
                    preview,
                    f'USB RGB  saved: {count}  next: {remaining:.1f}s  q/Esc: quit',
                    (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
                cv2.imshow('USB RGB capture', preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if not args.no_preview:
            cv2.destroyAllWindows()
        print(f'stopped; saved {count} images in {output_dir}')


if __name__ == '__main__':
    main()
