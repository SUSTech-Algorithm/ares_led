"""根据已标注的单色图像生成仅供 R2 使用的色相模型。"""

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
from typing import Dict, Tuple

import cv2
import numpy as np
import yaml

from .classifier import ColorModel, detect_candidates, load_config, select_winner
from .profiles import (
    CAMERA_PROFILES,
    DEFAULT_CAMERA_PROFILE,
    dataset_path,
    detector_config_path,
)


CLASSES = ('RED', 'GREEN', 'CYAN', 'BLUE', 'PURPLE')


def _save_validated_config(
        config: Dict, calibrated: Dict, output: Path) -> None:
    """仅以原子方式发布已经通过自检的检测器配置。"""
    published = dict(config)
    published['calibrated'] = True
    published['colors'] = calibrated
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=output.parent,
                prefix=f'.{output.name}.',
                suffix='.tmp',
                delete=False) as stream:
            temporary_path = Path(stream.name)
            yaml.safe_dump(
                published, stream, sort_keys=False, allow_unicode=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def hue_distance(values, center):
    delta = np.abs(values.astype(np.float32) - center)
    return np.minimum(delta, 180.0 - delta)


def _bounded(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))


def _normalized_bgr(pixels: np.ndarray) -> Dict[str, np.ndarray]:
    values = pixels.astype(np.float32)
    total = np.maximum(np.sum(values, axis=1), 1.0)
    return {
        'b': values[:, 0] / total,
        'g': values[:, 1] / total,
        'r': values[:, 2] / total,
    }


def _channel_constraints(name: str, ratios: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
    def lo(channel: str) -> float:
        return round(_bounded(np.percentile(ratios[channel], 5) - 0.025, 0.0, 1.0), 3)

    def hi(channel: str) -> float:
        return round(_bounded(np.percentile(ratios[channel], 95) + 0.035, 0.0, 1.0), 3)

    if name == 'GREEN':
        return {'g': max(lo('g'), 0.45)}, {'b': min(hi('b'), 0.42)}
    if name == 'CYAN':
        return (
            {'b': max(lo('b'), 0.40), 'g': lo('g')},
            {'r': hi('r'), 'g': 0.55, 'b': 0.70},
        )
    if name == 'BLUE':
        return {'b': max(lo('b'), 0.50)}, {'g': min(hi('g'), 0.40)}
    return {}, {}


def _collect_color_pixels(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'cannot read {path}')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    smooth = cv2.GaussianBlur(value, (0, 0), 0.8)
    broad = cv2.GaussianBlur(value, (0, 0), 2.6)
    dog = smooth - broad
    mask = (
        (hsv[:, :, 1] >= 55) &
        (hsv[:, :, 2] >= 45) &
        (dog >= max(3.0, float(np.percentile(dog, 92))))
    )
    if np.count_nonzero(mask) < 20:
        mask = (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 55)
    return hsv[:, :, 0][mask], hsv[:, :, 1][mask], image[mask]


def _prepare_detection_image(
        image: np.ndarray, processing_scale: float) -> np.ndarray:
    """按部署时的比例缩放校准自检图像。"""
    scale = float(np.clip(processing_scale, 0.1, 1.0))
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def calibrate(dataset: Path, base_config: Path, output: Path) -> None:
    with base_config.open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    calibrated = {}
    summaries = {}
    for name in CLASSES:
        hue_rows = []
        saturation_rows = []
        bgr_rows = []
        for path in sorted((dataset / name).glob('*.jpg')):
            hues, saturations, bgr_pixels = _collect_color_pixels(path)
            if len(hues) == 0:
                continue
            hue_rows.append(hues)
            saturation_rows.append(saturations)
            bgr_rows.append(bgr_pixels)
        if not hue_rows:
            raise RuntimeError(f'no calibration pixels found for {name}')
        hues = np.concatenate(hue_rows)
        saturations = np.concatenate(saturation_rows)
        bgr_pixels = np.concatenate(bgr_rows)
        values = cv2.cvtColor(
            bgr_pixels.reshape((-1, 1, 3)),
            cv2.COLOR_BGR2HSV).reshape((-1, 3))[:, 2]
        weights = (
            saturations.astype(np.float32) *
            values.astype(np.float32))
        histogram = np.bincount(hues, weights=weights, minlength=180)
        peak = int(np.argmax(histogram))
        local = hue_distance(hues, peak) <= 24.0
        angles = hues[local].astype(np.float32) * (2.0 * np.pi / 180.0)
        local_weights = weights[local]
        angle = np.arctan2(
            np.sum(np.sin(angles) * local_weights),
            np.sum(np.cos(angles) * local_weights))
        center = float((angle * 180.0 / (2.0 * np.pi)) % 180.0)
        local_hue_distances = hue_distance(hues[local], center)
        radius = _bounded(np.percentile(local_hue_distances, 98), 6.0, 22.0)
        local_saturations = saturations[local]
        local_values = values[local]
        ratios = _normalized_bgr(bgr_pixels[local])
        channel_min, channel_max = _channel_constraints(name, ratios)
        calibrated[name] = {
            'hue_center': round(center, 2),
            'hue_radius': round(radius, 2),
            'min_saturation': round(
                _bounded(np.percentile(local_saturations, 5) - 8.0, 55.0, 150.0), 1),
            'min_value': round(
                _bounded(np.percentile(local_values, 5) - 8.0, 40.0, 110.0), 1),
        }
        if channel_min:
            calibrated[name]['channel_min'] = channel_min
        if channel_max:
            calibrated[name]['channel_max'] = channel_max
        summaries[name] = {
            'images': len(hue_rows),
            'pixels': int(len(hues)),
            'peak': peak,
            'center': center,
            'radius': radius,
            'hue_p05_p50_p95': [
                round(float(value), 2)
                for value in np.percentile(hues[local], (5, 50, 95))
            ],
            'sat_p05_p50': [
                round(float(value), 1)
                for value in np.percentile(local_saturations, (5, 50))
            ],
            'val_p05_p50': [
                round(float(value), 1)
                for value in np.percentile(local_values, (5, 50))
            ],
            'bgr_ratio_p05_p50_p95': {
                key: [
                    round(float(value), 3)
                    for value in np.percentile(channel, (5, 50, 95))
                ]
                for key, channel in ratios.items()
            },
        }
    base = load_config(str(base_config))
    proposed = replace(base, colors=tuple(
        ColorModel(name=name, **{
            key: float(value) for key, value in calibrated[name].items()
            if key not in {'name', 'channel_min', 'channel_max'}
        }, **{
            key: value for key, value in calibrated[name].items()
            if key in {'channel_min', 'channel_max'}
        }) for name in CLASSES))
    valid = True
    rejected_by = None
    for expected in (*CLASSES, 'NONE'):
        for path in sorted((dataset / expected).glob('*.jpg')):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'cannot read {path}')
            work = _prepare_detection_image(
                image, proposed.processing_scale)
            candidates = detect_candidates(work, proposed)
            winner = select_winner(candidates, proposed)
            if expected == 'NONE':
                valid = not candidates
            else:
                margin = math.inf
                if len(candidates) > 1:
                    margin = candidates[0].score / max(candidates[1].score, 1e-9)
                valid = (
                    winner is not None and winner.color == expected and
                    all(item.color == expected for item in candidates) and
                    margin >= 1.10
                )
            if not valid:
                rejected_by = path
                break
        if not valid:
            break
    if not valid:
        raise RuntimeError(
            f'proposed model failed self-check at {rejected_by}; '
            f'active detector config was not changed')
    _save_validated_config(config, calibrated, output)
    for name in CLASSES:
        summary = summaries[name]
        print(
            f'{name}: images={summary["images"]} pixels={summary["pixels"]} '
            f'peak={summary["peak"]} center={summary["center"]:.2f} '
            f'radius={summary["radius"]:.2f} '
            f'hue_p05/p50/p95={summary["hue_p05_p50_p95"]} '
            f'sat_p05/p50={summary["sat_p05_p50"]} '
            f'val_p05/p50={summary["val_p05_p50"]} '
            f'bgr_ratio={summary["bgr_ratio_p05_p50_p95"]}')
    print(f'wrote camera color model: {output}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Calibrate R2 camera colour models')
    parser.add_argument(
        '--camera-profile',
        choices=CAMERA_PROFILES,
        default=DEFAULT_CAMERA_PROFILE,
        help='选择独立的数据集和 detector 配置。')
    parser.add_argument(
        '--dataset', type=Path,
        help='覆盖默认数据集路径；默认 camera_data/<camera-profile>。')
    parser.add_argument(
        '--base-config', type=Path,
        help='覆盖默认 detector；默认 config/cameras/<camera-profile>/detector.yaml。')
    parser.add_argument(
        '--output', type=Path,
        help='输出路径；默认覆盖当前 profile 的 detector.yaml。')
    args = parser.parse_args()
    config = args.base_config or detector_config_path(args.camera_profile)
    output = args.output or config
    dataset = args.dataset or dataset_path(args.camera_profile)
    calibrate(dataset.resolve(), config.resolve(), output.resolve())


if __name__ == '__main__':
    main()
