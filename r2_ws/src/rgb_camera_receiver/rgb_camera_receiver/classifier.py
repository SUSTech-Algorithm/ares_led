"""与协议无关的单色可寻址 LED 灯带检测。

检测器特意将摄像头侧的颜色观测与 R1 协议分离。有效灯带不能仅是一条彩色线：
它必须由一列紧凑且颜色相近的光点构成，点间距需符合透视规律，相邻光点之间还应有
可见的亮度谷值。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ColorModel:
    name: str
    hue_center: float
    hue_radius: float
    min_saturation: float
    min_value: float
    channel_min: Mapping[str, float] = None
    channel_max: Mapping[str, float] = None


@dataclass(frozen=True)
class DetectorConfig:
    colors: Tuple[ColorModel, ...]
    processing_scale: float = 1.0
    min_blob_area: int = 1
    max_blob_area: int = 300
    max_blob_aspect: float = 3.5
    min_blob_compactness: float = 0.12
    dog_sigma_small: float = 0.8
    dog_sigma_large: float = 2.6
    min_dog_response: float = 5.0
    max_points_per_color: int = 160
    max_pair_hypotheses: int = 2000
    max_line_hypotheses: int = 20
    min_dots: int = 6
    min_length_pixels: float = 35.0
    line_distance_pixels: float = 3.5
    max_gap_ratio: float = 3.2
    max_spacing_trend_error: float = 0.32
    min_coverage: float = 0.55
    min_valley_contrast: float = 0.10
    min_periodic_dot_quality: float = 0.38
    min_periodic_color_quality: float = 0.45
    continuous_min_area: int = 30
    continuous_min_length: float = 80.0
    continuous_min_aspect: float = 8.0
    continuous_min_color_quality: float = 0.75
    min_score: float = 0.04
    winner_margin: float = 1.0


@dataclass(frozen=True)
class StripDetection:
    color: str
    confidence: float
    score: float
    corners: np.ndarray
    dot_count: int
    length: float
    residual: float
    spacing_cv: float
    line_quality: float = 0.0
    dot_quality: float = 0.0
    periodic_quality: float = 0.0
    color_quality: float = 0.0
    valley_quality: float = 0.0
    peak_centers: Optional[np.ndarray] = None
    mode: str = 'periodic'

    def scaled(self, factor: float) -> 'StripDetection':
        peaks = None if self.peak_centers is None else self.peak_centers * factor
        return StripDetection(
            color=self.color,
            confidence=self.confidence,
            score=self.score,
            corners=self.corners * factor,
            dot_count=self.dot_count,
            length=self.length * factor,
            residual=self.residual * factor,
            spacing_cv=self.spacing_cv,
            line_quality=self.line_quality,
            dot_quality=self.dot_quality,
            periodic_quality=self.periodic_quality,
            color_quality=self.color_quality,
            valley_quality=self.valley_quality,
            peak_centers=peaks,
            mode=self.mode,
        )


@dataclass(frozen=True)
class _LightPoint:
    center: np.ndarray
    radius: float
    response: float
    shape_quality: float
    color_quality: float

    @property
    def quality(self) -> float:
        response_quality = max(0.0, min(self.response / 28.0, 1.0))
        return float(np.sqrt(
            max(self.shape_quality, 0.0) *
            max(response_quality, 0.0) *
            max(self.color_quality, 0.0)))


@dataclass(frozen=True)
class _ColorComponent:
    center: np.ndarray
    axis: np.ndarray
    length: float
    width: float
    area: int
    aspect: float
    corners: np.ndarray
    color_quality: float


def load_config(path: str) -> DetectorConfig:
    with Path(path).open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    color_rows = raw.get('colors', {})
    colors = tuple(
        ColorModel(
            name=str(name).upper(),
            hue_center=float(values['hue_center']),
            hue_radius=float(values['hue_radius']),
            min_saturation=float(values.get('min_saturation', 80)),
            min_value=float(values.get('min_value', 60)),
            channel_min={
                str(channel).lower(): float(value)
                for channel, value in values.get('channel_min', {}).items()
            },
            channel_max={
                str(channel).lower(): float(value)
                for channel, value in values.get('channel_max', {}).items()
            },
        )
        for name, values in color_rows.items()
    )
    if not colors:
        raise ValueError('detector config must define at least one color')
    dots = raw.get('dots', {})
    geometry = raw.get('geometry', {})
    selection = raw.get('selection', {})
    processing = raw.get('processing', {})
    return DetectorConfig(
        colors=colors,
        processing_scale=float(np.clip(
            processing.get('scale', raw.get('processing_scale', 1.0)),
            0.1, 1.0)),
        min_blob_area=int(dots.get('min_blob_area', 1)),
        max_blob_area=int(dots.get('max_blob_area', 300)),
        max_blob_aspect=float(dots.get('max_blob_aspect', 3.5)),
        min_blob_compactness=float(dots.get('min_blob_compactness', 0.12)),
        dog_sigma_small=float(dots.get('dog_sigma_small', 0.8)),
        dog_sigma_large=float(dots.get('dog_sigma_large', 2.6)),
        min_dog_response=float(dots.get('min_dog_response', 5.0)),
        max_points_per_color=int(dots.get('max_points_per_color', 160)),
        max_pair_hypotheses=int(geometry.get('max_pair_hypotheses', 2000)),
        max_line_hypotheses=int(geometry.get('max_line_hypotheses', 20)),
        min_dots=int(geometry.get('min_dots', 6)),
        min_length_pixels=float(geometry.get('min_length_pixels', 35)),
        line_distance_pixels=float(geometry.get('line_distance_pixels', 3.5)),
        max_gap_ratio=float(geometry.get('max_gap_ratio', 3.2)),
        max_spacing_trend_error=float(
            geometry.get('max_spacing_trend_error', 0.32)),
        min_coverage=float(geometry.get('min_coverage', 0.55)),
        min_valley_contrast=float(
            geometry.get('min_valley_contrast', 0.10)),
        min_periodic_dot_quality=float(
            geometry.get('min_periodic_dot_quality', 0.38)),
        min_periodic_color_quality=float(
            geometry.get('min_periodic_color_quality', 0.45)),
        continuous_min_area=int(
            geometry.get('continuous_min_area', 30)),
        continuous_min_length=float(
            geometry.get('continuous_min_length', 80)),
        continuous_min_aspect=float(
            geometry.get('continuous_min_aspect', 8.0)),
        continuous_min_color_quality=float(
            geometry.get('continuous_min_color_quality', 0.75)),
        min_score=float(selection.get('min_score', 0.04)),
        winner_margin=float(selection.get('winner_margin', 1.0)),
    )


def _hue_distance(hue: np.ndarray, center: float) -> np.ndarray:
    delta = np.abs(hue.astype(np.float32) - center)
    return np.minimum(delta, 180.0 - delta)


def _normalized_channels(bgr: np.ndarray) -> Dict[str, np.ndarray]:
    values = bgr.astype(np.float32)
    total = np.maximum(np.sum(values, axis=2), 1.0)
    return {
        'b': values[:, :, 0] / total,
        'g': values[:, :, 1] / total,
        'r': values[:, :, 2] / total,
    }


def _channel_constraints_mask(
        bgr: np.ndarray,
        model: ColorModel) -> np.ndarray:
    if not model.channel_min and not model.channel_max:
        return np.ones(bgr.shape[:2], dtype=bool)
    channels = _normalized_channels(bgr)
    mask = np.ones(bgr.shape[:2], dtype=bool)
    for name, minimum in (model.channel_min or {}).items():
        if name not in channels:
            continue
        mask &= channels[name] >= minimum
    for name, maximum in (model.channel_max or {}).items():
        if name not in channels:
            continue
        mask &= channels[name] <= maximum
    return mask


def _channel_constraints_quality(
        bgr_pixels: np.ndarray,
        model: ColorModel) -> float:
    if not model.channel_min and not model.channel_max:
        return 1.0
    if len(bgr_pixels) == 0:
        return 0.0
    values = bgr_pixels.astype(np.float32)
    total = np.maximum(np.sum(values, axis=1), 1.0)
    channels = {
        'b': values[:, 0] / total,
        'g': values[:, 1] / total,
        'r': values[:, 2] / total,
    }
    ok = np.ones(len(values), dtype=bool)
    margins = []
    for name, minimum in (model.channel_min or {}).items():
        if name not in channels:
            continue
        channel = channels[name]
        ok &= channel >= minimum
        margins.append(np.clip((channel - minimum) / 0.08, 0.0, 1.0))
    for name, maximum in (model.channel_max or {}).items():
        if name not in channels:
            continue
        channel = channels[name]
        ok &= channel <= maximum
        margins.append(np.clip((maximum - channel) / 0.08, 0.0, 1.0))
    if not margins:
        return 1.0
    pass_fraction = float(np.count_nonzero(ok)) / len(ok)
    margin_quality = float(np.median(np.min(np.stack(margins, axis=1), axis=1)))
    return float(np.clip(0.65 * pass_fraction + 0.35 * margin_quality, 0.0, 1.0))


def color_masks(
        hsv: np.ndarray,
        bgr: np.ndarray,
        config: DetectorConfig) -> Dict[str, np.ndarray]:
    hue_axis = np.arange(180, dtype=np.float32)
    normalized_distances = np.stack([
        _hue_distance(hue_axis, model.hue_center) /
        max(model.hue_radius, 1e-6)
        for model in config.colors
    ], axis=1)
    hue_owner_lut = np.argmin(normalized_distances, axis=1).astype(np.uint8)
    hue_owner = hue_owner_lut[hsv[:, :, 0]]
    masks: Dict[str, np.ndarray] = {}
    for model_index, model in enumerate(config.colors):
        low = model.hue_center - model.hue_radius
        high = model.hue_center + model.hue_radius
        saturation = int(np.clip(model.min_saturation, 0, 255))
        value = int(np.clip(model.min_value, 0, 255))
        if low < 0:
            first = cv2.inRange(
                hsv, (0, saturation, value),
                (int(np.floor(high)), 255, 255))
            second = cv2.inRange(
                hsv, (int(np.ceil(180 + low)), saturation, value),
                (179, 255, 255))
            mask = cv2.bitwise_or(first, second)
        elif high >= 180:
            first = cv2.inRange(
                hsv, (int(np.ceil(low)), saturation, value),
                (179, 255, 255))
            second = cv2.inRange(
                hsv, (0, saturation, value),
                (int(np.floor(high - 180)), 255, 255))
            mask = cv2.bitwise_or(first, second)
        else:
            mask = cv2.inRange(
                hsv, (int(np.ceil(low)), saturation, value),
                (int(np.floor(high)), 255, 255))
        masks[model.name] = (
            mask.astype(bool) &
            (hue_owner == model_index) &
            _channel_constraints_mask(bgr, model)
        )
    return masks


def _point_color_quality(
        hsv: np.ndarray,
        bgr: np.ndarray,
        center: np.ndarray,
        model: ColorModel,
        all_colors: Sequence[ColorModel],
        radius: int) -> float:
    height, width = hsv.shape[:2]
    x, y = np.round(center).astype(int)
    radius = max(2, min(radius, 8))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = hsv[y0:y1, x0:x1]
    bgr_patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
    disk = xx * xx + yy * yy <= radius * radius
    pixels = patch[disk]
    bright = pixels[pixels[:, 2] >= model.min_value]
    if len(bright) == 0:
        return 0.0
    saturated = bright[bright[:, 1] >= model.min_saturation]
    if len(saturated) == 0:
        return 0.0
    distances = np.stack([
        _hue_distance(saturated[:, 0], item.hue_center) /
        max(item.hue_radius, 1e-6)
        for item in all_colors
    ], axis=1)
    nearest = np.argmin(distances, axis=1)
    own_index = list(all_colors).index(model)
    own = (nearest == own_index) & (distances[:, own_index] <= 1.0)
    if not np.any(own):
        return 0.0
    channel_quality = _channel_constraints_quality(bgr_patch[disk][own], model)
    if channel_quality <= 0.0:
        return 0.0
    hue_quality = 1.0 - float(np.median(
        np.clip(distances[own, own_index], 0.0, 1.0)))
    purity = float(np.count_nonzero(own)) / len(saturated)
    # 白色灯具通常只有一圈很薄的彩色边缘。要求峰值周围有足够比例的彩色像素，
    # 可以排除这种误检，同时仍允许 LED 中心因过曝而呈白色。
    colored_fraction = min(1.0, len(saturated) / max(len(bright) * 0.45, 1.0))
    saturation_quality = min(
        1.0, float(np.median(saturated[own, 1])) / 180.0)
    return float(np.clip(
        purity * colored_fraction *
        (0.65 * hue_quality + 0.35 * saturation_quality) *
        channel_quality,
        0.0, 1.0))


def _extract_light_points(
        hsv: np.ndarray,
        bgr: np.ndarray,
        mask: np.ndarray,
        smooth: np.ndarray,
        dog: np.ndarray,
        model: ColorModel,
        config: DetectorConfig,
        recover_merged: bool = True) -> List[_LightPoint]:
    response_mask = mask & (dog >= config.min_dog_response)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        response_mask.astype(np.uint8), 8)
    points: List[_LightPoint] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < config.min_blob_area or area > config.max_blob_area:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > config.max_blob_aspect:
            continue
        ys, xs = np.nonzero(labels[y:y + height, x:x + width] == label)
        if len(xs) == 0:
            continue
        xs = xs.astype(np.float32) + x
        ys = ys.astype(np.float32) + y
        weights = np.maximum(dog[ys.astype(int), xs.astype(int)], 1e-3)
        center = np.array([
            np.average(xs, weights=weights),
            np.average(ys, weights=weights),
        ], dtype=np.float32)
        if len(xs) >= 3:
            coordinates = np.column_stack((xs, ys))
            covariance = np.cov(coordinates, rowvar=False, aweights=weights)
            eigenvalues = np.linalg.eigvalsh(covariance)
            compactness = float(
                (eigenvalues[0] + 0.35) / (eigenvalues[-1] + 0.35))
        else:
            compactness = 1.0 / aspect
        if compactness < config.min_blob_compactness:
            continue
        radius = max(1.0, float(np.sqrt(area / np.pi)))
        response = float(np.percentile(weights, 80))
        color_quality = _point_color_quality(
            hsv, bgr, center, model, config.colors,
            int(np.ceil(radius + 2.0)))
        if color_quality <= 0.0:
            continue
        points.append(_LightPoint(
            center=center,
            radius=radius,
            response=response,
            shape_quality=float(np.sqrt(compactness)),
            color_quality=color_quality,
        ))
    if recover_merged:
        # 光晕可能把相邻 LED 连成一个区域。仅当更快的连续灯带路径尚未解释同一颜色时，
        # 才根据局部极大值恢复各个灯珠。
        local_maxima = (
            (smooth >= cv2.dilate(
                smooth, np.ones((7, 7), np.uint8)) - 1e-4) &
            (dog >= config.min_dog_response)
        )
        peak_count, _, _, peak_centroids = cv2.connectedComponentsWithStats(
            local_maxima.astype(np.uint8), 8)
        image_height, image_width = dog.shape
        for label in range(1, peak_count):
            center = peak_centroids[label].astype(np.float32)
            cx, cy = np.round(center).astype(int)
            radius = 4
            x0, x1 = max(0, cx - radius), min(
                image_width, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(
                image_height, cy + radius + 1)
            patch = dog[y0:y1, x0:x1]
            peak = float(dog[cy, cx])
            footprint = patch >= max(
                config.min_dog_response, peak * 0.30)
            ys, xs = np.nonzero(footprint)
            if len(xs) >= 3:
                coordinates = np.column_stack((
                    xs.astype(np.float32) + x0,
                    ys.astype(np.float32) + y0,
                ))
                covariance = np.cov(coordinates, rowvar=False)
                eigenvalues = np.linalg.eigvalsh(covariance)
                compactness = float(
                    (eigenvalues[0] + 0.35) /
                    (eigenvalues[-1] + 0.35))
            else:
                compactness = 1.0
            if compactness < config.min_blob_compactness:
                continue
            color_quality = _point_color_quality(
                hsv, bgr, center, model, config.colors, radius)
            if color_quality <= 0.0:
                continue
            points.append(_LightPoint(
                center=center,
                radius=max(
                    1.0, float(np.sqrt(max(len(xs), 1) / np.pi))),
                response=peak,
                shape_quality=float(np.sqrt(compactness)),
                color_quality=color_quality,
            ))
    points.sort(key=lambda item: item.quality, reverse=True)
    unique: List[_LightPoint] = []
    for point in points:
        if any(
                np.linalg.norm(point.center - old.center) <
                max(2.5, min(point.radius + old.radius, 4.0))
                for old in unique):
            continue
        unique.append(point)
        if len(unique) >= config.max_points_per_color:
            break
    return unique


def _extract_color_components(
        hsv: np.ndarray,
        bgr: np.ndarray,
        mask: np.ndarray,
        model: ColorModel,
        config: DetectorConfig) -> List[_ColorComponent]:
    mask_u8 = mask.astype(np.uint8)
    nonzero = cv2.findNonZero(mask_u8)
    if nonzero is None:
        return []
    offset_x, offset_y, crop_width, crop_height = cv2.boundingRect(nonzero)
    cropped_mask = mask_u8[
        offset_y:offset_y + crop_height,
        offset_x:offset_x + crop_width]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cropped_mask, 8)
    components: List[_ColorComponent] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 5:
            continue
        x, y, width, height = stats[label, :4]
        ys, xs = np.nonzero(
            labels[y:y + height, x:x + width] == label)
        ys = ys + y + offset_y
        xs = xs + x + offset_x
        coordinates = np.column_stack((xs, ys)).astype(np.float32)
        center = np.mean(coordinates, axis=0)
        centered = coordinates - center
        if len(coordinates) < 3:
            continue
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0].astype(np.float32)
        if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
            axis = -axis
        normal = np.array((-axis[1], axis[0]), dtype=np.float32)
        along = centered @ axis
        across = centered @ normal
        low, high = float(np.min(along)), float(np.max(along))
        length = high - low
        half_width = max(1.0, float(np.percentile(np.abs(across), 90)))
        width = half_width * 2.0
        aspect = length / max(width, 1.0)
        pixels = hsv[ys, xs]
        distances = _hue_distance(
            pixels[:, 0], model.hue_center) / max(model.hue_radius, 1e-6)
        hue_quality = max(
            0.0, 1.0 - float(np.median(np.clip(distances, 0.0, 1.0))))
        saturation_quality = min(
            1.0, float(np.median(pixels[:, 1])) / 180.0)
        channel_quality = _channel_constraints_quality(bgr[ys, xs], model)
        color_quality = float(np.sqrt(
            max(hue_quality * saturation_quality * channel_quality, 0.0)))
        corners = np.array([
            center + axis * low - normal * (half_width + 1.0),
            center + axis * high - normal * (half_width + 1.0),
            center + axis * high + normal * (half_width + 1.0),
            center + axis * low + normal * (half_width + 1.0),
        ], dtype=np.float32)
        components.append(_ColorComponent(
            center=center,
            axis=axis,
            length=length,
            width=width,
            area=area,
            aspect=aspect,
            corners=corners,
            color_quality=color_quality,
        ))
    components.sort(key=lambda item: item.length, reverse=True)
    return components


def _component_light_points(
        components: Sequence[_ColorComponent],
        dog: np.ndarray,
        config: DetectorConfig) -> List[_LightPoint]:
    points = []
    height, width = dog.shape
    for component in components:
        if component.area > config.max_blob_area:
            continue
        if component.aspect > config.max_blob_aspect:
            continue
        x, y = np.round(component.center).astype(int)
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        response = float(dog[y, x])
        if response < config.min_dog_response:
            continue
        shape_quality = float(np.sqrt(
            min(1.0, 1.0 / max(component.aspect, 1.0))))
        points.append(_LightPoint(
            center=component.center,
            radius=max(1.0, float(np.sqrt(component.area / np.pi))),
            response=response,
            shape_quality=shape_quality,
            color_quality=component.color_quality,
        ))
    points.sort(key=lambda item: item.quality, reverse=True)
    return points[:config.max_points_per_color]


def _continuous_proposals(
        components: Sequence[_ColorComponent],
        color: str,
        config: DetectorConfig) -> List[StripDetection]:
    proposals = []
    for component in components:
        if component.area < config.continuous_min_area:
            continue
        if component.length < config.continuous_min_length:
            continue
        if component.aspect < config.continuous_min_aspect:
            continue
        if component.color_quality < config.continuous_min_color_quality:
            continue
        line_quality = float(np.clip(
            (component.aspect - config.continuous_min_aspect) /
            max(18.0 - config.continuous_min_aspect, 1e-6),
            0.0, 1.0))
        length_quality = float(np.clip(
            (component.length - config.continuous_min_length) / 100.0,
            0.0, 1.0))
        support_quality = float(np.sqrt(max(length_quality, 0.04)))
        score = float(
            (0.35 + 0.65 * line_quality) *
            support_quality *
            component.color_quality)
        proposals.append(StripDetection(
            color=color,
            confidence=float(np.sqrt(max(
                component.color_quality *
                (0.35 + 0.65 * line_quality), 0.0))),
            score=score,
            corners=component.corners,
            dot_count=0,
            length=component.length,
            residual=component.width * 0.5,
            spacing_cv=0.0,
            line_quality=line_quality,
            dot_quality=1.0,
            periodic_quality=support_quality,
            color_quality=component.color_quality,
            valley_quality=0.0,
            peak_centers=None,
            mode='continuous',
        ))
    return proposals


def _candidate_axis(candidate: StripDetection) -> Tuple[np.ndarray, np.ndarray]:
    start = (candidate.corners[0] + candidate.corners[3]) * 0.5
    end = (candidate.corners[1] + candidate.corners[2]) * 0.5
    axis = end - start
    axis /= max(float(np.linalg.norm(axis)), 1e-6)
    return start, axis


def _is_short_continuous_bar(
        candidate: StripDetection,
        components: Sequence[_ColorComponent],
        config: DetectorConfig) -> bool:
    start, axis = _candidate_axis(candidate)
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    candidate_interval = np.array((0.0, candidate.length))
    for component in components:
        if abs(float(axis @ component.axis)) < float(
                np.cos(np.deg2rad(15.0))):
            continue
        if abs(float((component.center - start) @ normal)) > max(
                8.0, component.width):
            continue
        component_center = float((component.center - start) @ axis)
        component_interval = np.array((
            component_center - component.length * 0.5,
            component_center + component.length * 0.5,
        ))
        overlap = min(candidate_interval[1], component_interval[1]) - max(
            candidate_interval[0], component_interval[0])
        if (
                component.length >= candidate.length * 0.60 and
                overlap >= candidate.length * 0.55):
            valid_continuous_strip = (
                component.area >= config.continuous_min_area and
                component.length >= config.continuous_min_length and
                component.aspect >= config.continuous_min_aspect and
                component.color_quality >=
                config.continuous_min_color_quality
            )
            return not valid_continuous_strip
    return False


def _line_hypotheses(
        points: Sequence[_LightPoint],
        config: DetectorConfig) -> List[np.ndarray]:
    if len(points) < config.min_dots:
        return []
    centers = np.asarray([item.center for item in points], dtype=np.float32)
    first, second = np.triu_indices(len(centers), 1)
    vectors = centers[second] - centers[first]
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.flatnonzero(lengths >= config.min_length_pixels)
    if len(valid) == 0:
        return []
    # 较长的点对更可能覆盖完整灯带。固定计算预算可使杂乱场景的结果保持确定，
    # 并避免旧版无上限的 O(n²) 循环。
    point_quality = np.asarray(
        [item.quality for item in points], dtype=np.float32)
    pair_quality = np.sqrt(
        point_quality[first] * point_quality[second])
    pair_rank = lengths * (0.35 + 0.65 * pair_quality)
    order = valid[np.argsort(pair_rank[valid])[::-1]]
    order = order[:config.max_pair_hypotheses]
    ranked: List[Tuple[float, int, float, np.ndarray]] = []
    for pair_index in order:
        start_index = first[pair_index]
        end_index = second[pair_index]
        axis = vectors[pair_index] / lengths[pair_index]
        normal = np.array((-axis[1], axis[0]), dtype=np.float32)
        relative = centers - centers[start_index]
        projection = relative @ axis
        distance = np.abs(relative @ normal)
        selected = np.flatnonzero(
            (distance <= config.line_distance_pixels) &
            (projection >= -config.line_distance_pixels) &
            (projection <= lengths[pair_index] + config.line_distance_pixels))
        if len(selected) < config.min_dots:
            continue
        qualities = np.asarray(
            [points[index].quality for index in selected], dtype=np.float32)
        density = len(selected) / max(float(lengths[pair_index]), 1.0)
        rank_score = float(
            len(selected) * np.median(qualities) * np.sqrt(density))
        ranked.append((
            rank_score, len(selected), float(lengths[pair_index]), selected))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    hypotheses: List[np.ndarray] = []
    seen = set()
    for _, _, _, selected in ranked:
        key = tuple(selected.tolist())
        if key in seen:
            continue
        seen.add(key)
        hypotheses.append(selected)
        if len(hypotheses) >= config.max_line_hypotheses:
            break
    return hypotheses


def _regular_chain(
        projections: np.ndarray,
        points: Sequence[_LightPoint],
        config: DetectorConfig) -> Optional[np.ndarray]:
    """从含噪直线中选择符合透视规律的清晰子序列。"""
    count = len(projections)
    if count < config.min_dots:
        return None
    best: Optional[Tuple[Tuple[float, ...], np.ndarray]] = None
    point_qualities = np.asarray(
        [item.quality for item in points], dtype=np.float32)
    # 量化后的短程差值能够显现重复的物理间距，无需尝试把每个可能点对都作为链起点。
    gap_histogram: Dict[float, int] = {}
    for offset in range(1, min(5, count)):
        for gap in projections[offset:] - projections[:-offset]:
            gap = float(gap)
            if gap < 2.0 or gap > config.min_length_pixels:
                continue
            quantized = round(gap * 2.0) / 2.0
            gap_histogram[quantized] = gap_histogram.get(quantized, 0) + 1
    seed_gaps = [
        row[0] for row in sorted(
            gap_histogram.items(), key=lambda row: (row[1], -row[0]),
            reverse=True)[:10]
    ]
    for initial_gap in seed_gaps:
        for first in range(count):
            chain = [first]
            gaps: List[float] = []
            last = first
            expected = initial_gap
            while last + 1 < count:
                if len(gaps) >= 2:
                    recent = gaps[-5:]
                    expected = float(
                        recent[-1] +
                        (recent[-1] - recent[0]) / max(len(recent) - 1, 1))
                expected = max(expected, 1.0)
                target = projections[last] + expected
                insertion = int(np.searchsorted(
                    projections, target, side='left'))
                choices = []
                for following in range(
                        max(last + 1, insertion - 3),
                        min(count, insertion + 4)):
                    gap = float(projections[following] - projections[last])
                    normalized_error = abs(gap - expected) / expected
                    if normalized_error > 0.46:
                        continue
                    choices.append((
                        normalized_error - 0.08 * point_qualities[following],
                        following,
                        gap,
                    ))
                if not choices:
                    break
                _, following, gap = min(choices)
                chain.append(following)
                gaps.append(gap)
                last = following
            if len(chain) < config.min_dots:
                continue
            chain_array = np.asarray(chain, dtype=np.int32)
            chain_gaps = np.diff(projections[chain_array])
            span = float(
                projections[chain_array[-1]] - projections[chain_array[0]])
            if span < config.min_length_pixels:
                continue
            if best is not None:
                if len(chain_array) < best[0][0]:
                    continue
            gap_axis = np.arange(len(chain_gaps), dtype=np.float32)
            centered_axis = gap_axis - float(np.mean(gap_axis))
            slope = float(
                (centered_axis @ chain_gaps) /
                max(centered_axis @ centered_axis, 1e-6))
            intercept = float(np.mean(chain_gaps) - slope * np.mean(gap_axis))
            predicted = intercept + slope * gap_axis
            if np.any(predicted <= 0.5):
                continue
            error = float(
                np.sqrt(np.mean(np.square(chain_gaps - predicted))) /
                max(float(np.median(chain_gaps)), 1e-6))
            mean_quality = float(np.mean(point_qualities[chain_array]))
            rank = (
                float(len(chain_array)),
                -error,
                span,
                mean_quality,
            )
            row = (rank, chain_array)
            if best is None or row[0] > best[0]:
                best = row
    if best is None:
        return None
    return best[1]


def _sample_image(image: np.ndarray, positions: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    rounded = np.round(positions).astype(int)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    return image[rounded[:, 1], rounded[:, 0]].astype(np.float32)


def _disambiguate_green_cyan_blue(
        color: str,
        bgr: np.ndarray,
        positions: np.ndarray) -> str:
    if color not in {'GREEN', 'CYAN', 'BLUE'}:
        return color
    pixels = _sample_image(bgr, positions)
    if len(pixels) == 0:
        return color
    total = np.maximum(np.sum(pixels, axis=1), 1.0)
    ratios = pixels / total[:, None]
    b_ratio, g_ratio, r_ratio = np.median(ratios, axis=0)
    if r_ratio > 0.28:
        return color
    if g_ratio > b_ratio * 1.03:
        return 'GREEN'
    if b_ratio >= 0.55 and b_ratio > g_ratio * 1.55:
        return 'BLUE'
    if r_ratio >= 0.075 and b_ratio > g_ratio * 1.04:
        return 'BLUE'
    if r_ratio >= 0.18 and b_ratio >= g_ratio * 0.95:
        return 'BLUE'
    if b_ratio >= 0.36 and g_ratio >= 0.28:
        return 'CYAN'
    return color


def _fit_candidate(
        points: Sequence[_LightPoint],
        indexes: np.ndarray,
        bgr: np.ndarray,
        dog: np.ndarray,
        config: DetectorConfig,
        color: str) -> Optional[StripDetection]:
    selected_points = [points[index] for index in indexes]
    selected = np.asarray(
        [item.center for item in selected_points], dtype=np.float32)
    center = np.mean(selected, axis=0)
    centered = selected - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    projection = centered @ axis
    residuals = np.abs(centered @ normal)
    order = np.argsort(projection)
    projection = projection[order]
    selected = selected[order]
    selected_points = [selected_points[index] for index in order]
    raw_gaps = np.diff(projection)
    raw_consistent = False
    if len(raw_gaps) >= config.min_dots - 1 and np.all(raw_gaps > 0.75):
        raw_axis = np.arange(len(raw_gaps), dtype=np.float32)
        centered_raw_axis = raw_axis - float(np.mean(raw_axis))
        raw_slope = float(
            (centered_raw_axis @ raw_gaps) /
            max(centered_raw_axis @ centered_raw_axis, 1e-6))
        raw_intercept = float(
            np.mean(raw_gaps) - raw_slope * np.mean(raw_axis))
        raw_predicted = raw_intercept + raw_slope * raw_axis
        raw_error = float(
            np.sqrt(np.mean(np.square(raw_gaps - raw_predicted))) /
            max(float(np.median(raw_gaps)), 1e-6))
        raw_consistent = (
            np.all(raw_predicted > 0.5) and
            float(np.max(raw_gaps) / np.min(raw_gaps)) <=
            config.max_gap_ratio and
            raw_error <= config.max_spacing_trend_error
        )
    if not raw_consistent:
        chain = _regular_chain(projection, selected_points, config)
        if chain is None:
            return None
        projection = projection[chain]
        selected = selected[chain]
        selected_points = [selected_points[index] for index in chain]
    center = np.mean(selected, axis=0)
    centered = selected - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0].astype(np.float32)
    if axis[0] < 0 or (abs(axis[0]) < 1e-6 and axis[1] < 0):
        axis = -axis
    normal = np.array((-axis[1], axis[0]), dtype=np.float32)
    projection = centered @ axis
    order = np.argsort(projection)
    projection = projection[order]
    selected = selected[order]
    selected_points = [selected_points[index] for index in order]
    residuals = np.abs(centered @ normal)
    residual = float(np.percentile(residuals, 90))
    length = float(projection[-1] - projection[0])
    if length < config.min_length_pixels:
        return None
    if residual > config.line_distance_pixels * 1.25:
        return None
    gaps = np.diff(projection)
    if len(gaps) < config.min_dots - 1 or np.any(gaps <= 0.75):
        return None
    median_gap = float(np.median(gaps))
    if float(np.max(gaps) / max(np.min(gaps), 1e-6)) > config.max_gap_ratio:
        return None
    gap_indexes = np.arange(len(gaps), dtype=np.float32)
    centered_gap_indexes = gap_indexes - float(np.mean(gap_indexes))
    slope = float(
        (centered_gap_indexes @ gaps) /
        max(centered_gap_indexes @ centered_gap_indexes, 1e-6))
    intercept = float(np.mean(gaps) - slope * np.mean(gap_indexes))
    predicted_gaps = intercept + slope * gap_indexes
    if np.any(predicted_gaps <= 0.5):
        return None
    trend_error = float(
        np.sqrt(np.mean(np.square(gaps - predicted_gaps))) /
        max(median_gap, 1e-6))
    if trend_error > config.max_spacing_trend_error:
        return None
    spacing_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
    expected = max(length / max(float(np.mean(gaps)), 1.0) + 1.0, 1.0)
    coverage = min(1.0, len(selected) / expected)
    if coverage < config.min_coverage:
        return None

    peak_response = _sample_image(dog, selected)
    midpoints = (selected[:-1] + selected[1:]) * 0.5
    valley_response = _sample_image(dog, midpoints)
    pair_peaks = np.minimum(peak_response[:-1], peak_response[1:])
    contrasts = (pair_peaks - valley_response) / np.maximum(pair_peaks, 1.0)
    valley_contrast = float(np.median(np.clip(contrasts, 0.0, 1.0)))
    if valley_contrast < config.min_valley_contrast:
        return None

    line_quality = float(np.exp(
        -np.square(residual / max(config.line_distance_pixels, 1e-6))))
    trend_quality = float(np.exp(
        -np.square(trend_error /
                   max(config.max_spacing_trend_error, 1e-6))))
    periodic_quality = float(np.clip(
        trend_quality * np.sqrt(coverage), 0.0, 1.0))
    dot_quality = float(np.median(
        [item.quality for item in selected_points]))
    color_values = np.asarray(
        [item.color_quality for item in selected_points], dtype=np.float32)
    color_quality = float(np.median(color_values))
    color_consistency = float(
        np.count_nonzero(color_values >= color_quality * 0.65) /
        len(color_values))
    color_quality *= color_consistency
    valley_quality = float(np.clip(
        (valley_contrast - config.min_valley_contrast) /
        max(0.55 - config.min_valley_contrast, 1e-6),
        0.0, 1.0))
    # 此处有意使用乘法：没有独立灯珠的彩色线，或只有彩色边缘的周期性白色结构，
    # 其得分必须保持较低。
    score = float(
        line_quality * dot_quality * periodic_quality *
        color_quality * valley_quality)
    half_width = max(
        3.0,
        residual + float(np.median(
            [item.radius for item in selected_points])) + 1.5)
    low, high = float(projection[0]), float(projection[-1])
    corners = np.array([
        center + axis * low - normal * half_width,
        center + axis * high - normal * half_width,
        center + axis * high + normal * half_width,
        center + axis * low + normal * half_width,
    ], dtype=np.float32)
    confidence = float(np.sqrt(max(color_quality * periodic_quality, 0.0)))
    color = _disambiguate_green_cyan_blue(color, bgr, selected)
    return StripDetection(
        color=color,
        confidence=confidence,
        score=score,
        corners=corners,
        dot_count=len(selected),
        length=length,
        residual=residual,
        spacing_cv=spacing_cv,
        line_quality=line_quality,
        dot_quality=dot_quality,
        periodic_quality=periodic_quality,
        color_quality=color_quality,
        valley_quality=valley_quality,
        peak_centers=selected,
        mode='periodic',
    )


def _overlap(first: StripDetection, second: StripDetection) -> float:
    box_a = cv2.boundingRect(np.round(first.corners).astype(np.int32))
    box_b = cv2.boundingRect(np.round(second.corners).astype(np.int32))
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(min(aw * ah, bw * bh), 1)


def _same_physical_strip(first: StripDetection, second: StripDetection) -> bool:
    def endpoints(item: StripDetection) -> np.ndarray:
        return np.array([
            (item.corners[0] + item.corners[3]) * 0.5,
            (item.corners[1] + item.corners[2]) * 0.5,
        ])

    first_ends = endpoints(first)
    second_ends = endpoints(second)
    first_axis = first_ends[1] - first_ends[0]
    second_axis = second_ends[1] - second_ends[0]
    first_axis /= max(float(np.linalg.norm(first_axis)), 1e-6)
    second_axis /= max(float(np.linalg.norm(second_axis)), 1e-6)
    if abs(float(first_axis @ second_axis)) < float(np.cos(np.deg2rad(20.0))):
        return False
    normal = np.array((-first_axis[1], first_axis[0]), dtype=np.float32)
    perpendicular = max(
        abs(float((second_ends[0] - first_ends[0]) @ normal)),
        abs(float((second_ends[1] - first_ends[0]) @ normal)),
    )
    if perpendicular > max(9.0, min(first.length, second.length) * 0.14):
        return False
    first_projection = np.sort((first_ends - first_ends[0]) @ first_axis)
    second_projection = np.sort((second_ends - first_ends[0]) @ first_axis)
    overlap = min(first_projection[1], second_projection[1]) - max(
        first_projection[0], second_projection[0])
    if overlap >= min(first.length, second.length) * 0.25:
        return True
    gap = max(first_projection[0], second_projection[0]) - min(
        first_projection[1], second_projection[1])
    # 曝光和白平衡可能把一条物理单色灯带拆成相邻的色相区域。
    # 若共线间隙不超过较短区域的大约一倍，则视为同一灯带，并仅保留更强的颜色类别。
    return gap <= max(25.0, min(first.length, second.length) * 2.50)


def detect_proposals(
        bgr: np.ndarray,
        config: DetectorConfig) -> List[StripDetection]:
    """返回应用最终分数阈值前结构有效的候选。"""
    if bgr is None or bgr.size == 0:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    smooth = cv2.GaussianBlur(
        value, (0, 0), config.dog_sigma_small)
    broad = cv2.GaussianBlur(
        value, (0, 0), config.dog_sigma_large)
    dog = smooth - broad
    masks = color_masks(hsv, bgr, config)
    raw: List[StripDetection] = []
    for model in config.colors:
        components = _extract_color_components(
            hsv, bgr, masks[model.name], model, config)
        continuous = _continuous_proposals(
            components, model.name, config)
        raw.extend(continuous)
        if continuous:
            continue
        points = _component_light_points(components, dog, config)
        for indexes in _line_hypotheses(points, config):
            candidate = _fit_candidate(
                points, indexes, bgr, dog, config, model.name)
            if candidate is None:
                continue
            if candidate.dot_quality < config.min_periodic_dot_quality:
                continue
            if candidate.color_quality < config.min_periodic_color_quality:
                continue
            if _is_short_continuous_bar(candidate, components, config):
                continue
            raw.append(candidate)
    return _deduplicate_candidates(raw)


def _deduplicate_candidates(
        candidates: Sequence[StripDetection]) -> List[StripDetection]:
    """合并同一单色段的重复检测，同时保留协议所需的相邻异色段。"""
    raw = sorted(candidates, key=lambda item: item.score, reverse=True)
    kept: List[StripDetection] = []
    for candidate in raw:
        if any(
                _overlap(candidate, existing) >= 0.55 or
                (
                    candidate.color == existing.color and
                    (
                        _same_physical_strip(candidate, existing) or
                        _same_physical_strip(existing, candidate)
                    )
                )
                for existing in kept):
            continue
        kept.append(candidate)
    return kept


def detect_candidates(
        bgr: np.ndarray,
        config: DetectorConfig) -> List[StripDetection]:
    return [
        item for item in detect_proposals(bgr, config)
        if item.score >= config.min_score
    ]


def select_winner(
        candidates: Sequence[StripDetection],
        config: DetectorConfig) -> Optional[StripDetection]:
    if not candidates:
        return None
    if len(candidates) > 1:
        # 同色反光不会造成颜色类别判定歧义。
        different = next(
            (item for item in candidates[1:]
             if item.color != candidates[0].color),
            None)
        if different is not None:
            ratio = candidates[0].score / max(different.score, 1e-9)
            if ratio < config.winner_margin:
                return None
    return candidates[0]


def detection_metrics(item: StripDetection) -> Mapping[str, float]:
    return {
        'line': item.line_quality,
        'dots': item.dot_quality,
        'periodic': item.periodic_quality,
        'color': item.color_quality,
        'valley': item.valley_quality,
    }


def annotate(
        bgr: np.ndarray,
        candidates: Sequence[StripDetection],
        winner: Optional[StripDetection]) -> np.ndarray:
    output = bgr.copy()
    for rank, candidate in enumerate(candidates, 1):
        selected = candidate is winner
        draw_color = (0, 255, 255) if selected else (255, 180, 0)
        corners = np.round(candidate.corners).astype(np.int32)
        cv2.polylines(output, [corners], True, draw_color, 2)
        if candidate.peak_centers is not None:
            for point in np.round(candidate.peak_centers).astype(np.int32):
                cv2.circle(output, tuple(point), 3, draw_color, 1)
        anchor = tuple(corners[np.argmin(corners[:, 1])])
        text = (
            f'#{rank} {candidate.color} score={candidate.score:.3f} '
            f'conf={candidate.confidence:.3f} dots={candidate.dot_count} '
            f'{candidate.mode[0].upper()} '
            f'P={candidate.periodic_quality:.2f} '
            f'V={candidate.valley_quality:.2f}')
        cv2.putText(
            output, text, anchor, cv2.FONT_HERSHEY_SIMPLEX,
            0.48, draw_color, 2)
    if winner is None:
        title = f'NO DETECTION ({len(candidates)} candidates)'
    else:
        different = next(
            (item for item in candidates[1:] if item.color != winner.color),
            None)
        margin = (
            winner.score / max(different.score, 1e-9)
            if different is not None else float('inf'))
        title = (
            f'SELECTED {winner.color} confidence={winner.confidence:.3f} '
            f'margin={margin:.2f}')
    cv2.putText(
        output, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (255, 255, 255), 2)
    return output
