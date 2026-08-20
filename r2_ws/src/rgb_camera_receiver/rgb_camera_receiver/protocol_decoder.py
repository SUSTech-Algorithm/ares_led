import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from rgb_comm_protocol import FixedColorProtocol

from .classifier import StripDetection


@dataclass(frozen=True)
class PairingConfig:
    min_pair_score: float = 0.04
    max_angle_degrees: float = 18.0
    max_cross_distance_pixels: float = 35.0
    min_center_distance_ratio: float = 0.35
    max_center_distance_ratio: float = 3.0
    min_length_ratio: float = 0.35
    max_length_ratio: float = 2.8
    max_pairs: int = 20


@dataclass(frozen=True)
class ProtocolDetection:
    command_id: int
    symbols: Tuple[str, str]
    first: StripDetection
    second: StripDetection
    score: float
    confidence: float
    geometry_quality: float
    angle_degrees: float
    cross_distance: float
    center_distance_ratio: float


def load_pairing_config(path: str) -> PairingConfig:
    with Path(path).open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    row = raw.get('pairing', {})
    return PairingConfig(
        min_pair_score=float(row.get('min_pair_score', 0.04)),
        max_angle_degrees=float(row.get('max_angle_degrees', 18.0)),
        max_cross_distance_pixels=float(
            row.get('max_cross_distance_pixels', 35.0)),
        min_center_distance_ratio=float(
            row.get('min_center_distance_ratio', 0.35)),
        max_center_distance_ratio=float(
            row.get('max_center_distance_ratio', 3.0)),
        min_length_ratio=float(row.get('min_length_ratio', 0.35)),
        max_length_ratio=float(row.get('max_length_ratio', 2.8)),
        max_pairs=int(row.get('max_pairs', 20)),
    )


def _axis(candidate: StripDetection) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = (candidate.corners[0] + candidate.corners[3]) * 0.5
    end = (candidate.corners[1] + candidate.corners[2]) * 0.5
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.array((1.0, 0.0), dtype=np.float32)
    else:
        direction = direction / norm
    center = (start + end) * 0.5
    return center.astype(np.float32), direction.astype(np.float32), start.astype(np.float32)


def _ordered_pair(
        first: StripDetection,
        second: StripDetection) -> Tuple[StripDetection, StripDetection]:
    center_a, axis_a, _ = _axis(first)
    center_b, axis_b, _ = _axis(second)
    axis = axis_a + axis_b
    if float(np.linalg.norm(axis)) <= 1e-6:
        axis = axis_a
    if float(np.dot(center_b - center_a, axis)) < 0.0:
        return second, first
    return first, second


def _pair_geometry(
        first: StripDetection,
        second: StripDetection,
        config: PairingConfig) -> Optional[Tuple[float, float, float, float]]:
    center_a, axis_a, _ = _axis(first)
    center_b, axis_b, _ = _axis(second)
    dot = abs(float(np.dot(axis_a, axis_b)))
    dot = max(-1.0, min(1.0, dot))
    angle = math.degrees(math.acos(dot))
    if angle > config.max_angle_degrees:
        return None

    average_axis = axis_a + axis_b
    if float(np.linalg.norm(average_axis)) <= 1e-6:
        average_axis = axis_a
    average_axis = average_axis / max(float(np.linalg.norm(average_axis)), 1e-6)
    normal = np.array((-average_axis[1], average_axis[0]), dtype=np.float32)
    delta = center_b - center_a
    center_distance = abs(float(np.dot(delta, average_axis)))
    cross_distance = abs(float(np.dot(delta, normal)))
    if cross_distance > config.max_cross_distance_pixels:
        return None

    median_length = max((first.length + second.length) * 0.5, 1.0)
    center_distance_ratio = center_distance / median_length
    if not (
            config.min_center_distance_ratio <= center_distance_ratio <=
            config.max_center_distance_ratio):
        return None
    length_ratio = first.length / max(second.length, 1e-6)
    length_ratio = max(length_ratio, 1.0 / max(length_ratio, 1e-6))
    if not (config.min_length_ratio <= length_ratio <= config.max_length_ratio):
        return None

    angle_quality = max(0.0, 1.0 - angle / max(config.max_angle_degrees, 1e-6))
    cross_quality = max(
        0.0, 1.0 - cross_distance / max(config.max_cross_distance_pixels, 1e-6))
    # 相邻段的中心距离通常接近一段灯带的长度。考虑到透视和局部遮挡的变化，
    # 此项采用软评分。
    distance_quality = math.exp(-abs(math.log(max(center_distance_ratio, 1e-6))))
    geometry_quality = float(np.clip(
        0.45 * angle_quality + 0.35 * cross_quality + 0.20 * distance_quality,
        0.0, 1.0))
    return geometry_quality, angle, cross_distance, center_distance_ratio


def decode_protocol_candidates(
        strip_candidates: Sequence[StripDetection],
        protocol: FixedColorProtocol,
        config: PairingConfig) -> List[ProtocolDetection]:
    pairs: List[ProtocolDetection] = []
    for index, left in enumerate(strip_candidates):
        for right in strip_candidates[index + 1:]:
            if left.color == right.color:
                continue
            geometry = _pair_geometry(left, right, config)
            if geometry is None:
                continue
            ordered_left, ordered_right = _ordered_pair(left, right)
            symbols = (ordered_left.color, ordered_right.color)
            command_id = protocol.decode(symbols)
            if command_id is None:
                continue
            geometry_quality, angle, cross_distance, center_distance_ratio = geometry
            score = math.sqrt(max(left.score, 0.0) * max(right.score, 0.0))
            score *= max(geometry_quality, 0.0)
            if score < config.min_pair_score:
                continue
            confidence = math.sqrt(
                max(ordered_left.confidence, 0.0) *
                max(ordered_right.confidence, 0.0) *
                max(geometry_quality, 0.0))
            pairs.append(ProtocolDetection(
                command_id=int(command_id),
                symbols=symbols,
                first=ordered_left,
                second=ordered_right,
                score=float(score),
                confidence=float(confidence),
                geometry_quality=geometry_quality,
                angle_degrees=angle,
                cross_distance=cross_distance,
                center_distance_ratio=center_distance_ratio,
            ))
    pairs.sort(key=lambda item: item.score, reverse=True)
    return pairs[:max(config.max_pairs, 1)]


def select_protocol_winner(
        candidates: Sequence[ProtocolDetection],
        margin: float) -> Optional[ProtocolDetection]:
    if not candidates:
        return None
    different = next(
        (item for item in candidates[1:]
         if item.command_id != candidates[0].command_id),
        None)
    if different is not None:
        ratio = candidates[0].score / max(different.score, 1e-9)
        if ratio < margin:
            return None
    return candidates[0]


def annotate_protocol(
        bgr: np.ndarray,
        protocol_candidates: Sequence[ProtocolDetection],
        winner: Optional[ProtocolDetection],
        state: str,
        confirmed_count: int,
        required_count: int) -> np.ndarray:
    output = bgr.copy()
    for rank, item in enumerate(protocol_candidates, 1):
        selected = item is winner
        color = (0, 255, 255) if selected else (0, 165, 255)
        for segment in (item.first, item.second):
            corners = np.round(segment.corners).astype(np.int32)
            cv2.polylines(output, [corners], True, color, 2)
        center_a, _, _ = _axis(item.first)
        center_b, _, _ = _axis(item.second)
        cv2.line(
            output,
            tuple(np.round(center_a).astype(np.int32)),
            tuple(np.round(center_b).astype(np.int32)),
            color,
            2)
        anchor = tuple(np.round((center_a + center_b) * 0.5).astype(np.int32))
        text = (
            f'#{rank} id={item.command_id} {item.symbols[0]}-{item.symbols[1]} '
            f'score={item.score:.3f} geo={item.geometry_quality:.2f}')
        cv2.putText(
            output, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    if winner is None:
        title = f'{state}: no valid two-segment command'
    else:
        title = (
            f'{state}: id={winner.command_id} '
            f'{winner.symbols[0]}-{winner.symbols[1]} '
            f'{confirmed_count}/{required_count} '
            f'score={winner.score:.3f}')
    cv2.putText(
        output, title, (12, 58), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (255, 255, 255), 2)
    return output
