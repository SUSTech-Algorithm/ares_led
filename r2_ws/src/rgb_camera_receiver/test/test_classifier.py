from pathlib import Path

import cv2
import numpy as np

from rgb_camera_receiver.classifier import (
    StripDetection,
    _deduplicate_candidates,
    annotate,
    detect_candidates,
    load_config,
    select_winner,
)


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = load_config(str(
    PACKAGE / 'config' / 'cameras' / 'usb_rgb' / 'detector.yaml'))


def color_bgr(name: str, saturation: int = 220, value: int = 180):
    model = next(item for item in CONFIG.colors if item.name == name)
    pixel = np.uint8([[[int(round(model.hue_center)), saturation, value]]])
    return tuple(int(x) for x in cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0])


def dotted_strip(bgr=None):
    if bgr is None:
        bgr = color_bgr('BLUE')
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    for index in range(24):
        cv2.circle(image, (45 + index * 10, 80 + index // 8), 3, bgr, -1)
    return image


def detection(color: str, start: float, end: float, score: float = 0.2):
    corners = np.array([
        [start, 96.0],
        [end, 96.0],
        [end, 104.0],
        [start, 104.0],
    ], dtype=np.float32)
    return StripDetection(
        color=color,
        confidence=0.9,
        score=score,
        corners=corners,
        dot_count=8,
        length=end - start,
        residual=1.0,
        spacing_cv=0.1,
    )


def test_detects_single_blue_dot_train():
    candidates = detect_candidates(dotted_strip(), CONFIG)
    winner = select_winner(candidates, CONFIG)
    assert winner is not None
    assert winner.color == 'BLUE'


def test_rejects_short_colored_bar():
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.rectangle(image, (35, 100), (95, 108), color_bgr('GREEN'), -1)
    assert detect_candidates(image, CONFIG) == []


def test_accepts_long_merged_led_strip():
    image = np.zeros((240, 360, 3), dtype=np.uint8)
    cv2.line(image, (45, 130), (300, 125), color_bgr('GREEN'), 4)
    candidates = detect_candidates(image, CONFIG)
    winner = select_winner(candidates, CONFIG)
    assert winner is not None
    assert winner.color == 'GREEN'


def test_keeps_adjacent_different_color_protocol_segments():
    red = detection('RED', 40.0, 120.0)
    blue = detection('BLUE', 130.0, 210.0, score=0.18)

    kept = _deduplicate_candidates([red, blue])

    assert [item.color for item in kept] == ['RED', 'BLUE']


def test_merges_adjacent_fragments_of_same_color():
    strong = detection('RED', 40.0, 120.0)
    weak = detection('RED', 130.0, 210.0, score=0.18)

    kept = _deduplicate_candidates([strong, weak])

    assert len(kept) == 1
    assert kept[0] is strong


def test_annotation_keeps_original_shape():
    image = dotted_strip(color_bgr('RED'))
    candidates = detect_candidates(image, CONFIG)
    rendered = annotate(image, candidates, select_winner(candidates, CONFIG))
    assert rendered.shape == image.shape


def test_complete_calibration_dataset():
    dataset = PACKAGE.parents[2] / 'camera_data' / 'usb_rgb'
    if not dataset.exists():
        return
    counts = 0
    for expected in ('BLUE', 'CYAN', 'GREEN', 'PURPLE', 'RED', 'NONE'):
        for path in sorted((dataset / expected).glob('*.jpg')):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if CONFIG.processing_scale < 1.0:
                image = cv2.resize(
                    image, None,
                    fx=CONFIG.processing_scale,
                    fy=CONFIG.processing_scale,
                    interpolation=cv2.INTER_AREA)
            candidates = detect_candidates(image, CONFIG)
            winner = select_winner(candidates, CONFIG)
            if expected == 'NONE':
                assert candidates == [], path
            else:
                assert winner is not None, path
                assert winner.color == expected, path
                wrong = [item for item in candidates if item.color != expected]
                assert wrong == [], path
            counts += 1
    assert counts > 0
