from pathlib import Path

import numpy as np
import pytest
import yaml

import rgb_camera_receiver.calibrate as calibration
from rgb_camera_receiver.calibrate import _save_validated_config


def test_save_validated_config_publishes_calibrated_model(tmp_path: Path):
    output = tmp_path / 'detector.yaml'
    original = {
        'calibrated': False,
        'processing_scale': 0.5,
        'colors': {'OLD': {'hue_center': 1.0}},
    }
    colors = {'RED': {'hue_center': 2.0}}

    _save_validated_config(original, colors, output)

    with output.open('r', encoding='utf-8') as stream:
        saved = yaml.safe_load(stream)
    assert saved['calibrated'] is True
    assert saved['colors'] == colors
    assert saved['processing_scale'] == 0.5
    assert original['calibrated'] is False
    assert original['colors'] != colors


def test_failed_self_check_does_not_overwrite_active_config(
        tmp_path: Path, monkeypatch):
    dataset = tmp_path / 'dataset'
    for name in calibration.CLASSES:
        directory = dataset / name
        directory.mkdir(parents=True)
        (directory / 'sample.jpg').touch()
    output = tmp_path / 'detector.yaml'
    output.write_text('calibrated: true\nmarker: keep\n', encoding='utf-8')
    base_config = (
        Path(__file__).resolve().parents[1] /
        'config' / 'cameras' / 'usb_rgb' / 'detector.yaml')

    monkeypatch.setattr(
        calibration,
        '_collect_color_pixels',
        lambda _path: (
            np.full(32, 10, dtype=np.uint8),
            np.full(32, 200, dtype=np.uint8),
            np.tile(np.array([[20, 30, 200]], dtype=np.uint8), (32, 1)),
        ))
    monkeypatch.setattr(
        calibration.cv2, 'imread',
        lambda *_args, **_kwargs: np.zeros((8, 8, 3), dtype=np.uint8))
    detected_shapes = []

    def reject_candidates(image, _config):
        detected_shapes.append(image.shape[:2])
        return []

    monkeypatch.setattr(calibration, 'detect_candidates', reject_candidates)

    with pytest.raises(RuntimeError, match='active detector config was not changed'):
        calibration.calibrate(dataset, base_config, output)

    # 随附配置的 processing.scale 为 0.4；校准自检必须与部署时一致。
    assert detected_shapes == [(3, 3)]
    assert output.read_text(encoding='utf-8') == (
        'calibrated: true\nmarker: keep\n')
