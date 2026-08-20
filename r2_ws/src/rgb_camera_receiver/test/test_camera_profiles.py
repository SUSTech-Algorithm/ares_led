from pathlib import Path

import yaml

import rgb_camera_receiver.profiles as profiles
from rgb_camera_receiver.profiles import (
    detector_config_path,
    validate_camera_profile,
    workspace_root,
)


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / 'config'


def load_yaml(path: Path):
    with path.open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream) or {}


def test_detector_configs_are_profile_scoped():
    assert not (CONFIG / 'detector.yaml').exists()
    assert not (CONFIG / 'receiver.yaml').exists()
    for profile in ('usb_rgb', 'odin1'):
        root = CONFIG / 'cameras' / profile
        assert (root / 'receiver.yaml').is_file()
        assert (root / 'detector.yaml').is_file()


def test_only_usb_rgb_is_calibrated_now():
    assert load_yaml(CONFIG / 'cameras' / 'usb_rgb' / 'detector.yaml')['calibrated']
    assert not load_yaml(CONFIG / 'cameras' / 'odin1' / 'detector.yaml')['calibrated']


def test_profile_validation_rejects_unknown_camera():
    try:
        validate_camera_profile('unknown')
    except ValueError as error:
        assert 'usb_rgb' in str(error)
        assert 'odin1' in str(error)
    else:
        raise AssertionError('unknown camera_profile was accepted')


def test_detector_config_path_uses_profile_directory():
    path = detector_config_path('usb_rgb')
    assert path.parts[-4:] == (
        'config',
        'cameras',
        'usb_rgb',
        'detector.yaml',
    )


def test_workspace_root_is_found_from_colcon_install_layout(
        tmp_path, monkeypatch):
    root = tmp_path / 'LED'
    installed_module = (
        root / 'r2_ws' / 'install' / 'rgb_camera_receiver' / 'lib' /
        'python3.10' / 'site-packages' / 'rgb_camera_receiver' / 'profiles.py')
    (root / 'camera_data').mkdir(parents=True)
    monkeypatch.delenv('LED_WORKSPACE_ROOT', raising=False)
    monkeypatch.setattr(profiles, '__file__', str(installed_module))
    monkeypatch.chdir(tmp_path)

    assert workspace_root() == root


def test_workspace_root_honors_valid_override(tmp_path, monkeypatch):
    root = tmp_path / 'custom_led'
    (root / 'camera_data').mkdir(parents=True)
    monkeypatch.setenv('LED_WORKSPACE_ROOT', str(root))

    assert workspace_root() == root
