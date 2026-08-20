"""R2 摄像头 profile 路径管理。

不同摄像头的成像颜色、噪声和输入方式可能完全不同，因此训练数据、识别参数和后续算法
都按 profile 隔离。当前可用 profile 是 USB RGB 摄像头；Odin1 先预留数据和配置位置。
"""

import os
from pathlib import Path
from typing import Tuple

import yaml

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
except ModuleNotFoundError:
    class PackageNotFoundError(Exception):
        pass

    def get_package_share_directory(_package_name: str) -> str:
        raise PackageNotFoundError()


CAMERA_PROFILES: Tuple[str, ...] = ('usb_rgb', 'odin1')
DEFAULT_CAMERA_PROFILE = 'usb_rgb'


def validate_camera_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if normalized not in CAMERA_PROFILES:
        choices = ', '.join(CAMERA_PROFILES)
        raise ValueError(f'unknown camera_profile {profile!r}; choices: {choices}')
    return normalized


def _workspace_candidates(start: Path):
    """生成可能的 LED 工作区根目录，兼容源码和 colcon 安装布局。"""
    for base in (start.resolve(), Path.cwd().resolve()):
        current = base if base.is_dir() else base.parent
        for directory in (current, *current.parents):
            yield directory
            if directory.name == 'r2_ws':
                yield directory.parent


def workspace_root() -> Path:
    """定位包含 camera_data 的项目根目录。

    可通过 LED_WORKSPACE_ROOT 显式指定；否则从当前模块路径和当前工作目录
    向上查找。该方式在源码目录和 ros2 run 的 install 目录中均可工作。
    """
    override = os.environ.get('LED_WORKSPACE_ROOT')
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / 'camera_data').is_dir():
            raise RuntimeError(
                f'LED_WORKSPACE_ROOT has no camera_data directory: {root}')
        return root

    checked = set()
    for candidate in _workspace_candidates(Path(__file__)):
        if candidate in checked:
            continue
        checked.add(candidate)
        if (candidate / 'camera_data').is_dir():
            return candidate
    raise RuntimeError(
        'cannot locate LED workspace containing camera_data; '
        'set LED_WORKSPACE_ROOT to the LED project root')


def config_root() -> Path:
    try:
        return Path(get_package_share_directory('rgb_camera_receiver')) / 'config'
    except PackageNotFoundError:
        return Path(__file__).resolve().parents[1] / 'config'


def camera_config_dir(profile: str) -> Path:
    return config_root() / 'cameras' / validate_camera_profile(profile)


def detector_config_path(profile: str) -> Path:
    return camera_config_dir(profile) / 'detector.yaml'


def require_calibrated_detector(path: Path, profile: str) -> Path:
    """确认当前 profile 已有可用于实时识别的 detector。"""
    if not path.is_file():
        raise RuntimeError(
            f'camera_profile={profile} has no detector config: {path}')
    with path.open('r', encoding='utf-8') as stream:
        raw = yaml.safe_load(stream) or {}
    if not bool(raw.get('calibrated', False)):
        raise RuntimeError(
            f'camera_profile={profile} detector is not calibrated; '
            f'collect and label camera_data/{profile} first')
    return path


def dataset_path(profile: str) -> Path:
    return workspace_root() / 'camera_data' / validate_camera_profile(profile)


def results_path(profile: str) -> Path:
    return workspace_root() / 'camera_results' / validate_camera_profile(profile)
