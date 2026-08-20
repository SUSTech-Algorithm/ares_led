from pathlib import Path

import numpy as np

from rgb_comm_protocol import FixedColorProtocol

from rgb_camera_receiver.classifier import StripDetection
from rgb_camera_receiver.protocol_decoder import (
    PairingConfig,
    decode_protocol_candidates,
    select_protocol_winner,
)


PACKAGE = Path(__file__).resolve().parents[1]
PROTOCOL = FixedColorProtocol(
    config_path=str(PACKAGE.parents[2] / 'shared' / 'src' /
                    'rgb_comm_protocol' / 'config' / 'rgb_protocol.yaml'))
CONFIG = PairingConfig(min_pair_score=0.01)


def strip(color: str, center_x: float, center_y: float = 100.0) -> StripDetection:
    length = 80.0
    width = 8.0
    corners = np.array([
        [center_x - length / 2, center_y - width / 2],
        [center_x + length / 2, center_y - width / 2],
        [center_x + length / 2, center_y + width / 2],
        [center_x - length / 2, center_y + width / 2],
    ], dtype=np.float32)
    return StripDetection(
        color=color,
        confidence=0.9,
        score=0.20,
        corners=corners,
        dot_count=8,
        length=length,
        residual=2.0,
        spacing_cv=0.1,
        line_quality=0.9,
        dot_quality=0.9,
        periodic_quality=0.9,
        color_quality=0.9,
        valley_quality=0.9,
    )


def test_decodes_two_segment_command():
    candidates = decode_protocol_candidates(
        [strip('RED', 100), strip('BLUE', 190)], PROTOCOL, CONFIG)
    winner = select_protocol_winner(candidates, margin=1.0)
    assert winner is not None
    assert winner.command_id == 1
    assert winner.symbols == ('RED', 'BLUE')


def test_reverse_order_has_same_meaning():
    candidates = decode_protocol_candidates(
        [strip('RED', 190), strip('BLUE', 100)], PROTOCOL, CONFIG)
    winner = select_protocol_winner(candidates, margin=1.0)
    assert winner is not None
    assert winner.command_id == 1
    assert set(winner.symbols) == {'RED', 'BLUE'}


def test_invalid_protocol_pair_is_rejected():
    candidates = decode_protocol_candidates(
        [strip('CYAN', 100), strip('PURPLE', 190)], PROTOCOL, CONFIG)
    assert candidates == []


def test_non_collinear_pair_is_rejected():
    candidates = decode_protocol_candidates(
        [strip('RED', 100, 80), strip('BLUE', 190, 180)], PROTOCOL, CONFIG)
    assert candidates == []
