import pytest

from rgb_led_sender.mapping import build_wled_state_json


def test_build_wled_state_json_lights_two_segments_and_clears_rest():
    payload = build_wled_state_json(
        [(255, 0, 0), (0, 0, 255)],
        brightness=40.0,
        pixel_count=11,
    )

    assert payload == (
        '{"on":true,"bri":40,"seg":['
        '{"id":0,"start":0,"stop":1,"col":[[255,0,0]],"fx":0},'
        '{"id":1,"start":1,"stop":2,"col":[[0,0,255]],"fx":0},'
        '{"id":2,"start":2,"stop":11,"col":[[0,0,0]],"fx":0}'
        ']}'
    )


def test_build_wled_state_json_rejects_wrong_shape():
    with pytest.raises(ValueError):
        build_wled_state_json([(255, 0, 0)], brightness=40.0, pixel_count=11)
    with pytest.raises(ValueError):
        build_wled_state_json(
            [(255, 0, 0), (0, 0, 255)],
            brightness=40.0,
            pixel_count=1,
        )
