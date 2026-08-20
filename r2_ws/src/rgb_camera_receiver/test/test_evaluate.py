from rgb_camera_receiver.evaluate import _evaluation_passed


def test_evaluation_requires_negative_samples_to_pass():
    assert _evaluation_passed(0, True, 20.0, 'passed')
    assert not _evaluation_passed(
        0, True, 20.0, 'not_accepted_no_negative_samples')
    assert not _evaluation_passed(0, True, 20.0, 'failed')
