import numpy as np

from sura.motion.pdr import StepDetector, StepDetectorConfig, pdr_controls


def test_step_detector_and_controls_are_causal():
    config = StepDetectorConfig(sampling_hz=10.0, threshold=0.5, refractory_seconds=0.2)
    detector = StepDetector(config)
    samples = [9.81, 9.81, 11.0, 9.81, 9.81]
    detections = [detector.update(value) for value in samples]
    assert detections == [False, False, True, False, False]

    acceleration = np.zeros((5, 3))
    acceleration[:, 2] = samples
    heading = np.zeros(5)
    controls = pdr_controls(
        acceleration,
        heading,
        heading_offset=0.0,
        step_length=0.7,
        detector=StepDetector(config),
    )
    assert controls.shape == (5, 2)
    assert np.count_nonzero(np.any(controls != 0, axis=1)) == 1
