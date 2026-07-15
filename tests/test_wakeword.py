import unittest

import torch

from src.wakeword.openwakeword_detector import OpenWakeWordDetector, is_detected, waveform_to_pcm16


class FakeWakeWordModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def predict(self, audio):
        return {"hey_jarvis": self.score}


class WakeWordTests(unittest.TestCase):
    def test_threshold_detection(self) -> None:
        self.assertTrue(is_detected(0.7, 0.5))
        self.assertFalse(is_detected(0.3, 0.5))

    def test_waveform_to_pcm16(self) -> None:
        pcm = waveform_to_pcm16(torch.tensor([[0.0, 1.0, -1.0]]))

        self.assertEqual(pcm.dtype.name, "int16")
        self.assertEqual(pcm.tolist(), [0, 32767, -32767])

    def test_detector_uses_fake_model(self) -> None:
        detector = OpenWakeWordDetector(model=FakeWakeWordModel(0.8), threshold=0.5)

        result = detector.predict_frame(torch.zeros(1, 1280), sample_rate=16000)

        self.assertTrue(result.detected)
        self.assertEqual(result.score, 0.8)


if __name__ == "__main__":
    unittest.main()
