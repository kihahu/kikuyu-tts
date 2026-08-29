from __future__ import annotations

import unittest

import numpy as np

from scripts.prepare_anv_kikuyu_mms_tts import audio_quality, resample_audio


class AnvAudioQualityTest(unittest.TestCase):
    def test_resample_audio_produces_requested_length_and_mono_output(self) -> None:
        samples = np.ones((800, 2), dtype=np.float32)

        result = resample_audio(samples, source_rate=16000, target_rate=8000)

        self.assertEqual(result.shape, (400,))
        self.assertTrue(np.isfinite(result).all())

    def test_audio_quality_reports_silence_and_nonfinite_samples(self) -> None:
        samples = np.array([0.0, 0.0, np.nan], dtype=np.float32)

        rms, peak, nonfinite_ratio = audio_quality(samples)

        self.assertEqual(rms, 0.0)
        self.assertEqual(peak, 0.0)
        self.assertAlmostEqual(nonfinite_ratio, 1 / 3)


if __name__ == "__main__":
    unittest.main()
