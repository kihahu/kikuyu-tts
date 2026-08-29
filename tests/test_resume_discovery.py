from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.bootstrap_mms_kikuyu_tts_finetune import (
    checkpoint_step_from_files,
    latest_local_checkpoint_step,
)
from scripts.train_mms_asr_kik import find_latest_checkpoint


class ResumeDiscoveryTest(unittest.TestCase):
    def test_asr_chooses_latest_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "checkpoint-100").mkdir()
            (root / "checkpoint-100" / "trainer_state.json").write_text("{}", encoding="utf-8")
            (root / "checkpoint-200").mkdir()

            result = find_latest_checkpoint(root)

            self.assertEqual(result, root / "checkpoint-100")

    def test_tts_chooses_latest_generator_discriminator_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "G_100.pth").touch()
            (root / "D_100.pth").touch()
            (root / "G_200.pth").touch()

            self.assertEqual(latest_local_checkpoint_step(root), 100)

    def test_hub_checkpoint_discovery_requires_both_vits_files(self) -> None:
        files = [
            "mms_vits_finetune/vits/logs/run/G_100.pth",
            "mms_vits_finetune/vits/logs/run/D_100.pth",
            "mms_vits_finetune/vits/logs/run/G_200.pth",
        ]

        self.assertEqual(
            checkpoint_step_from_files(files, "mms_vits_finetune/vits/logs/run"),
            100,
        )


if __name__ == "__main__":
    unittest.main()
