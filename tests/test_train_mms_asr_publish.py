from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import train_mms_asr_kik as train


class FakeHubApi:
    def __init__(self) -> None:
        self.uploaded_files: dict[str, str] = {}
        self.created_repo: dict[str, object] | None = None

    def whoami(self, token: str) -> dict[str, str]:
        return {"name": "tester"}

    def create_repo(self, **kwargs):
        self.created_repo = kwargs
        return "https://huggingface.co/tester/model"

    def upload_file(self, *, path_or_fileobj, path_in_repo: str, **kwargs):
        self.uploaded_files[path_in_repo] = Path(path_or_fileobj).read_text(encoding="utf-8")
        return type("CommitInfo", (), {"oid": "abc123"})()


class FakeProcessor:
    def save_pretrained(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "processor_marker.txt").write_text("saved", encoding="utf-8")


class PublishHelpersTest(unittest.TestCase):
    def test_require_hf_token_fails_when_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN is not set"):
                train._require_hf_token()

    def test_matching_required_files_accepts_exact_and_glob_patterns(self) -> None:
        present, missing = train._matching_required_files(
            ["config.json", "model.safetensors", "checkpoint-10/config.json"],
            ["config.json", "*.safetensors", "trainer_state.json"],
        )

        self.assertEqual(present, ["config.json", "*.safetensors"])
        self.assertEqual(missing, ["trainer_state.json"])

    def test_preflight_uploads_and_reads_back_sentinel(self) -> None:
        api = FakeHubApi()

        def fake_download(*, repo_id: str, filename: str, **kwargs) -> str:
            self.assertEqual(repo_id, "tester/model")
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
                f.write(api.uploaded_files[filename])
                return f.name

        with patch.object(train, "hf_hub_download", side_effect=fake_download):
            train.preflight_hub_repo(
                api=api,
                repo_id="tester/model",
                private=True,
                token="secret",
            )

        self.assertIsNotNone(api.created_repo)
        assert api.created_repo is not None
        self.assertEqual(api.created_repo["repo_id"], "tester/model")
        self.assertTrue(api.created_repo["private"])
        sentinel_path, sentinel_text = next(iter(api.uploaded_files.items()))
        self.assertTrue(sentinel_path.startswith(".publish_preflight/"))
        self.assertEqual(json.loads(sentinel_text)["repo_id"], "tester/model")

    def test_checkpoint_callback_keeps_local_checkpoint_when_upload_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoint-400"
            checkpoint_dir.mkdir()
            args = type("Args", (), {"output_dir": temp_dir})()
            state = type("State", (), {"global_step": 400})()
            callback = train.HubCheckpointUploadCallback(
                api=FakeHubApi(),
                repo_id="tester/model",
                token="secret",
                processor=FakeProcessor(),
                upload_every_n_saves=1,
            )

            with patch.object(train, "upload_folder_to_hub", side_effect=RuntimeError("upload failed")):
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    callback.on_save(args, state, object())

            self.assertTrue(checkpoint_dir.is_dir())
            self.assertEqual((checkpoint_dir / "processor_marker.txt").read_text(encoding="utf-8"), "saved")


if __name__ == "__main__":
    unittest.main()
