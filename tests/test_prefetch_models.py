"""Tests for scripts/prefetch_models.py."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "prefetch_models.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prefetch_models_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prefetch_models():
    return _load_module()


def _llm_config(**kwargs):
    from src.config import LLMConfig

    return LLMConfig(**kwargs)


class TestPrefetchWhisper:
    def test_downloads_model(self, prefetch_models):
        mock_download = MagicMock()
        fake_utils = MagicMock(download_model=mock_download)
        with patch.dict(sys.modules, {"faster_whisper.utils": fake_utils}):
            prefetch_models.prefetch_whisper("large-v3-turbo")
        mock_download.assert_called_once_with(
            "large-v3-turbo", revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
        )

    def test_missing_dependency_does_not_raise(self, prefetch_models):
        with patch.dict(sys.modules, {"faster_whisper.utils": None}):
            prefetch_models.prefetch_whisper("large-v3-turbo")


class TestPrefetchLLM:
    def test_disabled_skips(self, prefetch_models):
        cfg = _llm_config(enabled=False)
        prefetch_models.prefetch_llm(cfg)  # must not raise, must not touch network

    def test_llama_cpp_resolves_model_path(self, prefetch_models):
        cfg = _llm_config(enabled=True, provider="llama_cpp", model_repo="x/y", model_file="m.gguf")
        with patch(
            "src.llm_adapter.LlamaCppAdapter._resolve_model_path",
            return_value=Path("/tmp/m.gguf"),
        ) as mock_resolve:
            prefetch_models.prefetch_llm(cfg)
        mock_resolve.assert_called_once()

    def test_llama_cpp_failure_propagates(self, prefetch_models):
        from src.llm_adapter import LLMAdapterError

        cfg = _llm_config(enabled=True, provider="llama_cpp")
        with patch(
            "src.llm_adapter.LlamaCppAdapter._resolve_model_path",
            side_effect=LLMAdapterError("boom"),
        ):
            with pytest.raises(LLMAdapterError):
                prefetch_models.prefetch_llm(cfg)

    def test_qwen_local_downloads_snapshot(self, prefetch_models):
        cfg = _llm_config(enabled=True, provider="qwen_local", model="Qwen/Qwen2.5-7B-Instruct")
        mock_snapshot = MagicMock()
        fake_hub = MagicMock(snapshot_download=mock_snapshot)
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
            prefetch_models.prefetch_llm(cfg)
        mock_snapshot.assert_called_once_with(repo_id="Qwen/Qwen2.5-7B-Instruct")

    def test_qwen_local_missing_dependency_does_not_raise(self, prefetch_models):
        cfg = _llm_config(enabled=True, provider="qwen_local")
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            prefetch_models.prefetch_llm(cfg)

    def test_cloud_provider_needs_no_prefetch(self, prefetch_models):
        cfg = _llm_config(enabled=True, provider="openai")
        prefetch_models.prefetch_llm(cfg)  # must not raise, must not touch network


class TestMain:
    def test_main_prefetches_whisper_and_llm(self, prefetch_models, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("transcription:\n  model: tiny\n")

        fake_config = MagicMock()
        fake_config.with_hardware_profile.return_value = fake_config
        fake_config.transcription.model = "tiny"
        fake_config.scoring.llm = MagicMock()

        with (
            patch.object(prefetch_models.Config, "load", return_value=fake_config) as mock_load,
            patch.object(prefetch_models, "prefetch_whisper") as mock_whisper,
            patch.object(prefetch_models, "prefetch_llm") as mock_llm,
            patch.object(prefetch_models, "prefetch_keybert"),
            patch.object(prefetch_models, "prefetch_face_detector"),
            patch("sys.argv", ["prefetch_models.py", "--config", str(config_file)]),
        ):
            rc = prefetch_models.main()

        mock_load.assert_called_once_with(str(config_file))
        mock_whisper.assert_called_once_with("tiny")
        mock_llm.assert_called_once_with(fake_config.scoring.llm)
        assert rc == 0

    def test_main_respects_skip_flags(self, prefetch_models, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("transcription:\n  model: tiny\n")
        fake_config = MagicMock()
        fake_config.with_hardware_profile.return_value = fake_config

        with (
            patch.object(prefetch_models.Config, "load", return_value=fake_config),
            patch.object(prefetch_models, "prefetch_whisper") as mock_whisper,
            patch.object(prefetch_models, "prefetch_llm") as mock_llm,
            patch.object(prefetch_models, "prefetch_keybert"),
            patch.object(prefetch_models, "prefetch_face_detector"),
            patch(
                "sys.argv",
                [
                    "prefetch_models.py",
                    "--config",
                    str(config_file),
                    "--skip-whisper",
                    "--skip-llm",
                ],
            ),
        ):
            rc = prefetch_models.main()

        mock_whisper.assert_not_called()
        mock_llm.assert_not_called()
        assert rc == 0

    def test_main_returns_1_on_failure(self, prefetch_models, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("transcription:\n  model: tiny\n")
        fake_config = MagicMock()
        fake_config.with_hardware_profile.return_value = fake_config

        with (
            patch.object(prefetch_models.Config, "load", return_value=fake_config),
            patch.object(prefetch_models, "prefetch_whisper", side_effect=RuntimeError("boom")),
            patch("sys.argv", ["prefetch_models.py", "--config", str(config_file)]),
        ):
            rc = prefetch_models.main()

        assert rc == 1


class TestPrefetchWhisperCpp:
    def _tconfig(self, **kwargs):
        from src.config import TranscriptionConfig

        return TranscriptionConfig(engine="whisper_cpp", model="large-v3-turbo", **kwargs)

    def test_existing_model_is_verified(self, prefetch_models, tmp_path):
        model = tmp_path / "ggml-large-v3-turbo.bin"
        model.write_bytes(b"x")
        with patch.object(prefetch_models, "verify_file") as mock_verify:
            path = prefetch_models.prefetch_whisper_cpp_model(
                self._tconfig(whisper_cpp_model_path=str(model))
            )
        assert path == model
        assert mock_verify.call_args.args[1].file == "ggml-large-v3-turbo.bin"

    def test_missing_model_is_downloaded_and_linked(self, prefetch_models, tmp_path):
        cached = tmp_path / "hf" / "ggml-large-v3-turbo.bin"
        cached.parent.mkdir()
        cached.write_bytes(b"x")
        target = tmp_path / "models" / "ggml-large-v3-turbo.bin"
        with patch.object(prefetch_models, "download_pinned_file", return_value=cached) as dl:
            prefetch_models.prefetch_whisper_cpp_model(
                self._tconfig(whisper_cpp_model_path=str(target))
            )
        assert dl.call_args.args[0].repo == "ggerganov/whisper.cpp"
        assert target.resolve() == cached.resolve()

    def test_unpinned_missing_model_raises(self, prefetch_models, tmp_path):
        cfg = self._tconfig(whisper_cpp_model_path=str(tmp_path / "x.bin"))
        cfg.model = "no-such-model"
        with pytest.raises(RuntimeError, match="not in config/models.lock.yaml"):
            prefetch_models.prefetch_whisper_cpp_model(cfg)

    def test_main_uses_whisper_cpp_engine(self, prefetch_models, tmp_path):
        fake_config = MagicMock()
        fake_config.with_hardware_profile.return_value = fake_config
        fake_config.transcription.engine = "whisper_cpp"
        with (
            patch.object(prefetch_models.Config, "load", return_value=fake_config),
            patch.object(prefetch_models, "prefetch_whisper_cpp_model") as mock_cpp,
            patch.object(prefetch_models, "prefetch_whisper") as mock_fw,
            patch.object(prefetch_models, "prefetch_llm"),
            patch.object(prefetch_models, "prefetch_face_detector"),
            patch("sys.argv", ["prefetch_models.py"]),
        ):
            assert prefetch_models.main() == 0
        mock_cpp.assert_called_once_with(fake_config.transcription)
        mock_fw.assert_not_called()


class TestPrefetchSmallModels:
    """KeyBERT's embeddings and the face detector used to download mid-job."""

    def test_keybert_downloads_embedding_model(self, prefetch_models):
        with patch("huggingface_hub.snapshot_download") as download:
            prefetch_models.prefetch_keybert("keybert")
        download.assert_called_once_with(repo_id=prefetch_models.KEYBERT_MODEL)

    def test_keybert_skipped_for_statistical(self, prefetch_models):
        with patch("huggingface_hub.snapshot_download") as download:
            prefetch_models.prefetch_keybert("statistical")
        download.assert_not_called()

    def test_face_detector_downloads_the_pinned_model(self, prefetch_models):
        with patch("src.face_cropping.face_detector_model") as model:
            prefetch_models.prefetch_face_detector()
        model.assert_called_once_with()
