"""
Tests for the whisper.cpp transcription engine (whisper-cli subprocess path).
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import TranscriptionConfig
from src.transcription import (
    TranscriptionError,
    _validate_whisper_cpp_setup,
    resolve_whisper_cpp_binary,
    transcribe_chunk_whisper_cpp,
)


class TestResolveWhisperCppBinary:
    def test_empty_binary_returns_none(self):
        assert resolve_whisper_cpp_binary("") is None

    def test_existing_path_resolves(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\n")
        assert resolve_whisper_cpp_binary(str(binary)) == str(binary.resolve())

    def test_missing_binary_uses_path(self):
        assert resolve_whisper_cpp_binary("definitely-not-a-real-binary-xyz") is None


class TestValidateWhisperCppSetup:
    def test_valid_setup(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "ggml-model.bin"
        model.write_text("fake")
        config = TranscriptionConfig(
            engine="whisper_cpp",
            whisper_cpp_binary=str(binary),
            whisper_cpp_model_path=str(model),
        )
        _validate_whisper_cpp_setup(config)

    def test_missing_binary_raises(self, tmp_path):
        model = tmp_path / "ggml-model.bin"
        model.write_text("fake")
        config = TranscriptionConfig(
            engine="whisper_cpp",
            whisper_cpp_binary=str(tmp_path / "nope"),
            whisper_cpp_model_path=str(model),
        )
        with pytest.raises(TranscriptionError, match="whisper-cli was not found"):
            _validate_whisper_cpp_setup(config)

    def test_missing_model_raises(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\n")
        config = TranscriptionConfig(
            engine="whisper_cpp",
            whisper_cpp_binary=str(binary),
            whisper_cpp_model_path=str(tmp_path / "nope.bin"),
        )
        with pytest.raises(TranscriptionError, match="model was not found"):
            _validate_whisper_cpp_setup(config)


class TestTranscribeChunkWhisperCpp:
    def _make_env(self, tmp_path):
        binary = tmp_path / "whisper-cli"
        binary.write_text("#!/bin/sh\n")
        model = tmp_path / "ggml-model.bin"
        model.write_text("fake")
        audio = tmp_path / "chunk.wav"
        audio.write_bytes(b"\x00" * 16)
        return binary, model, audio

    def _sample_json(self, text="привет мир"):
        return {
            "result": {"language": "ru"},
            "transcription": [
                {
                    "timestamps": {"from": "00:00:00,000", "to": "00:00:02,500"},
                    "offsets": {"from": 0, "to": 2500},
                    "text": text,
                },
                {
                    "timestamps": {"from": "00:00:02,500", "to": "00:00:05,000"},
                    "offsets": {"from": 2500, "to": 5000},
                    "text": " ",
                },
            ],
        }

    def test_parses_json_and_converts_ms_to_seconds(self, tmp_path):
        binary, model, audio = self._make_env(tmp_path)

        def fake_run(cmd, **kwargs):
            out_file = Path(cmd[cmd.index("-of") + 1] + ".json")
            out_file.write_text(json.dumps(self._sample_json()), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("src.transcription.subprocess.run", side_effect=fake_run) as run_mock:
            segments = transcribe_chunk_whisper_cpp(str(binary), str(model), audio, language="ru")

        assert len(segments) == 1
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == 2.5
        assert segments[0]["text"] == "привет мир"
        assert segments[0]["words"] == []
        assert run_mock.call_count == 1

    def test_passes_language_flag(self, tmp_path):
        binary, model, audio = self._make_env(tmp_path)

        def fake_run(cmd, **kwargs):
            out_file = Path(cmd[cmd.index("-of") + 1] + ".json")
            out_file.write_text(json.dumps(self._sample_json()), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("src.transcription.subprocess.run", side_effect=fake_run) as run_mock:
            transcribe_chunk_whisper_cpp(str(binary), str(model), audio, language="ru")

        cmd = run_mock.call_args.args[0]
        assert "-l" in cmd and "ru" in cmd

    def test_passes_decoding_flags(self, tmp_path):
        """Threads, beam size and context carry-over come from arguments."""
        binary, model, audio = self._make_env(tmp_path)

        def fake_run(cmd, **kwargs):
            out_file = Path(cmd[cmd.index("-of") + 1] + ".json")
            out_file.write_text(json.dumps(self._sample_json()), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("src.transcription.subprocess.run", side_effect=fake_run) as run_mock:
            transcribe_chunk_whisper_cpp(
                str(binary),
                str(model),
                audio,
                language="ru",
                threads=12,
                beam_size=2,
                max_context=64,
            )

        cmd = run_mock.call_args.args[0]
        assert cmd[cmd.index("-t") + 1] == "12"
        assert cmd[cmd.index("-bs") + 1] == "2"
        assert cmd[cmd.index("-mc") + 1] == "64"

    def test_context_carry_over_disabled_by_default(self, tmp_path):
        """Default -mc 0: carry-over degrades long chunks into fragments."""
        binary, model, audio = self._make_env(tmp_path)

        def fake_run(cmd, **kwargs):
            out_file = Path(cmd[cmd.index("-of") + 1] + ".json")
            out_file.write_text(json.dumps(self._sample_json()), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("src.transcription.subprocess.run", side_effect=fake_run) as run_mock:
            transcribe_chunk_whisper_cpp(str(binary), str(model), audio, language="ru")

        cmd = run_mock.call_args.args[0]
        assert cmd[cmd.index("-mc") + 1] == "0"
        assert TranscriptionConfig().whisper_cpp_max_context == 0

    def test_auto_language_detects_and_passes_flag(self, tmp_path):
        binary, model, audio = self._make_env(tmp_path)

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffmpeg":
                slice_path = Path(cmd[-1])
                slice_path.write_bytes(b"\x00" * 16)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "-l" in cmd and "auto" in cmd:
                # detection run: report detected language
                stem = cmd[cmd.index("-of") + 1]
                Path(stem + ".json").write_text(
                    json.dumps(
                        {
                            "result": {"language": "ru"},
                            "transcription": [{"offsets": {"from": 0, "to": 100}, "text": "x"}],
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            stem = cmd[cmd.index("-of") + 1]
            Path(stem + ".json").write_text(json.dumps(self._sample_json()), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("src.transcription.subprocess.run", side_effect=fake_run) as run_mock:
            segments = transcribe_chunk_whisper_cpp(str(binary), str(model), audio, language="auto")

        assert len(segments) == 1
        real_run_cmd = run_mock.call_args_list[-1].args[0]
        assert "-l" in real_run_cmd and "ru" in real_run_cmd
        # detect slice is cleaned up
        assert not (tmp_path / "chunk.detect.wav").exists()

    def test_auto_language_detection_failure_omits_flag(self, tmp_path):
        binary, model, audio = self._make_env(tmp_path)

        def fake_run(cmd, **kwargs):
            if cmd[0] == "ffmpeg":
                slice_path = Path(cmd[-1])
                slice_path.write_bytes(b"\x00" * 16)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "-l" in cmd and "auto" in cmd:
                # detection run fails
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
            stem = cmd[cmd.index("-of") + 1]
            Path(stem + ".json").write_text(json.dumps(self._sample_json()), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("src.transcription.subprocess.run", side_effect=fake_run) as run_mock:
            segments = transcribe_chunk_whisper_cpp(str(binary), str(model), audio, language="auto")

        assert len(segments) == 1
        real_run_cmd = run_mock.call_args_list[-1].args[0]
        assert "-l" not in real_run_cmd

    def test_removes_output_json_file(self, tmp_path):
        binary, model, audio = self._make_env(tmp_path)

        def fake_run(cmd, **kwargs):
            out_file = Path(cmd[cmd.index("-of") + 1] + ".json")
            out_file.write_text(json.dumps(self._sample_json()), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("src.transcription.subprocess.run", side_effect=fake_run):
            transcribe_chunk_whisper_cpp(str(binary), str(model), audio, language="ru")

        assert not (tmp_path / "chunk.json").exists()

    def test_nonzero_exit_raises(self, tmp_path):
        binary, model, audio = self._make_env(tmp_path)

        with patch(
            "src.transcription.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, stdout="", stderr="boom"),
        ):
            with pytest.raises(TranscriptionError, match="exit code 1"):
                transcribe_chunk_whisper_cpp(str(binary), str(model), audio)

    def test_missing_json_output_raises(self, tmp_path):
        binary, model, audio = self._make_env(tmp_path)

        with patch(
            "src.transcription.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""),
        ):
            with pytest.raises(TranscriptionError, match="did not produce JSON"):
                transcribe_chunk_whisper_cpp(str(binary), str(model), audio)


def _make_job(tmp_path):
    """Create the minimal artifacts transcribe() needs."""
    audio_dir = tmp_path / "artifacts" / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "raw.wav").write_bytes(b"dummy audio")
    (tmp_path / "artifacts" / "meta.json").write_text(json.dumps({"duration_sec": 5.0}))
    return audio_dir / "raw.wav"


def _cpu_gpu_config():
    from src.gpu_utils import GPUConfig

    return GPUConfig(
        backend="cpu",
        whisper_device="cpu",
        whisper_compute_type="int8",
        cleanup_fn=lambda: None,
    )


class TestTranscriptCacheKey:
    def test_fallback_engine_hash_matches_faster_whisper_config(self):
        """whisper.cpp settings do not leak into the faster-whisper key."""
        from src.transcription import compute_model_config_hash

        cpp_cfg = TranscriptionConfig(
            engine="whisper_cpp", whisper_cpp_model_path="models/ggml.bin"
        )
        fw_cfg = TranscriptionConfig(engine="faster_whisper")
        assert compute_model_config_hash(cpp_cfg, "faster_whisper") == (
            compute_model_config_hash(fw_cfg)
        )
        assert compute_model_config_hash(cpp_cfg) != compute_model_config_hash(fw_cfg)

    def test_whisper_cpp_decoding_settings_change_key(self):
        from src.transcription import compute_model_config_hash

        base = TranscriptionConfig(engine="whisper_cpp", whisper_cpp_model_path="m.bin")
        with_context = base.model_copy(update={"whisper_cpp_max_context": -1})
        wider_beam = base.model_copy(update={"whisper_cpp_beam_size": 5})
        assert compute_model_config_hash(base) != compute_model_config_hash(with_context)
        assert compute_model_config_hash(base) != compute_model_config_hash(wider_beam)


class TestTranscribeEngineSelection:
    @pytest.mark.parametrize("missing", ["binary", "model"])
    def test_missing_whisper_cpp_fails_with_instructions(self, tmp_path, missing):
        """No silent switch to faster-whisper: on Intel GPUs that is the CPU."""
        from src.config import Config
        from src.transcription import TranscriptionError, transcribe

        _make_job(tmp_path)
        binary = tmp_path / "whisper-cli"
        model = tmp_path / "ggml.bin"
        if missing == "model":
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
        else:
            model.write_bytes(b"\0")
        cfg = Config(
            work_dir=tmp_path,
            transcription=TranscriptionConfig(
                engine="whisper_cpp",
                whisper_cpp_binary=str(binary),
                whisper_cpp_model_path=str(model),
            ),
        )
        faster_whisper = MagicMock()
        with (
            patch("src.transcription.get_auto_gpu_config", return_value=_cpu_gpu_config()),
            patch("src.transcription.load_whisper_model", faster_whisper),
            pytest.raises(TranscriptionError) as excinfo,
        ):
            transcribe(cfg)

        hint = "install-whisper-cpp" if missing == "binary" else "prefetch-models"
        assert hint in str(excinfo.value)
        faster_whisper.assert_not_called()

    def test_whisper_cpp_receives_decoding_settings(self, tmp_path):
        from src.config import Config
        from src.transcription import transcribe

        audio = _make_job(tmp_path)
        cfg = Config(
            work_dir=tmp_path,
            transcription=TranscriptionConfig(
                engine="whisper_cpp",
                whisper_cpp_threads=12,
                whisper_cpp_beam_size=3,
                whisper_cpp_max_context=0,
            ),
        )
        segments = [{"start": 0.0, "end": 5.0, "text": "ok", "words": []}]

        with (
            patch("src.transcription.get_auto_gpu_config", return_value=_cpu_gpu_config()),
            patch("src.transcription._validate_whisper_cpp_setup"),
            patch("src.transcription.chunk_audio_by_duration", return_value=[audio]),
            patch("src.transcription._get_chunk_duration", return_value=5.0),
            patch(
                "src.transcription.transcribe_chunk_whisper_cpp", return_value=segments
            ) as chunk_mock,
        ):
            transcript = transcribe(cfg)

        kwargs = chunk_mock.call_args.kwargs
        assert (kwargs["threads"], kwargs["beam_size"], kwargs["max_context"]) == (12, 3, 0)
        assert transcript[0]["id"] == "seg_0001"


class TestWhisperCppDevice:
    """whisper-cli names its backend on stderr; a run on the CPU is a notice."""

    def test_device_from_stderr(self):
        from src.transcription import whisper_cpp_device

        assert whisper_cpp_device("whisper_backend_init_gpu: using Vulkan0 backend\n") == "Vulkan0"
        assert whisper_cpp_device("whisper_backend_init_gpu: no GPU found\n") == "CPU"
        assert whisper_cpp_device("") is None


class TestTranscriptionGpuFields:
    """The transcription_gpu log must not claim a faster-whisper device for whisper.cpp."""

    def test_whisper_cpp_omits_whisper_device(self):
        from src.transcription import transcription_gpu_fields

        fields = transcription_gpu_fields(_cpu_gpu_config(), "whisper_cpp")
        assert fields == {"engine": "whisper_cpp", "backend": "cpu"}
        assert "whisper_device" not in fields

    def test_faster_whisper_keeps_whisper_device(self):
        from src.gpu_utils import GPUConfig
        from src.transcription import transcription_gpu_fields

        gpu = GPUConfig(
            backend="cuda",
            whisper_device="cuda",
            whisper_compute_type="float16",
            cleanup_fn=lambda: None,
        )
        assert transcription_gpu_fields(gpu, "faster_whisper") == {
            "engine": "faster_whisper",
            "backend": "cuda",
            "whisper_device": "cuda",
        }
