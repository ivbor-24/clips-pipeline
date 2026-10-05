"""
Tests for P2-05: Audio Denoising Feature

Tests cover:
- Configuration (denoise, denoise_method)
- denoise_audio() function
- Integration with ingest()
- Meta.json updates
- Transcription using denoised audio
"""

import json
from unittest.mock import MagicMock, patch

from src.config import Config, PreprocessingConfig


class TestDenoiseConfig:
    """Test denoise configuration."""

    def test_default_denoise_disabled(self):
        """Test that denoise is disabled by default."""
        config = PreprocessingConfig()
        assert config.denoise is False
        assert config.denoise_method == "afftdn"

    def test_denoise_enabled(self):
        """Test enabling denoise."""
        config = PreprocessingConfig(denoise=True)
        assert config.denoise is True

    def test_denoise_method_afftdn(self):
        """Test afftdn method."""
        config = PreprocessingConfig(denoise=True, denoise_method="afftdn")
        assert config.denoise_method == "afftdn"

    def test_denoise_method_rnnoise(self):
        """Test rnnoise method."""
        config = PreprocessingConfig(denoise=True, denoise_method="rnnoise")
        assert config.denoise_method == "rnnoise"

    def test_denoise_from_yaml(self, tmp_path):
        """Test loading denoise config from YAML."""
        yaml_content = """
preprocessing:
  denoise: true
  denoise_method: afftdn
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.load(str(config_file))
        assert config.preprocessing.denoise is True
        assert config.preprocessing.denoise_method == "afftdn"


class TestDenoiseAudioFunction:
    """Test denoise_audio() function."""

    def test_denoise_disabled_skips(self, tmp_path):
        """Test that denoise is skipped when disabled."""
        from src.ingestion import denoise_audio

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        config = PreprocessingConfig(denoise=False)

        with patch("subprocess.run") as mock_run:
            denoise_audio(input_path, output_path, config)
            mock_run.assert_not_called()

    def test_denoise_afftdn_command(self, tmp_path):
        """Test afftdn ffmpeg command."""
        from src.ingestion import denoise_audio

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        config = PreprocessingConfig(denoise=True, denoise_method="afftdn")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            denoise_audio(input_path, output_path, config)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]

            assert "ffmpeg" in cmd
            assert "-af" in cmd
            assert "afftdn=nf=-20:tn=1" in cmd
            assert str(input_path) in cmd
            assert str(output_path) in cmd

    def test_denoise_rnnoise_command(self, tmp_path):
        """Test rnnoise ffmpeg command."""
        from src.ingestion import denoise_audio

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "output.wav"
        input_path.write_bytes(b"fake audio")

        config = PreprocessingConfig(denoise=True, denoise_method="rnnoise")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            denoise_audio(input_path, output_path, config)

            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]

            assert "ffmpeg" in cmd
            assert "-af" in cmd
            assert "arnndn=m=rnnoise-default" in cmd

    def test_denoise_creates_output_dir(self, tmp_path):
        """Test that output directory is created."""
        from src.ingestion import denoise_audio

        input_path = tmp_path / "input.wav"
        output_path = tmp_path / "subdir" / "output.wav"
        input_path.write_bytes(b"fake audio")

        config = PreprocessingConfig(denoise=True)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            denoise_audio(input_path, output_path, config)

            assert output_path.parent.exists()


class TestIngestWithDenoise:
    """Test ingest() integration with denoise."""

    @patch("src.ingestion.get_video_metadata")
    @patch("src.ingestion.preprocess_video")
    @patch("src.ingestion.extract_audio")
    @patch("src.ingestion.denoise_audio")
    @patch("src.ingestion.compute_file_hash")
    def test_ingest_with_denoise_enabled(
        self, mock_hash, mock_denoise, mock_extract, mock_preprocess, mock_meta, tmp_path
    ):
        """Test that ingest() calls denoise_audio when enabled."""
        from src.ingestion import ingest

        mock_hash.return_value = "abc123"
        mock_meta.return_value = {"duration_sec": 100, "resolution": [1920, 1080], "fps": 30}

        config = Config()
        config.work_dir = tmp_path
        config.preprocessing.denoise = True

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake video")

        result = ingest(str(input_file), config)

        mock_denoise.assert_called_once()
        assert result["denoised"] is True
        assert result["denoise_method"] == "afftdn"

    @patch("src.ingestion.get_video_metadata")
    @patch("src.ingestion.preprocess_video")
    @patch("src.ingestion.extract_audio")
    @patch("src.ingestion.denoise_audio")
    @patch("src.ingestion.compute_file_hash")
    def test_ingest_with_denoise_disabled(
        self, mock_hash, mock_denoise, mock_extract, mock_preprocess, mock_meta, tmp_path
    ):
        """Test that ingest() does not call denoise_audio when disabled."""
        from src.ingestion import ingest

        mock_hash.return_value = "abc123"
        mock_meta.return_value = {"duration_sec": 100, "resolution": [1920, 1080], "fps": 30}

        config = Config()
        config.work_dir = tmp_path
        config.preprocessing.denoise = False

        input_file = tmp_path / "input.mp4"
        input_file.write_bytes(b"fake video")

        result = ingest(str(input_file), config)

        mock_denoise.assert_not_called()
        assert result["denoised"] is False
        assert result["denoise_method"] is None


class TestTranscriptionWithDenoisedAudio:
    """Test transcription uses denoised audio when available."""

    def test_transcription_uses_denoised_when_exists(self, tmp_path):
        """Test that transcription prefers denoised.wav over raw.wav."""
        artifacts_dir = tmp_path / "artifacts"
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True)

        raw_path = audio_dir / "raw.wav"
        denoised_path = audio_dir / "denoised.wav"
        raw_path.write_bytes(b"raw audio")
        denoised_path.write_bytes(b"denoised audio")

        meta_path = artifacts_dir / "meta.json"
        meta_path.write_text(json.dumps({"duration_sec": 100}))

        config = Config()
        config.work_dir = tmp_path

        with patch("src.transcription.compute_audio_hash") as mock_hash:
            mock_hash.return_value = "abc123"
            with patch("src.transcription.load_transcript_from_cache") as mock_cache:
                mock_cache.return_value = None
                with patch("src.transcription.load_whisper_model") as mock_load:
                    mock_load.return_value = MagicMock()
                    with patch("src.transcription.chunk_audio_by_duration") as mock_chunk:
                        mock_chunk.return_value = []

                        from src.transcription import transcribe

                        transcribe(config, dry_run=False)

                        mock_hash.assert_called_once_with(denoised_path)

    def test_transcription_uses_raw_when_no_denoised(self, tmp_path):
        """Test that transcription uses raw.wav when denoised.wav doesn't exist."""
        artifacts_dir = tmp_path / "artifacts"
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True)

        raw_path = audio_dir / "raw.wav"
        raw_path.write_bytes(b"raw audio")

        meta_path = artifacts_dir / "meta.json"
        meta_path.write_text(json.dumps({"duration_sec": 100}))

        config = Config()
        config.work_dir = tmp_path

        with patch("src.transcription.compute_audio_hash") as mock_hash:
            mock_hash.return_value = "abc123"
            with patch("src.transcription.load_transcript_from_cache") as mock_cache:
                mock_cache.return_value = None
                with patch("src.transcription.load_whisper_model") as mock_load:
                    mock_load.return_value = MagicMock()
                    with patch("src.transcription.chunk_audio_by_duration") as mock_chunk:
                        mock_chunk.return_value = []

                        from src.transcription import transcribe

                        transcribe(config, dry_run=False)

                        mock_hash.assert_called_once_with(raw_path)
