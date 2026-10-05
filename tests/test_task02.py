"""
Tests for TASK-02: Ingestion & Preprocessing
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestIngestionModule:
    """Test ingestion module imports and structure."""

    def test_module_exists(self):
        """Test that ingestion module exists."""
        from src import ingestion

        assert ingestion is not None

    def test_ingestion_error_exists(self):
        """Test that IngestionError exception exists."""
        from src.ingestion import IngestionError

        assert IngestionError is not None
        assert issubclass(IngestionError, Exception)

    def test_compute_file_hash_function_exists(self):
        """Test that compute_file_hash function exists."""
        from src.ingestion import compute_file_hash

        assert callable(compute_file_hash)

    def test_get_video_metadata_function_exists(self):
        """Test that get_video_metadata function exists."""
        from src.ingestion import get_video_metadata

        assert callable(get_video_metadata)

    def test_preprocess_video_function_exists(self):
        """Test that preprocess_video function exists."""
        from src.ingestion import preprocess_video

        assert callable(preprocess_video)

    def test_extract_audio_function_exists(self):
        """Test that extract_audio function exists."""
        from src.ingestion import extract_audio

        assert callable(extract_audio)

    def test_ingest_function_exists(self):
        """Test that main ingest function exists."""
        from src.ingestion import ingest

        assert callable(ingest)


class TestComputeFileHash:
    """Test file hash computation."""

    def test_hash_computation(self, tmp_path):
        """Test that hash is computed correctly."""
        from src.ingestion import compute_file_hash

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        hash_result = compute_file_hash(test_file)

        # Hash should be a 64-character hex string (SHA256)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_same_content_same_hash(self, tmp_path):
        """Test that identical files produce same hash."""
        from src.ingestion import compute_file_hash

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        content = "identical content"

        file1.write_text(content)
        file2.write_text(content)

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        """Test that different files produce different hashes."""
        from src.ingestion import compute_file_hash

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content A")
        file2.write_text("content B")

        assert compute_file_hash(file1) != compute_file_hash(file2)


class TestGetVideoMetadata:
    """Test video metadata extraction."""

    @patch("src.ingestion.subprocess.run")
    def test_metadata_extraction_success(self, mock_run):
        """Test successful metadata extraction."""
        from src.ingestion import get_video_metadata

        # Mock ffprobe output
        mock_data = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "pix_fmt": "yuv420p",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                },
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "120.5"},
        }

        mock_run.return_value = MagicMock(stdout=json.dumps(mock_data), stderr="")

        result = get_video_metadata(Path("test.mp4"))

        assert result["duration_sec"] == 120.5
        assert result["resolution"] == [1920, 1080]
        assert result["fps"] == 30.0
        assert (result["video_codec"], result["pix_fmt"], result["audio_codec"]) == (
            "h264",
            "yuv420p",
            "aac",
        )

    @patch("src.ingestion.subprocess.run")
    def test_metadata_extraction_no_video_stream(self, mock_run):
        """Test failure when no video stream found."""
        from src.ingestion import IngestionError, get_video_metadata

        mock_data = {"streams": [{"codec_type": "audio"}], "format": {}}

        mock_run.return_value = MagicMock(stdout=json.dumps(mock_data), stderr="")

        with pytest.raises(IngestionError, match="No video stream"):
            get_video_metadata(Path("test.mp4"))

    @patch("src.ingestion.subprocess.run")
    def test_metadata_extraction_ffprobe_failure(self, mock_run):
        """Test failure when ffprobe command fails."""
        import subprocess

        from src.ingestion import IngestionError, get_video_metadata

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")

        with pytest.raises(IngestionError):
            get_video_metadata(Path("test.mp4"))


class TestPreprocessVideo:
    """Test video preprocessing."""

    @patch("src.ingestion.subprocess.run")
    def test_preprocess_video_success(self, mock_run, tmp_path):
        """Test successful video preprocessing."""
        from src.config import PreprocessingConfig
        from src.ingestion import preprocess_video

        config = PreprocessingConfig()
        mock_run.return_value = MagicMock(stdout="", stderr="")

        # Should not raise
        preprocess_video(Path("input.mp4"), tmp_path / "output" / "prep.mp4", config)

        # Verify ffmpeg was called
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args[0]
        assert "-vf" in call_args
        assert "scale=-2:'min(1080,ih)'" in call_args


class TestPreprocessRemux:
    """A compatible source is remuxed, not re-encoded."""

    H264_1080 = {
        "resolution": [1920, 1080],
        "video_codec": "h264",
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
    }

    def _cmd(self, source_meta, **config_kwargs):
        from src.config import PreprocessingConfig
        from src.ingestion import preprocess_video

        with patch("src.ingestion.subprocess.run") as run_mock:
            preprocess_video(
                Path("in.mp4"),
                Path("out/prep.mp4"),
                PreprocessingConfig(**config_kwargs),
                source_meta=source_meta,
            )
        return run_mock.call_args[0][0]

    def test_h264_1080p_is_remuxed(self):
        cmd = self._cmd(self.H264_1080)
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert cmd[cmd.index("-c:a") + 1] == "copy"
        assert "-vf" not in cmd and "-crf" not in cmd
        assert "+faststart" in cmd

    def test_non_aac_audio_is_encoded_on_remux(self):
        cmd = self._cmd({**self.H264_1080, "audio_codec": "pcm_s16le"})
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert cmd[cmd.index("-c:a") + 1] == "aac"

    @pytest.mark.parametrize(
        "override",
        [
            {"resolution": [3840, 2160]},
            {"video_codec": "hevc"},
            {"pix_fmt": "yuv420p10le"},
        ],
    )
    def test_incompatible_source_is_reencoded(self, override):
        cmd = self._cmd({**self.H264_1080, **override})
        assert cmd[cmd.index("-c:v") + 1] == "h264"
        assert "scale=-2:'min(1080,ih)'" in cmd

    def test_copy_can_be_disabled(self):
        cmd = self._cmd(self.H264_1080, copy_compatible_video=False)
        assert "-vf" in cmd

    def test_without_metadata_reencodes(self):
        cmd = self._cmd(None)
        assert "-vf" in cmd


class TestExtractAudio:
    """Test audio extraction."""

    @patch("src.ingestion.subprocess.run")
    def test_extract_audio_success(self, mock_run):
        """Test successful audio extraction."""
        from src.config import PreprocessingConfig
        from src.ingestion import extract_audio

        config = PreprocessingConfig()
        mock_run.return_value = MagicMock(stdout="", stderr="")

        # Should not raise
        extract_audio(Path("video.mp4"), Path("audio/raw.wav"), config)

        # Verify ffmpeg was called with correct params
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args[0]
        assert "-vn" in call_args  # No video
        assert "-acodec" in call_args
        assert "pcm_s16le" in call_args  # PCM16
        assert "-ar" in call_args
        assert "16000" in call_args  # 16kHz
        assert "-ac" in call_args
        assert "1" in call_args  # Mono


class TestIngestFunction:
    """Test main ingest function."""

    @patch("src.ingestion.compute_file_hash")
    @patch("src.ingestion.get_video_metadata")
    @patch("src.ingestion.preprocess_video")
    @patch("src.ingestion.extract_audio")
    @patch("src.ingestion.save_metadata")
    def test_ingest_local_file(
        self, mock_save_meta, mock_extract, mock_preprocess, mock_get_meta, mock_hash, tmp_path
    ):
        """Test ingestion of local file."""
        from src.config import Config
        from src.ingestion import ingest

        # Create a dummy input file
        input_file = tmp_path / "test_video.mp4"
        input_file.write_bytes(b"dummy video content")

        # Mock dependencies
        mock_hash.return_value = "abc123"
        mock_get_meta.return_value = {"duration_sec": 60.0, "resolution": [1920, 1080], "fps": 30.0}

        cfg = Config.load("config/config.yaml")

        result = ingest(str(input_file), cfg, dry_run=False)

        assert result is not None
        assert result["source"] == str(input_file)
        assert "source_hash" in result
        assert result["duration_sec"] == 60.0

    def test_ingest_dry_run(self):
        """Test dry run mode."""
        from src.config import Config
        from src.ingestion import ingest

        cfg = Config.load("config/config.yaml")

        result = ingest("test.mp4", cfg, dry_run=True)

        assert result is not None
        assert result["status"] == "dry_run_passed"

    def test_ingest_missing_file(self):
        """Test ingestion with missing file."""
        from src.config import Config
        from src.ingestion import IngestionError, ingest

        cfg = Config.load("config/config.yaml")

        with pytest.raises(IngestionError, match="not found"):
            ingest("/nonexistent/path/video.mp4", cfg, dry_run=False)

    @patch("src.ingestion.download_youtube_video")
    @patch("src.ingestion.compute_file_hash")
    @patch("src.ingestion.get_video_metadata")
    @patch("src.ingestion.preprocess_video")
    @patch("src.ingestion.extract_audio")
    @patch("src.ingestion.save_metadata")
    def test_ingest_youtube_url(
        self,
        mock_save_meta,
        mock_extract,
        mock_preprocess,
        mock_get_meta,
        mock_hash,
        mock_download,
        tmp_path,
    ):
        """Test ingestion from YouTube URL."""
        from src.config import Config
        from src.ingestion import ingest

        # Mock download to return a temp file
        temp_file = tmp_path / "downloaded.mp4"
        temp_file.write_bytes(b"dummy")
        mock_download.return_value = temp_file

        mock_hash.return_value = "xyz789"
        mock_get_meta.return_value = {"duration_sec": 120.0, "resolution": [1280, 720], "fps": 25.0}

        cfg = Config.load("config/config.yaml")

        result = ingest("https://youtube.com/watch?v=test123", cfg, dry_run=False)

        assert result is not None
        assert "youtube" in result["source"].lower() or "https" in result["source"]
        mock_download.assert_called_once()

    @patch("src.ingestion.subprocess.run")
    def test_download_youtube_video_preserves_downloaded_file(self, mock_run, tmp_path):
        """Downloaded file must survive temp_dir cleanup (C1 regression)."""
        import subprocess

        from src.ingestion import download_youtube_video

        temp_dir = tmp_path / "temp_download"
        temp_dir.mkdir()
        output_path = tmp_path / "video" / "source.mp4"

        def fake_run(cmd, **kwargs):
            (temp_dir / "downloaded.mp4").write_bytes(b"dummy")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        mock_run.side_effect = fake_run

        result = download_youtube_video("https://youtube.com/watch?v=test", output_path, temp_dir)

        # Downloaded file must be moved OUT of temp_dir and still exist
        assert result.exists()
        assert result.read_bytes() == b"dummy"
        # temp_dir must be cleaned up
        assert not temp_dir.exists()


class TestSaveMetadata:
    """Test metadata saving."""

    def test_save_metadata_creates_file(self, tmp_path):
        """Test that save_metadata creates JSON file."""
        from src.ingestion import save_metadata

        output_path = tmp_path / "meta.json"
        meta = {"source": "test.mp4", "duration_sec": 60.0, "resolution": [1920, 1080]}

        save_metadata(meta, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded == meta

    def test_save_metadata_creates_directory(self, tmp_path):
        """Test that save_metadata creates parent directories."""
        from src.ingestion import save_metadata

        output_path = tmp_path / "nested" / "dir" / "meta.json"
        meta = {"test": "data"}

        save_metadata(meta, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()


class TestIntegrationScenarios:
    """Integration tests for various scenarios."""

    def test_broken_file_handling(self, tmp_path):
        """Test graceful handling of corrupted/broken files."""
        from src.config import Config
        from src.ingestion import IngestionError, ingest

        # Create a broken file (not a valid video)
        broken_file = tmp_path / "broken.mp4"
        broken_file.write_bytes(b"not a video file at all")

        cfg = Config.load("config/config.yaml")

        # Should fail gracefully with IngestionError
        with pytest.raises(IngestionError):
            ingest(str(broken_file), cfg, dry_run=False)
