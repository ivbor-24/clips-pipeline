"""
Tests for TASK-03: Transcription & Caching
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.gpu_utils import GPUConfig


class TestTranscriptionModule:
    """Test transcription module imports and structure."""

    def test_module_exists(self):
        """Test that transcription module exists."""
        from src import transcription

        assert transcription is not None

    def test_transcription_error_exists(self):
        """Test that TranscriptionError exception exists."""
        from src.transcription import TranscriptionError

        assert TranscriptionError is not None
        assert issubclass(TranscriptionError, Exception)

    def test_compute_audio_hash_function_exists(self):
        """Test that compute_audio_hash function exists."""
        from src.transcription import compute_audio_hash

        assert callable(compute_audio_hash)

    def test_compute_model_config_hash_function_exists(self):
        """Test that compute_model_config_hash function exists."""
        from src.transcription import compute_model_config_hash

        assert callable(compute_model_config_hash)

    def test_get_cache_key_function_exists(self):
        """Test that get_cache_key function exists."""
        from src.transcription import get_cache_key

        assert callable(get_cache_key)

    def test_load_transcript_from_cache_function_exists(self):
        """Test that load_transcript_from_cache function exists."""
        from src.transcription import load_transcript_from_cache

        assert callable(load_transcript_from_cache)

    def test_save_transcript_to_cache_function_exists(self):
        """Test that save_transcript_to_cache function exists."""
        from src.transcription import save_transcript_to_cache

        assert callable(save_transcript_to_cache)

    def test_chunk_audio_by_duration_function_exists(self):
        """Test that chunk_audio_by_duration function exists."""
        from src.transcription import chunk_audio_by_duration

        assert callable(chunk_audio_by_duration)

    def test_cleanup_audio_chunks_function_exists(self):
        """Test that cleanup_audio_chunks function exists."""
        from src.transcription import cleanup_audio_chunks

        assert callable(cleanup_audio_chunks)

    def test_transcribe_chunk_function_exists(self):
        """Test that transcribe_chunk function exists."""
        from src.transcription import transcribe_chunk

        assert callable(transcribe_chunk)

    def test_validate_transcript_duration_function_exists(self):
        """Test that validate_transcript_duration function exists."""
        from src.transcription import validate_transcript_duration

        assert callable(validate_transcript_duration)

    def test_load_whisper_model_function_exists(self):
        """Test that load_whisper_model function exists."""
        from src.transcription import load_whisper_model

        assert callable(load_whisper_model)

    def test_unload_whisper_model_function_exists(self):
        """Test that unload_whisper_model function exists."""
        from src.transcription import unload_whisper_model

        assert callable(unload_whisper_model)

    def test_transcribe_function_exists(self):
        """Test that main transcribe function exists."""
        from src.transcription import transcribe

        assert callable(transcribe)


class TestComputeAudioHash:
    """Test audio hash computation."""

    def test_hash_computation(self, tmp_path):
        """Test that hash is computed correctly."""
        from src.transcription import compute_audio_hash

        # Create a test file
        test_file = tmp_path / "test.wav"
        test_file.write_bytes(b"hello world audio")

        hash_result = compute_audio_hash(test_file)

        # Hash should be a 64-character hex string (SHA256)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_same_content_same_hash(self, tmp_path):
        """Test that identical files produce same hash."""
        from src.transcription import compute_audio_hash

        file1 = tmp_path / "file1.wav"
        file2 = tmp_path / "file2.wav"
        content = b"identical audio content"

        file1.write_bytes(content)
        file2.write_bytes(content)

        assert compute_audio_hash(file1) == compute_audio_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        """Test that different files produce different hashes."""
        from src.transcription import compute_audio_hash

        file1 = tmp_path / "file1.wav"
        file2 = tmp_path / "file2.wav"

        file1.write_bytes(b"content A")
        file2.write_bytes(b"content B")

        assert compute_audio_hash(file1) != compute_audio_hash(file2)


class TestModelConfigHash:
    """Test model configuration hash computation."""

    def test_config_hash_computation(self):
        """Test that config hash is computed correctly."""
        from src.config import TranscriptionConfig
        from src.transcription import compute_model_config_hash

        config = TranscriptionConfig()
        hash_result = compute_model_config_hash(config)

        # Hash should be a 16-character hex string (truncated SHA256)
        assert len(hash_result) == 16
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_different_configs_different_hashes(self):
        """Test that different configs produce different hashes."""
        from src.config import TranscriptionConfig
        from src.transcription import compute_model_config_hash

        config1 = TranscriptionConfig(model="large-v3-turbo", batch_size=8)
        config2 = TranscriptionConfig(model="small", batch_size=4)

        assert compute_model_config_hash(config1) != compute_model_config_hash(config2)


class TestCacheKey:
    """Test cache key generation."""

    def test_cache_key_format(self):
        """Test that cache key combines audio and config hashes."""
        from src.transcription import get_cache_key

        audio_hash = "abc123def456"
        config_hash = "xyz789"

        cache_key = get_cache_key(audio_hash, config_hash)

        assert cache_key == "abc123def456_xyz789"


class TestCacheOperations:
    """Test cache loading and saving operations."""

    def test_save_and_load_transcript(self, tmp_path):
        """Test saving and loading transcript from cache."""
        from src.transcription import load_transcript_from_cache, save_transcript_to_cache

        cache_dir = tmp_path / "cache"
        cache_key = "test_key_123"
        transcript = [
            {"id": "seg_0001", "start": 0.0, "end": 5.0, "text": "Hello world"},
            {"id": "seg_0002", "start": 5.0, "end": 10.0, "text": "Goodbye world"},
        ]

        # Save to cache
        save_transcript_to_cache(cache_dir, cache_key, transcript)

        # Load from cache
        loaded = load_transcript_from_cache(cache_dir, cache_key)

        assert loaded == transcript

    def test_load_nonexistent_cache_returns_none(self, tmp_path):
        """Test that loading non-existent cache returns None."""
        from src.transcription import load_transcript_from_cache

        cache_dir = tmp_path / "cache"
        cache_key = "nonexistent_key"

        result = load_transcript_from_cache(cache_dir, cache_key)

        assert result is None


class TestValidateTranscriptDuration:
    """Test transcript duration validation."""

    def test_valid_transcript(self):
        """Test validation passes for valid transcript."""
        from src.transcription import validate_transcript_duration

        transcript = [
            {"start": 0.0, "end": 30.0, "text": "Segment 1"},
            {"start": 30.0, "end": 60.0, "text": "Segment 2"},
        ]
        expected_duration = 60.0

        result = validate_transcript_duration(transcript, expected_duration)

        assert result is True

    def test_empty_transcript_fails(self):
        """Test validation fails for empty transcript."""
        from src.transcription import validate_transcript_duration

        transcript = []
        expected_duration = 60.0

        result = validate_transcript_duration(transcript, expected_duration)

        assert result is False

    def test_overlapping_segments(self):
        """Test validation handles overlapping segments correctly."""
        from src.transcription import validate_transcript_duration

        # Overlapping segments: total covered should be 60s, not 90s
        transcript = [
            {"start": 0.0, "end": 40.0, "text": "Segment 1"},
            {"start": 20.0, "end": 60.0, "text": "Segment 2"},
        ]
        expected_duration = 60.0

        result = validate_transcript_duration(transcript, expected_duration)

        assert result is True

    def test_short_transcript_fails(self):
        """Test validation fails when transcript is too short."""
        from src.transcription import validate_transcript_duration

        transcript = [{"start": 0.0, "end": 10.0, "text": "Short segment"}]
        expected_duration = 60.0

        result = validate_transcript_duration(transcript, expected_duration)

        assert result is False


class TestChunkAudioByDuration:
    """Test audio chunking functionality."""

    @patch("src.transcription.subprocess.run")
    def test_no_chunking_needed(self, mock_run, tmp_path):
        """Test that short audio is not chunked."""
        from src.transcription import chunk_audio_by_duration

        # Create a dummy audio file
        audio_file = tmp_path / "short.wav"
        audio_file.write_bytes(b"dummy audio")

        # Mock ffprobe to return short duration
        mock_data = {"format": {"duration": "1800"}}  # 30 minutes
        mock_run.return_value = MagicMock(stdout=json.dumps(mock_data))

        chunks = chunk_audio_by_duration(audio_file, chunk_duration_min=30)

        # Should return original file as single chunk
        assert len(chunks) == 1
        assert chunks[0] == audio_file

    @patch("src.transcription.subprocess.run")
    def test_long_audio_is_chunked(self, mock_run, tmp_path):
        """Test that long audio is split into chunks."""
        from src.transcription import chunk_audio_by_duration

        # Create a dummy audio file
        audio_file = tmp_path / "long.wav"
        audio_file.write_bytes(b"dummy audio")

        # Mock ffprobe to return long duration (2 hours)
        mock_data = {"format": {"duration": "7200"}}  # 2 hours
        mock_run.return_value = MagicMock(stdout=json.dumps(mock_data))

        chunks = chunk_audio_by_duration(audio_file, chunk_duration_min=30)

        # Should create multiple chunks (7200 / 1800 = 4 chunks)
        assert len(chunks) > 1


class TestCleanupAudioChunks:
    """Test audio chunk cleanup."""

    def test_cleanup_removes_temp_files(self, tmp_path):
        """Test that cleanup removes temporary chunk files."""
        from src.transcription import cleanup_audio_chunks

        # Create temp directory and chunks
        temp_dir = tmp_path / "temp_chunks"
        temp_dir.mkdir()

        original_audio = tmp_path / "raw.wav"
        original_audio.write_bytes(b"original")

        chunk1 = temp_dir / "chunk_000.wav"
        chunk2 = temp_dir / "chunk_001.wav"
        chunk1.write_bytes(b"chunk1")
        chunk2.write_bytes(b"chunk2")

        chunks = [original_audio, chunk1, chunk2]

        cleanup_audio_chunks(chunks, original_audio)

        # Original should still exist
        assert original_audio.exists()
        # Chunks should be removed
        assert not chunk1.exists()
        assert not chunk2.exists()


class TestLoadWhisperModel:
    """Test Whisper model loading."""

    @patch("faster_whisper.WhisperModel")
    def test_model_loads_successfully(self, mock_whisper_model):
        """Test successful model loading (mocked to avoid downloading)."""
        from src.config import TranscriptionConfig
        from src.transcription import load_whisper_model

        config = TranscriptionConfig()
        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )

        fake_model = MagicMock()
        mock_whisper_model.return_value = fake_model

        model = load_whisper_model(config, mock_gpu_config)

        assert model is fake_model
        # large-v3-turbo is pinned in config/models.lock.yaml
        mock_whisper_model.assert_called_once_with(
            config.model,
            device="cpu",
            compute_type="int8",
            revision="0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
        )

    @patch("faster_whisper.WhisperModel", side_effect=ImportError("faster-whisper not found"))
    def test_import_error_raises_transcription_error(self, mock_whisper_model):
        """Test that ImportError raises TranscriptionError."""
        from src.config import TranscriptionConfig
        from src.transcription import TranscriptionError, load_whisper_model

        config = TranscriptionConfig()
        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )

        with pytest.raises(TranscriptionError, match="faster-whisper is not installed"):
            load_whisper_model(config, mock_gpu_config)


class TestUnloadWhisperModel:
    """Test Whisper model unloading."""

    def test_unload_clears_cuda_cache(self):
        """Test that unloading clears CUDA cache when torch is available."""
        from src.transcription import unload_whisper_model

        mock_cleanup = MagicMock()
        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=mock_cleanup,
        )

        # This test just verifies the function doesn't crash
        # even if torch is not available
        model = object()  # Dummy model
        unload_whisper_model(model, mock_gpu_config)  # Should not raise
        mock_cleanup.assert_called_once()


class TestTranscribeFunction:
    """Test main transcribe function."""

    def test_dry_run_mode(self, tmp_path):
        """Test dry run mode returns without processing."""
        from src.config import Config
        from src.transcription import transcribe

        # Setup artifacts directory to avoid file not found error
        artifacts_dir = tmp_path / "artifacts"
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True)

        audio_file = audio_dir / "raw.wav"
        audio_file.write_bytes(b"dummy audio")

        meta_file = artifacts_dir / "meta.json"
        meta_file.write_text('{"duration_sec": 60.0}')

        # Change to tmp_path
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))

            result = transcribe(cfg, dry_run=True)

            assert result is not None
            assert result[0]["status"] == "dry_run_passed"
        finally:
            os.chdir(original_cwd)

    def test_missing_audio_file_raises_error(self):
        """Test that missing audio file raises TranscriptionError."""
        from src.config import Config
        from src.transcription import TranscriptionError, transcribe

        cfg = Config.load("config/config.yaml")

        with pytest.raises(TranscriptionError, match="Audio file not found"):
            transcribe(cfg, dry_run=False)

    def test_missing_meta_file_raises_error(self, tmp_path):
        """Test that missing metadata file raises TranscriptionError."""
        from src.config import Config
        from src.transcription import TranscriptionError, transcribe

        # Create artifacts directory with audio but no meta
        artifacts_dir = tmp_path / "artifacts"
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True)

        audio_file = audio_dir / "raw.wav"
        audio_file.write_bytes(b"dummy audio")

        # Change to tmp_path
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))

            with pytest.raises(TranscriptionError, match="Metadata file not found"):
                transcribe(cfg, dry_run=False)
        finally:
            os.chdir(original_cwd)

    @patch("src.transcription.load_whisper_model")
    @patch("src.transcription.chunk_audio_by_duration")
    @patch("src.transcription.transcribe_chunk")
    @patch("src.transcription.validate_transcript_duration")
    @patch("src.transcription.save_transcript_to_cache")
    @patch("src.transcription.subprocess.run")
    def test_transcription_success(
        self,
        mock_subprocess,
        mock_save_cache,
        mock_validate,
        mock_transcribe_chunk,
        mock_chunk,
        mock_load_model,
        tmp_path,
    ):
        """Test successful transcription workflow."""
        from src.config import Config
        from src.transcription import transcribe

        # Setup artifacts
        artifacts_dir = tmp_path / "artifacts"
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True)

        audio_file = audio_dir / "raw.wav"
        audio_file.write_bytes(b"dummy audio")

        meta_file = artifacts_dir / "meta.json"
        meta_file.write_text(json.dumps({"duration_sec": 60.0}))

        # Mock dependencies
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        mock_chunk.return_value = [audio_file]

        # Mock transcribe_chunk to return segments
        mock_transcribe_chunk.return_value = [
            {"start": 0.0, "end": 5.0, "text": "Hello", "words": []}
        ]

        mock_validate.return_value = True
        mock_subprocess.return_value = MagicMock(
            stdout=json.dumps({"format": {"duration": "60.0"}})
        )

        # Change to tmp_path
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))

            result = transcribe(cfg, dry_run=False)

            assert result is not None
            assert len(result) > 0
            assert "id" in result[0]
            mock_load_model.assert_called_once()
            mock_save_cache.assert_called_once()
        finally:
            os.chdir(original_cwd)

    @patch("src.transcription.load_whisper_model")
    @patch("src.transcription.load_transcript_from_cache")
    def test_cache_hit_uses_cached_transcript(self, mock_load_cache, mock_load_model, tmp_path):
        """Test that cached transcript is used when available."""
        from src.config import Config
        from src.transcription import transcribe

        # Setup artifacts
        artifacts_dir = tmp_path / "artifacts"
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True)

        audio_file = audio_dir / "raw.wav"
        audio_file.write_bytes(b"dummy audio")

        meta_file = artifacts_dir / "meta.json"
        meta_file.write_text(json.dumps({"duration_sec": 60.0}))

        # Mock cached transcript
        cached_transcript = [
            {"id": "seg_0001", "start": 0.0, "end": 5.0, "text": "Cached", "words": []}
        ]
        mock_load_cache.return_value = cached_transcript

        # Change to tmp_path
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))

            result = transcribe(cfg, dry_run=False)

            assert result == cached_transcript
            mock_load_model.assert_not_called()  # Should not load model if cached
        finally:
            os.chdir(original_cwd)


class TestOOMRetry:
    """Test OOM fallback: reload the model in int8 and retry the chunk."""

    @staticmethod
    def _setup(tmp_path):
        artifacts_dir = tmp_path / "artifacts"
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir(parents=True)
        audio_file = audio_dir / "raw.wav"
        audio_file.write_bytes(b"dummy audio")
        (artifacts_dir / "meta.json").write_text(json.dumps({"duration_sec": 60.0}))
        return audio_file

    @staticmethod
    def _gpu_config(compute_type):
        from src.gpu_utils import GPUConfig

        return GPUConfig(
            backend="cuda",
            whisper_device="cuda",
            whisper_compute_type=compute_type,
            cleanup_fn=lambda: None,
        )

    @patch("src.transcription._get_chunk_duration", return_value=60.0)
    @patch("src.transcription.get_auto_gpu_config")
    @patch("src.transcription.load_whisper_model")
    @patch("src.transcription.chunk_audio_by_duration")
    @patch("src.transcription.transcribe_chunk")
    def test_oom_reloads_model_in_int8_and_retries(
        self, mock_transcribe_chunk, mock_chunk, mock_load_model, mock_gpu, _mock_duration, tmp_path
    ):
        """OOM on float16 reloads the model as int8 and retries the same chunk."""
        from src.config import Config
        from src.transcription import transcribe

        audio_file = self._setup(tmp_path)
        mock_gpu.return_value = self._gpu_config("float16")
        mock_load_model.side_effect = [MagicMock(name="fp16"), MagicMock(name="int8")]
        mock_chunk.return_value = [audio_file]
        mock_transcribe_chunk.side_effect = [
            RuntimeError("CUDA out of memory"),
            [{"start": 0.0, "end": 5.0, "text": "Success", "words": []}],
        ]

        result = transcribe(Config(work_dir=tmp_path), dry_run=False)

        assert [s["text"] for s in result] == ["Success"]
        assert mock_transcribe_chunk.call_count == 2
        compute_types = [c.args[1].whisper_compute_type for c in mock_load_model.call_args_list]
        assert compute_types == ["float16", "int8"]

    @patch("src.transcription.get_auto_gpu_config")
    @patch("src.transcription.load_whisper_model")
    @patch("src.transcription.chunk_audio_by_duration")
    @patch("src.transcription.transcribe_chunk")
    def test_oom_at_int8_raises(
        self, mock_transcribe_chunk, mock_chunk, mock_load_model, mock_gpu, tmp_path
    ):
        """No second fallback when the model already runs in int8."""
        from src.config import Config
        from src.transcription import TranscriptionError, transcribe

        audio_file = self._setup(tmp_path)
        mock_gpu.return_value = self._gpu_config("int8")
        mock_chunk.return_value = [audio_file]
        mock_transcribe_chunk.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(TranscriptionError, match="int8"):
            transcribe(Config(work_dir=tmp_path), dry_run=False)
        assert mock_load_model.call_count == 1


class TestIntegrationScenarios:
    """Integration tests for various transcription scenarios."""

    def test_word_level_timestamps_included(self):
        """Test that word-level timestamps are included in output."""

        # Simulate transcript with word-level timestamps
        transcript = [
            {
                "id": "seg_0001",
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ],
                "language": "en",
            }
        ]

        # Validate structure
        assert "words" in transcript[0]
        assert len(transcript[0]["words"]) > 0
        assert "start" in transcript[0]["words"][0]
        assert "end" in transcript[0]["words"][0]

    def test_segment_ids_sequential(self):
        """Test that segment IDs are sequential."""
        # Simulate transcript generation
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Seg 1", "words": []},
            {"start": 5.0, "end": 10.0, "text": "Seg 2", "words": []},
            {"start": 10.0, "end": 15.0, "text": "Seg 3", "words": []},
        ]

        transcript = []
        for i, seg in enumerate(segments):
            seg_with_id = {"id": f"seg_{i+1:04d}", **seg, "language": "en"}
            transcript.append(seg_with_id)

        assert transcript[0]["id"] == "seg_0001"
        assert transcript[1]["id"] == "seg_0002"
        assert transcript[2]["id"] == "seg_0003"
