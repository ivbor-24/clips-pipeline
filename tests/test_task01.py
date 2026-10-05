"""
Tests for TASK-01: Project Skeleton & CLI Orchestration
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.gpu_utils import GPUConfig


class TestConfigLoading:
    """Test configuration loading and validation."""

    def test_load_valid_config(self):
        """Test loading a valid configuration file."""
        config_path = "config/config.yaml"
        assert Path(config_path).exists(), f"Config file not found: {config_path}"

        cfg = Config.load(config_path)
        assert cfg is not None
        assert cfg.input.cache_dir == "artifacts/cache"
        assert cfg.transcription.model == "large-v3-turbo"
        assert cfg.scoring.min_score_threshold == 0.45

    def test_base_config_is_machine_neutral(self):
        """Machine-specific settings live in profiles, not in config.yaml."""
        cfg = Config.load("config/config.yaml")
        assert cfg.transcription.engine == "faster_whisper"
        assert cfg.transcription.language == "auto"
        assert cfg.scoring.llm.max_segments is None
        assert cfg.rendering.render_timeout_sec == 300

    def test_intel_arc_profile(self):
        """The Intel Arc profile switches to whisper.cpp with hardware-only settings."""
        cfg = Config.load("config/config.yaml").load_profile("intel_arc")
        assert cfg.transcription.engine == "whisper_cpp"
        assert cfg.transcription.language == "auto"
        assert cfg.transcription.whisper_cpp_max_context == 0
        assert cfg.transcription.whisper_cpp_model_path.endswith("ggml-large-v3-turbo.bin")
        # Hardware only: language, scoring and fonts come from config.yaml.
        assert cfg.scoring.strategy == "llm_windows"
        assert cfg.scoring.llm.llm_weight == 0.4
        assert cfg.rendering.render_timeout_sec == 900
        assert cfg.rendering.subtitle_fonts_dir is None
        # untouched sections keep base values
        assert cfg.scoring.llm.provider == "llama_cpp"

    def test_old_profile_name_still_loads(self):
        """Existing jobs may name the profile by its old name local_arc."""
        cfg = Config.load("config/config.yaml").load_profile("local_arc")
        assert cfg.transcription.engine == "whisper_cpp"

    def test_load_missing_config(self):
        """Test graceful failure when config file is missing."""
        with pytest.raises(FileNotFoundError):
            Config.load("nonexistent/config.yaml")

    def test_config_sections_exist(self):
        """Test that all required config sections are present."""
        cfg = Config.load("config/config.yaml")

        # Check all sections exist
        assert hasattr(cfg, "input")
        assert hasattr(cfg, "preprocessing")
        assert hasattr(cfg, "transcription")
        assert hasattr(cfg, "scoring")
        assert hasattr(cfg, "cropping")
        assert hasattr(cfg, "rendering")
        assert hasattr(cfg, "output")
        assert hasattr(cfg, "logging")

    def test_config_defaults(self):
        """Test that default values are set correctly."""
        cfg = Config.load("config/config.yaml")

        # Preprocessing defaults
        assert cfg.preprocessing.audio_sample_rate == 16000
        assert cfg.preprocessing.audio_channels == 1
        assert cfg.preprocessing.video_max_height == 1080

        # Scoring defaults
        assert cfg.scoring.min_duration == 45
        assert cfg.scoring.max_duration == 90
        assert cfg.scoring.max_clips_per_video == 12
        assert cfg.scoring.strategy == "llm_windows"

        # Cropping defaults
        assert cfg.cropping.output_width == 1080
        assert cfg.cropping.output_height == 1920


class TestEmptyConfig:
    """Test behavior with empty or invalid configurations."""

    def test_empty_yaml_file(self, tmp_path):
        """Test graceful exit on empty config file."""
        empty_config = tmp_path / "empty.yaml"
        empty_config.write_text("")

        # Should fail gracefully with appropriate error
        with pytest.raises(Exception):
            Config.load(str(empty_config))

    def test_partial_config_uses_defaults(self, tmp_path):
        """Test that missing fields use defaults."""
        partial_config = tmp_path / "partial.yaml"
        partial_config.write_text("""
input:
  cache_dir: "custom_cache"
""")

        cfg = Config.load(str(partial_config))
        assert cfg.input.cache_dir == "custom_cache"
        # Other sections should use defaults
        assert cfg.transcription.model == "large-v3-turbo"


class TestLoggingSetup:
    """Test logging configuration."""

    def test_log_directory_creation(self, tmp_path):
        """Test that log directory can be created."""

        from src.logger import setup_logging

        # Setup should create directories if needed
        log_path = tmp_path / "artifacts" / "logs" / "test_pipeline.log"
        setup_logging(log_file=str(log_path))

        assert log_path.parent.exists()


class TestDirectoryStructure:
    """Test project directory structure."""

    def test_required_directories_exist(self):
        """Test that the source directories exist.

        Runtime directories (artifacts/, output/) are made by the pipeline itself;
        the test does not create them in the checkout, where they hold real results.
        """
        for dir_path in ["config", "src", "tests"]:
            assert Path(dir_path).is_dir()

    def test_cli_module_exists(self):
        """Test that CLI module exists."""
        from src import cli

        assert hasattr(cli, "app")
        assert hasattr(cli, "process")

    def test_config_module_exists(self):
        """Test that config module exists."""
        from src import config

        assert hasattr(config, "Config")

    def test_logger_module_exists(self):
        """Test that logger module exists."""
        from src import logger

        assert hasattr(logger, "setup_logging")
        assert hasattr(logger, "get_logger")


class TestPipelineOrchestrator:
    """Test the end-to-end pipeline orchestrator."""

    def test_cleanup_vram_calls_gc_and_torch(self):
        """Test that _cleanup_vram calls gc.collect and gpu_config.cleanup_fn."""
        import gc as gc_module

        from src.pipeline import _cleanup_vram

        mock_cleanup = MagicMock()
        gpu_config = GPUConfig(
            backend="cuda",
            whisper_device="cuda",
            whisper_compute_type="float16",
            cleanup_fn=mock_cleanup,
        )

        with patch.object(gc_module, "collect") as mock_gc:
            _cleanup_vram(gpu_config)
            mock_gc.assert_called_once()
            mock_cleanup.assert_called_once()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.artifact_validator.is_valid_meta", return_value=False)
    @patch("src.artifact_validator.is_valid_transcript", return_value=False)
    @patch("src.artifact_validator.is_valid_scored_segments", return_value=False)
    @patch("src.artifact_validator.is_valid_crop_params", return_value=False)
    @patch("src.artifact_validator.is_valid_manifest", return_value=False)
    @patch("src.artifact_validator.get_expected_duration_from_meta", return_value=0.0)
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_run_pipeline_calls_all_stages(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect,
        mock_render,
        mock_review,
        mock_dur,
        mock_manifest,
        mock_crop,
        mock_scored,
        mock_trans,
        mock_meta,
        mock_get_gpu,
    ):
        """Test that run_pipeline calls all stages in order."""
        from src.pipeline import run_pipeline

        cfg = Config.load("config/config.yaml")

        mock_ingest.return_value = {"status": "ok"}
        mock_transcribe.return_value = [{"status": "ok"}]
        mock_score.return_value = [{"status": "ok"}]
        mock_detect.return_value = [{"status": "ok"}]
        mock_render.return_value = [{"status": "ok"}]
        mock_review.return_value = {"clips": {}}

        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu_config
        run_pipeline(cfg, "test.mp4", resume=False, force=False)

        mock_ingest.assert_called_once_with("test.mp4", cfg, dry_run=False)
        mock_transcribe.assert_called_once_with(cfg, dry_run=False)
        mock_score.assert_called_once_with(cfg, dry_run=False)
        mock_detect.assert_called_once_with(cfg, dry_run=False)
        mock_render.assert_called_once_with(cfg, dry_run=False, reuse=True)
        mock_review.assert_called_once()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.artifact_validator.is_valid_meta", return_value=False)
    @patch("src.artifact_validator.is_valid_transcript", return_value=False)
    @patch("src.artifact_validator.is_valid_scored_segments", return_value=False)
    @patch("src.artifact_validator.is_valid_crop_params", return_value=False)
    @patch("src.artifact_validator.is_valid_manifest", return_value=False)
    @patch("src.artifact_validator.get_expected_duration_from_meta", return_value=0.0)
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_run_pipeline_stops_on_ingestion_error(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect,
        mock_render,
        mock_review,
        mock_dur,
        mock_manifest,
        mock_crop,
        mock_scored,
        mock_trans,
        mock_meta,
        mock_get_gpu,
    ):
        """Test that pipeline stops when ingestion fails."""
        from src.ingestion import IngestionError
        from src.pipeline import IngestionError as PipelineIngestionError
        from src.pipeline import run_pipeline

        cfg = Config.load("config/config.yaml")

        mock_ingest.side_effect = IngestionError("test failure")

        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu_config
        with pytest.raises(PipelineIngestionError):
            run_pipeline(cfg, "test.mp4", resume=False, force=False)

        mock_transcribe.assert_not_called()
        mock_render.assert_not_called()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.artifact_validator.is_valid_meta", return_value=False)
    @patch("src.artifact_validator.is_valid_transcript", return_value=False)
    @patch("src.artifact_validator.is_valid_scored_segments", return_value=False)
    @patch("src.artifact_validator.is_valid_crop_params", return_value=False)
    @patch("src.artifact_validator.is_valid_manifest", return_value=False)
    @patch("src.artifact_validator.get_expected_duration_from_meta", return_value=0.0)
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_run_pipeline_stops_on_transcription_error(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect,
        mock_render,
        mock_review,
        mock_dur,
        mock_manifest,
        mock_crop,
        mock_scored,
        mock_trans,
        mock_meta,
        mock_get_gpu,
    ):
        """Test that pipeline stops when transcription fails."""
        from src.pipeline import TranscriptionError as PipelineTranscriptionError
        from src.pipeline import run_pipeline
        from src.transcription import TranscriptionError

        cfg = Config.load("config/config.yaml")

        mock_ingest.return_value = {"status": "ok"}
        mock_transcribe.side_effect = TranscriptionError("test failure")

        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu_config
        with pytest.raises(PipelineTranscriptionError):
            run_pipeline(cfg, "test.mp4", resume=False, force=False)

        mock_score.assert_not_called()
        mock_render.assert_not_called()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.artifact_validator.is_valid_meta", return_value=False)
    @patch("src.artifact_validator.is_valid_transcript", return_value=False)
    @patch("src.artifact_validator.is_valid_scored_segments", return_value=False)
    @patch("src.artifact_validator.is_valid_crop_params", return_value=False)
    @patch("src.artifact_validator.is_valid_manifest", return_value=False)
    @patch("src.artifact_validator.get_expected_duration_from_meta", return_value=0.0)
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_run_pipeline_continues_on_review_error(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect,
        mock_render,
        mock_review,
        mock_dur,
        mock_manifest,
        mock_crop,
        mock_scored,
        mock_trans,
        mock_meta,
        mock_get_gpu,
    ):
        """Test that pipeline continues when review fails (non-critical)."""
        from src.pipeline import run_pipeline

        cfg = Config.load("config/config.yaml")

        mock_ingest.return_value = {"status": "ok"}
        mock_transcribe.return_value = [{"status": "ok"}]
        mock_score.return_value = [{"status": "ok"}]
        mock_detect.return_value = [{"status": "ok"}]
        mock_render.return_value = [{"status": "ok"}]
        mock_review.side_effect = Exception("review failure")

        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu_config
        run_pipeline(cfg, "test.mp4", resume=False, force=False)

        mock_review.assert_called_once()

    @patch("src.pipeline._cleanup_vram")
    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.artifact_validator.is_valid_meta", return_value=False)
    @patch("src.artifact_validator.is_valid_transcript", return_value=False)
    @patch("src.artifact_validator.is_valid_scored_segments", return_value=False)
    @patch("src.artifact_validator.is_valid_crop_params", return_value=False)
    @patch("src.artifact_validator.is_valid_manifest", return_value=False)
    @patch("src.artifact_validator.get_expected_duration_from_meta", return_value=0.0)
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_vram_cleanup_called_after_transcription_and_face_cropping(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect,
        mock_render,
        mock_review,
        mock_dur,
        mock_manifest,
        mock_crop,
        mock_scored,
        mock_trans,
        mock_meta,
        mock_get_gpu,
        mock_cleanup,
    ):
        """Test that VRAM cleanup is called after transcription and face_cropping."""
        from src.pipeline import run_pipeline

        cfg = Config.load("config/config.yaml")

        mock_ingest.return_value = {"status": "ok"}
        mock_transcribe.return_value = [{"status": "ok"}]
        mock_score.return_value = [{"status": "ok"}]
        mock_detect.return_value = [{"status": "ok"}]
        mock_render.return_value = [{"status": "ok"}]
        mock_review.return_value = {"clips": {}}

        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu_config
        run_pipeline(cfg, "test.mp4", resume=False, force=False)

        assert mock_cleanup.call_count == 2

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.artifact_validator.is_valid_meta", return_value=True)
    @patch("src.artifact_validator.is_valid_transcript", return_value=True)
    @patch("src.artifact_validator.is_valid_scored_segments", return_value=True)
    @patch("src.artifact_validator.is_valid_crop_params", return_value=True)
    @patch("src.artifact_validator.is_valid_manifest", return_value=True)
    @patch("src.artifact_validator.get_expected_duration_from_meta", return_value=100.0)
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_run_pipeline_skips_stages_on_resume(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect,
        mock_render,
        mock_review,
        mock_dur,
        mock_manifest,
        mock_crop,
        mock_scored,
        mock_trans,
        mock_meta,
        mock_exists,
        mock_get_gpu,
    ):
        """Test that run_pipeline skips stages when resume=True and artifacts are valid."""
        from src.pipeline import run_pipeline

        cfg = Config.load("config/config.yaml")

        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu_config
        run_pipeline(cfg, "test.mp4", resume=True, force=False)

        mock_ingest.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_score.assert_not_called()
        mock_detect.assert_not_called()
        mock_render.assert_not_called()
        mock_review.assert_called_once()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("src.artifact_validator.is_valid_meta", return_value=True)
    @patch("src.artifact_validator.is_valid_transcript", return_value=True)
    @patch("src.artifact_validator.is_valid_scored_segments", return_value=True)
    @patch("src.artifact_validator.is_valid_crop_params", return_value=True)
    @patch("src.artifact_validator.is_valid_manifest", return_value=True)
    @patch("src.artifact_validator.get_expected_duration_from_meta", return_value=100.0)
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_run_pipeline_force_ignores_resume(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect,
        mock_render,
        mock_review,
        mock_dur,
        mock_manifest,
        mock_crop,
        mock_scored,
        mock_trans,
        mock_meta,
        mock_exists,
        mock_get_gpu,
    ):
        """Test that force=True overrides resume and re-runs all stages."""
        from src.pipeline import run_pipeline

        cfg = Config.load("config/config.yaml")

        mock_ingest.return_value = {"status": "ok"}
        mock_transcribe.return_value = [{"status": "ok"}]
        mock_score.return_value = [{"status": "ok"}]
        mock_detect.return_value = [{"status": "ok"}]
        mock_render.return_value = [{"status": "ok"}]
        mock_review.return_value = {"clips": {}}

        mock_gpu_config = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu_config
        run_pipeline(cfg, "test.mp4", resume=True, force=True)

        mock_ingest.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_score.assert_called_once()
        mock_detect.assert_called_once()
        mock_render.assert_called_once()
        mock_review.assert_called_once()

    def test_process_accepts_force_parameter(self):
        """Test that process() accepts the force parameter."""
        import inspect

        from src.cli import process

        sig = inspect.signature(process)
        params = list(sig.parameters.keys())
        assert "force" in params


class TestCliForceMode:
    """Test force mode functionality."""

    def test_force_parameter_exists(self):
        """Test that force parameter exists in process()."""
        import inspect

        from src.cli import process

        sig = inspect.signature(process)
        params = list(sig.parameters.keys())
        assert "force" in params


class TestRichProgress:
    """Test Rich progress display in pipeline."""

    def test_progress_uses_rich_progress_class(self):
        """Test that RichProgressCallback uses rich.progress.Progress."""
        import inspect

        from src.cli import RichProgressCallback

        source = inspect.getsource(RichProgressCallback)
        assert "Progress(" in source
        assert "SpinnerColumn()" in source
        assert "TimeElapsedColumn()" in source

    def test_progress_shows_running_status(self):
        """Test that running stages show yellow status."""
        import inspect

        from src.cli import RichProgressCallback

        source = inspect.getsource(RichProgressCallback)
        assert "[yellow]" in source

    def test_progress_shows_complete_status(self):
        """Test that complete stages show green status."""
        import inspect

        from src.cli import RichProgressCallback

        source = inspect.getsource(RichProgressCallback)
        assert "[green]" in source

    def test_progress_shows_skipped_status(self):
        """Test that skipped stages show blue status."""
        import inspect

        from src.cli import RichProgressCallback

        source = inspect.getsource(RichProgressCallback)
        assert "[blue]" in source

    def test_progress_shows_error_status(self):
        """Test that failed stages show red status."""
        import inspect

        from src.cli import RichProgressCallback

        source = inspect.getsource(RichProgressCallback)
        assert "[red]" in source

    def test_progress_displays_elapsed_time(self):
        """Test that progress display format includes TimeElapsedColumn."""
        import inspect

        from src.cli import RichProgressCallback

        source = inspect.getsource(RichProgressCallback)
        assert "TimeElapsedColumn()" in source
