"""
Tests for the pipeline orchestrator module (src/pipeline.py).

Verifies:
- Stages are called in correct order
- progress_callback is called with correct arguments
- Exceptions are raised correctly
- Resume/force logic works
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.gpu_utils import GPUConfig
from src.pipeline import (
    STAGE_EXCEPTIONS,
    FaceCroppingError,
    IngestionError,
    PipelineError,
    RenderingError,
    ScoringError,
    TranscriptionError,
    run_pipeline,
)


@pytest.fixture
def mock_gpu_config():
    return GPUConfig(
        backend="cpu",
        whisper_device="cpu",
        whisper_compute_type="int8",
        cleanup_fn=MagicMock(),
    )


@pytest.fixture
def basic_config(tmp_path):
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config" / "config.yaml"
    config_path.write_text(
        "input:\n"
        "  cache_dir: artifacts/cache\n"
        "scoring:\n"
        "  min_duration: 1\n"
        "  max_duration: 15\n"
        "output:\n"
        "  clips_dir: output/clips\n"
        "  manifest_file: output/manifest.json\n"
        "  review_file: artifacts/review.json\n"
    )
    return Config.load(str(config_path))


def _create_valid_meta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "source": "test.mp4",
                "duration_sec": 10.0,
                "resolution": [1920, 1080],
                "fps": 30.0,
                "source_hash": "sha256:abc",
            },
            f,
        )


def _create_valid_transcript(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [
                {"id": "seg_0001", "start": 0.0, "end": 5.0, "text": "Hello world"},
                {"id": "seg_0002", "start": 5.0, "end": 10.0, "text": "Test segment"},
            ],
            f,
        )


def _create_valid_scored_segments(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [
                {"id": "seg_0001", "start": 0.0, "end": 5.0, "score": 0.8},
            ],
            f,
        )


def _create_valid_crop_params(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            [
                {
                    "clip_id": "clip_001",
                    "start": 0.0,
                    "end": 5.0,
                    "frames": [],
                    "fallback": "center_crop",
                },
            ],
            f,
        )


def _create_valid_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"total_clips": 1, "clips": [{"clip_id": "clip_001"}]}, f)


class TestPipelineExceptions:
    def test_exception_hierarchy(self):
        assert issubclass(IngestionError, PipelineError)
        assert issubclass(TranscriptionError, PipelineError)
        assert issubclass(ScoringError, PipelineError)
        assert issubclass(FaceCroppingError, PipelineError)
        assert issubclass(RenderingError, PipelineError)

    def test_stage_exceptions_mapping(self):
        assert STAGE_EXCEPTIONS["ingestion"] is IngestionError
        assert STAGE_EXCEPTIONS["transcription"] is TranscriptionError
        assert STAGE_EXCEPTIONS["scoring"] is ScoringError
        assert STAGE_EXCEPTIONS["face_cropping"] is FaceCroppingError
        assert STAGE_EXCEPTIONS["rendering"] is RenderingError


class TestRunPipelineStageOrder:
    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_stages_called_in_order(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        call_order = []

        mock_ingest.side_effect = lambda *a, **kw: call_order.append("ingestion")
        mock_transcribe.side_effect = lambda *a, **kw: call_order.append("transcription")
        mock_score.side_effect = lambda *a, **kw: call_order.append("scoring")
        mock_detect_faces.side_effect = lambda *a, **kw: call_order.append("face_cropping")
        mock_render.side_effect = lambda *a, **kw: call_order.append("rendering")
        mock_review.side_effect = lambda *a, **kw: call_order.append("review")

        run_pipeline(basic_config, "test.mp4")

        assert call_order == [
            "ingestion",
            "transcription",
            "scoring",
            "face_cropping",
            "rendering",
            "review",
        ]

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_each_stage_called_once(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config

        run_pipeline(basic_config, "test.mp4")

        mock_ingest.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_score.assert_called_once()
        mock_detect_faces.assert_called_once()
        mock_render.assert_called_once()
        mock_review.assert_called_once()


class TestProgressCallback:
    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_callback_called_for_each_stage(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        callback = MagicMock()

        run_pipeline(basic_config, "test.mp4", progress_callback=callback)

        expected_stages = [
            "ingestion",
            "transcription",
            "scoring",
            "face_cropping",
            "rendering",
            "review",
        ]
        for stage in expected_stages:
            callback.assert_any_call(stage, "started", None)
            callback.assert_any_call(stage, "completed", 100.0)

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_callback_started_before_completed(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        events = []

        def callback(stage, status, progress):
            events.append((stage, status))

        run_pipeline(basic_config, "test.mp4", progress_callback=callback)

        for stage in [
            "ingestion",
            "transcription",
            "scoring",
            "face_cropping",
            "rendering",
            "review",
        ]:
            started_idx = events.index((stage, "started"))
            completed_idx = events.index((stage, "completed"))
            assert started_idx < completed_idx

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_no_callback_does_not_raise(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config

        run_pipeline(basic_config, "test.mp4")


class TestPipelineExceptionsRaised:
    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.ingestion.ingest")
    def test_ingestion_error(
        self, mock_ingest, mock_get_gpu, basic_config, mock_gpu_config, monkeypatch, tmp_path
    ):
        from src.ingestion import IngestionError as ModuleIngestionError

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_ingest.side_effect = ModuleIngestionError("video not found")

        with pytest.raises(IngestionError, match="video not found"):
            run_pipeline(basic_config, "test.mp4")

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_transcription_error(
        self,
        mock_ingest,
        mock_transcribe,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        from src.transcription import TranscriptionError as ModuleTranscriptionError

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_transcribe.side_effect = ModuleTranscriptionError("whisper failed")

        with pytest.raises(TranscriptionError, match="whisper failed"):
            run_pipeline(basic_config, "test.mp4")

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_scoring_error(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        from src.scoring import ScoringError as ModuleScoringError

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_score.side_effect = ModuleScoringError("scoring failed")

        with pytest.raises(ScoringError, match="scoring failed"):
            run_pipeline(basic_config, "test.mp4")

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_face_cropping_error(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        from src.face_cropping import FaceCroppingError as ModuleFaceCroppingError

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_detect_faces.side_effect = ModuleFaceCroppingError("no face")

        with pytest.raises(FaceCroppingError, match="no face"):
            run_pipeline(basic_config, "test.mp4")

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_rendering_error(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        from src.rendering import RenderingError as ModuleRenderingError

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_render.side_effect = ModuleRenderingError("ffmpeg failed")

        with pytest.raises(RenderingError, match="ffmpeg failed"):
            run_pipeline(basic_config, "test.mp4")

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.ingestion.ingest")
    def test_unexpected_error_wrapped(
        self, mock_ingest, mock_get_gpu, basic_config, mock_gpu_config, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_ingest.side_effect = RuntimeError("something broke")

        with pytest.raises(IngestionError, match="Unexpected error"):
            run_pipeline(basic_config, "test.mp4")

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.ingestion.ingest")
    def test_failed_callback_on_error(
        self, mock_ingest, mock_get_gpu, basic_config, mock_gpu_config, monkeypatch, tmp_path
    ):
        from src.ingestion import IngestionError as ModuleIngestionError

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_ingest.side_effect = ModuleIngestionError("fail")
        callback = MagicMock()

        with pytest.raises(IngestionError):
            run_pipeline(basic_config, "test.mp4", progress_callback=callback)

        callback.assert_any_call("ingestion", "started", None)
        callback.assert_any_call("ingestion", "failed", None)


class TestResumeForceLogic:
    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_resume_skips_valid_artifacts(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config

        _create_valid_meta(tmp_path / "artifacts" / "meta.json")
        _create_valid_transcript(tmp_path / "artifacts" / "transcript.json")
        _create_valid_scored_segments(tmp_path / "artifacts" / "scored_segments.json")
        _create_valid_crop_params(tmp_path / "artifacts" / "crop_params.json")
        _create_valid_manifest(tmp_path / "output" / "manifest.json")

        callback = MagicMock()
        run_pipeline(basic_config, "test.mp4", resume=True, progress_callback=callback)

        mock_ingest.assert_not_called()
        mock_transcribe.assert_not_called()
        mock_score.assert_not_called()
        mock_detect_faces.assert_not_called()
        mock_render.assert_not_called()

        for stage in ["ingestion", "transcription", "scoring", "face_cropping", "rendering"]:
            callback.assert_any_call(stage, "skipped", None)

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_force_reruns_all_stages(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config

        _create_valid_meta(tmp_path / "artifacts" / "meta.json")
        _create_valid_transcript(tmp_path / "artifacts" / "transcript.json")
        _create_valid_scored_segments(tmp_path / "artifacts" / "scored_segments.json")
        _create_valid_crop_params(tmp_path / "artifacts" / "crop_params.json")
        _create_valid_manifest(tmp_path / "output" / "manifest.json")

        run_pipeline(basic_config, "test.mp4", resume=True, force=True)

        mock_ingest.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_score.assert_called_once()
        mock_detect_faces.assert_called_once()
        mock_render.assert_called_once()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_no_resume_runs_all_stages(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config

        run_pipeline(basic_config, "test.mp4", resume=False, force=False)

        mock_ingest.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_score.assert_called_once()
        mock_detect_faces.assert_called_once()
        mock_render.assert_called_once()


class TestVramCleanup:
    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_cleanup_after_transcription_and_face_cropping(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config

        run_pipeline(basic_config, "test.mp4")

        assert mock_gpu_config.cleanup_fn.call_count == 2


class TestReviewNonCritical:
    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_review_failure_does_not_raise(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        basic_config,
        mock_gpu_config,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = mock_gpu_config
        mock_review.side_effect = RuntimeError("review failed")
        callback = MagicMock()

        run_pipeline(basic_config, "test.mp4", progress_callback=callback)

        callback.assert_any_call("review", "failed", None)
