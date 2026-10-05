"""
TASK-E2E-04: End-to-End Integration Test on Synthetic Video

Generates a synthetic video with ffmpeg, runs it through the full pipeline,
and validates all output artifacts. Works without GPU by mocking
Whisper and face detector model loading.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.gpu_utils import GPUConfig


@pytest.fixture
def mock_gpu_config():
    """Return a mock GPUConfig for testing on CPU."""
    return GPUConfig(
        backend="cpu",
        whisper_device="cpu",
        whisper_compute_type="int8",
        cleanup_fn=MagicMock(),
    )


@pytest.fixture
def synthetic_video(tmp_path):
    """
    Generate a 10-second synthetic video with ffmpeg.

    Video: 1920x1080, 30fps, blue background with a white rectangle (fake face).
    Audio: 440 Hz sine wave, 16kHz, mono.
    """
    video_path = tmp_path / "synthetic.mp4"

    # Generate video: blue background + white rectangle
    video_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=1920x1080:r=30:d=10",
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=200x200:r=30:d=10",
        "-filter_complex",
        "[1:v]setpts=PTS-STARTPTS,format=yuva420p[fg];"
        "[0:v][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:shortest=1[outv]",
        "-map",
        "[outv]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-t",
        "10",
        str(video_path),
    ]
    subprocess.run(video_cmd, capture_output=True, check=True, timeout=60)

    # Add audio: 440 Hz sine wave
    audio_path = tmp_path / "audio.wav"
    audio_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=10:sample_rate=16000",
        "-c:a",
        "pcm_s16le",
        "-ac",
        "1",
        str(audio_path),
    ]
    subprocess.run(audio_cmd, capture_output=True, check=True, timeout=30)

    # Merge video + audio
    final_path = tmp_path / "final.mp4"
    merge_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(final_path),
    ]
    subprocess.run(merge_cmd, capture_output=True, check=True, timeout=30)

    return final_path


def _make_mock_transcript():
    """Create a synthetic transcript that matches a 10-second video."""
    return [
        {
            "id": "seg_0001",
            "start": 0.0,
            "end": 3.5,
            "text": "Это определение ключевого понятия в научной области.",
            "words": [
                {"word": "Это", "start": 0.0, "end": 0.2},
                {"word": "определение", "start": 0.3, "end": 1.0},
                {"word": "ключевого", "start": 1.1, "end": 1.7},
                {"word": "понятия", "start": 1.8, "end": 2.3},
            ],
        },
        {
            "id": "seg_0002",
            "start": 3.5,
            "end": 6.5,
            "text": "Важно понимать, что этот термин означает фундаментальный принцип.",
            "words": [
                {"word": "Важно", "start": 3.5, "end": 3.8},
                {"word": "понимать", "start": 3.9, "end": 4.5},
                {"word": "термин", "start": 5.0, "end": 5.5},
            ],
        },
        {
            "id": "seg_0003",
            "start": 6.5,
            "end": 10.0,
            "text": "Как это работает? Рассмотрим пример подробнее.",
            "words": [
                {"word": "Как", "start": 6.5, "end": 6.7},
                {"word": "работает", "start": 7.0, "end": 7.5},
                {"word": "пример", "start": 8.0, "end": 8.5},
            ],
        },
    ]


def _mock_transcribe_side_effect(config, dry_run=False):
    """Mock for transcribe() that also saves the file to disk."""
    import json

    transcript = _make_mock_transcript()
    transcript_path = config.work_dir / "artifacts" / "transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    return transcript


def _mock_detect_faces_side_effect(config, dry_run=False):
    """Mock for detect_faces() that also saves the file to disk."""
    import json

    crop_params = _make_mock_crop_params()
    crop_path = config.work_dir / "artifacts" / "crop_params.json"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    with open(crop_path, "w", encoding="utf-8") as f:
        json.dump(crop_params, f, indent=2)
    return crop_params


def _mock_review_side_effect(
    clips_dir="output/clips",
    review_output="artifacts/review.json",
    auto_mode=False,
    player="auto",
    work_dir=None,
):
    """Mock for review_and_export() that also saves the file to disk."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    review_path = Path(review_output)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    review_data = {
        "review_completed_at": datetime.now(timezone.utc).isoformat(),
        "total_clips": 3,
        "summary": {"keep": 0, "reject": 0, "edit": 0, "pending": 3},
        "clips": [
            {
                "clip_id": "clip_001",
                "video_path": "output/clips/clip_001.mp4",
                "flag": "pending",
                "metadata": {},
            },
            {
                "clip_id": "clip_002",
                "video_path": "output/clips/clip_002.mp4",
                "flag": "pending",
                "metadata": {},
            },
            {
                "clip_id": "clip_003",
                "video_path": "output/clips/clip_003.mp4",
                "flag": "pending",
                "metadata": {},
            },
        ],
    }

    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2)

    return review_data


def _make_mock_crop_params():
    """Create synthetic crop params with center_crop fallback."""
    return [
        {
            "clip_id": "clip_001",
            "start": 0.0,
            "end": 6.5,
            "frames": [],
            "fallback": "center_crop",
            "fallback_ratio": 1.0,
        }
    ]


@pytest.fixture
def pipeline_env(tmp_path, monkeypatch):
    """
    Set up working directory, config, and logging for integration tests.

    Returns:
        tuple: (cfg, logger)
    """
    from src.logger import get_logger, setup_logging

    monkeypatch.chdir(tmp_path)

    # Create required directories
    (tmp_path / "artifacts" / "logs").mkdir(parents=True)
    (tmp_path / "artifacts" / "cache").mkdir(parents=True)
    (tmp_path / "output" / "clips").mkdir(parents=True)
    (tmp_path / "config").mkdir(parents=True)

    # Copy config with adjusted thresholds for short synthetic video
    src_config = Path(__file__).parent.parent / "config" / "config.yaml"
    if src_config.exists():
        config_text = src_config.read_text()
        config_text = config_text.replace("min_duration: 45", "min_duration: 1")
        config_text = config_text.replace("max_duration: 90", "max_duration: 15")
        (tmp_path / "config" / "config.yaml").write_text(config_text)
    else:
        (tmp_path / "config" / "config.yaml").write_text(
            "input:\n  cache_dir: artifacts/cache\n"
            "scoring:\n  min_duration: 1\n  max_duration: 15\n"
        )

    cfg = Config.load(str(tmp_path / "config" / "config.yaml"))

    setup_logging(
        log_file=str(tmp_path / "artifacts" / "logs" / "pipeline.log"),
        level="INFO",
        json_format=True,
    )
    logger = get_logger("integration_test")

    return cfg, logger


class TestFullPipeline:
    """Integration test: run full pipeline on synthetic video."""

    @patch("src.review.review_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.transcription.transcribe")
    def test_full_pipeline_produces_artifacts(
        self,
        mock_transcribe,
        mock_detect_faces,
        mock_review,
        synthetic_video,
        tmp_path,
        pipeline_env,
        mock_gpu_config,
    ):
        """
        Run the full pipeline on synthetic video and validate all artifacts.

        Steps:
        1. Generate synthetic video (fixture).
        2. Run ingestion (real).
        3. Mock transcribe() to return synthetic transcript.
        4. Run scoring (real).
        5. Mock detect_faces() to return center_crop fallback.
        6. Run rendering (real).
        7. Validate all output artifacts.
        """
        cfg, logger = pipeline_env

        # Set up mocks
        mock_transcribe.side_effect = _mock_transcribe_side_effect
        mock_detect_faces.side_effect = _mock_detect_faces_side_effect
        mock_review.side_effect = _mock_review_side_effect

        from src.pipeline import run_pipeline

        with patch("src.pipeline.get_auto_gpu_config", return_value=mock_gpu_config):
            run_pipeline(cfg, str(synthetic_video), resume=False, force=False)

        # === Validate artifacts ===

        # 1. meta.json
        meta_path = tmp_path / "artifacts" / "meta.json"
        assert meta_path.exists(), "meta.json not created"
        meta = json.loads(meta_path.read_text())
        assert meta["duration_sec"] > 0, "meta duration should be > 0"
        assert "source_hash" in meta

        # 2. transcript.json (from mock, but saved by pipeline)
        transcript_path = tmp_path / "artifacts" / "transcript.json"
        assert transcript_path.exists(), "transcript.json not created"
        transcript = json.loads(transcript_path.read_text())
        assert isinstance(transcript, list)
        assert len(transcript) == 3

        # 3. scored_segments.json
        scored_path = tmp_path / "artifacts" / "scored_segments.json"
        assert scored_path.exists(), "scored_segments.json not created"
        scored = json.loads(scored_path.read_text())
        assert isinstance(scored, list)
        for seg in scored:
            assert "id" in seg
            assert "start" in seg
            assert "end" in seg
            assert "score" in seg

        # 4. crop_params.json (from mock, but saved by pipeline)
        crop_path = tmp_path / "artifacts" / "crop_params.json"
        assert crop_path.exists(), "crop_params.json not created"
        crop = json.loads(crop_path.read_text())
        assert isinstance(crop, list)

        # 5. manifest.json
        manifest_path = tmp_path / "output" / "manifest.json"
        assert manifest_path.exists(), "manifest.json not created"
        manifest = json.loads(manifest_path.read_text())
        assert "total_clips" in manifest
        assert manifest["total_clips"] >= 0
        assert "clips" in manifest
        assert isinstance(manifest["clips"], list)

        # 6. review.json
        review_path = tmp_path / "artifacts" / "review.json"
        assert review_path.exists(), "review.json not created"
        review_data = json.loads(review_path.read_text())
        assert "clips" in review_data

    @patch("src.review.review_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.transcription.transcribe")
    def test_resume_skips_stages(
        self,
        mock_transcribe,
        mock_detect_faces,
        mock_review,
        synthetic_video,
        tmp_path,
        pipeline_env,
        mock_gpu_config,
    ):
        """
        Run pipeline twice: first full run, then resume=True.
        Second run should skip all stages.
        """
        cfg, logger = pipeline_env

        # Set up mocks
        mock_transcribe.side_effect = _mock_transcribe_side_effect
        mock_detect_faces.side_effect = _mock_detect_faces_side_effect
        mock_review.side_effect = _mock_review_side_effect

        from src.pipeline import run_pipeline

        # First run: full pipeline
        with patch("src.pipeline.get_auto_gpu_config", return_value=mock_gpu_config):
            run_pipeline(cfg, str(synthetic_video), resume=False, force=False)

        # Verify artifacts exist after first run
        assert (tmp_path / "artifacts" / "meta.json").exists()
        assert (tmp_path / "artifacts" / "transcript.json").exists()
        assert (tmp_path / "artifacts" / "scored_segments.json").exists()
        assert (tmp_path / "artifacts" / "crop_params.json").exists()
        assert (tmp_path / "output" / "manifest.json").exists()
        assert (tmp_path / "artifacts" / "review.json").exists()

        # Reset mocks to verify they are NOT called during resume
        mock_transcribe.reset_mock()
        mock_detect_faces.reset_mock()
        mock_review.reset_mock()

        # Second run: resume=True
        with patch("src.pipeline.get_auto_gpu_config", return_value=mock_gpu_config):
            run_pipeline(cfg, str(synthetic_video), resume=True, force=False)

        # Verify stages were skipped (mocks not called)
        mock_transcribe.assert_not_called()
        mock_detect_faces.assert_not_called()
        mock_review.assert_called_once()  # review always runs

    @patch("src.review.review_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.transcription.transcribe")
    def test_force_recomputes_all(
        self,
        mock_transcribe,
        mock_detect_faces,
        mock_review,
        synthetic_video,
        tmp_path,
        pipeline_env,
        mock_gpu_config,
    ):
        """
        Run pipeline twice: first full run, then force=True.
        Second run should re-execute all stages even with valid artifacts.
        """
        cfg, logger = pipeline_env

        # Set up mocks
        mock_transcribe.side_effect = _mock_transcribe_side_effect
        mock_detect_faces.side_effect = _mock_detect_faces_side_effect
        mock_review.side_effect = _mock_review_side_effect

        from src.pipeline import run_pipeline

        # First run: full pipeline
        with patch("src.pipeline.get_auto_gpu_config", return_value=mock_gpu_config):
            run_pipeline(cfg, str(synthetic_video), resume=False, force=False)

        # Reset mocks
        mock_transcribe.reset_mock()
        mock_detect_faces.reset_mock()
        mock_review.reset_mock()

        # Second run: force=True (should re-run everything)
        with patch("src.pipeline.get_auto_gpu_config", return_value=mock_gpu_config):
            run_pipeline(cfg, str(synthetic_video), resume=True, force=True)

        # Verify all stages were re-executed
        mock_transcribe.assert_called_once()
        mock_detect_faces.assert_called_once()
        mock_review.assert_called_once()
