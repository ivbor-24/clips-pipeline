"""
TASK-P1-03: Tests for isolated output directories per video.

Verifies that:
- Different videos produce different output directories
- Timestamp format is correct
- All output files go to the isolated directory
- Backward compatibility (isolated=false uses old paths)
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli import apply_isolated_output, compute_output_dir
from src.config import Config, OutputConfig


class TestComputeOutputDir:

    def test_basic_video_name(self):
        cfg = Config()
        result = compute_output_dir("/path/to/my_video.mp4", cfg)
        assert result.parent == Path("output")
        assert result.name.startswith("my_video_")

    def test_timestamp_format(self):
        cfg = Config()
        result = compute_output_dir("/path/to/lecture.mp4", cfg)
        dir_name = result.name
        ts_part = dir_name.replace("lecture_", "")
        datetime.strptime(ts_part, "%Y-%m-%d_%H-%M-%S")

    def test_different_videos_different_dirs(self):
        cfg = Config()
        d1 = compute_output_dir("/path/to/video_a.mp4", cfg)
        d2 = compute_output_dir("/path/to/video_b.mp4", cfg)
        assert d1 != d2

    def test_special_characters_sanitized(self):
        cfg = Config()
        result = compute_output_dir("/path/to/my video (1).mp4", cfg)
        dir_name = result.name
        assert " " not in dir_name
        assert "(" not in dir_name
        assert ")" not in dir_name

    def test_isolated_false_returns_parent_of_clips_dir(self):
        cfg = Config(output=OutputConfig(isolated=False, clips_dir="output/clips"))
        result = compute_output_dir("/path/to/video.mp4", cfg)
        assert result == Path("output")

    def test_custom_template(self):
        cfg = Config(
            output=OutputConfig(
                output_dir_template="custom_{video_name}_{timestamp}",
                timestamp_format="%Y%m%d",
            )
        )
        result = compute_output_dir("/path/to/test.mp4", cfg)
        assert result.name.startswith("custom_test_")
        ts_part = result.name.replace("custom_test_", "")
        datetime.strptime(ts_part, "%Y%m%d")

    def test_url_input(self):
        cfg = Config()
        result = compute_output_dir("https://youtube.com/watch?v=abc123", cfg)
        assert result.parent == Path("output")
        assert "watch" in result.name


class TestApplyIsolatedOutput:

    def test_paths_updated(self):
        cfg = Config()
        output_dir = Path("output/test_video_2026-01-01_12-00-00")
        apply_isolated_output(cfg, output_dir)

        assert cfg.work_dir == output_dir
        assert cfg.output.clips_dir == "clips"
        assert cfg.output.manifest_file == "manifest.json"
        assert cfg.output.review_file == "review.json"


class TestOutputConfigValidation:

    def test_defaults(self):
        oc = OutputConfig()
        assert oc.output_dir_template == "{video_name}_{timestamp}"
        assert oc.timestamp_format == "%Y-%m-%d_%H-%M-%S"
        assert oc.isolated is True

    def test_old_config_still_works(self):
        oc = OutputConfig(
            clips_dir="output/clips",
            manifest_file="output/manifest.json",
            review_file="artifacts/review.json",
        )
        assert oc.clips_dir == "output/clips"
        assert oc.isolated is True

    def test_custom_values(self):
        oc = OutputConfig(
            output_dir_template="{video_name}",
            timestamp_format="%Y%m%d",
            isolated=False,
        )
        assert oc.output_dir_template == "{video_name}"
        assert oc.isolated is False


class TestConfigYamlLoading:

    def test_load_with_new_fields(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "output:\n"
            "  clips_dir: output/clips\n"
            "  manifest_file: output/manifest.json\n"
            "  review_file: artifacts/review.json\n"
            "  isolated: true\n"
            "  output_dir_template: '{video_name}_{timestamp}'\n"
            "  timestamp_format: '%Y-%m-%d_%H-%M-%S'\n"
        )
        cfg = Config.load(str(config_file))
        assert cfg.output.isolated is True
        assert cfg.output.output_dir_template == "{video_name}_{timestamp}"

    def test_load_without_new_fields_backward_compat(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "output:\n"
            "  clips_dir: output/clips\n"
            "  manifest_file: output/manifest.json\n"
            "  review_file: artifacts/review.json\n"
        )
        cfg = Config.load(str(config_file))
        assert cfg.output.isolated is True
        assert cfg.output.output_dir_template == "{video_name}_{timestamp}"
        assert cfg.output.timestamp_format == "%Y-%m-%d_%H-%M-%S"


class TestIsolatedOutputIntegration:

    @pytest.fixture
    def mock_gpu_config(self):
        from src.gpu_utils import GPUConfig

        return GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )

    @pytest.fixture
    def pipeline_env(self, tmp_path, monkeypatch):
        from src.logger import get_logger, setup_logging

        monkeypatch.chdir(tmp_path)

        (tmp_path / "artifacts" / "logs").mkdir(parents=True)
        (tmp_path / "artifacts" / "cache").mkdir(parents=True)
        (tmp_path / "output").mkdir(parents=True)
        (tmp_path / "config").mkdir(parents=True)

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
        logger = get_logger("test_isolated")
        return cfg, logger

    def _make_mock_transcript(self):
        return [
            {"id": "seg_0001", "start": 0.0, "end": 3.5, "text": "Test segment one.", "words": []},
            {"id": "seg_0002", "start": 3.5, "end": 6.5, "text": "Test segment two.", "words": []},
            {
                "id": "seg_0003",
                "start": 6.5,
                "end": 10.0,
                "text": "Test segment three.",
                "words": [],
            },
        ]

    def _mock_transcribe(self, config, dry_run=False):
        transcript = self._make_mock_transcript()
        p = config.work_dir / "artifacts" / "transcript.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(transcript, f)
        return transcript

    def _mock_detect_faces(self, config, dry_run=False):
        crop = [
            {
                "clip_id": "clip_001",
                "start": 0.0,
                "end": 6.5,
                "frames": [],
                "fallback": "center_crop",
                "fallback_ratio": 1.0,
            }
        ]
        p = config.work_dir / "artifacts" / "crop_params.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(crop, f)
        return crop

    def _mock_review(
        self, clips_dir=None, review_output=None, auto_mode=False, player="auto", work_dir=None
    ):
        if work_dir is None:
            work_dir = Path(".")
        if review_output is None:
            review_output = str(work_dir / "artifacts" / "review.json")
        p = Path(review_output)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "review_completed_at": "2026-01-01",
            "total_clips": 0,
            "summary": {"keep": 0, "reject": 0, "edit": 0, "pending": 0},
            "clips": [],
        }
        with open(p, "w") as f:
            json.dump(data, f)
        return data

    def _mock_ingest(self, input_source, config, dry_run=False):
        meta = {
            "source": str(input_source),
            "source_hash": "abc123",
            "duration_sec": 10.0,
            "resolution": [1920, 1080],
            "fps": 30,
            "audio_sample_rate": 16000,
            "audio_channels": 1,
        }
        p = config.work_dir / "artifacts" / "meta.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(meta, f)
        (config.work_dir / "artifacts" / "video").mkdir(parents=True, exist_ok=True)
        return meta

    def _mock_render(self, config, dry_run=False, reuse=True):
        manifest_path = config.work_dir / config.output.manifest_file
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "source": "test",
            "source_hash": "abc123",
            "total_clips": 0,
            "clips": [],
            "generated_at": "2026-01-01T00:00:00Z",
            "pipeline_version": "0.1.0-mvp",
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        return []

    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_output_goes_to_isolated_dir(
        self,
        mock_ingest,
        mock_transcribe,
        mock_detect_faces,
        mock_render,
        mock_review,
        tmp_path,
        pipeline_env,
        mock_gpu_config,
    ):
        cfg, logger = pipeline_env

        mock_ingest.side_effect = self._mock_ingest
        mock_transcribe.side_effect = self._mock_transcribe
        mock_detect_faces.side_effect = self._mock_detect_faces
        mock_render.side_effect = self._mock_render
        mock_review.side_effect = self._mock_review

        video_path = tmp_path / "test_lecture.mp4"
        video_path.write_bytes(b"\x00" * 100)

        output_dir = compute_output_dir(str(video_path), cfg)
        apply_isolated_output(cfg, output_dir)

        from src.pipeline import run_pipeline

        with patch("src.pipeline.get_auto_gpu_config", return_value=mock_gpu_config):
            run_pipeline(cfg, str(video_path), resume=False, force=False)

        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "review.json").exists()
        assert not (tmp_path / "output" / "manifest.json").exists()

    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_two_videos_two_dirs(
        self,
        mock_ingest,
        mock_transcribe,
        mock_detect_faces,
        mock_render,
        mock_review,
        tmp_path,
        pipeline_env,
        mock_gpu_config,
    ):
        cfg1, logger1 = pipeline_env

        mock_ingest.side_effect = self._mock_ingest
        mock_transcribe.side_effect = self._mock_transcribe
        mock_detect_faces.side_effect = self._mock_detect_faces
        mock_render.side_effect = self._mock_render
        mock_review.side_effect = self._mock_review

        video_a = tmp_path / "lecture_a.mp4"
        video_a.write_bytes(b"\x00" * 100)
        video_b = tmp_path / "lecture_b.mp4"
        video_b.write_bytes(b"\x00" * 100)

        dir_a = compute_output_dir(str(video_a), cfg1)
        dir_b = compute_output_dir(str(video_b), cfg1)

        assert dir_a != dir_b
        assert "lecture_a" in dir_a.name
        assert "lecture_b" in dir_b.name

        apply_isolated_output(cfg1, dir_a)
        from src.pipeline import run_pipeline

        with patch("src.pipeline.get_auto_gpu_config", return_value=mock_gpu_config):
            run_pipeline(cfg1, str(video_a), resume=False, force=False)
        assert (dir_a / "manifest.json").exists()
