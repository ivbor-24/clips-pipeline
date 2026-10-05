"""
TASK-UI-02: Tests for work_dir parameterization (per-job isolation).

Verifies:
- work_dir field exists on Config with default Path(".")
- All modules use config.work_dir for artifact paths
- Two jobs with different work_dir don't collide
- Backward compatibility (default work_dir = ".")
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.config import Config


class TestWorkDirConfig:

    def test_work_dir_default(self):
        cfg = Config()
        assert cfg.work_dir == Path(".")

    def test_work_dir_custom(self):
        cfg = Config(work_dir=Path("/tmp/job_123"))
        assert cfg.work_dir == Path("/tmp/job_123")

    def test_work_dir_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("work_dir: /tmp/custom_work_dir\n")
        cfg = Config.load(str(config_file))
        assert cfg.work_dir == Path("/tmp/custom_work_dir")

    def test_work_dir_omitted_in_yaml_uses_default(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("scoring:\n  min_duration: 1\n")
        cfg = Config.load(str(config_file))
        assert cfg.work_dir == Path(".")


class TestWorkDirBackwardCompatibility:

    def test_default_work_dir_resolves_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = Config()
        resolved = cfg.work_dir.resolve()
        assert resolved == tmp_path.resolve()

    def test_artifacts_path_with_default_work_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = Config()
        artifacts = cfg.work_dir / "artifacts" / "meta.json"
        assert artifacts.resolve() == (tmp_path / "artifacts" / "meta.json").resolve()


class TestWorkDirIsolation:

    def test_two_configs_different_paths(self, tmp_path):
        job_a = tmp_path / "job_a"
        job_b = tmp_path / "job_b"
        cfg_a = Config(work_dir=job_a)
        cfg_b = Config(work_dir=job_b)

        assert (
            cfg_a.work_dir / "artifacts" / "meta.json" != cfg_b.work_dir / "artifacts" / "meta.json"
        )
        assert (
            cfg_a.work_dir / "output" / "manifest.json"
            != cfg_b.work_dir / "output" / "manifest.json"
        )

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.review.review_and_export")
    @patch("src.rendering.render_and_export")
    @patch("src.face_cropping.detect_faces")
    @patch("src.scoring.score_transcript")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_pipeline_uses_work_dir_for_artifact_checks(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        tmp_path,
        monkeypatch,
    ):
        from src.gpu_utils import GPUConfig
        from src.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        mock_gpu = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu

        work_dir = tmp_path / "isolated_job"
        work_dir.mkdir(parents=True)

        meta_path = work_dir / "artifacts" / "meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
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

        transcript_path = work_dir / "artifacts" / "transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(
                [
                    {"id": "seg_0001", "start": 0.0, "end": 5.0, "text": "Hello"},
                    {"id": "seg_0002", "start": 5.0, "end": 10.0, "text": "World"},
                ],
                f,
            )

        scored_path = work_dir / "artifacts" / "scored_segments.json"
        with open(scored_path, "w") as f:
            json.dump(
                [
                    {"id": "seg_0001", "start": 0.0, "end": 5.0, "score": 0.8},
                ],
                f,
            )

        crop_path = work_dir / "artifacts" / "crop_params.json"
        with open(crop_path, "w") as f:
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

        manifest_path = work_dir / "output" / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({"total_clips": 1, "clips": [{"clip_id": "clip_001"}]}, f)

        cfg = Config(work_dir=work_dir, output={"manifest_file": "output/manifest.json"})

        callback = MagicMock()
        run_pipeline(cfg, "test.mp4", resume=True, progress_callback=callback)

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
    def test_two_jobs_no_collision(
        self,
        mock_ingest,
        mock_transcribe,
        mock_score,
        mock_detect_faces,
        mock_render,
        mock_review,
        mock_get_gpu,
        tmp_path,
        monkeypatch,
    ):
        from src.gpu_utils import GPUConfig
        from src.pipeline import run_pipeline

        monkeypatch.chdir(tmp_path)
        mock_gpu = GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )
        mock_get_gpu.return_value = mock_gpu

        job_a_dir = tmp_path / "job_a"
        job_a_dir.mkdir(parents=True)
        (job_a_dir / "artifacts").mkdir(parents=True)

        job_b_dir = tmp_path / "job_b"
        job_b_dir.mkdir(parents=True)

        meta_a = {
            "source": "a.mp4",
            "duration_sec": 10.0,
            "resolution": [1920, 1080],
            "fps": 30.0,
            "source_hash": "sha256:aaa",
        }
        with open(job_a_dir / "artifacts" / "meta.json", "w") as f:
            json.dump(meta_a, f)

        cfg_a = Config(work_dir=job_a_dir)
        cfg_b = Config(work_dir=job_b_dir)

        callback_a = MagicMock()
        run_pipeline(cfg_a, "a.mp4", resume=True, progress_callback=callback_a)
        mock_ingest.assert_not_called()
        callback_a.assert_any_call("ingestion", "skipped", None)

        callback_b = MagicMock()
        run_pipeline(cfg_b, "b.mp4", resume=True, progress_callback=callback_b)
        mock_ingest.assert_called_once()
        callback_b.assert_any_call("ingestion", "started", None)


class TestWorkDirApplyIsolated:

    def test_apply_isolated_sets_work_dir(self):
        from src.cli import apply_isolated_output

        cfg = Config()
        output_dir = Path("output/my_video_2026-01-01")
        apply_isolated_output(cfg, output_dir)

        assert cfg.work_dir == output_dir

    def test_apply_isolated_paths_are_relative_to_work_dir(self):
        from src.cli import apply_isolated_output

        cfg = Config()
        output_dir = Path("output/my_video_2026-01-01")
        apply_isolated_output(cfg, output_dir)

        assert cfg.work_dir / cfg.output.clips_dir == output_dir / "clips"
        assert cfg.work_dir / cfg.output.manifest_file == output_dir / "manifest.json"
        assert cfg.work_dir / cfg.output.review_file == output_dir / "review.json"
