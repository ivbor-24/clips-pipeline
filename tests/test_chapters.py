"""
Tests for P2-13: Chapters Mode
"""

import json
from pathlib import Path

import pytest


class TestChaptersModule:
    """Test chapters module imports and structure."""

    def test_module_exists(self):
        from src import chapters

        assert chapters is not None

    def test_chapters_error_exists(self):
        from src.chapters import ChaptersError

        assert ChaptersError is not None
        assert issubclass(ChaptersError, Exception)

    def test_format_timestamp_mmss_exists(self):
        from src.chapters import format_timestamp_mmss

        assert callable(format_timestamp_mmss)

    def test_format_chapters_youtube_exists(self):
        from src.chapters import format_chapters_youtube

        assert callable(format_chapters_youtube)

    def test_validate_chapters_exists(self):
        from src.chapters import validate_chapters

        assert callable(validate_chapters)

    def test_generate_chapters_exists(self):
        from src.chapters import generate_chapters

        assert callable(generate_chapters)


class TestFormatTimestampMmss:
    """Test MM:SS timestamp formatting."""

    def test_zero(self):
        from src.chapters import format_timestamp_mmss

        assert format_timestamp_mmss(0) == "00:00"

    def test_seconds(self):
        from src.chapters import format_timestamp_mmss

        assert format_timestamp_mmss(45) == "00:45"

    def test_minutes(self):
        from src.chapters import format_timestamp_mmss

        assert format_timestamp_mmss(125) == "02:05"

    def test_large_value(self):
        from src.chapters import format_timestamp_mmss

        assert format_timestamp_mmss(3661) == "61:01"


class TestFormatChaptersYoutube:
    """Test YouTube chapters format."""

    def test_single_chapter(self):
        from src.chapters import format_chapters_youtube

        chapters = [{"timestamp": "00:00", "title": "Introduction"}]
        result = format_chapters_youtube(chapters)
        assert result == "00:00 Introduction"

    def test_multiple_chapters(self):
        from src.chapters import format_chapters_youtube

        chapters = [
            {"timestamp": "00:00", "title": "Intro"},
            {"timestamp": "05:30", "title": "Main Topic"},
            {"timestamp": "12:00", "title": "Conclusion"},
        ]
        result = format_chapters_youtube(chapters)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "00:00 Intro" in lines[0]
        assert "05:30 Main Topic" in lines[1]
        assert "12:00 Conclusion" in lines[2]

    def test_empty_chapters(self):
        from src.chapters import format_chapters_youtube

        result = format_chapters_youtube([])
        assert result == ""


class TestValidateChapters:
    """Test chapters validation."""

    def test_valid_chapters(self):
        from src.chapters import validate_chapters

        chapters = [
            {"start_sec": 0, "title": "Intro"},
            {"start_sec": 300, "title": "Part 2"},
            {"start_sec": 600, "title": "Part 3"},
        ]
        assert validate_chapters(chapters, 900) is True

    def test_empty_chapters(self):
        from src.chapters import validate_chapters

        assert validate_chapters([], 900) is False

    def test_first_chapter_not_zero(self):
        from src.chapters import validate_chapters

        chapters = [
            {"start_sec": 10, "title": "Intro"},
            {"start_sec": 300, "title": "Part 2"},
        ]
        assert validate_chapters(chapters, 900) is False

    def test_chapter_out_of_bounds(self):
        from src.chapters import validate_chapters

        chapters = [
            {"start_sec": 0, "title": "Intro"},
            {"start_sec": 1000, "title": "Part 2"},
        ]
        assert validate_chapters(chapters, 900) is False

    def test_chapters_overlap(self):
        from src.chapters import validate_chapters

        chapters = [
            {"start_sec": 0, "title": "Intro"},
            {"start_sec": 300, "title": "Part 2"},
            {"start_sec": 200, "title": "Part 3"},
        ]
        assert validate_chapters(chapters, 900) is False


class TestChaptersConfig:
    """Test ChaptersConfig."""

    def test_default_values(self):
        from src.config import ChaptersConfig

        config = ChaptersConfig()
        assert config.enabled is False
        assert config.min_minutes == 5
        assert config.max_minutes == 7
        assert config.output_format == "both"

    def test_custom_values(self):
        from src.config import ChaptersConfig

        config = ChaptersConfig(min_minutes=3, max_minutes=10, enabled=True)
        assert config.min_minutes == 3
        assert config.max_minutes == 10
        assert config.enabled is True

    def test_config_loads_from_yaml(self):
        from src.config import Config

        config = Config.load("config/config.yaml")
        assert hasattr(config, "chapters")
        assert config.chapters.min_minutes == 5
        assert config.chapters.max_minutes == 7


class TestGenerateChaptersDryRun:
    """Test generate_chapters dry run."""

    def test_dry_run_returns_status(self, tmp_path):
        from src.chapters import generate_chapters
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "transcript.json").write_text(
            json.dumps([{"start": 0, "end": 30, "text": "Hello world"}])
        )
        (artifacts_dir / "meta.json").write_text(
            json.dumps({"duration_sec": 600, "source": "test.mp4"})
        )

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = Config()
            config.work_dir = tmp_path
            result = generate_chapters(config, dry_run=True)
            assert result is not None
            assert result[0]["status"] == "dry_run_passed"
        finally:
            os.chdir(old_cwd)

    def test_missing_transcript_raises_error(self, tmp_path):
        from src.chapters import ChaptersError, generate_chapters
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "meta.json").write_text(json.dumps({"duration_sec": 600}))

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = Config()
            config.work_dir = tmp_path
            with pytest.raises(ChaptersError):
                generate_chapters(config, dry_run=False)
        finally:
            os.chdir(old_cwd)

    def test_missing_meta_raises_error(self, tmp_path):
        from src.chapters import ChaptersError, generate_chapters
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "transcript.json").write_text(json.dumps([]))

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = Config()
            config.work_dir = tmp_path
            with pytest.raises(ChaptersError):
                generate_chapters(config, dry_run=False)
        finally:
            os.chdir(old_cwd)


class TestFallbackChapters:
    """Test fallback chapters generation."""

    def test_fallback_generates_chapters(self, tmp_path):
        from src.chapters import generate_chapters
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        transcript = [
            {"start": i * 60, "end": (i + 1) * 60, "text": f"Segment {i}"} for i in range(20)
        ]
        (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))
        (artifacts_dir / "meta.json").write_text(
            json.dumps({"duration_sec": 1200, "source": "test.mp4"})
        )

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = Config()
            config.work_dir = tmp_path
            config.scoring.llm.enabled = False

            result = generate_chapters(config, dry_run=False)
            assert result is not None
            assert len(result) > 0
            assert result[0]["start_sec"] == 0

            chapters_json = artifacts_dir / "chapters.json"
            assert chapters_json.exists()

            chapters_txt = artifacts_dir / "chapters.txt"
            assert chapters_txt.exists()
        finally:
            os.chdir(old_cwd)

    def test_fallback_chapters_have_correct_structure(self, tmp_path):
        from src.chapters import generate_chapters
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        transcript = [{"start": 0, "end": 60, "text": "Test segment"}]
        (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))
        (artifacts_dir / "meta.json").write_text(
            json.dumps({"duration_sec": 300, "source": "test.mp4"})
        )

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = Config()
            config.work_dir = tmp_path
            config.scoring.llm.enabled = False

            result = generate_chapters(config, dry_run=False)
            for ch in result:
                assert "timestamp" in ch
                assert "start_sec" in ch
                assert "title" in ch
                assert "summary" in ch
        finally:
            os.chdir(old_cwd)


class TestChaptersPipeline:
    """Test chapters pipeline integration."""

    def test_run_chapters_pipeline_exists(self):
        from src.pipeline import run_chapters_pipeline

        assert callable(run_chapters_pipeline)

    def test_chapters_error_in_pipeline(self):
        from src.pipeline import ChaptersError

        assert ChaptersError is not None

    def test_chapters_pipeline_dry_run(self, tmp_path):
        from src.config import Config
        from src.pipeline import run_chapters_pipeline

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        audio_dir = artifacts_dir / "audio"
        audio_dir.mkdir()
        (audio_dir / "raw.wav").write_bytes(b"fake audio")
        (artifacts_dir / "transcript.json").write_text(
            json.dumps([{"start": 0, "end": 30, "text": "Hello"}])
        )
        (artifacts_dir / "meta.json").write_text(
            json.dumps({"duration_sec": 600, "source": "test.mp4"})
        )

        import os

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            config = Config()
            config.work_dir = tmp_path

            callback_calls = []

            def callback(stage, status, progress):
                callback_calls.append((stage, status, progress))

            run_chapters_pipeline(
                config=config,
                input_source="test.mp4",
                dry_run=True,
                progress_callback=callback,
            )
        finally:
            os.chdir(old_cwd)


class TestJobTypeModel:
    """Test JobType model."""

    def test_job_type_enum(self):
        from src.api.models.job import JobType

        assert JobType.CLIPS.value == "clips"
        assert JobType.CHAPTERS.value == "chapters"

    def test_job_has_job_type(self):
        from src.api.models.job import Job

        assert hasattr(Job, "job_type")


class TestJobSchemas:
    """Test job schemas with job_type."""

    def test_job_create_default_type(self):
        from src.api.schemas.job import JobCreate

        job = JobCreate(input_source="test.mp4")
        assert job.job_type == "clips"

    def test_job_create_chapters_type(self):
        from src.api.schemas.job import JobCreate

        job = JobCreate(input_source="test.mp4", job_type="chapters")
        assert job.job_type == "chapters"

    def test_job_response_has_type(self):
        from src.api.schemas.job import JobResponse

        fields = JobResponse.model_fields
        assert "job_type" in fields


class TestChaptersApiSchemas:
    """Test chapters API schemas."""

    def test_chapters_response(self):
        from src.api.schemas.artifacts import ChaptersResponse

        resp = ChaptersResponse(
            chapters=[{"timestamp": "00:00", "title": "Intro"}], source="test.mp4"
        )
        assert len(resp.chapters) == 1
        assert resp.source == "test.mp4"

    def test_chapters_text_response(self):
        from src.api.schemas.artifacts import ChaptersTextResponse

        resp = ChaptersTextResponse(text="00:00 Intro\n05:00 Part 2", format="youtube")
        assert "Intro" in resp.text
        assert resp.format == "youtube"


class TestPromptFile:
    """Test chapter detection prompt file."""

    def test_prompt_file_exists(self):
        prompt_path = Path("config/prompts/chapter_detection_v1.txt")
        assert prompt_path.exists()

    def test_prompt_has_placeholders(self):
        prompt_path = Path("config/prompts/chapter_detection_v1.txt")
        content = prompt_path.read_text()
        assert "{{CHAPTER_COUNT_INSTRUCTION}}" in content
        assert "{{MIN_MINUTES}}" in content
        assert "{{TRANSCRIPT}}" in content


class TestBuildChaptersPrompt:
    """Test prompt construction respects target_count / min-max minutes."""

    def test_duration_based_instruction_by_default(self):
        from src.chapters import _build_chapters_prompt
        from src.config import ChaptersConfig

        config = ChaptersConfig(min_minutes=4, max_minutes=8)
        prompt = _build_chapters_prompt([{"start": 0, "text": "hello"}], config)
        assert "Each chapter should be 4 to 8 minutes long" in prompt
        assert "exactly" not in prompt

    def test_target_count_overrides_duration(self):
        from src.chapters import _build_chapters_prompt
        from src.config import ChaptersConfig

        config = ChaptersConfig(target_count=6)
        prompt = _build_chapters_prompt([{"start": 0, "text": "hello"}], config)
        assert "Create exactly 6 chapters" in prompt
        assert "minutes long" not in prompt.split("**Guidelines:**")[0]


class TestChaptersTargetCountFallback:
    """Test fallback chapter generation respects target_count."""

    def test_fallback_uses_target_count(self, tmp_path):
        from src.chapters import generate_chapters
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        transcript = [
            {"start": i * 30, "end": (i + 1) * 30, "text": f"Segment {i}"} for i in range(40)
        ]
        (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))
        (artifacts_dir / "meta.json").write_text(
            json.dumps(
                {
                    "duration_sec": 1200,
                    "source": "test.mp4",
                }
            )
        )

        config = Config()
        config.work_dir = tmp_path
        config.scoring.llm.enabled = False
        config.chapters.target_count = 3

        result = generate_chapters(config, dry_run=False)
        assert len(result) == 3
