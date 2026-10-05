"""
Tests for B-Roll Suggestions Module
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestBrollModule:
    """Test broll module imports and structure."""

    def test_module_exists(self):
        from src import broll

        assert broll is not None

    def test_broll_error_exists(self):
        from src.broll import BrollError

        assert BrollError is not None
        assert issubclass(BrollError, Exception)

    def test_format_broll_text_exists(self):
        from src.broll import format_broll_text

        assert callable(format_broll_text)

    def test_validate_broll_suggestions_exists(self):
        from src.broll import validate_broll_suggestions

        assert callable(validate_broll_suggestions)

    def test_generate_broll_suggestions_exists(self):
        from src.broll import generate_broll_suggestions

        assert callable(generate_broll_suggestions)


class TestFormatBrollText:
    """Test plain-text b-roll artifact formatting."""

    def test_single_suggestion(self):
        from src.broll import format_broll_text

        suggestions = [{"timestamp": "00:15", "suggestion": "Brain scan close-up"}]
        result = format_broll_text(suggestions)
        assert result == "00:15 Brain scan close-up"

    def test_suggestion_with_keywords(self):
        from src.broll import format_broll_text

        suggestions = [
            {
                "timestamp": "00:15",
                "suggestion": "Brain scan close-up",
                "keywords": ["brain", "neuron"],
            }
        ]
        result = format_broll_text(suggestions)
        assert result == "00:15 Brain scan close-up [brain, neuron]"

    def test_multiple_suggestions(self):
        from src.broll import format_broll_text

        suggestions = [
            {"timestamp": "00:15", "suggestion": "Intro visual"},
            {"timestamp": "05:30", "suggestion": "Main topic visual"},
        ]
        result = format_broll_text(suggestions)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "00:15 Intro visual" in lines[0]
        assert "05:30 Main topic visual" in lines[1]

    def test_empty_suggestions(self):
        from src.broll import format_broll_text

        assert format_broll_text([]) == ""


class TestValidateBrollSuggestions:
    """Test b-roll suggestion validation."""

    def test_valid_suggestions(self):
        from src.broll import validate_broll_suggestions

        suggestions = [
            {"start_sec": 0, "suggestion": "Intro visual"},
            {"start_sec": 300, "suggestion": "Later visual"},
        ]
        assert validate_broll_suggestions(suggestions, 900) is True

    def test_empty_suggestions(self):
        from src.broll import validate_broll_suggestions

        assert validate_broll_suggestions([], 900) is False

    def test_suggestion_out_of_bounds(self):
        from src.broll import validate_broll_suggestions

        suggestions = [{"start_sec": 1000, "suggestion": "Out of bounds"}]
        assert validate_broll_suggestions(suggestions, 900) is False

    def test_suggestion_negative_start(self):
        from src.broll import validate_broll_suggestions

        suggestions = [{"start_sec": -5, "suggestion": "Invalid"}]
        assert validate_broll_suggestions(suggestions, 900) is False

    def test_suggestion_missing_text(self):
        from src.broll import validate_broll_suggestions

        suggestions = [{"start_sec": 10, "suggestion": ""}]
        assert validate_broll_suggestions(suggestions, 900) is False


class TestBrollConfig:
    """Test BrollConfig."""

    def test_default_values(self):
        from src.config import BrollConfig

        config = BrollConfig()
        assert config.enabled is False
        assert config.max_suggestions is None
        assert config.prompt_file == "config/prompts/broll_suggestions_v1.txt"

    def test_custom_values(self):
        from src.config import BrollConfig

        config = BrollConfig(max_suggestions=5, enabled=True)
        assert config.max_suggestions == 5
        assert config.enabled is True

    def test_config_loads_from_yaml(self):
        from src.config import Config

        config = Config.load("config/config.yaml")
        assert hasattr(config, "broll")
        assert config.broll.max_suggestions is None
        assert config.broll.enabled is False


class TestGenerateBrollSuggestionsDryRun:
    """Test generate_broll_suggestions dry run."""

    def test_dry_run_returns_status(self, tmp_path):
        from src.broll import generate_broll_suggestions
        from src.config import Config

        config = Config()
        config.work_dir = tmp_path
        result = generate_broll_suggestions(config, dry_run=True)
        assert result is not None
        assert result[0]["status"] == "dry_run_passed"

    def test_missing_transcript_raises_error(self, tmp_path):
        from src.broll import BrollError, generate_broll_suggestions
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "meta.json").write_text(json.dumps({"duration_sec": 600}))

        config = Config()
        config.work_dir = tmp_path
        with pytest.raises(BrollError):
            generate_broll_suggestions(config, dry_run=False)

    def test_missing_meta_raises_error(self, tmp_path):
        from src.broll import BrollError, generate_broll_suggestions
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "transcript.json").write_text(json.dumps([]))

        config = Config()
        config.work_dir = tmp_path
        with pytest.raises(BrollError):
            generate_broll_suggestions(config, dry_run=False)


class TestFallbackBrollSuggestions:
    """Test fallback (non-LLM) b-roll suggestion generation."""

    def test_fallback_generates_suggestions(self, tmp_path):
        from src.broll import generate_broll_suggestions
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        transcript = [
            {"start": i * 60, "end": (i + 1) * 60, "text": f"Segment {i} about neurons"}
            for i in range(20)
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

        from src.broll import _DEFAULT_FALLBACK_COUNT

        result = generate_broll_suggestions(config, dry_run=False)
        assert result is not None
        assert len(result) > 0
        assert len(result) <= _DEFAULT_FALLBACK_COUNT

        broll_json = artifacts_dir / "broll_suggestions.json"
        assert broll_json.exists()

        broll_txt = artifacts_dir / "broll_suggestions.txt"
        assert broll_txt.exists()
        assert broll_txt.read_text() != ""

    def test_fallback_suggestions_have_correct_structure(self, tmp_path):
        from src.broll import generate_broll_suggestions
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        transcript = [{"start": 0, "end": 60, "text": "Test segment about the topic"}]
        (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))
        (artifacts_dir / "meta.json").write_text(
            json.dumps(
                {
                    "duration_sec": 300,
                    "source": "test.mp4",
                }
            )
        )

        config = Config()
        config.work_dir = tmp_path
        config.scoring.llm.enabled = False

        result = generate_broll_suggestions(config, dry_run=False)
        for s in result:
            assert "timestamp" in s
            assert "start_sec" in s
            assert "suggestion" in s
            assert "keywords" in s

    def test_fallback_respects_max_suggestions(self, tmp_path):
        from src.broll import generate_broll_suggestions
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        transcript = [
            {"start": i * 10, "end": (i + 1) * 10, "text": f"Segment {i}"} for i in range(100)
        ]
        (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))
        (artifacts_dir / "meta.json").write_text(
            json.dumps(
                {
                    "duration_sec": 1000,
                    "source": "test.mp4",
                }
            )
        )

        config = Config()
        config.work_dir = tmp_path
        config.scoring.llm.enabled = False
        config.broll.max_suggestions = 5

        result = generate_broll_suggestions(config, dry_run=False)
        assert len(result) <= 5

    def test_empty_transcript_produces_no_suggestions(self, tmp_path):
        from src.broll import generate_broll_suggestions
        from src.config import Config

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "transcript.json").write_text(json.dumps([]))
        (artifacts_dir / "meta.json").write_text(
            json.dumps(
                {
                    "duration_sec": 300,
                    "source": "test.mp4",
                }
            )
        )

        config = Config()
        config.work_dir = tmp_path
        config.scoring.llm.enabled = False

        result = generate_broll_suggestions(config, dry_run=False)
        assert result == []


class TestBuildBrollPrompt:
    """Test prompt construction respects max_suggestions / model discretion."""

    def test_explicit_count_instruction(self):
        from src.broll import _build_broll_prompt
        from src.config import BrollConfig

        config = BrollConfig(max_suggestions=7)
        prompt = _build_broll_prompt([{"start": 0, "text": "hello"}], config)
        assert "at most 7 moments" in prompt
        assert "use your judgment" not in prompt

    def test_none_count_defers_to_model(self):
        from src.broll import _build_broll_prompt
        from src.config import BrollConfig

        config = BrollConfig(max_suggestions=None)
        prompt = _build_broll_prompt([{"start": 0, "text": "hello"}], config)
        assert "use your judgment" in prompt
        assert "at most" not in prompt


class TestBrollPipelineIntegration:
    """Test b-roll stage integration into the draft (chapters) pipeline."""

    def test_pipeline_module_imports_broll(self):
        import src.pipeline as pipeline_module

        assert pipeline_module is not None

    def _mock_gpu_config(self):
        from src.gpu_utils import GPUConfig

        return GPUConfig(
            backend="cpu",
            whisper_device="cpu",
            whisper_compute_type="int8",
            cleanup_fn=MagicMock(),
        )

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.broll.generate_broll_suggestions")
    @patch("src.chapters.generate_chapters")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_broll_not_called_in_clips_pipeline(
        self,
        mock_ingest,
        mock_transcribe,
        mock_generate_chapters,
        mock_generate_broll,
        mock_get_gpu,
        tmp_path,
        monkeypatch,
    ):
        """broll must never run as part of the clips (final) pipeline."""
        from src.config import Config

        with (
            patch("src.review.review_and_export"),
            patch("src.rendering.render_and_export"),
            patch("src.face_cropping.detect_faces"),
            patch("src.scoring.score_transcript"),
        ):
            from src.pipeline import run_pipeline

            monkeypatch.chdir(tmp_path)
            mock_get_gpu.return_value = self._mock_gpu_config()

            config = Config()
            config.work_dir = tmp_path
            config.broll.enabled = True  # even if enabled, clips pipeline must not use it

            run_pipeline(config=config, input_source="test.mp4")

            mock_generate_broll.assert_not_called()
            mock_generate_chapters.assert_not_called()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.broll.generate_broll_suggestions")
    @patch("src.chapters.generate_chapters")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_broll_disabled_by_default_not_called(
        self,
        mock_ingest,
        mock_transcribe,
        mock_generate_chapters,
        mock_generate_broll,
        mock_get_gpu,
        tmp_path,
        monkeypatch,
    ):
        from src.config import Config
        from src.pipeline import run_chapters_pipeline

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = self._mock_gpu_config()

        config = Config()
        config.work_dir = tmp_path
        assert config.broll.enabled is False

        run_chapters_pipeline(config=config, input_source="test.mp4")

        mock_generate_broll.assert_not_called()

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.broll.generate_broll_suggestions")
    @patch("src.chapters.generate_chapters")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_broll_called_when_enabled(
        self,
        mock_ingest,
        mock_transcribe,
        mock_generate_chapters,
        mock_generate_broll,
        mock_get_gpu,
        tmp_path,
        monkeypatch,
    ):
        from src.config import Config
        from src.pipeline import run_chapters_pipeline

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = self._mock_gpu_config()

        config = Config()
        config.work_dir = tmp_path
        config.broll.enabled = True

        run_chapters_pipeline(config=config, input_source="test.mp4")

        mock_generate_broll.assert_called_once_with(config, dry_run=False)

    @patch("src.pipeline.get_auto_gpu_config")
    @patch("src.broll.generate_broll_suggestions")
    @patch("src.chapters.generate_chapters")
    @patch("src.transcription.transcribe")
    @patch("src.ingestion.ingest")
    def test_broll_failure_does_not_fail_pipeline(
        self,
        mock_ingest,
        mock_transcribe,
        mock_generate_chapters,
        mock_generate_broll,
        mock_get_gpu,
        tmp_path,
        monkeypatch,
    ):
        from src.config import Config
        from src.pipeline import run_chapters_pipeline

        monkeypatch.chdir(tmp_path)
        mock_get_gpu.return_value = self._mock_gpu_config()

        config = Config()
        config.work_dir = tmp_path
        config.broll.enabled = True
        mock_generate_broll.side_effect = RuntimeError("boom")

        # Should not raise even though the broll stage fails.
        run_chapters_pipeline(config=config, input_source="test.mp4")


class TestPromptFile:
    """Test b-roll suggestions prompt file."""

    def test_prompt_file_exists(self):
        prompt_path = Path("config/prompts/broll_suggestions_v1.txt")
        assert prompt_path.exists()

    def test_prompt_has_placeholders(self):
        prompt_path = Path("config/prompts/broll_suggestions_v1.txt")
        content = prompt_path.read_text()
        assert "{{COUNT_INSTRUCTION}}" in content
        assert "{{TRANSCRIPT}}" in content
