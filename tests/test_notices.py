"""Notices: fallbacks that change a job's result reach the job page."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.notices import add_notice, clear_notices, read_notices


class TestNoticesFile:
    def test_add_and_read(self, tmp_path):
        add_notice(tmp_path, "scoring", "llm_failed", "Clips picked by heuristics.", error="OOM")
        assert read_notices(tmp_path) == [
            {
                "stage": "scoring",
                "code": "llm_failed",
                "message": "Clips picked by heuristics.",
                "error": "OOM",
            }
        ]

    def test_clear_only_own_stage(self, tmp_path):
        add_notice(tmp_path, "scoring", "a", "A")
        add_notice(tmp_path, "chapters", "b", "B")
        clear_notices(tmp_path, "scoring")
        assert [n["code"] for n in read_notices(tmp_path)] == ["b"]

    def test_no_file(self, tmp_path):
        assert read_notices(tmp_path) == []
        clear_notices(tmp_path, "scoring")  # nothing to clear, no error
        assert not (tmp_path / "artifacts" / "notices.json").exists()

    def test_broken_file_is_ignored(self, tmp_path):
        path = tmp_path / "artifacts" / "notices.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        assert read_notices(tmp_path) == []


def _scoring_job(tmp_path, strategy):
    from src.config import Config, LLMConfig, ScoringConfig

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    transcript = [
        {"id": f"seg_{i:03d}", "start": i * 5.0, "end": i * 5.0 + 5, "text": f"t{i}"}
        for i in range(20)
    ]
    (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))
    return Config(
        work_dir=tmp_path,
        scoring=ScoringConfig(
            strategy=strategy,
            term_extraction_method="statistical",
            llm=LLMConfig(enabled=True, provider="openai", api_key="x"),
        ),
    )


class TestScoringFallbackNotice:
    @pytest.mark.parametrize("strategy", ["segment_merge", "llm_windows"])
    def test_llm_failure_is_reported(self, tmp_path, strategy):
        from src.scoring import score_transcript

        cfg = _scoring_job(tmp_path, strategy)
        with patch(
            "src.llm_adapter.create_llm_adapter", side_effect=RuntimeError("model did not load")
        ):
            score_transcript(cfg, dry_run=False)

        notices = read_notices(tmp_path)
        assert [(n["stage"], n["code"]) for n in notices] == [
            ("scoring", "llm_failed_heuristics_used")
        ]
        assert "heuristics" in notices[0]["message"]
        assert notices[0]["error"] == "model did not load"

    def test_rerun_with_working_llm_clears_it(self, tmp_path):
        from src.scoring import score_transcript

        cfg = _scoring_job(tmp_path, "segment_merge")
        with patch("src.llm_adapter.create_llm_adapter", side_effect=RuntimeError("no model")):
            score_transcript(cfg, dry_run=False)
        assert read_notices(tmp_path)

        adapter = MagicMock()
        adapter.analyze_segments.side_effect = lambda batch: []
        with patch("src.llm_adapter.create_llm_adapter", return_value=adapter):
            score_transcript(cfg, dry_run=False)
        assert read_notices(tmp_path) == []
