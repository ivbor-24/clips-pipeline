"""
Tests for TASK-04: Semantic Scoring Engine
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestScoringModule:
    """Test scoring module imports and structure."""

    def test_module_exists(self):
        """Test that scoring module exists."""
        from src import scoring

        assert scoring is not None

    def test_scoring_error_exists(self):
        """Test that ScoringError exception exists."""
        from src.scoring import ScoringError

        assert ScoringError is not None
        assert issubclass(ScoringError, Exception)

    def test_scored_segment_dataclass_exists(self):
        """Test that ScoredSegment dataclass exists."""
        from src.scoring import ScoredSegment

        assert ScoredSegment is not None

    def test_load_transcript_function_exists(self):
        """Test that load_transcript function exists."""
        from src.scoring import load_transcript

        assert callable(load_transcript)

    def test_detect_language_function_exists(self):
        """Test that detect_language function exists."""
        from src.scoring import detect_language

        assert callable(detect_language)

    def test_extract_terms_statistical_function_exists(self):
        """Test that extract_terms_statistical function exists."""
        from src.scoring import extract_terms_statistical

        assert callable(extract_terms_statistical)

    def test_try_keybert_extraction_function_exists(self):
        """Test that try_keybert_extraction function exists."""
        from src.scoring import try_keybert_extraction

        assert callable(try_keybert_extraction)

    def test_extract_terms_function_exists(self):
        """Test that extract_terms function exists."""
        from src.scoring import extract_terms

        assert callable(extract_terms)

    def test_compute_term_density_score_function_exists(self):
        """Test that compute_term_density_score function exists."""
        from src.scoring import compute_term_density_score

        assert callable(compute_term_density_score)

    def test_compute_definition_score_function_exists(self):
        """Test that compute_definition_score function exists."""
        from src.scoring import compute_definition_score

        assert callable(compute_definition_score)

    def test_compute_rhetorical_score_function_exists(self):
        """Test that compute_rhetorical_score function exists."""
        from src.scoring import compute_rhetorical_score

        assert callable(compute_rhetorical_score)

    def test_compute_self_containment_score_function_exists(self):
        """Test that compute_self_containment_score function exists."""
        from src.scoring import compute_self_containment_score

        assert callable(compute_self_containment_score)

    def test_filter_by_duration_function_exists(self):
        """Test that filter_by_duration function exists."""
        from src.scoring import filter_by_duration

        assert callable(filter_by_duration)

    def test_filter_by_score_function_exists(self):
        """Test that filter_by_score function exists."""
        from src.scoring import filter_by_score

        assert callable(filter_by_score)

    def test_select_top_segments_function_exists(self):
        """Test that select_top_segments function exists."""
        from src.scoring import select_top_segments

        assert callable(select_top_segments)

    def test_score_segment_function_exists(self):
        """Test that score_segment function exists."""
        from src.scoring import score_segment

        assert callable(score_segment)

    def test_extract_global_terms_function_exists(self):
        """Test that extract_global_terms function exists."""
        from src.scoring import extract_global_terms

        assert callable(extract_global_terms)

    def test_score_transcript_function_exists(self):
        """Test that main score_transcript function exists."""
        from src.scoring import score_transcript

        assert callable(score_transcript)


class TestDetectLanguage:
    """Test language detection."""

    def test_detects_russian(self):
        """Test Russian language detection."""
        from src.scoring import detect_language

        text = "Квантовая запутанность — это физическое явление"
        lang = detect_language(text)
        assert lang == "ru"

    def test_detects_english(self):
        """Test English language detection."""
        from src.scoring import detect_language

        text = "Quantum entanglement is a physical phenomenon"
        lang = detect_language(text)
        assert lang == "en"

    def test_empty_text_returns_unknown(self):
        """Test empty text returns unknown."""
        from src.scoring import detect_language

        lang = detect_language("")
        assert lang == "unknown"

    def test_mixed_text_majority_wins(self):
        """Test mixed text uses majority language."""
        from src.scoring import detect_language

        # Mostly Russian (more Cyrillic characters)
        text = "Квантовая запутанность это явление физики"
        lang = detect_language(text)
        assert lang == "ru"


class TestExtractTermsStatistical:
    """Test statistical term extraction."""

    def test_extracts_frequent_words(self):
        """Test extraction of frequent meaningful words."""
        from src.scoring import extract_terms_statistical

        text = "Quantum quantum quantum entanglement entanglement physics physics the the the"
        terms = extract_terms_statistical(text, top_n=3)

        assert "quantum" in terms or "entanglement" in terms

    def test_filters_stop_words(self):
        """Test that stop words are filtered."""
        from src.scoring import extract_terms_statistical

        text = "the is are was were have has had и в на с со за под"
        terms = extract_terms_statistical(text, top_n=5)

        # Should return empty or very few terms since all are stop words
        assert len(terms) <= 2

    def test_respects_top_n(self):
        """Test that top_n limits results."""
        from src.scoring import extract_terms_statistical

        text = "physics chemistry biology mathematics astronomy geology meteorology"
        terms = extract_terms_statistical(text, top_n=3)

        assert len(terms) <= 3


class TestTermExtractionMethod:
    def test_old_yake_value_reads_as_statistical(self):
        """Configs and settings.yaml written before the yake package was dropped."""
        from src.config import ScoringConfig

        assert ScoringConfig(term_extraction_method="yake").term_extraction_method == "statistical"

    def test_unknown_method_is_rejected(self):
        from pydantic import ValidationError

        from src.config import ScoringConfig

        with pytest.raises(ValidationError):
            ScoringConfig(term_extraction_method="tfidf")


class TestTryKeybertExtraction:
    """Test KeyBERT extraction fallback."""

    def test_returns_none_when_keybert_unavailable(self):
        """Test graceful fallback when KeyBERT not installed."""
        from src.scoring import try_keybert_extraction

        result = try_keybert_extraction("Some scientific text here", top_n=5)

        # Since KeyBERT is likely not installed, should return None
        assert result is None or isinstance(result, list)


class TestComputeDefinitionScore:
    """Test definition marker scoring."""

    def test_russian_definition_detected(self):
        """Test Russian definition markers."""
        from src.scoring import compute_definition_score

        text = "Квантовая запутанность — это физическое явление"
        score = compute_definition_score(text)
        assert score > 0.0

    def test_english_definition_detected(self):
        """Test English definition markers."""
        from src.scoring import compute_definition_score

        text = "Quantum entanglement is a physical phenomenon"
        score = compute_definition_score(text)
        assert score > 0.0

    def test_no_definition_markers(self):
        """Test text without definition markers."""
        from src.scoring import compute_definition_score

        text = "The experiment was conducted yesterday in the lab"
        score = compute_definition_score(text)
        assert score == 0.0

    def test_empty_text_returns_zero(self):
        """Test empty text returns zero score."""
        from src.scoring import compute_definition_score

        score = compute_definition_score("")
        assert score == 0.0

    def test_multiple_markers_increase_score(self):
        """Test multiple markers increase score."""
        from src.scoring import compute_definition_score

        text = "Это явление, то есть квантовая запутанность, представляет собой особый случай"
        score = compute_definition_score(text)
        # Multiple markers should give higher score
        assert score >= 0.5


class TestComputeRhetoricalScore:
    """Test rhetorical/engagement scoring."""

    def test_question_mark_detected(self):
        """Test question mark detection."""
        from src.scoring import compute_rhetorical_score

        text = "What is quantum entanglement?"
        score = compute_rhetorical_score(text)
        assert score > 0.0

    def test_question_words_detected(self):
        """Test question word detection."""
        from src.scoring import compute_rhetorical_score

        text = "Почему это происходит? Как это работает?"
        score = compute_rhetorical_score(text)
        assert score > 0.0

    def test_emphasis_markers_detected(self):
        """Test emphasis marker detection."""
        from src.scoring import compute_rhetorical_score

        text = "Важно отметить, что это ключевой момент"
        score = compute_rhetorical_score(text)
        assert score > 0.0

    def test_no_rhetorical_devices(self):
        """Test text without rhetorical devices."""
        from src.scoring import compute_rhetorical_score

        text = "The data shows a correlation between variables"
        score = compute_rhetorical_score(text)
        assert score == 0.0


class TestComputeSelfContainmentScore:
    """Test self-containment scoring."""

    def test_low_pronoun_ratio_high_score(self):
        """Test low pronoun ratio gives high score."""
        from src.scoring import compute_self_containment_score

        text = "Quantum entanglement demonstrates non-local correlations between particles"
        score, is_contained = compute_self_containment_score(text)
        assert score > 0.5
        assert is_contained is True

    def test_high_pronoun_ratio_low_score(self):
        """Test high pronoun ratio gives low score."""
        from src.scoring import compute_self_containment_score

        text = "It is what it is, and they said it would be how they wanted it to be"
        score, is_contained = compute_self_containment_score(text)
        assert score < 0.5
        assert is_contained is False

    def test_russian_pronouns_detected(self):
        """Test Russian pronoun detection."""
        from src.scoring import compute_self_containment_score

        text = "Он сказал ей, что они пойдут туда, где она была"
        score, is_contained = compute_self_containment_score(text)
        assert score < 0.5

    def test_empty_text_returns_zero(self):
        """Test empty text returns zero score."""
        from src.scoring import compute_self_containment_score

        score, is_contained = compute_self_containment_score("")
        assert score == 0.0
        assert is_contained is False


class TestComputeTermDensityScore:
    """Test term density scoring."""

    def test_high_term_density(self):
        """Test high term density detection."""
        from src.scoring import compute_term_density_score

        text = "Quantum entanglement demonstrates quantum superposition and quantum decoherence"
        terms = ["quantum", "entanglement", "superposition", "decoherence"]
        score, found = compute_term_density_score(text, terms)

        assert score > 0.0
        assert len(found) > 0

    def test_no_terms_found(self):
        """Test when no terms match."""
        from src.scoring import compute_term_density_score

        text = "The weather is nice today"
        terms = ["quantum", "physics", "entanglement"]
        score, found = compute_term_density_score(text, terms)

        assert score == 0.0
        assert len(found) == 0

    def test_empty_text_returns_zero(self):
        """Test empty text returns zero score."""
        from src.scoring import compute_term_density_score

        score, found = compute_term_density_score("", ["term"])
        assert score == 0.0
        assert len(found) == 0


class TestFilterByDuration:
    """Test duration filtering."""

    def test_filters_short_segments(self):
        """Test short segments are filtered out."""
        from src.scoring import ScoredSegment, filter_by_duration

        segments = [
            ScoredSegment(id="s1", start=0, end=30, text="Short", score=0.8),
            ScoredSegment(id="s2", start=100, end=160, text="Good", score=0.7),
        ]

        filtered = filter_by_duration(segments, min_duration=45, max_duration=90)

        assert len(filtered) == 1
        assert filtered[0].id == "s2"

    def test_filters_long_segments(self):
        """Test long segments are filtered out."""
        from src.scoring import ScoredSegment, filter_by_duration

        segments = [
            ScoredSegment(id="s1", start=0, end=200, text="Too long", score=0.9),
            ScoredSegment(id="s2", start=300, end=360, text="Good", score=0.7),
        ]

        filtered = filter_by_duration(segments, min_duration=45, max_duration=90)

        assert len(filtered) == 1
        assert filtered[0].id == "s2"

    def test_keeps_valid_segments(self):
        """Test valid segments are kept."""
        from src.scoring import ScoredSegment, filter_by_duration

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="Perfect", score=0.8),
            ScoredSegment(id="s2", start=100, end=145, text="Good", score=0.7),
        ]

        filtered = filter_by_duration(segments, min_duration=45, max_duration=90)

        assert len(filtered) == 2


class TestFilterByScore:
    """Test score threshold filtering."""

    def test_filters_low_scores(self):
        """Test low-scoring segments are filtered out."""
        from src.scoring import ScoredSegment, filter_by_score

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="Low", score=0.3),
            ScoredSegment(id="s2", start=100, end=160, text="High", score=0.8),
        ]

        filtered = filter_by_score(segments, min_threshold=0.55)

        assert len(filtered) == 1
        assert filtered[0].id == "s2"

    def test_keeps_high_scores(self):
        """Test high-scoring segments are kept."""
        from src.scoring import ScoredSegment, filter_by_score

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="Good", score=0.7),
            ScoredSegment(id="s2", start=100, end=160, text="Better", score=0.9),
        ]

        filtered = filter_by_score(segments, min_threshold=0.55)

        assert len(filtered) == 2


class TestSelectTopSegments:
    """Test top segment selection."""

    def test_selects_top_n(self):
        """Test selects exactly N top segments."""
        from src.scoring import ScoredSegment, select_top_segments

        segments = [
            ScoredSegment(id="s1", start=0, end=60, text="First", score=0.5),
            ScoredSegment(id="s2", start=100, end=160, text="Second", score=0.9),
            ScoredSegment(id="s3", start=200, end=260, text="Third", score=0.7),
        ]

        top = select_top_segments(segments, max_count=2)

        assert len(top) == 2
        assert top[0].score == 0.9  # Highest first
        assert top[1].score == 0.7  # Second highest


class TestScoreSegment:
    """Test individual segment scoring."""

    def test_composite_scoring(self):
        """Test composite score calculation."""
        from src.config import ScoringConfig
        from src.scoring import ScoredSegment, score_segment

        config = ScoringConfig()
        segment = {
            "id": "test_seg",
            "start": 0.0,
            "end": 60.0,
            "text": "Квантовая запутанность — это явление, когда частицы связаны",
        }

        global_terms = ["квантовая", "запутанность", "явление", "частицы"]

        result = score_segment(segment, global_terms, config)

        assert isinstance(result, ScoredSegment)
        assert result.id == "test_seg"
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.tags, list)

    def test_definition_gets_tag(self):
        """Test definition segments get proper tag."""
        from src.config import ScoringConfig
        from src.scoring import score_segment

        config = ScoringConfig()
        segment = {
            "id": "def_seg",
            "start": 0.0,
            "end": 60.0,
            "text": "Квантовая запутанность — это физическое явление",
        }

        result = score_segment(segment, [], config)

        assert "definition" in result.tags

    def test_tags_included(self):
        """Test tags are properly included."""
        from src.config import ScoringConfig
        from src.scoring import score_segment

        config = ScoringConfig()
        segment = {
            "id": "tagged_seg",
            "start": 0.0,
            "end": 60.0,
            "text": "Что такое квантовая запутанность? Это важно!",
        }

        result = score_segment(segment, ["квантовая", "запутанность"], config)

        assert len(result.tags) > 0


class TestExtractGlobalTerms:
    """Test global term extraction from transcript."""

    def test_extracts_from_full_transcript(self):
        """Test term extraction from full transcript."""
        from src.scoring import extract_global_terms

        transcript = [
            {"text": "Quantum physics studies quantum mechanics"},
            {"text": "Entanglement is a quantum phenomenon"},
            {"text": "Physics explains the universe"},
        ]

        terms = extract_global_terms(transcript, method="statistical", top_n=5)

        assert len(terms) <= 5
        # "quantum" appears multiple times, should be extracted
        assert any("quantum" in t.lower() for t in terms if t)


class TestLoadTranscript:
    """Test transcript loading."""

    def test_loads_valid_json(self, tmp_path):
        """Test loading valid transcript JSON."""
        from src.scoring import load_transcript

        transcript_file = tmp_path / "transcript.json"
        transcript_data = [{"id": "seg_001", "start": 0.0, "end": 5.0, "text": "Hello"}]
        transcript_file.write_text(json.dumps(transcript_data))

        loaded = load_transcript(transcript_file)

        assert len(loaded) == 1
        assert loaded[0]["id"] == "seg_001"

    def test_missing_file_raises_error(self):
        """Test missing file raises ScoringError."""
        from src.scoring import ScoringError, load_transcript

        with pytest.raises(ScoringError, match="not found"):
            load_transcript(Path("/nonexistent/path.json"))


class TestScoreTranscriptFunction:
    """Test main score_transcript function."""

    def test_dry_run_mode(self, tmp_path):
        """Test dry run mode returns without processing."""
        from src.config import Config
        from src.scoring import score_transcript

        # Setup artifacts directory
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        transcript_file = artifacts_dir / "transcript.json"
        transcript_file.write_text('[{"id": "seg_001", "start": 0, "end": 60, "text": "Test"}]')

        # Change to tmp_path
        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))

            result = score_transcript(cfg, dry_run=True)

            assert result is not None
            assert result[0]["status"] == "dry_run_passed"
        finally:
            os.chdir(original_cwd)

    def test_missing_transcript_raises_error(self):
        """Test missing transcript raises ScoringError."""
        from src.config import Config
        from src.scoring import ScoringError, score_transcript

        cfg = Config.load("config/config.yaml")

        with pytest.raises(ScoringError, match="Transcript file not found"):
            score_transcript(cfg, dry_run=False)

    def test_full_scoring_pipeline(self, tmp_path):
        """Test complete scoring pipeline."""
        from src.config import Config
        from src.scoring import score_transcript

        # Setup artifacts directory
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create a realistic transcript with varying content
        transcript_data = [
            {
                "id": "seg_001",
                "start": 0.0,
                "end": 60.0,
                "text": "Квантовая запутанность — это физическое явление, при котором две или более частиц становятся взаимосвязанными",
            },
            {
                "id": "seg_002",
                "start": 60.0,
                "end": 120.0,
                "text": "Что происходит при измерении? Важно понять, что состояние частиц определяется только в момент измерения",
            },
            {
                "id": "seg_003",
                "start": 120.0,
                "end": 180.0,
                "text": "Эйнштейн называл это «жутким дальнодействием» и сомневался в полноте квантовой механики",
            },
            {
                "id": "seg_004",
                "start": 180.0,
                "end": 240.0,
                "text": "Она сказала ему, что они пойдут туда, где он был раньше",  # High pronoun, low self-containment
            },
            {
                "id": "seg_005",
                "start": 240.0,
                "end": 300.0,
                "text": "Белловские неравенства представляют собой математический критерий для проверки локального реализма",
            },
        ]

        transcript_file = artifacts_dir / "transcript.json"
        transcript_file.write_text(json.dumps(transcript_data, ensure_ascii=False))

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))

            result = score_transcript(cfg, dry_run=False)

            assert result is not None
            assert isinstance(result, list)
            # Should return top segments (max 5 by default)
            assert len(result) <= 5

            # Each result should have required fields
            for seg in result:
                assert "id" in seg
                assert "start" in seg
                assert "end" in seg
                assert "score" in seg
                assert "tags" in seg
                assert "self_contained" in seg

            # Output file should be created
            scored_file = artifacts_dir / "scored_segments.json"
            assert scored_file.exists()

        finally:
            os.chdir(original_cwd)

    def test_short_segments_merge_into_clips(self, tmp_path):
        """Short sentence-level segments must not all be filtered out (C2 regression)."""
        from src.config import Config
        from src.scoring import score_transcript

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # 12 short (~10s) segments — realistic faster-whisper output
        transcript_data = [
            {
                "id": f"seg_{i:03d}",
                "start": float(i * 10),
                "end": float((i + 1) * 10),
                "text": f"Квантовая запутанность — это физическое явление, при котором частицы взаимосвязаны ({i})",
            }
            for i in range(12)
        ]

        transcript_file = artifacts_dir / "transcript.json"
        transcript_file.write_text(json.dumps(transcript_data, ensure_ascii=False))

        import os

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            cfg = Config.load(str(Path(original_cwd) / "config" / "config.yaml"))

            result = score_transcript(cfg, dry_run=False)

            assert isinstance(result, list)
            # The key regression: short segments must NOT all be dropped
            assert len(result) >= 1
            for seg in result:
                assert "start" in seg and "end" in seg and "score" in seg
        finally:
            os.chdir(original_cwd)


class TestPronounResilience:
    """Test resilience to ASR errors and pronoun usage."""

    def test_handles_asr_errors_gracefully(self):
        """Test scoring handles garbled text gracefully."""
        from src.config import ScoringConfig
        from src.scoring import score_segment

        config = ScoringConfig()
        # Simulated ASR errors
        segment = {
            "id": "asr_seg",
            "start": 0.0,
            "end": 60.0,
            "text": "uhm... like... you know... quantum something... it's... uh...",
        }

        result = score_segment(segment, ["quantum"], config)

        # Should still produce a valid score, just low
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_high_pronoun_segments_filtered(self):
        """Test segments with many pronouns get lower scores."""
        from src.config import ScoringConfig
        from src.scoring import score_segment

        config = ScoringConfig()

        # Low pronoun segment
        good_segment = {
            "id": "good",
            "start": 0.0,
            "end": 60.0,
            "text": "Квантовая механика описывает поведение элементарных частиц",
        }

        # High pronoun segment
        bad_segment = {
            "id": "bad",
            "start": 0.0,
            "end": 60.0,
            "text": "Она сказала ему, что он должен сделать это там, где они были",
        }

        good_result = score_segment(good_segment, ["квантовая", "механика"], config)
        bad_result = score_segment(bad_segment, [], config)

        # Good segment should have higher self-containment
        assert good_result.self_containment_score > bad_result.self_containment_score


class TestMergeSegmentsForClips:
    """B4: greedy merge of short segments into clips (score = max of members)."""

    @staticmethod
    def _seg(seg_id, start, end, score, tags=()):
        from src.scoring import ScoredSegment

        return ScoredSegment(
            id=seg_id, start=start, end=end, text=seg_id, score=score, tags=list(tags)
        )

    def test_clip_score_is_max_of_members(self):
        from src.scoring import merge_segments_for_clips

        segments = [
            self._seg("a", 0.0, 10.0, 0.2, ["definition"]),
            self._seg("b", 10.0, 20.0, 0.8, ["engaging", "definition"]),
            self._seg("c", 20.0, 30.0, 0.3),
        ]

        clips = merge_segments_for_clips(segments, max_duration=90)

        assert len(clips) == 1
        assert (clips[0].start, clips[0].end) == (0.0, 30.0)
        assert clips[0].score == 0.8
        assert clips[0].tags == ["definition", "engaging"]
        assert clips[0].id == "clip_001"

    def test_new_clip_when_max_duration_exceeded(self):
        from src.scoring import merge_segments_for_clips

        segments = [
            self._seg("a", 0.0, 50.0, 0.4),
            self._seg("b", 50.0, 80.0, 0.5),
            self._seg("c", 80.0, 95.0, 0.9),
        ]

        clips = merge_segments_for_clips(segments, max_duration=90)

        assert [(c.start, c.end, c.score) for c in clips] == [(0.0, 80.0, 0.5), (80.0, 95.0, 0.9)]
        assert [c.id for c in clips] == ["clip_001", "clip_002"]

    def test_input_order_does_not_matter(self):
        from src.scoring import merge_segments_for_clips

        segments = [
            self._seg("b", 10.0, 20.0, 0.1),
            self._seg("a", 0.0, 10.0, 0.7),
        ]

        clips = merge_segments_for_clips(segments, max_duration=90)

        assert [(c.start, c.end, c.score) for c in clips] == [(0.0, 20.0, 0.7)]

    def test_empty_input(self):
        from src.scoring import merge_segments_for_clips

        assert merge_segments_for_clips([], max_duration=90) == []


class TestLLMCandidateCap:
    """B4: max_segments sends only the top heuristic candidates, in batches."""

    def test_top_n_candidates_are_batched(self, tmp_path):
        from src.config import Config, LLMConfig, ScoringConfig
        from src.scoring import LLM_SEGMENT_BATCH_SIZE, ScoredSegment, score_transcript

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        transcript = [
            {"id": f"seg_{i:03d}", "start": i * 5.0, "end": i * 5.0 + 5, "text": f"t{i}"}
            for i in range(40)
        ]
        (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))

        def fake_score(segment, terms, cfg):
            idx = int(segment["id"][4:])
            return ScoredSegment(
                id=segment["id"],
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                score=idx / 100,
            )

        cfg = Config(
            work_dir=tmp_path,
            scoring=ScoringConfig(
                strategy="segment_merge",
                term_extraction_method="statistical",
                llm=LLMConfig(enabled=True, provider="openai", api_key="x", max_segments=20),
            ),
        )
        adapter = MagicMock()
        adapter.analyze_segments.side_effect = lambda batch: []

        with (
            patch("src.scoring.score_segment", side_effect=fake_score),
            patch("src.llm_adapter.create_llm_adapter", return_value=adapter),
        ):
            score_transcript(cfg, dry_run=False)

        batches = [c.args[0] for c in adapter.analyze_segments.call_args_list]
        assert [len(b) for b in batches] == [LLM_SEGMENT_BATCH_SIZE, 20 - LLM_SEGMENT_BATCH_SIZE]
        sent = [s["id"] for b in batches for s in b]
        # highest heuristic scores first: seg_039 .. seg_020
        assert sent == [f"seg_{i:03d}" for i in range(39, 19, -1)]
        assert set(batches[0][0]) == {"id", "start", "end", "text"}

    def test_no_cap_sends_everything(self, tmp_path):
        from src.config import Config, LLMConfig, ScoringConfig
        from src.scoring import score_transcript

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)
        transcript = [
            {"id": f"seg_{i:03d}", "start": i * 5.0, "end": i * 5.0 + 5, "text": f"t{i}"}
            for i in range(20)
        ]
        (artifacts_dir / "transcript.json").write_text(json.dumps(transcript))
        cfg = Config(
            work_dir=tmp_path,
            scoring=ScoringConfig(
                strategy="segment_merge",
                term_extraction_method="statistical",
                llm=LLMConfig(enabled=True, provider="openai", api_key="x"),
            ),
        )
        adapter = MagicMock()
        adapter.analyze_segments.side_effect = lambda batch: []

        with patch("src.llm_adapter.create_llm_adapter", return_value=adapter):
            score_transcript(cfg, dry_run=False)

        sent = [s["id"] for c in adapter.analyze_segments.call_args_list for s in c.args[0]]
        assert sent == [t["id"] for t in transcript]
