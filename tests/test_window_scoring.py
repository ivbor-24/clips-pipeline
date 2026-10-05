"""
Tests for the llm_windows scoring strategy (src/window_scoring.py).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config, LLMConfig, ScoringConfig
from src.window_scoring import (
    ClipCandidate,
    Window,
    build_windows,
    candidate_from_result,
    fit_boundaries,
    format_windows,
    is_continuation,
    pad_bounds,
    score_windows,
    select_clips,
    to_scored_segments,
)


def _transcript(n=60, seg_len=5.0, gap=0.5):
    """n sentences of seg_len seconds separated by gap."""
    step = seg_len + gap
    return [
        {
            "id": f"seg_{i + 1:04d}",
            "start": i * step,
            "end": i * step + seg_len,
            "text": f"Фраза номер {i}.",
        }
        for i in range(n)
    ]


def _cand(start, end, score, first=0, last=0):
    return ClipCandidate(
        window_id="w",
        first=first,
        last=last,
        start=start,
        end=end,
        score=score,
        hook=score,
        complete=score,
        value=score,
    )


class TestBuildWindows:
    def test_windows_overlap_and_cover_everything(self):
        transcript = _transcript(60)  # 60 * 5.5 s = 330 s
        windows = build_windows(transcript, window_sec=100, step_sec=50)

        assert windows[0].start == 0.0
        assert windows[-1].segments[-1]["id"] == "seg_0060"
        for w in windows:
            assert w.end - w.start <= 100
        for a, b in zip(windows, windows[1:]):
            assert 45 <= b.start - a.start <= 55  # ~step, on segment bounds
            assert b.start < a.end  # overlap
        covered = {s["id"] for w in windows for s in w.segments}
        assert covered == {s["id"] for s in transcript}

    def test_ids_and_order(self):
        windows = build_windows(list(reversed(_transcript(30))), 100, 50)
        assert [w.id for w in windows][:2] == ["w_001", "w_002"]
        assert windows[0].segments[0]["id"] == "seg_0001"

    def test_segment_longer_than_window_does_not_loop(self):
        transcript = [
            {"id": "a", "start": 0.0, "end": 300.0, "text": "long"},
            {"id": "b", "start": 300.0, "end": 305.0, "text": "short"},
        ]
        windows = build_windows(transcript, 100, 50)
        assert [[s["id"] for s in w.segments] for w in windows] == [["a"], ["b"]]

    def test_empty(self):
        assert build_windows([], 100, 50) == []


def test_format_windows_numbers_lines_with_offsets():
    w = Window(id="w_007", segments=_transcript(3))
    text = format_windows([w])
    assert text.splitlines() == [
        "### Window w_007",
        "[0 | 0s] Фраза номер 0.",
        "[1 | 6s] Фраза номер 1.",
        "[2 | 11s] Фраза номер 2.",
    ]


class TestCandidateFromResult:
    def setup_method(self):
        self.transcript = _transcript(40)
        self.index_of = {s["id"]: i for i, s in enumerate(self.transcript)}
        self.window = Window(id="w_002", segments=self.transcript[10:28])

    def test_maps_lines_to_absolute_segments(self):
        cand = candidate_from_result(
            self.window,
            {
                "start_line": 2,
                "end_line": 12,
                "hook": 1.0,
                "complete": 0.5,
                "value": 0.0,
                "title": "Заголовок",
            },
            self.index_of,
        )
        assert (cand.first, cand.last) == (12, 22)
        assert cand.start == self.transcript[12]["start"]
        assert cand.end == self.transcript[22]["end"]
        assert cand.score == round(0.4 * 1.0 + 0.35 * 0.5, 3)
        assert cand.title == "Заголовок"

    def test_clamps_out_of_range_lines_and_ratings(self):
        cand = candidate_from_result(
            self.window,
            {"start_line": -3, "end_line": 99, "hook": 7, "complete": "x", "value": None},
            self.index_of,
        )
        assert (cand.first, cand.last) == (10, 27)
        assert (cand.hook, cand.complete, cand.value) == (1.0, 0.0, 0.0)

    @pytest.mark.parametrize(
        "result",
        [
            {"start_line": 5, "end_line": 2},
            {"start_line": "abc", "end_line": 3},
            {},
        ],
    )
    def test_unusable_answers(self, result):
        assert candidate_from_result(self.window, result, self.index_of) is None


def _lines(texts, seg_len=5.0, gap=0.5):
    step = seg_len + gap
    return [
        {"id": f"s{i}", "start": i * step, "end": i * step + seg_len, "text": text}
        for i, text in enumerate(texts)
    ]


class TestFitBoundaries:
    def setup_method(self):
        self.transcript = _transcript(60)  # sentences of 5 s, 0.5 s gaps

    def test_too_long_is_trimmed_at_the_end(self):
        cand = fit_boundaries(_cand(0, 0, 0.5, first=4, last=30), self.transcript, 45, 90)
        assert cand.first == 4
        assert cand.duration <= 90
        assert "trimmed_to_max" in cand.tags

    def test_too_short_is_extended_forward(self):
        cand = fit_boundaries(_cand(0, 0, 0.5, first=4, last=6), self.transcript, 45, 90)
        assert cand.first == 4
        assert 45 <= cand.duration <= 90
        assert "extended_to_min" in cand.tags

    def test_within_limits_untouched(self):
        cand = fit_boundaries(_cand(0, 0, 0.5, first=4, last=14), self.transcript, 45, 90)
        assert (cand.first, cand.last, cand.tags) == (4, 14, [])

    def test_start_moves_back_to_sentence_start(self):
        texts = [
            "Раньше было иначе.",
            "Почему Чингисхан не пошёл",
            "по этому пути,",
            "а выбрал войну?",
        ] + [f"Фраза {i}." for i in range(20)]
        cand = fit_boundaries(_cand(0, 0, 0.5, first=2, last=12), _lines(texts), 45, 90)
        assert cand.first == 1
        assert "start_to_sentence" in cand.tags

    def test_end_moves_forward_to_sentence_end(self):
        texts = [f"Фраза {i}." for i in range(10)] + ["и вот так", "всё и закончилось."]
        cand = fit_boundaries(_cand(0, 0, 0.5, first=1, last=10), _lines(texts), 45, 90)
        assert cand.last == 11
        assert "end_to_sentence" in cand.tags

    def test_extension_to_min_stops_on_a_sentence_end(self):
        texts = [f"Фраза {i}." for i in range(8)] + [
            "основателя династии,",
            "который правил долго.",
        ]
        # lines 0..7 = 43.5 s; min 45 needs line 8, which ends mid-sentence
        cand = fit_boundaries(_cand(0, 0, 0.5, first=0, last=5), _lines(texts), 45, 90)
        assert cand.last == 9
        assert cand.duration >= 45

    def test_trailing_question_is_trimmed(self):
        texts = [f"Фраза {i}." for i in range(10)] + ["А что вы имеете в виду под самости?"]
        cand = fit_boundaries(_cand(0, 0, 0.5, first=0, last=10), _lines(texts), 45, 90)
        assert cand.last == 9
        assert "question_end_trimmed" in cand.tags

    def test_trailing_question_trimmed_slightly_below_min(self):
        texts = [f"Фраза {i}." for i in range(8)] + ["А там южнее, да?"]
        # lines 0..7 = 43.5 s: below 45 but above 0.8 * 45
        cand = fit_boundaries(_cand(0, 0, 0.5, first=0, last=8), _lines(texts), 45, 90)
        assert cand.last == 7

    def test_trailing_question_kept_if_clip_would_be_far_too_short(self):
        texts = [f"Фраза {i}." for i in range(6)] + ["Почему?"]
        cand = fit_boundaries(_cand(0, 0, 0.5, first=0, last=6), _lines(texts), 45, 90)
        assert cand.last == 6

    def test_continuation_opening_is_penalised(self):
        texts = ["То есть, тем самым он претендует на трон."] + [f"Фраза {i}." for i in range(12)]
        cand = _cand(0, 0, 0.8, first=0, last=9)
        cand = fit_boundaries(cand, _lines(texts), 45, 90)
        assert cand.hook == pytest.approx(0.55)
        assert cand.complete == pytest.approx(0.55)
        assert cand.score == round(0.4 * 0.55 + 0.35 * 0.55 + 0.25 * 0.8, 3)
        assert "continuation_start" in cand.tags

    def test_unfixable_mid_sentence_start_is_penalised(self):
        texts = ["вот и третье это ломка сознания"] + [f"Фраза {i}." for i in range(12)]
        cand = fit_boundaries(_cand(0, 0, 0.8, first=0, last=9), _lines(texts), 45, 90)
        assert cand.first == 0
        assert "continuation_start" in cand.tags
        assert cand.hook == pytest.approx(0.55)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("То есть, тем самым он претендует", True),
        ("И, соответственно, когда Ахмат Гирей", True),
        ("Он сказал, что", True),
        ("Нет, потому что вот смотрите, то же самое Абул Хаир.", True),
        ("Вот это большой-большой вопрос.", True),
        ("Почему Чингисхан не пошел по этому пути?", False),
        ("Кочевание это признак элиты.", False),
        ("Тогда вопрос: что было за Уралом?", False),
    ],
)
def test_is_continuation(text, expected):
    assert is_continuation(text) is expected


def test_pad_bounds_stops_at_neighbouring_lines():
    transcript = [
        {"id": "a", "start": 0.0, "end": 10.0, "text": "a"},
        {"id": "b", "start": 10.1, "end": 20.0, "text": "b"},
        {"id": "c", "start": 21.0, "end": 30.0, "text": "c"},
    ]
    cand = pad_bounds(_cand(10.1, 20.0, 0.5, first=1, last=1), transcript)
    assert cand.start == 10.0  # previous line ends at 10.0
    assert cand.end == 20.4


class TestSelectClips:
    def test_best_non_overlapping_first(self):
        cands = [
            _cand(0, 60, 0.9),
            _cand(30, 90, 0.95),  # overlaps both neighbours
            _cand(61, 120, 0.8),
            _cand(200, 260, 0.7),
            _cand(300, 360, 0.2),  # below threshold
        ]
        selected = select_clips(cands, max_clips=10, min_score=0.45)
        assert [(c.start, c.score) for c in selected] == [(30, 0.95), (200, 0.7)]

    def test_small_overlap_tolerated_and_limit(self):
        cands = [_cand(0, 60, 0.9), _cand(59, 120, 0.8), _cand(130, 190, 0.7)]
        selected = select_clips(cands, max_clips=2, min_score=0.0)
        assert [c.start for c in selected] == [0, 59]


def test_to_scored_segments_contract():
    cand = _cand(10.0, 70.0, 0.81)
    cand.title = "Хук"
    out = to_scored_segments([cand])
    assert out[0]["id"] == "clip_001"
    assert {"id", "start", "end", "score", "tags", "self_contained"} <= set(out[0])
    assert out[0]["title"] == "Хук"
    assert out[0]["tags"][0] == "llm_window"


class TestScoreWindows:
    def test_end_to_end_with_fake_llm(self):
        from src.scoring import run_llm_batches

        transcript = _transcript(80)  # ~440 s
        scoring = ScoringConfig(max_clips_per_video=3, min_score_threshold=0.3)
        prompts = []

        def analyze(payload, prompt):
            prompts.append(prompt)
            # every window: lines 1..10 (~55 s); later windows score higher
            return [
                {
                    "id": w["id"],
                    "start_line": 1,
                    "end_line": 10,
                    "hook": 0.5 + int(w["id"][2:]) / 100,
                    "complete": 0.8,
                    "value": 0.6,
                }
                for w in payload
            ]

        selected, candidates, debug = score_windows(
            transcript,
            scoring,
            analyze,
            run_llm_batches,
            "{{MIN_DURATION}}-{{MAX_DURATION}}\n{{WINDOWS}}",
        )

        assert prompts[0].startswith("45-90\n### Window w_001")
        assert len(candidates) == len(debug["windows"])
        assert 1 <= len(selected) <= 3
        assert [c.score for c in selected] == sorted((c.score for c in selected), reverse=True)
        for a in selected:
            for b in selected:
                if a is not b:
                    assert min(a.end, b.end) - max(a.start, b.start) <= 2.0

    def test_no_usable_answers_raises(self):
        from src.scoring import run_llm_batches
        from src.window_scoring import WindowScoringError

        with pytest.raises(WindowScoringError):
            score_windows(
                _transcript(30),
                ScoringConfig(),
                lambda payload, prompt: [],
                run_llm_batches,
                "{{WINDOWS}}",
            )


class TestScoreTranscriptWindows:
    def _config(self, tmp_path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "transcript.json").write_text(
            json.dumps(_transcript(60), ensure_ascii=False), encoding="utf-8"
        )
        return Config(
            work_dir=tmp_path,
            scoring=ScoringConfig(
                strategy="llm_windows",
                term_extraction_method="statistical",
                max_clips_per_video=4,
                llm=LLMConfig(enabled=True, provider="openai", api_key="x"),
            ),
        )

    def test_windows_strategy_writes_clips_and_debug(self, tmp_path):
        from src.scoring import score_transcript

        adapter = MagicMock()
        adapter.analyze_segments.side_effect = lambda payload, prompt: [
            {
                "id": w["id"],
                "start_line": 0,
                "end_line": 12,
                "hook": 0.9,
                "complete": 0.9,
                "value": 0.9,
                "title": "t",
            }
            for w in payload
        ]
        with patch("src.llm_adapter.create_llm_adapter", return_value=adapter):
            result = score_transcript(self._config(tmp_path))

        saved = json.loads((tmp_path / "artifacts" / "scored_segments.json").read_text())
        assert saved == result
        assert result and all("llm_window" in c["tags"] for c in result)
        debug = json.loads((tmp_path / "artifacts" / "llm_analysis.json").read_text())
        assert debug["strategy"] == "llm_windows"
        assert debug["prompt_file"].endswith("segment_analysis_v3.txt")
        adapter.unload.assert_called_once()

    def test_llm_failure_falls_back_to_heuristics_without_llm(self, tmp_path):
        from src.llm_adapter import LLMUnavailableError
        from src.scoring import score_transcript

        adapter = MagicMock()
        adapter.analyze_segments.side_effect = LLMUnavailableError("load failed")
        with patch("src.llm_adapter.create_llm_adapter", return_value=adapter) as factory:
            result = score_transcript(self._config(tmp_path))

        assert result  # segment_merge clips
        assert all("llm_window" not in c["tags"] for c in result)
        assert factory.call_count == 1  # the segment path does not retry the LLM


class TestLlamaCppThinkingAndSeed:
    def test_no_think_and_seed_are_passed(self, tmp_path):
        from src.llm_adapter import LlamaCppAdapter

        adapter = LlamaCppAdapter(
            LLMConfig(
                provider="llama_cpp",
                disable_thinking=True,
                seed=42,
                min_tokens_per_sec=0,
            )
        )
        adapter._llm = MagicMock()
        adapter._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": '{"windows": [{"id": "w_001"}]}'}}],
            "usage": {"completion_tokens": 10},
        }

        result = adapter.analyze_segments([{"id": "w_001"}], prompt="PROMPT")

        kwargs = adapter._llm.create_chat_completion.call_args.kwargs
        assert kwargs["messages"][1]["content"] == "PROMPT\n/no_think"
        assert kwargs["seed"] == 42
        assert result == [{"id": "w_001"}]

    def test_defaults_do_not_change_prompt(self):
        from src.llm_adapter import LlamaCppAdapter

        adapter = LlamaCppAdapter(LLMConfig(provider="llama_cpp", min_tokens_per_sec=0))
        adapter._llm = MagicMock()
        adapter._llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "[]"}}],
            "usage": {},
        }
        adapter.analyze_segments([], prompt="PROMPT")
        kwargs = adapter._llm.create_chat_completion.call_args.kwargs
        assert kwargs["messages"][1]["content"] == "PROMPT"
        assert "seed" not in kwargs
