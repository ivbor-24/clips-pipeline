"""
Tests for subtitle cues and styles (src/subtitles.py).
"""

import pytest

from src.config import RenderingConfig, SubtitleStyle
from src.subtitles import (
    ass_color,
    ass_style_fields,
    build_cues,
    resolve_subtitle_style,
    wrap_text,
)

LONG = (
    "Для Москвы все испортила ситуация 1571-1572 года, когда состоялся крупный "
    "крымский поход на Москву, причем не просто этот крымский поход состоялся, "
    "а когда крымские войска дошли до Москвы."
)


class TestStyle:
    def test_typewriter_default_look(self):
        style = resolve_subtitle_style(RenderingConfig())
        assert style.font == "Nimbus Mono PS"
        assert (style.box, style.box_color, style.box_opacity) == (True, "#4D4D4D", 0.75)
        assert (style.max_lines, style.anchor) == (3, "bottom")

    def test_overrides_apply_on_top_of_preset(self):
        cfg = RenderingConfig(subtitle_style_overrides={"box_opacity": 0.5, "font_size": 56})
        style = resolve_subtitle_style(cfg)
        assert (style.box_opacity, style.font_size, style.font) == (0.5, 56, "Nimbus Mono PS")

    def test_unknown_override_is_rejected_at_load(self):
        with pytest.raises(ValueError):
            RenderingConfig(subtitle_style_overrides={"box_opacty": 0.5})

    def test_legacy_preset_has_no_style_object(self):
        assert resolve_subtitle_style(RenderingConfig(subtitle_style="modern")) is None

    @pytest.mark.parametrize(
        "color,opacity,expected",
        [
            ("#FFFFFF", 1.0, "&H00FFFFFF"),
            ("#4D4D4D", 0.75, "&H404D4D4D"),
            ("#102030", 0.0, "&HFF302010"),  # BGR order, fully transparent
        ],
    )
    def test_ass_color(self, color, opacity, expected):
        assert ass_color(color, opacity) == expected

    def test_bad_color(self):
        with pytest.raises(ValueError):
            ass_color("white")

    def test_ass_fields_box_and_safe_area(self):
        fields = ass_style_fields(SubtitleStyle(), 1080, 1920)
        assert fields["border_style"] == 4  # one box behind the whole cue
        assert fields["back_colour"] == fields["outline_colour"] == "&H404D4D4D"
        assert fields["alignment"] == 2  # bottom center
        assert (fields["margin_l"], fields["margin_r"]) == (140, 140)
        assert fields["margin_v"] == 480  # clear of the platforms' bottom overlay
        assert fields["bold"] == -1

    def test_ass_fields_without_box(self):
        fields = ass_style_fields(SubtitleStyle(box=False, outline=3, anchor="top"), 1080, 1920)
        assert (fields["border_style"], fields["outline"], fields["alignment"]) == (1, 3, 8)


class TestWrap:
    def test_wraps_on_words(self):
        assert wrap_text("один два три четыре", 9) == ["один два", "три", "четыре"]

    def test_explicit_newline_kept(self):
        assert wrap_text("a b\nc", 20) == ["a b", "c"]


class TestBuildCues:
    def test_long_segment_splits_into_short_cues(self):
        transcript = [{"start": 100.0, "end": 112.0, "text": LONG}]
        cues = build_cues(transcript, 100.0, 130.0, max_chars_per_line=26, max_lines=3)

        assert len(cues) >= 2
        assert all(1 <= len(c.lines) <= 3 for c in cues)
        assert all(len(line) <= 26 for c in cues for line in c.lines)
        assert " ".join(" ".join(c.lines) for c in cues) == LONG
        assert cues[0].start == 0.0
        assert cues[-1].end == pytest.approx(12.0)
        for a, b in zip(cues, cues[1:]):
            assert a.end == b.start  # no gaps inside a segment

    def test_short_tail_is_balanced(self):
        text = "слово " * 13 + "хвост."
        cues = build_cues([{"start": 0, "end": 10, "text": text}], 0, 10, 26, 3)
        lengths = [len(" ".join(c.lines)) for c in cues]
        assert min(lengths) > 10

    def test_word_timestamps_are_used(self):
        seg = {
            "start": 10.0,
            "end": 14.0,
            "text": "раз два",
            "words": [
                {"word": "раз", "start": 10.0, "end": 10.5},
                {"word": "два", "start": 13.0, "end": 14.0},
            ],
        }
        cues = build_cues([seg], 10.0, 20.0, 26, 3)
        assert (cues[0].start, cues[0].end) == (0.0, 4.0)

    def test_clip_bounds_cut_words_outside(self):
        seg = {"start": 0.0, "end": 10.0, "text": "a b c d e f g h i j"}
        cues = build_cues([seg], 5.0, 10.0, 26, 3)
        assert " ".join(cues[0].lines) == "f g h i j"
        assert cues[0].start == pytest.approx(0.0, abs=0.6)

    def test_segments_outside_clip_ignored(self):
        assert build_cues([{"start": 0, "end": 5, "text": "x"}], 10, 20, 26, 3) == []
