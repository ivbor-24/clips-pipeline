"""
Tests for P2-08: ASS Subtitles
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

_PROBE = {
    "streams": [{"codec_type": "video", "width": 1920, "height": 1080}],
    "format": {"duration": 10},
}


def _fake_ffmpeg(cmd, **kwargs):
    """ffprobe answers with a 1920x1080 video; ffmpeg 'writes' its output file."""
    if cmd[0] == "ffmpeg" and cmd[-1].endswith(".mp4"):
        Path(cmd[-1]).write_bytes(b"fake output")
    return MagicMock(stdout=json.dumps(_PROBE), returncode=0)


class TestASSModuleFunctions:
    """Test ASS module functions exist."""

    def test_format_timestamp_ass_exists(self):
        """Test that format_timestamp_ass function exists."""
        from src.rendering import format_timestamp_ass

        assert callable(format_timestamp_ass)

    def test_get_ass_style_preset_exists(self):
        """Test that get_ass_style_preset function exists."""
        from src.rendering import get_ass_style_preset

        assert callable(get_ass_style_preset)

    def test_generate_ass_from_transcript_exists(self):
        """Test that generate_ass_from_transcript function exists."""
        from src.rendering import generate_ass_from_transcript

        assert callable(generate_ass_from_transcript)

    def test_save_ass_file_exists(self):
        """Test that save_ass_file function exists."""
        from src.rendering import save_ass_file

        assert callable(save_ass_file)


class TestFormatTimestampASS:
    """Test ASS timestamp formatting."""

    def test_format_zero_seconds(self):
        """Test formatting zero seconds."""
        from src.rendering import format_timestamp_ass

        result = format_timestamp_ass(0)
        assert result == "0:00:00.00"

    def test_format_simple_seconds(self):
        """Test formatting simple seconds value."""
        from src.rendering import format_timestamp_ass

        result = format_timestamp_ass(12.345)
        assert result == "0:00:12.34"

    def test_format_minutes(self):
        """Test formatting with minutes."""
        from src.rendering import format_timestamp_ass

        result = format_timestamp_ass(75.5)
        assert result == "0:01:15.50"

    def test_format_hours(self):
        """Test formatting with hours."""
        from src.rendering import format_timestamp_ass

        result = format_timestamp_ass(3665.123)
        assert result == "1:01:05.12"

    def test_format_edge_case(self):
        """Test formatting edge case."""
        from src.rendering import format_timestamp_ass

        result = format_timestamp_ass(59.99)
        assert result == "0:00:59.99"


class TestGetAssStylePreset:
    """Test ASS style preset retrieval."""

    def test_default_style(self):
        """Test default style preset."""
        from src.config import RenderingConfig
        from src.rendering import get_ass_style_preset

        config = RenderingConfig()
        style = get_ass_style_preset("default", config)

        assert style["fontname"] == "Arial"
        assert style["fontsize"] == 48
        assert style["bold"] == 0
        assert style["outline"] == 2
        assert style["shadow"] == 1

    def test_modern_style(self):
        """Test modern style preset."""
        from src.config import RenderingConfig
        from src.rendering import get_ass_style_preset

        config = RenderingConfig()
        style = get_ass_style_preset("modern", config)

        assert style["fontsize"] == 52
        assert style["bold"] == 1
        assert style["border_style"] == 3
        assert style["outline"] == 0
        assert style["shadow"] == 0

    def test_minimal_style(self):
        """Test minimal style preset."""
        from src.config import RenderingConfig
        from src.rendering import get_ass_style_preset

        config = RenderingConfig()
        style = get_ass_style_preset("minimal", config)

        assert style["fontsize"] == 40
        assert style["bold"] == 0
        assert style["outline"] == 1
        assert style["shadow"] == 0

    def test_unknown_style_falls_back_to_default(self):
        """Test that unknown style falls back to default."""
        from src.config import RenderingConfig
        from src.rendering import get_ass_style_preset

        config = RenderingConfig()
        style = get_ass_style_preset("unknown_style", config)
        default_style = get_ass_style_preset("default", config)

        assert style == default_style

    def test_custom_config_overrides(self):
        """Test that custom config values are used."""
        from src.config import RenderingConfig
        from src.rendering import get_ass_style_preset

        config = RenderingConfig(
            subtitle_font="Roboto", subtitle_fontsize=60, subtitle_primary_color="&H00FF0000"
        )
        style = get_ass_style_preset("default", config)

        assert style["fontname"] == "Roboto"
        assert style["fontsize"] == 60
        assert style["primary_colour"] == "&H00FF0000"


class TestGenerateAssFromTranscript:
    """Test ASS generation from transcript."""

    def test_empty_transcript(self):
        """Test generating ASS from empty transcript."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        result = generate_ass_from_transcript([], 0, 10, config)
        assert result == ""

    def test_single_segment(self):
        """Test generating ASS with single segment."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        transcript = [{"start": 0, "end": 5, "text": "Hello world"}]

        result = generate_ass_from_transcript(transcript, 0, 10, config)

        assert "[Script Info]" in result
        assert "[V4+ Styles]" in result
        assert "[Events]" in result
        assert "Hello world" in result
        assert "Dialogue:" in result

    def test_multiple_segments(self):
        """Test generating ASS with multiple segments."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        transcript = [
            {"start": 0, "end": 3, "text": "First segment"},
            {"start": 5, "end": 8, "text": "Second segment"},
        ]

        result = generate_ass_from_transcript(transcript, 0, 10, config)

        assert "First segment" in result
        assert "Second segment" in result
        assert result.count("Dialogue:") == 2

    def test_segments_outside_clip_range(self):
        """Test that segments outside clip range are filtered."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        transcript = [{"start": 100, "end": 105, "text": "Outside segment"}]

        result = generate_ass_from_transcript(transcript, 0, 10, config)
        assert result == ""

    def test_timestamps_adjusted_relative_to_clip_start(self):
        """Test that timestamps are adjusted relative to clip start."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        transcript = [{"start": 50, "end": 55, "text": "Adjusted segment"}]

        result = generate_ass_from_transcript(transcript, 50, 60, config)

        assert "0:00:00.00" in result
        assert "0:00:05.00" in result

    def test_script_info_section(self):
        """Test Script Info section structure."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        transcript = [{"start": 0, "end": 5, "text": "Test"}]

        result = generate_ass_from_transcript(transcript, 0, 10, config, 1080, 1920)

        assert "PlayResX: 1080" in result
        assert "PlayResY: 1920" in result
        assert "ScriptType: v4.00+" in result

    def test_styles_section_format(self):
        """Test V4+ Styles section format."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        transcript = [{"start": 0, "end": 5, "text": "Test"}]

        result = generate_ass_from_transcript(transcript, 0, 10, config)

        assert "Format: Name, Fontname, Fontsize" in result
        assert "Style: Default," in result

    def test_newlines_converted_to_ass_format(self):
        """Test that newlines in text are converted to ASS format."""
        from src.config import RenderingConfig
        from src.rendering import generate_ass_from_transcript

        config = RenderingConfig()
        transcript = [{"start": 0, "end": 5, "text": "Line 1\nLine 2"}]

        result = generate_ass_from_transcript(transcript, 0, 10, config)

        assert "\\N" in result


class TestSaveAssFile:
    """Test ASS file saving."""

    def test_ass_file_saved(self, tmp_path):
        """Test that ASS file is saved correctly."""
        from src.rendering import save_ass_file

        ass_content = "[Script Info]\nTitle: Test\n"
        output_path = tmp_path / "test.ass"

        save_ass_file(ass_content, output_path)

        assert output_path.exists()
        assert output_path.read_text() == ass_content

    def test_ass_file_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        from src.rendering import save_ass_file

        ass_content = "[Script Info]\n"
        output_path = tmp_path / "subdir" / "nested" / "test.ass"

        save_ass_file(ass_content, output_path)

        assert output_path.exists()


class TestRenderClipWithAss:
    """Test render_clip with ASS subtitles."""

    def test_render_clip_uses_ass_filter_for_ass_format(self, tmp_path):
        """Test that render_clip uses ass filter for ASS format."""
        from src.config import Config
        from src.rendering import render_clip

        config = Config()

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video")

        subtitle_path = tmp_path / "clip.ass"
        subtitle_path.write_text("[Script Info]\n")

        output_path = tmp_path / "output.mp4"

        clip_params = {"start": 0, "end": 10, "crop_frames": [], "fallback": "center_crop"}

        with (
            patch("src.rendering.get_video_info") as mock_info,
            patch("subprocess.run") as mock_run,
        ):

            mock_info.return_value = {"width": 1920, "height": 1080}
            mock_run.return_value = MagicMock(returncode=0)
            mock_run.return_value.stdout = json.dumps(
                {"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]}
            )

            output_path.write_bytes(b"fake output")

            render_clip(
                video_path=video_path,
                clip_params=clip_params,
                subtitle_path=subtitle_path,
                output_path=output_path,
                config=config,
                subtitle_format="ass",
            )

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            vf_index = cmd.index("-vf") + 1
            vf_value = cmd[vf_index]

            assert "ass=" in vf_value

    def test_render_clip_uses_subtitles_filter_for_srt_format(self, tmp_path):
        """Test that render_clip uses subtitles filter for SRT format."""
        from src.config import Config
        from src.rendering import render_clip

        config = Config()

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video")

        subtitle_path = tmp_path / "clip.srt"
        subtitle_path.write_text("1\n00:00:00,000 --> 00:00:05,000\nTest")

        output_path = tmp_path / "output.mp4"

        clip_params = {"start": 0, "end": 10, "crop_frames": [], "fallback": "center_crop"}

        with (
            patch("src.rendering.get_video_info") as mock_info,
            patch("subprocess.run") as mock_run,
        ):

            mock_info.return_value = {"width": 1920, "height": 1080}
            mock_run.return_value = MagicMock(returncode=0)

            output_path.write_bytes(b"fake output")

            render_clip(
                video_path=video_path,
                clip_params=clip_params,
                subtitle_path=subtitle_path,
                output_path=output_path,
                config=config,
                subtitle_format="srt",
            )

            call_args = mock_run.call_args
            cmd = call_args[0][0]
            vf_index = cmd.index("-vf") + 1
            vf_value = cmd[vf_index]

            assert "subtitles=" in vf_value


class TestRenderAndExportWithAss:
    """Test render_and_export with ASS format."""

    def test_ass_format_generates_ass_files(self, tmp_path):
        """Test that ASS format generates .ass files."""
        from src.config import Config
        from src.rendering import render_and_export

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        video_dir = artifacts_dir / "video"
        video_dir.mkdir()
        (video_dir / "prep.mp4").write_bytes(b"fake video")

        (artifacts_dir / "scored_segments.json").write_text(
            json.dumps([{"start": 0, "end": 10, "score": 0.7, "tags": [], "self_contained": True}])
        )
        (artifacts_dir / "crop_params.json").write_text(
            json.dumps([{"start": 0, "end": 10, "frames": [], "fallback": "center_crop"}])
        )
        (artifacts_dir / "transcript.json").write_text(
            json.dumps([{"start": 0, "end": 10, "text": "Test transcript"}])
        )
        (artifacts_dir / "meta.json").write_text(json.dumps({"source_hash": "sha256:test"}))

        import os

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            config = Config()
            config.rendering.subtitle_format = "ass"

            with patch("subprocess.run", side_effect=_fake_ffmpeg):
                output_dir = tmp_path / "output" / "clips"
                render_and_export(config, dry_run=False)

                ass_file = output_dir / "clip_001.ass"
                assert ass_file.exists()

                content = ass_file.read_text()
                assert "[Script Info]" in content
                assert "[V4+ Styles]" in content
                assert "[Events]" in content
        finally:
            os.chdir(original_cwd)

    def test_srt_format_generates_srt_files(self, tmp_path):
        """Test that SRT format still generates .srt files."""
        from src.config import Config
        from src.rendering import render_and_export

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        video_dir = artifacts_dir / "video"
        video_dir.mkdir()
        (video_dir / "prep.mp4").write_bytes(b"fake video")

        (artifacts_dir / "scored_segments.json").write_text(
            json.dumps([{"start": 0, "end": 10, "score": 0.7, "tags": [], "self_contained": True}])
        )
        (artifacts_dir / "crop_params.json").write_text(
            json.dumps([{"start": 0, "end": 10, "frames": [], "fallback": "center_crop"}])
        )
        (artifacts_dir / "transcript.json").write_text(
            json.dumps([{"start": 0, "end": 10, "text": "Test transcript"}])
        )
        (artifacts_dir / "meta.json").write_text(json.dumps({"source_hash": "sha256:test"}))

        import os

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            config = Config()
            config.rendering.subtitle_format = "srt"

            with patch("subprocess.run", side_effect=_fake_ffmpeg):
                output_dir = tmp_path / "output" / "clips"
                render_and_export(config, dry_run=False)

                srt_file = output_dir / "clip_001.srt"
                assert srt_file.exists()
        finally:
            os.chdir(original_cwd)


class TestRenderingConfigAssFields:
    """Test RenderingConfig ASS fields."""

    def test_default_subtitle_format(self):
        """Test default subtitle format is srt."""
        from src.config import RenderingConfig

        config = RenderingConfig()
        assert config.subtitle_format == "srt"

    def test_default_subtitle_style(self):
        """Default style: typewriter font on a translucent gray box."""
        from src.config import RenderingConfig

        config = RenderingConfig()
        assert config.subtitle_style == "typewriter"

    def test_ass_format_accepted(self):
        """Test that ass format is accepted."""
        from src.config import RenderingConfig

        config = RenderingConfig(subtitle_format="ass")
        assert config.subtitle_format == "ass"

    def test_modern_style_accepted(self):
        """Test that modern style is accepted."""
        from src.config import RenderingConfig

        config = RenderingConfig(subtitle_style="modern")
        assert config.subtitle_style == "modern"

    def test_minimal_style_accepted(self):
        """Test that minimal style is accepted."""
        from src.config import RenderingConfig

        config = RenderingConfig(subtitle_style="minimal")
        assert config.subtitle_style == "minimal"

    def test_custom_font(self):
        """Test custom font setting."""
        from src.config import RenderingConfig

        config = RenderingConfig(subtitle_font="Roboto")
        assert config.subtitle_font == "Roboto"

    def test_custom_fontsize(self):
        """Test custom fontsize setting."""
        from src.config import RenderingConfig

        config = RenderingConfig(subtitle_fontsize=60)
        assert config.subtitle_fontsize == 60
