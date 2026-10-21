"""
Tests for TASK-06: Rendering & Export Module
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestRenderingModule:
    """Test rendering module imports and structure."""

    def test_module_exists(self):
        """Test that rendering module exists."""
        from src import rendering

        assert rendering is not None

    def test_rendering_error_exists(self):
        """Test that RenderingError exception exists."""
        from src.rendering import RenderingError

        assert RenderingError is not None
        assert issubclass(RenderingError, Exception)

    def test_load_json_file_function_exists(self):
        """Test that load_json_file function exists."""
        from src.rendering import load_json_file

        assert callable(load_json_file)

    def test_format_timestamp_srt_function_exists(self):
        """Test that format_timestamp_srt function exists."""
        from src.rendering import format_timestamp_srt

        assert callable(format_timestamp_srt)

    def test_generate_srt_from_transcript_function_exists(self):
        """Test that generate_srt_from_transcript function exists."""
        from src.rendering import generate_srt_from_transcript

        assert callable(generate_srt_from_transcript)

    def test_save_srt_file_function_exists(self):
        """Test that save_srt_file function exists."""
        from src.rendering import save_srt_file

        assert callable(save_srt_file)

    def test_build_crop_filter_function_exists(self):
        """Test that build_crop_filter function exists."""
        from src.rendering import build_crop_filter

        assert callable(build_crop_filter)

    def test_build_audio_loudnorm_filter_function_exists(self):
        """Test that build_audio_loudnorm_filter function exists."""
        from src.rendering import build_audio_loudnorm_filter

        assert callable(build_audio_loudnorm_filter)

    def test_render_clip_function_exists(self):
        """Test that render_clip function exists."""
        from src.rendering import render_clip

        assert callable(render_clip)

    def test_get_video_info_function_exists(self):
        """Test that get_video_info function exists."""
        from src.rendering import get_video_info

        assert callable(get_video_info)

    def test_validate_output_file_function_exists(self):
        """Test that validate_output_file function exists."""
        from src.rendering import validate_output_file

        assert callable(validate_output_file)

    def test_delete_invalid_file_function_exists(self):
        """Test that delete_invalid_file function exists."""
        from src.rendering import delete_invalid_file

        assert callable(delete_invalid_file)

    def test_generate_clip_metadata_function_exists(self):
        """Test that generate_clip_metadata function exists."""
        from src.rendering import generate_clip_metadata

        assert callable(generate_clip_metadata)

    def test_save_clip_metadata_function_exists(self):
        """Test that save_clip_metadata function exists."""
        from src.rendering import save_clip_metadata

        assert callable(save_clip_metadata)

    def test_generate_manifest_function_exists(self):
        """Test that generate_manifest function exists."""
        from src.rendering import generate_manifest

        assert callable(generate_manifest)

    def test_render_and_export_function_exists(self):
        """Test that main render_and_export function exists."""
        from src.rendering import render_and_export

        assert callable(render_and_export)


class TestFormatTimestampSRT:
    """Test SRT timestamp formatting."""

    def test_format_zero_seconds(self):
        """Test formatting zero seconds."""
        from src.rendering import format_timestamp_srt

        result = format_timestamp_srt(0)
        assert result == "00:00:00,000"

    def test_format_simple_seconds(self):
        """Test formatting simple seconds value."""
        from src.rendering import format_timestamp_srt

        result = format_timestamp_srt(12.345)
        assert result == "00:00:12,345"

    def test_format_minutes(self):
        """Test formatting with minutes."""
        from src.rendering import format_timestamp_srt

        result = format_timestamp_srt(75.5)  # 1 min 15.5 sec
        assert result == "00:01:15,500"

    def test_format_hours(self):
        """Test formatting with hours."""
        from src.rendering import format_timestamp_srt

        result = format_timestamp_srt(3665.123)  # 1 hour 1 min 5.123 sec
        assert result == "01:01:05,123"

    def test_format_edge_case_milliseconds(self):
        """Test formatting edge case milliseconds."""
        from src.rendering import format_timestamp_srt

        result = format_timestamp_srt(1.999)
        assert result == "00:00:01,999"


class TestGenerateSRTFromTranscript:
    """Test SRT generation from transcript."""

    def test_empty_transcript(self):
        """Test generating SRT from empty transcript."""
        from src.rendering import generate_srt_from_transcript

        result = generate_srt_from_transcript([], 0, 10)
        assert result == ""

    def test_single_segment(self):
        """Test generating SRT with single segment."""
        from src.rendering import generate_srt_from_transcript

        transcript = [{"start": 0, "end": 5, "text": "Hello world"}]

        result = generate_srt_from_transcript(transcript, 0, 10)

        assert "1" in result
        assert "00:00:00,000 --> 00:00:05,000" in result
        assert "Hello world" in result

    def test_multiple_segments(self):
        """Test generating SRT with multiple segments."""
        from src.rendering import generate_srt_from_transcript

        transcript = [
            {"start": 0, "end": 3, "text": "First segment"},
            {"start": 5, "end": 8, "text": "Second segment"},
        ]

        result = generate_srt_from_transcript(transcript, 0, 10)

        assert "First segment" in result
        assert "Second segment" in result
        assert "2" in result  # Two cues

    def test_segments_outside_clip_range(self):
        """Test that segments outside clip range are filtered."""
        from src.rendering import generate_srt_from_transcript

        transcript = [{"start": 100, "end": 105, "text": "Outside segment"}]

        result = generate_srt_from_transcript(transcript, 0, 10)
        assert result == ""

    def test_timestamps_adjusted_relative_to_clip_start(self):
        """Test that timestamps are adjusted relative to clip start."""
        from src.rendering import generate_srt_from_transcript

        transcript = [{"start": 50, "end": 55, "text": "Adjusted segment"}]

        result = generate_srt_from_transcript(transcript, 50, 60)

        # Should start at 00:00:00 relative to clip start
        assert "00:00:00,000 --> 00:00:05,000" in result


class TestBuildCropFilter:
    """Test crop filter building for ffmpeg."""

    def test_empty_params_center_strip_no_pad(self):
        """Empty params: full-height centered 9:16 strip, no letterbox pad."""
        from src.rendering import build_crop_filter

        crop_params = {"frames": [], "segments": [], "fallback": "center_crop"}

        filter_str = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        assert filter_str.startswith("crop=")
        # 1080 * 9/16 = 607.5 -> 608 wide strip, full height
        assert filter_str.startswith("crop=608:1080:")
        # centered: (1920-608)//2 = 656
        assert filter_str.startswith("crop=608:1080:656:0,")
        assert "scale=1080:1920" in filter_str
        assert "pad=" not in filter_str
        assert "setsar=1" in filter_str

    def test_frames_face_center_shifts_strip(self):
        """Face center from frames shifts the strip horizontally."""
        from src.rendering import build_crop_filter

        # Face at x=1600 (right side) -> strip pinned to right edge
        crop_params = {
            "frames": [{"x": 1600, "y": 400, "w": 200, "h": 300, "conf": 0.9}],
            "segments": [],
            "fallback": "none",
        }

        filter_str = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        # strip width 608, max x = 1920-608 = 1312
        assert filter_str.startswith("crop=608:1080:1312:0,")

    def test_target_dimensions_in_filter(self):
        """Test that target dimensions are in filter."""
        from src.rendering import build_crop_filter

        crop_params = {"frames": [], "segments": [], "fallback": "none"}

        filter_str = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        assert "1080:1920" in filter_str

    def test_segments_switch_the_window_on_their_start(self):
        """One crop filter whose window switches at the next segment's start (clip time)."""
        from src.rendering import build_crop_filter

        crop_params = {
            "frames": [],
            "segments": [
                {"start": 10.0, "end": 30.0, "cx": 500.0, "cy": 400.0, "has_face": True},
                {"start": 30.0, "end": 50.0, "cx": 1500.0, "cy": 300.0, "has_face": True},
            ],
            "fallback": "none",
        }

        filter_str = build_crop_filter(crop_params, 1920, 1080, 1080, 1920, clip_start=10.0)

        # x = cx - 304: 196, then 1196 from t = 20 s (a frame within 5 ms counts)
        assert filter_str == (
            "crop=w=608:h=1080:x='196+1000*gte(t,19.995)':y='0':exact=1,scale=1080:1920,setsar=1"
        )

    def test_path_pans_the_window(self):
        from src.rendering import build_crop_filter

        segment = {"start": 10.0, "end": 30.0, "cx": 800.0, "cy": 400.0, "has_face": True}
        segment["path"] = [[10.0, 800.0], [14.0, 800.0], [16.5, 1050.0], [30.0, 1300.0]]

        filter_str = build_crop_filter({"segments": [segment]}, 1920, 1080, 1080, 1920, 10.0)

        x = filter_str.split("x='")[1].split("'")[0]
        assert x == "496+250*clip((t-4.000)/2.500,0,1)+250*clip((t-6.500)/13.500,0,1)"

    def test_segments_without_face_use_center(self):
        """No-face segments fall back to centered strip."""
        from src.rendering import build_crop_filter

        crop_params = {
            "frames": [],
            "segments": [
                {"start": 0.0, "end": 5.0, "cx": None, "cy": None, "has_face": False},
                {"start": 5.0, "end": 10.0, "cx": None, "cy": None, "has_face": False},
            ],
            "fallback": "center_crop",
        }

        filter_str = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        assert filter_str.startswith("crop=w=608:h=1080:x='656':y='0':exact=1,")

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_window_position_on_every_frame(self):
        """ffmpeg moves the window as planned: still, a pan, a switch on the cut frame."""
        import subprocess

        from src.rendering import build_crop_filter

        # Each column of the 640x320 source holds its own x: red = x % 256, green = x // 256.
        source = "color=s=640x320:r=25:d=4,format=rgb24,geq=r='mod(X,256)':g='floor(X/256)':b=0"
        segments = [
            # strip 180 wide: x = cx - 90; pans 100 -> 300 between 1 and 2 s
            {"start": 0.0, "end": 2.0, "cx": 200.0, "cy": 160.0, "has_face": True},
            {"start": 2.0, "end": 4.0, "cx": 500.0, "cy": 160.0, "has_face": True},
        ]
        segments[0]["path"] = [[0.0, 100.0], [1.0, 100.0], [2.0, 300.0]]
        chain = build_crop_filter({"segments": segments}, 640, 320, 180, 320)
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", source, "-vf", chain]
            + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True,
            capture_output=True,
        )
        frame_size = 180 * 320 * 3
        frames = [
            result.stdout[i : i + frame_size] for i in range(0, len(result.stdout), frame_size)
        ]

        positions = [frame[0] + 256 * frame[1] for frame in frames]  # x of the left column
        expected = [10 if n < 25 else 10 + 200 * min(n - 25, 25) / 25 for n in range(50)]
        expected += [410] * 50
        assert len(positions) == 100
        assert all(abs(x - e) <= 1 for x, e in zip(positions, expected)), positions


class TestBuildAudioLoudnormFilter:
    """Test audio loudnorm filter building."""

    def test_default_loudnorm_values(self):
        """Test building loudnorm filter with default values."""
        from src.config import RenderingConfig
        from src.rendering import build_audio_loudnorm_filter

        config = RenderingConfig()
        result = build_audio_loudnorm_filter(config)

        assert "loudnorm=" in result
        assert "I=-14" in result
        assert "TP=-1.5" in result
        assert "LRA=11" in result


class TestRenderingConfig:
    """Test rendering configuration defaults and ffmpeg mapping."""

    def test_rendering_config_defaults(self):
        """Test default rendering config values."""
        from src.config import RenderingConfig

        config = RenderingConfig()
        assert config.video_codec == "h264"
        assert config.video_preset == "medium"
        assert config.render_timeout_sec == 300

    def test_resolve_video_encoder_mapping(self):
        """Test codec name to ffmpeg encoder mapping."""
        from src.rendering import resolve_video_encoder

        assert resolve_video_encoder("h264") == "libx264"
        assert resolve_video_encoder("hevc") == "libx265"
        assert resolve_video_encoder("av1") == "libsvtav1"
        assert resolve_video_encoder("copy") == "copy"
        assert resolve_video_encoder("libx264") == "libx264"


class TestValidateOutputFile:
    """Test output file validation."""

    def test_nonexistent_file_returns_false(self, tmp_path):
        """Test that nonexistent file returns False."""
        from src.rendering import validate_output_file

        fake_path = tmp_path / "nonexistent.mp4"
        result = validate_output_file(fake_path)
        assert result is False


class TestGenerateClipMetadata:
    """Test clip metadata generation."""

    def test_basic_metadata_generation(self):
        """Test generating basic clip metadata."""
        from src.rendering import generate_clip_metadata

        clip_params = {"start": 10, "end": 40}
        scored_segment = {
            "score": 0.75,
            "tags": ["definition", "terms:test,science"],
            "self_contained": True,
        }
        source_meta = {"source_hash": "sha256:abc123"}

        result = generate_clip_metadata(
            clip_params=clip_params,
            scored_segment=scored_segment,
            source_meta=source_meta,
            render_time_sec=2.5,
            clip_index=1,
        )

        assert result["clip_id"] == "clip_001"
        assert result["start_time"] == 10
        assert result["end_time"] == 40
        assert result["duration_sec"] == 30
        assert result["score"] == 0.75
        assert result["source_hash"] == "sha256:abc123"
        assert result["render_time_sec"] == 2.5
        assert result["resolution"] == [1080, 1920]
        assert result["self_contained"] is True

    def test_terms_extraction_from_tags(self):
        """Test extracting terms from tags."""
        from src.rendering import generate_clip_metadata

        scored_segment = {
            "score": 0.5,
            "tags": ["definition", "terms:quantum,physics,entanglement"],
        }

        result = generate_clip_metadata(
            clip_params={"start": 0, "end": 10},
            scored_segment=scored_segment,
            source_meta={},
            render_time_sec=1,
            clip_index=1,
        )

        assert "quantum" in result["terms_found"]
        assert "physics" in result["terms_found"]
        assert "entanglement" in result["terms_found"]


class TestGenerateManifest:
    """Test manifest generation."""

    def test_manifest_structure(self, tmp_path):
        """Test manifest has correct structure."""
        from src.rendering import generate_manifest

        clips_metadata = [
            {"clip_id": "clip_001", "score": 0.75},
            {"clip_id": "clip_002", "score": 0.65},
        ]
        source_meta = {
            "source": "test_video.mp4",
            "source_hash": "sha256:xyz",
            "duration_sec": 3600,
        }
        output_path = tmp_path / "manifest.json"

        generate_manifest(clips_metadata, source_meta, output_path)

        assert output_path.exists()

        with open(output_path) as f:
            manifest = json.load(f)

        assert manifest["source"] == "test_video.mp4"
        assert manifest["source_hash"] == "sha256:xyz"
        assert manifest["source_duration_sec"] == 3600
        assert manifest["total_clips"] == 2
        assert len(manifest["clips"]) == 2
        assert "generated_at" in manifest
        from src import __version__

        assert manifest["pipeline_version"] == __version__


class TestRenderAndExportDryRun:
    """Test render_and_export dry run mode."""

    def test_dry_run_returns_status(self, tmp_path):
        """Test dry run returns status without processing."""
        from src.config import Config
        from src.rendering import render_and_export

        # Create mock artifact files
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        video_dir = artifacts_dir / "video"
        video_dir.mkdir()
        (video_dir / "prep.mp4").write_bytes(b"fake video")

        # Create JSON artifacts
        (artifacts_dir / "scored_segments.json").write_text(
            json.dumps([{"start": 0, "end": 10, "score": 0.7}])
        )
        (artifacts_dir / "crop_params.json").write_text(
            json.dumps([{"start": 0, "end": 10, "frames": []}])
        )
        (artifacts_dir / "transcript.json").write_text(
            json.dumps([{"start": 0, "end": 10, "text": "test"}])
        )
        (artifacts_dir / "meta.json").write_text(json.dumps({"source_hash": "sha256:test"}))

        # Change to temp directory for test
        import os

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            config = Config()
            result = render_and_export(config, dry_run=True)

            assert result is not None
            assert len(result) == 1
            assert result[0]["status"] == "dry_run_passed"
        finally:
            os.chdir(original_cwd)


class TestSaveSRTFile:
    """Test SRT file saving."""

    def test_srt_file_saved(self, tmp_path):
        """Test that SRT file is saved correctly."""
        from src.rendering import save_srt_file

        srt_content = "1\n00:00:00,000 --> 00:00:05,000\nHello World"
        output_path = tmp_path / "test.srt"

        save_srt_file(srt_content, output_path)

        assert output_path.exists()
        assert output_path.read_text() == srt_content


class TestLoadJSONFile:
    """Test JSON file loading."""

    def test_load_valid_json(self, tmp_path):
        """Test loading valid JSON file."""
        from src.rendering import load_json_file

        data = {"key": "value", "number": 42}
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps(data))

        result = load_json_file(json_path)

        assert result == data

    def test_load_nonexistent_file_raises_error(self, tmp_path):
        """Test that loading nonexistent file raises error."""
        from src.rendering import RenderingError, load_json_file

        fake_path = tmp_path / "nonexistent.json"

        with pytest.raises(RenderingError):
            load_json_file(fake_path)


class TestDeleteInvalidFile:
    """Test invalid file deletion."""

    def test_delete_existing_file(self, tmp_path):
        """Test deleting an existing file."""
        from src.rendering import delete_invalid_file

        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("content")

        assert test_file.exists()
        delete_invalid_file(test_file)
        assert not test_file.exists()

    def test_delete_nonexistent_file_no_error(self, tmp_path):
        """Test that deleting nonexistent file doesn't raise error."""
        from src.rendering import delete_invalid_file

        fake_path = tmp_path / "nonexistent.txt"

        # Should not raise any exception
        delete_invalid_file(fake_path)


class TestIntegration:
    """Integration tests for rendering pipeline."""

    def test_full_pipeline_mock(self, tmp_path):
        """Test full rendering pipeline with mocked ffmpeg."""
        from src.config import Config
        from src.rendering import render_and_export

        # Create mock artifact files
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        video_dir = artifacts_dir / "video"
        video_dir.mkdir()
        (video_dir / "prep.mp4").write_bytes(b"fake video")

        # Create JSON artifacts
        scored_segments = [
            {
                "id": "clip_001",
                "start": 0,
                "end": 30,
                "score": 0.75,
                "tags": ["definition"],
                "self_contained": True,
            }
        ]
        (artifacts_dir / "scored_segments.json").write_text(json.dumps(scored_segments))
        (artifacts_dir / "crop_params.json").write_text(
            json.dumps([{"start": 0, "end": 30, "frames": [], "fallback": "center_crop"}])
        )
        (artifacts_dir / "transcript.json").write_text(
            json.dumps([{"start": 0, "end": 30, "text": "Test transcript"}])
        )
        (artifacts_dir / "meta.json").write_text(
            json.dumps({"source": "test.mp4", "source_hash": "sha256:test123", "duration_sec": 100})
        )

        # Change to temp directory
        import os

        original_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            config = Config()

            # Mock ffmpeg and ffprobe calls
            with patch("subprocess.run") as mock_run:
                # Mock successful ffprobe for video info
                mock_run.return_value = MagicMock(
                    stdout=json.dumps(
                        {"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]}
                    ),
                    returncode=0,
                )

                # Dry run should work without actual ffmpeg
                result = render_and_export(config, dry_run=True)

                assert result is not None
        finally:
            os.chdir(original_cwd)


class TestRenderClipCommand:
    """B4: ffmpeg command built by render_clip."""

    def _render(self, tmp_path, segments, with_subtitles=True):
        from src.config import Config
        from src.rendering import render_clip

        subtitle = tmp_path / "clip_001.srt"
        if with_subtitles:
            subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\nПривет\n", encoding="utf-8")
        output = tmp_path / "clip_001.mp4"
        clip_params = {"start": 100.0, "end": 160.0, "segments": segments, "crop_frames": []}

        def fake_run(cmd, **kwargs):
            Path(cmd[-1]).write_bytes(b"mp4")  # ffmpeg writes clip_001.part.mp4
            return MagicMock(returncode=0)

        with (
            patch("src.rendering.get_video_info", return_value={"width": 1920, "height": 1080}),
            patch("src.rendering.subprocess.run", side_effect=fake_run) as run_mock,
        ):
            ok = render_clip(tmp_path / "prep.mp4", clip_params, subtitle, output, Config())

        assert ok is True
        return run_mock.call_args.args[0]

    def test_input_seek_before_input(self, tmp_path):
        cmd = self._render(tmp_path, [])
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-ss") + 1] == "100.0"
        assert cmd[cmd.index("-t") + 1] == "60.0"

    def test_single_window_uses_vf_with_subtitles_last(self, tmp_path):
        segments = [{"start": 100.0, "end": 160.0, "cx": 1300.0, "cy": 400.0, "has_face": True}]
        cmd = self._render(tmp_path, segments)

        assert "-filter_complex" not in cmd
        vf = cmd[cmd.index("-vf") + 1]
        assert vf.startswith("crop=608:1080:")
        assert vf.split(",")[-1].startswith("subtitles=")
        assert "-af" in cmd

    def test_moving_window_uses_vf_with_subtitles_last(self, tmp_path):
        segments = [
            {"start": 100.0, "end": 130.0, "cx": 1300.0, "cy": 400.0, "has_face": True},
            {"start": 130.0, "end": 160.0, "cx": 800.0, "cy": 300.0, "has_face": True},
        ]
        cmd = self._render(tmp_path, segments)

        assert "-filter_complex" not in cmd and "-map" not in cmd
        vf = cmd[cmd.index("-vf") + 1]
        # clip-relative times (input seek resets the timeline to 0)
        assert vf.startswith("crop=w=608:h=1080:x='996-500*gte(t,29.995)':y='0':exact=1,")
        assert "trim=" not in vf and "concat=" not in vf
        assert vf.count("subtitles=") == 1 and vf.split(",")[-1].startswith("subtitles=")
        assert cmd[cmd.index("-af") + 1].startswith("loudnorm")

    def test_moving_window_without_subtitles(self, tmp_path):
        segments = [
            {"start": 100.0, "end": 130.0, "cx": 1300.0, "cy": 400.0, "has_face": True},
            {"start": 130.0, "end": 160.0, "cx": None, "cy": None, "has_face": False},
        ]
        cmd = self._render(tmp_path, segments, with_subtitles=False)

        vf = cmd[cmd.index("-vf") + 1]
        assert "subtitles=" not in vf
        assert vf.endswith("scale=1080:1920,setsar=1")
