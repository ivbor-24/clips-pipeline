"""
Tests for TASK-06: Rendering & Export Module
"""

import json
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

        filter_str, label = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        assert label is None
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

        filter_str, label = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        # strip width 608, max x = 1920-608 = 1312
        assert filter_str.startswith("crop=608:1080:1312:0,")

    def test_target_dimensions_in_filter(self):
        """Test that target dimensions are in filter."""
        from src.rendering import build_crop_filter

        crop_params = {"frames": [], "segments": [], "fallback": "none"}

        filter_str, _ = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        assert "1080:1920" in filter_str

    def test_multi_segment_graph_with_concat(self):
        """Multiple tracking segments produce trim/concat filtergraph."""
        from src.rendering import build_crop_filter

        crop_params = {
            "frames": [],
            "segments": [
                {"start": 10.0, "end": 30.0, "cx": 500.0, "cy": 400.0, "has_face": True},
                {"start": 30.0, "end": 50.0, "cx": 1500.0, "cy": 300.0, "has_face": True},
            ],
            "fallback": "none",
        }

        filter_str, label = build_crop_filter(crop_params, 1920, 1080, 1080, 1920, clip_start=10.0)

        assert label == "vout"
        assert "trim=start=0.0:end=20.0" in filter_str
        assert "trim=start=20.0:end=40.0" in filter_str
        assert "[v0]" in filter_str and "[v1]" in filter_str
        assert "concat=n=2:v=1:a=0[vout]" in filter_str

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

        filter_str, label = build_crop_filter(crop_params, 1920, 1080, 1080, 1920)

        assert label == "vout"
        assert "crop=608:1080:656:0" in filter_str


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

    def test_multi_segment_graph_maps_vout_and_aout(self, tmp_path):
        segments = [
            {"start": 100.0, "end": 130.0, "cx": 1300.0, "cy": 400.0, "has_face": True},
            {"start": 130.0, "end": 160.0, "cx": 800.0, "cy": 300.0, "has_face": True},
        ]
        cmd = self._render(tmp_path, segments)

        graph = cmd[cmd.index("-filter_complex") + 1]
        # clip-relative trims (input seek resets the timeline to 0)
        assert "trim=start=0.0:end=30.0" in graph
        assert "trim=start=30.0:end=60.0" in graph
        assert "concat=n=2:v=1:a=0" in graph
        # subtitles are burned once, after concat
        assert graph.count("subtitles=") == 1
        assert graph.index("concat=") < graph.index("subtitles=")
        assert "[0:a]loudnorm" in graph and "[aout]" in graph
        maps = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-map"]
        assert maps == ["[vout]", "[aout]"]
        assert "-vf" not in cmd and "-af" not in cmd

    def test_multi_segment_without_subtitles(self, tmp_path):
        segments = [
            {"start": 100.0, "end": 130.0, "cx": 1300.0, "cy": 400.0, "has_face": True},
            {"start": 130.0, "end": 160.0, "cx": None, "cy": None, "has_face": False},
        ]
        cmd = self._render(tmp_path, segments, with_subtitles=False)

        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "subtitles=" not in graph
        assert graph.split(";")[-2].endswith("concat=n=2:v=1:a=0[vout]")
