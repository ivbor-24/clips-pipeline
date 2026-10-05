"""
Tests for TASK-E2E-02: Artifact Validation Module
"""

import json

from src.artifact_validator import (
    get_expected_duration_from_meta,
    is_valid_crop_params,
    is_valid_json_file,
    is_valid_manifest,
    is_valid_meta,
    is_valid_scored_segments,
    is_valid_transcript,
)


class TestIsValidJsonFile:
    """Test L1: basic JSON file validation."""

    def test_valid_json_file(self, tmp_path):
        """Test that a valid JSON file returns True."""
        f = tmp_path / "valid.json"
        f.write_text('{"key": "value"}')
        assert is_valid_json_file(f) is True

    def test_nonexistent_file(self, tmp_path):
        """Test that a nonexistent file returns False."""
        f = tmp_path / "missing.json"
        assert is_valid_json_file(f) is False

    def test_empty_file(self, tmp_path):
        """Test that an empty file returns False."""
        f = tmp_path / "empty.json"
        f.touch()
        assert is_valid_json_file(f) is False

    def test_invalid_json(self, tmp_path):
        """Test that invalid JSON returns False."""
        f = tmp_path / "bad.json"
        f.write_text("{invalid json}")
        assert is_valid_json_file(f) is False

    def test_valid_list_json(self, tmp_path):
        """Test that a valid JSON list returns True."""
        f = tmp_path / "list.json"
        f.write_text('[{"a": 1}, {"b": 2}]')
        assert is_valid_json_file(f) is True


class TestIsValidMeta:
    """Test L1 + L2: meta.json validation."""

    def _write_meta(self, tmp_path, data):
        f = tmp_path / "meta.json"
        f.write_text(json.dumps(data))
        return f

    def test_valid_meta(self, tmp_path):
        """Test valid meta.json."""
        data = {
            "source": "test.mp4",
            "source_hash": "sha256:abc",
            "duration_sec": 120.5,
            "resolution": [1920, 1080],
            "fps": 30.0,
        }
        f = self._write_meta(tmp_path, data)
        assert is_valid_meta(f) is True

    def test_meta_missing_keys(self, tmp_path):
        """Test meta.json missing required keys."""
        data = {"source": "test.mp4"}
        f = self._write_meta(tmp_path, data)
        assert is_valid_meta(f) is False

    def test_meta_invalid_duration_zero(self, tmp_path):
        """Test meta.json with zero duration."""
        data = {
            "source": "test.mp4",
            "duration_sec": 0,
            "resolution": [1920, 1080],
            "fps": 30.0,
        }
        f = self._write_meta(tmp_path, data)
        assert is_valid_meta(f) is False

    def test_meta_invalid_duration_negative(self, tmp_path):
        """Test meta.json with negative duration."""
        data = {
            "source": "test.mp4",
            "duration_sec": -10,
            "resolution": [1920, 1080],
            "fps": 30.0,
        }
        f = self._write_meta(tmp_path, data)
        assert is_valid_meta(f) is False

    def test_meta_invalid_resolution(self, tmp_path):
        """Test meta.json with invalid resolution."""
        data = {
            "source": "test.mp4",
            "duration_sec": 120,
            "resolution": [1920],
            "fps": 30.0,
        }
        f = self._write_meta(tmp_path, data)
        assert is_valid_meta(f) is False

    def test_meta_not_dict(self, tmp_path):
        """Test meta.json that is a list instead of dict."""
        f = self._write_meta(tmp_path, [1, 2, 3])
        assert is_valid_meta(f) is False


class TestIsValidTranscript:
    """Test L1 + L2 + L3: transcript.json validation."""

    def _write_transcript(self, tmp_path, data):
        f = tmp_path / "transcript.json"
        f.write_text(json.dumps(data))
        return f

    def test_valid_transcript_no_duration_check(self, tmp_path):
        """Test valid transcript without L3 check."""
        data = [
            {"id": "seg_0001", "start": 0.0, "end": 5.0, "text": "Hello world"},
            {"id": "seg_0002", "start": 5.0, "end": 10.0, "text": "Second segment"},
        ]
        f = self._write_transcript(tmp_path, data)
        assert is_valid_transcript(f) is True

    def test_transcript_empty_list(self, tmp_path):
        """Test empty transcript list."""
        f = self._write_transcript(tmp_path, [])
        assert is_valid_transcript(f) is False

    def test_transcript_not_list(self, tmp_path):
        """Test transcript that is a dict instead of list."""
        f = self._write_transcript(tmp_path, {"key": "value"})
        assert is_valid_transcript(f) is False

    def test_transcript_missing_segment_keys(self, tmp_path):
        """Test transcript segment missing required keys."""
        data = [{"id": "seg_0001", "start": 0.0}]
        f = self._write_transcript(tmp_path, data)
        assert is_valid_transcript(f) is False

    def test_transcript_segment_not_dict(self, tmp_path):
        """Test transcript with non-dict segment."""
        data = ["not a dict"]
        f = self._write_transcript(tmp_path, data)
        assert is_valid_transcript(f) is False

    def test_transcript_duration_match(self, tmp_path):
        """Test L3: transcript duration matches expected."""
        data = [
            {"id": "seg_0001", "start": 0.0, "end": 49.0, "text": "Part 1"},
            {"id": "seg_0002", "start": 51.0, "end": 100.0, "text": "Part 2"},
        ]
        f = self._write_transcript(tmp_path, data)
        assert is_valid_transcript(f, expected_duration=98.0, tolerance_percent=5.0) is True

    def test_transcript_duration_mismatch(self, tmp_path):
        """Test L3: transcript duration does not match expected."""
        data = [
            {"id": "seg_0001", "start": 0.0, "end": 10.0, "text": "Short"},
        ]
        f = self._write_transcript(tmp_path, data)
        assert is_valid_transcript(f, expected_duration=100.0, tolerance_percent=2.0) is False

    def test_transcript_overlapping_segments(self, tmp_path):
        """Test L3: overlapping segments are merged correctly."""
        data = [
            {"id": "seg_0001", "start": 0.0, "end": 60.0, "text": "Part 1"},
            {"id": "seg_0002", "start": 30.0, "end": 100.0, "text": "Part 2"},
        ]
        f = self._write_transcript(tmp_path, data)
        assert is_valid_transcript(f, expected_duration=100.0, tolerance_percent=5.0) is True


class TestIsValidScoredSegments:
    """Test L1 + L2: scored_segments.json validation."""

    def _write_scored(self, tmp_path, data):
        f = tmp_path / "scored.json"
        f.write_text(json.dumps(data))
        return f

    def test_valid_scored_segments(self, tmp_path):
        """Test valid scored segments."""
        data = [
            {"id": "seg_0001", "start": 0.0, "end": 10.0, "score": 0.75, "tags": ["definition"]},
            {"id": "seg_0002", "start": 10.0, "end": 20.0, "score": 0.60, "tags": ["engaging"]},
        ]
        f = self._write_scored(tmp_path, data)
        assert is_valid_scored_segments(f) is True

    def test_scored_segments_empty(self, tmp_path):
        """Test empty scored segments."""
        f = self._write_scored(tmp_path, [])
        assert is_valid_scored_segments(f) is False

    def test_scored_segments_missing_keys(self, tmp_path):
        """Test scored segments missing required keys."""
        data = [{"id": "seg_0001", "start": 0.0}]
        f = self._write_scored(tmp_path, data)
        assert is_valid_scored_segments(f) is False

    def test_scored_segments_invalid_score_type(self, tmp_path):
        """Test scored segments with non-numeric score."""
        data = [{"id": "seg_0001", "start": 0.0, "end": 10.0, "score": "high"}]
        f = self._write_scored(tmp_path, data)
        assert is_valid_scored_segments(f) is False

    def test_scored_segments_not_list(self, tmp_path):
        """Test scored segments that is a dict."""
        f = self._write_scored(tmp_path, {"key": "value"})
        assert is_valid_scored_segments(f) is False


class TestIsValidCropParams:
    """Test L1 + L2: crop_params.json validation."""

    def _write_crop(self, tmp_path, data):
        f = tmp_path / "crop.json"
        f.write_text(json.dumps(data))
        return f

    def test_valid_crop_params(self, tmp_path):
        """Test valid crop params."""
        data = [
            {
                "clip_id": "clip_001",
                "start": 0.0,
                "end": 10.0,
                "frames": [{"time": 0.0, "x": 100, "y": 50, "w": 400, "h": 300, "conf": 0.95}],
                "fallback": "none",
                "fallback_ratio": 0.0,
            }
        ]
        f = self._write_crop(tmp_path, data)
        assert is_valid_crop_params(f) is True

    def test_crop_params_empty(self, tmp_path):
        """Test empty crop params."""
        f = self._write_crop(tmp_path, [])
        assert is_valid_crop_params(f) is False

    def test_crop_params_missing_keys(self, tmp_path):
        """Test crop params missing required keys."""
        data = [{"clip_id": "clip_001"}]
        f = self._write_crop(tmp_path, data)
        assert is_valid_crop_params(f) is False

    def test_crop_params_frames_not_list(self, tmp_path):
        """Test crop params with non-list frames."""
        data = [
            {
                "clip_id": "clip_001",
                "start": 0.0,
                "end": 10.0,
                "frames": "not a list",
                "fallback": "none",
            }
        ]
        f = self._write_crop(tmp_path, data)
        assert is_valid_crop_params(f) is False

    def test_crop_params_not_list(self, tmp_path):
        """Test crop params that is a dict."""
        f = self._write_crop(tmp_path, {"key": "value"})
        assert is_valid_crop_params(f) is False


class TestIsValidManifest:
    """Test L1 + L2: manifest.json validation."""

    def _write_manifest(self, tmp_path, data):
        f = tmp_path / "manifest.json"
        f.write_text(json.dumps(data))
        return f

    def test_valid_manifest(self, tmp_path):
        """Test valid manifest."""
        data = {
            "source": "test.mp4",
            "source_hash": "sha256:abc",
            "source_duration_sec": 120.0,
            "total_clips": 2,
            "clips": [
                {"clip_id": "clip_001", "score": 0.75},
                {"clip_id": "clip_002", "score": 0.60},
            ],
            "generated_at": "2026-05-12T10:00:00Z",
        }
        f = self._write_manifest(tmp_path, data)
        assert is_valid_manifest(f) is True

    def test_manifest_empty_clips(self, tmp_path):
        """Test manifest with zero clips (valid)."""
        data = {
            "source": "test.mp4",
            "total_clips": 0,
            "clips": [],
        }
        f = self._write_manifest(tmp_path, data)
        assert is_valid_manifest(f) is True

    def test_manifest_missing_keys(self, tmp_path):
        """Test manifest missing required keys."""
        data = {"source": "test.mp4"}
        f = self._write_manifest(tmp_path, data)
        assert is_valid_manifest(f) is False

    def test_manifest_invalid_total_clips(self, tmp_path):
        """Test manifest with negative total_clips."""
        data = {"total_clips": -1, "clips": []}
        f = self._write_manifest(tmp_path, data)
        assert is_valid_manifest(f) is False

    def test_manifest_clips_not_list(self, tmp_path):
        """Test manifest with non-list clips."""
        data = {"total_clips": 1, "clips": "not a list"}
        f = self._write_manifest(tmp_path, data)
        assert is_valid_manifest(f) is False

    def test_manifest_not_dict(self, tmp_path):
        """Test manifest that is a list."""
        f = self._write_manifest(tmp_path, [1, 2, 3])
        assert is_valid_manifest(f) is False


class TestGetExpectedDurationFromMeta:
    """Test helper function for extracting duration from meta.json."""

    def test_returns_duration_from_valid_meta(self, tmp_path):
        """Test that duration is extracted from valid meta."""
        f = tmp_path / "meta.json"
        data = {
            "source": "test.mp4",
            "duration_sec": 123.45,
            "resolution": [1920, 1080],
            "fps": 30.0,
        }
        f.write_text(json.dumps(data))
        assert get_expected_duration_from_meta(f) == 123.45

    def test_returns_zero_for_invalid_meta(self, tmp_path):
        """Test that 0.0 is returned for invalid meta."""
        f = tmp_path / "meta.json"
        f.write_text("not json")
        assert get_expected_duration_from_meta(f) == 0.0

    def test_returns_zero_for_missing_file(self, tmp_path):
        """Test that 0.0 is returned for missing file."""
        f = tmp_path / "missing.json"
        assert get_expected_duration_from_meta(f) == 0.0


class TestTranscriptReachesEnd:
    """The resume check must accept real lectures with pauses."""

    def test_lecture_with_pauses_is_valid(self):
        from src.artifact_validator import transcript_reaches_end

        # Speech covers ~62% of 25 min, but the transcript reaches the end.
        segments = [
            {"start": float(t), "end": float(t) + 30.0, "text": "..."} for t in range(0, 1500, 48)
        ]
        assert max(s["end"] for s in segments) >= 1470
        assert transcript_reaches_end(segments, 1500.0) is True

    def test_truncated_transcript_is_invalid(self):
        from src.artifact_validator import transcript_reaches_end

        # A 30-minute chunk failed: the transcript stops at 25 of 55 minutes.
        segments = [{"start": 0.0, "end": 1500.0, "text": "..."}]
        assert transcript_reaches_end(segments, 3300.0) is False

    def test_silent_outro_up_to_a_minute_is_fine(self):
        from src.artifact_validator import transcript_reaches_end

        segments = [{"start": 0.0, "end": 1445.0, "text": "..."}]
        assert transcript_reaches_end(segments, 1500.0) is True
        segments = [{"start": 0.0, "end": 1435.0, "text": "..."}]
        assert transcript_reaches_end(segments, 1500.0) is False

    def test_segments_past_the_video_are_invalid(self):
        from src.artifact_validator import transcript_reaches_end

        segments = [{"start": 0.0, "end": 200.0, "text": "..."}]
        assert transcript_reaches_end(segments, 180.0) is False

    def test_real_short_clip(self):
        """The 3-minute test clip on Arc: speech 172 s of 180 s, last segment ends at 179.98."""
        from src.artifact_validator import transcript_reaches_end

        segments = [
            {"start": 0.0, "end": 100.0, "text": "..."},
            {"start": 108.0, "end": 179.98, "text": "..."},
        ]
        assert transcript_reaches_end(segments, 180.0) is True
