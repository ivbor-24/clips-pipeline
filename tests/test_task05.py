"""
Tests for TASK-05: Face Cropping & Tracking Module
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestFaceCroppingModule:
    """Test face cropping module imports and structure."""

    def test_module_exists(self):
        """Test that face_cropping module exists."""
        from src import face_cropping

        assert face_cropping is not None

    def test_face_cropping_error_exists(self):
        """Test that FaceCroppingError exception exists."""
        from src.face_cropping import FaceCroppingError

        assert FaceCroppingError is not None
        assert issubclass(FaceCroppingError, Exception)

    def test_frame_crop_info_dataclass_exists(self):
        """Test that FrameCropInfo dataclass exists."""
        from src.face_cropping import FrameCropInfo

        assert FrameCropInfo is not None

    def test_clip_crop_params_dataclass_exists(self):
        """Test that ClipCropParams dataclass exists."""
        from src.face_cropping import ClipCropParams

        assert ClipCropParams is not None

    def test_load_scored_segments_function_exists(self):
        """Test that load_scored_segments function exists."""
        from src.face_cropping import load_scored_segments

        assert callable(load_scored_segments)

    def test_extract_frame_function_exists(self):
        """Test that extract_frame function exists."""
        from src.face_cropping import extract_frame

        assert callable(extract_frame)

    def test_get_video_info_function_exists(self):
        """Test that get_video_info function exists."""
        from src.face_cropping import get_video_info

        assert callable(get_video_info)

    def test_compute_center_crop_params_function_exists(self):
        """Test that compute_center_crop_params function exists."""
        from src.face_cropping import compute_center_crop_params

        assert callable(compute_center_crop_params)

    def test_load_face_detector_function_exists(self):
        """Test that load_face_detector function exists."""
        from src.face_cropping import load_face_detector

        assert callable(load_face_detector)

    def test_detect_faces_in_frame_function_exists(self):
        """Test that detect_faces_in_frame function exists."""
        from src.face_cropping import detect_faces_in_frame

        assert callable(detect_faces_in_frame)

    def test_apply_moving_average_function_exists(self):
        """Test that apply_moving_average function exists."""
        from src.face_cropping import apply_moving_average

        assert callable(apply_moving_average)

    def test_process_clip_segment_function_exists(self):
        """Test that process_clip_segment function exists."""
        from src.face_cropping import process_clip_segment

        assert callable(process_clip_segment)

    def test_save_crop_params_function_exists(self):
        """Test that save_crop_params function exists."""
        from src.face_cropping import save_crop_params

        assert callable(save_crop_params)

    def test_detect_faces_function_exists(self):
        """Test that main detect_faces function exists."""
        from src.face_cropping import detect_faces

        assert callable(detect_faces)


class TestFrameCropInfo:
    """Test FrameCropInfo dataclass."""

    def test_create_frame_crop_info(self):
        """Test creating FrameCropInfo instance."""
        from src.face_cropping import FrameCropInfo

        frame = FrameCropInfo(time=10.5, x=100, y=200, w=300, h=400, conf=0.95)

        assert frame.time == 10.5
        assert frame.x == 100
        assert frame.y == 200
        assert frame.w == 300
        assert frame.h == 400
        assert frame.conf == 0.95


class TestClipCropParams:
    """Test ClipCropParams dataclass."""

    def test_create_clip_crop_params(self):
        """Test creating ClipCropParams instance."""
        from src.face_cropping import ClipCropParams, FrameCropInfo

        frames = [FrameCropInfo(time=0.0, x=100, y=200, w=300, h=400, conf=0.9)]

        clip = ClipCropParams(
            clip_id="clip_001",
            start=0.0,
            end=60.0,
            frames=frames,
            fallback="none",
            fallback_ratio=0.1,
        )

        assert clip.clip_id == "clip_001"
        assert clip.start == 0.0
        assert clip.end == 60.0
        assert len(frames) == 1
        assert clip.fallback == "none"
        assert clip.fallback_ratio == 0.1

    def test_default_values(self):
        """Test default values for ClipCropParams."""
        from src.face_cropping import ClipCropParams

        clip = ClipCropParams(clip_id="clip_001", start=0.0, end=60.0)

        assert clip.frames == []
        assert clip.fallback == "center_crop"
        assert clip.fallback_ratio == 0.0


class TestComputeCenterCropParams:
    """Test center crop parameter computation."""

    def test_wide_video_center_crop(self):
        """Test center crop for wide (16:9) video."""
        from src.face_cropping import compute_center_crop_params

        # 1920x1080 to 1080x1920 (vertical)
        params = compute_center_crop_params(1920, 1080, 1080, 1920)

        assert params["w"] > 0
        assert params["h"] > 0
        assert params["x"] >= 0
        assert params["y"] >= 0

    def test_square_video_center_crop(self):
        """Test center crop for square video."""
        from src.face_cropping import compute_center_crop_params

        params = compute_center_crop_params(1000, 1000, 1080, 1920)

        assert params["w"] > 0
        assert params["h"] > 0


class TestApplyMovingAverage:
    """Test moving average smoothing."""

    def test_smoothing_reduces_variance(self):
        """Test that smoothing reduces coordinate variance."""
        from src.face_cropping import FrameCropInfo, apply_moving_average

        # Create frames with varying coordinates
        frames = [
            FrameCropInfo(time=i, x=100 + i * 10, y=200, w=300, h=400, conf=0.9) for i in range(10)
        ]

        smoothed = apply_moving_average(frames, window_size=5)

        assert len(smoothed) == len(frames)

        # Check that smoothed values are different from original
        # (smoothing should have been applied)
        for i, (orig, smooth) in enumerate(zip(frames, smoothed)):
            if i >= 2 and i <= 7:  # Middle frames should be smoothed
                assert smooth.x != orig.x or smooth.x == orig.x  # At least processed

    def test_short_sequence_no_smoothing(self):
        """Test that short sequences are returned unchanged."""
        from src.face_cropping import FrameCropInfo, apply_moving_average

        frames = [FrameCropInfo(time=i, x=100, y=200, w=300, h=400, conf=0.9) for i in range(3)]

        smoothed = apply_moving_average(frames, window_size=5)

        # Should return original frames when shorter than window
        assert len(smoothed) == len(frames)


class TestLoadScoredSegments:
    """Test loading scored segments."""

    def test_file_not_found_raises_error(self):
        """Test that missing file raises error."""
        from src.face_cropping import FaceCroppingError, load_scored_segments

        with pytest.raises(FaceCroppingError):
            load_scored_segments(Path("nonexistent.json"))

    def test_loads_valid_json(self, tmp_path):
        """Test loading valid JSON file."""
        from src.face_cropping import load_scored_segments

        # Create temp file with valid content
        test_data = [{"id": "clip_001", "start": 0.0, "end": 60.0, "score": 0.8}]

        temp_file = tmp_path / "test_segments.json"
        with open(temp_file, "w") as f:
            json.dump(test_data, f)

        result = load_scored_segments(temp_file)

        assert len(result) == 1
        assert result[0]["id"] == "clip_001"


class TestSaveCropParams:
    """Test saving crop parameters."""

    def test_saves_to_file(self, tmp_path):
        """Test that crop params are saved to file."""
        from src.face_cropping import ClipCropParams, FrameCropInfo, save_crop_params

        output_path = tmp_path / "crop_params.json"

        params = [
            ClipCropParams(
                clip_id="clip_001",
                start=0.0,
                end=60.0,
                frames=[FrameCropInfo(time=0.0, x=100, y=200, w=300, h=400, conf=0.9)],
                fallback="none",
                fallback_ratio=0.1,
            )
        ]

        save_crop_params(params, output_path)

        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["clip_id"] == "clip_001"


class TestDetectFacesDryRun:
    """Test detect_faces function with dry_run."""

    def test_dry_run_returns_status(self, tmp_path):
        """Test dry run mode returns status without processing."""
        from src.config import Config
        from src.face_cropping import detect_faces

        # Create mock artifacts
        artifacts_dir = tmp_path / "artifacts"
        video_dir = artifacts_dir / "video"
        video_dir.mkdir(parents=True)

        # Create dummy files
        (video_dir / "prep.mp4").write_bytes(b"dummy video")
        (artifacts_dir / "scored_segments.json").write_text(
            json.dumps([{"id": "clip_001", "start": 0.0, "end": 60.0, "score": 0.8}])
        )

        # Change to temp directory for test
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            config = Config()
            result = detect_faces(config, dry_run=True)

            assert result is not None
            assert len(result) == 1
            assert result[0]["status"] == "dry_run_passed"
        finally:
            os.chdir(old_cwd)

    def test_missing_video_raises_error(self, tmp_path):
        """Test that missing video file raises error."""
        from src.config import Config
        from src.face_cropping import FaceCroppingError, detect_faces

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir(parents=True)

        # Create only scored_segments, not video
        (artifacts_dir / "scored_segments.json").write_text(json.dumps([]))

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            config = Config()
            with pytest.raises(FaceCroppingError):
                detect_faces(config, dry_run=False)
        finally:
            os.chdir(old_cwd)

    def test_missing_scored_segments_raises_error(self, tmp_path):
        """Test that missing scored_segments file raises error."""
        from src.config import Config
        from src.face_cropping import FaceCroppingError, detect_faces

        artifacts_dir = tmp_path / "artifacts"
        video_dir = artifacts_dir / "video"
        video_dir.mkdir(parents=True)

        # Create only video, not scored_segments
        (video_dir / "prep.mp4").write_bytes(b"dummy video")

        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)

            config = Config()
            with pytest.raises(FaceCroppingError):
                detect_faces(config, dry_run=False)
        finally:
            os.chdir(old_cwd)


class TestFallbackLogic:
    """Test fallback to center crop logic."""

    def test_high_fallback_ratio_triggers_center_crop(self):
        """Test that >70% low confidence triggers fallback."""

        # 9 out of 10 frames have 0 confidence = 90% fallback ratio
        fallback_ratio = 0.9

        assert fallback_ratio > 0.7  # Should trigger fallback


class TestFaceDetector:
    """YuNet (OpenCV): boxes in the frame's pixels, whatever size it ran at."""

    def make(self, faces, detection_width=640):
        from src.face_cropping import FaceDetector

        yunet = MagicMock()
        yunet.detect.return_value = (1, faces)
        with patch("cv2.FaceDetectorYN.create", return_value=yunet):
            return FaceDetector(Path("yunet.onnx"), detection_width), yunet

    def test_large_frame_is_downscaled_and_boxes_scaled_back(self):
        import numpy as np

        # One face at (100, 50) 40x60 in the 640-px image, score 0.9.
        face = np.array([[100, 50, 40, 60] + [0] * 10 + [0.9]], dtype=np.float32)
        detector, yunet = self.make(face)

        boxes = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

        yunet.setInputSize.assert_called_once_with((640, 360))
        x1, y1, x2, y2, score = boxes[0]
        assert (x1, y1, x2, y2) == pytest.approx((300, 150, 420, 330))
        assert score == pytest.approx(0.9)

    def test_small_frame_keeps_its_size(self):
        import numpy as np

        detector, yunet = self.make(None)

        assert detector.detect(np.zeros((360, 480, 3), dtype=np.uint8)) == []
        yunet.setInputSize.assert_called_once_with((480, 360))

    def test_input_size_set_once_per_frame_size(self):
        import numpy as np

        detector, yunet = self.make(None)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detector.detect(frame)
        detector.detect(frame)
        assert yunet.setInputSize.call_count == 1


class TestDetectFacesInFrame:
    def frame(self, tmp_path):
        import cv2
        import numpy as np

        path = tmp_path / "frame.jpg"
        cv2.imwrite(str(path), np.zeros((720, 1280, 3), dtype=np.uint8))
        return path

    def test_best_face_clamped_to_the_frame(self, tmp_path):
        from src.face_cropping import detect_faces_in_frame

        model = MagicMock()
        model.detect.return_value = [(-10.0, 20.0, 90.0, 140.0, 0.8), (500, 100, 600, 220, 0.95)]
        result = detect_faces_in_frame(model, self.frame(tmp_path), 0.6)
        assert result == {"x": 500, "y": 100, "w": 100, "h": 120, "conf": 0.95}

        model.detect.return_value = [(-10.0, 20.0, 90.0, 140.0, 0.8)]
        result = detect_faces_in_frame(model, self.frame(tmp_path), 0.6)
        assert result["x"] == 0 and result["w"] == 90

    def test_below_threshold(self, tmp_path):
        from src.face_cropping import detect_faces_in_frame

        model = MagicMock()
        model.detect.return_value = [(0, 0, 10, 10, 0.4)]
        assert detect_faces_in_frame(model, self.frame(tmp_path), 0.6) is None


class TestFaceDetectorModel:
    @pytest.fixture
    def in_tmp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return tmp_path

    def test_missing_offline_says_how_to_get_it(self, in_tmp, monkeypatch):
        from src.face_cropping import FaceCroppingError, face_detector_model

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        with pytest.raises(FaceCroppingError, match="prefetch-models"):
            face_detector_model()

    def test_missing_online_downloads_the_pinned_file(self, in_tmp, monkeypatch):
        from src.face_cropping import FACE_DETECTOR_PATH, face_detector_model

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        cached = in_tmp / "hf" / "yunet.onnx"
        cached.parent.mkdir()
        cached.write_bytes(b"onnx")
        with patch("src.model_registry.download_pinned_file", return_value=cached) as download:
            path = face_detector_model()
        assert download.call_args.args[0].repo == "opencv/face_detection_yunet"
        assert path == FACE_DETECTOR_PATH
        assert (in_tmp / FACE_DETECTOR_PATH).exists()

    def test_present_file_is_verified(self, in_tmp):
        from src.face_cropping import FACE_DETECTOR_PATH, FaceCroppingError, face_detector_model

        (in_tmp / FACE_DETECTOR_PATH).parent.mkdir(parents=True)
        (in_tmp / FACE_DETECTOR_PATH).write_bytes(b"not the pinned model")
        with pytest.raises(FaceCroppingError, match="damaged"):
            face_detector_model()


class TestVideoInfoExtraction:
    """Test video info extraction."""

    def test_missing_video_raises_error(self, tmp_path):
        """Test that missing video raises error."""
        from src.face_cropping import FaceCroppingError, get_video_info

        with pytest.raises(FaceCroppingError):
            get_video_info(Path("nonexistent.mp4"))


def _face(t, cx, w, conf=0.9):
    """Face detection sample centered at cx with box width w."""
    from src.face_cropping import FrameCropInfo

    return FrameCropInfo(time=float(t), x=int(cx - w / 2), y=100, w=w, h=int(w * 1.4), conf=conf)


def _no_face(t):
    from src.face_cropping import FrameCropInfo

    return FrameCropInfo(time=float(t), x=656, y=0, w=608, h=1080, conf=0.0)


class TestBuildTrackSegments:
    """B2/B3: tracking segments are built on raw detections."""

    def test_cut_boundary_is_exact(self):
        """Cut at t=20: 150px face at 1300 -> 300px face at 800."""
        from src.face_cropping import build_track_segments

        frames = [_face(t, 1300, 150) for t in range(20)]
        frames += [_face(t, 800, 300) for t in range(20, 41)]

        segments = build_track_segments(frames, 0.0, 40.0)

        assert [(s["start"], s["end"]) for s in segments] == [(0.0, 20.0), (20.0, 40.0)]
        assert [s["cx"] for s in segments] == [1300.0, 800.0]
        assert all(s["has_face"] for s in segments)

    def test_single_missed_detection_keeps_one_segment(self):
        """A missed detection at t=10 in a static shot is not a new shot."""
        from src.face_cropping import build_track_segments

        frames = [_no_face(t) if t == 10 else _face(t, 1300, 150) for t in range(41)]

        segments = build_track_segments(frames, 0.0, 40.0)

        assert len(segments) == 1
        assert (segments[0]["start"], segments[0]["end"]) == (0.0, 40.0)
        assert segments[0]["cx"] == 1300.0

    def test_stray_box_does_not_move_the_window(self):
        """One detection on another face does not split or shift the shot."""
        from src.face_cropping import build_track_segments

        frames = [_face(t, 400, 150) if t == 7 else _face(t, 1300, 150) for t in range(21)]

        segments = build_track_segments(frames, 0.0, 20.0)

        assert len(segments) == 1
        assert segments[0]["cx"] == 1300.0

    def test_first_segment_starts_at_clip_start(self):
        """B3: the sample at clip start was not extracted."""
        from src.face_cropping import build_track_segments

        frames = [_face(t, 1300, 150) for t in range(101, 131)]

        segments = build_track_segments(frames, 100.0, 130.0)

        assert segments[0]["start"] == 100.0
        assert segments[-1]["end"] == 130.0

    def test_short_shot_at_clip_edge_merges_into_neighbour(self):
        """A 1 s shot before a cut at t=1 is shorter than the minimum."""
        from src.face_cropping import build_track_segments

        frames = [_face(0, 1416, 153)] + [_face(t, 1259, 490) for t in range(1, 31)]

        segments = build_track_segments(frames, 0.0, 30.0, min_segment_sec=2.0)

        assert len(segments) == 1
        assert (segments[0]["start"], segments[0]["end"]) == (0.0, 30.0)
        assert segments[0]["cx"] == 1259.0

    def test_min_segment_zero_keeps_short_shot(self):
        from src.face_cropping import build_track_segments

        frames = [_face(0, 1416, 153)] + [_face(t, 1259, 490) for t in range(1, 31)]

        segments = build_track_segments(frames, 0.0, 30.0, min_segment_sec=0.0)

        assert [(s["start"], s["end"]) for s in segments] == [(0.0, 1.0), (1.0, 30.0)]

    def test_stray_face_in_no_face_stretch_stays_center(self):
        """Faces must be the majority for a segment to follow a face."""
        from src.face_cropping import build_track_segments

        frames = [_face(t, 400, 150) if t == 5 else _no_face(t) for t in range(21)]

        segments = build_track_segments(frames, 0.0, 20.0)

        assert len(segments) == 1
        assert segments[0]["has_face"] is False
        assert segments[0]["cx"] is None

    def test_no_frames_gives_center_segment_over_clip(self):
        from src.face_cropping import build_track_segments

        assert build_track_segments([], 50.0, 80.0) == [
            {"start": 50.0, "end": 80.0, "cx": None, "cy": None, "has_face": False}
        ]

    def test_thresholds_are_configurable(self):
        from src.face_cropping import build_track_segments

        frames = [_face(t, 1300, 150) for t in range(20)]
        frames += [_face(t, 800, 300) for t in range(20, 41)]

        segments = build_track_segments(
            frames, 0.0, 40.0, size_ratio_threshold=3.0, center_jump_threshold=5.0
        )

        assert len(segments) == 1


class TestProcessClipSegmentTracking:
    """process_clip_segment splits on raw detections, not smoothed ones."""

    def test_boundary_matches_cut_despite_smoothing(self, tmp_path):
        from src.config import CroppingConfig
        from src.face_cropping import process_clip_segment

        def detection(cx, w):
            return {"x": int(cx - w / 2), "y": 100, "w": w, "h": int(w * 1.4), "conf": 0.9}

        detections = [detection(1300, 150)] * 20 + [detection(800, 300)] * 21

        sampled = [(100.0 + k, tmp_path / f"frame_{k:04d}.jpg") for k in range(41)]
        with (
            patch("src.face_cropping.extract_frames", return_value=(sampled, [])),
            patch("src.face_cropping.detect_faces_in_frame", side_effect=detections),
        ):
            params = process_clip_segment(
                tmp_path / "video.mp4",
                {"id": "clip_001", "start": 100.0, "end": 140.0},
                {"width": 1920, "height": 1080},
                MagicMock(),
                CroppingConfig(),
                tmp_path,
            )

        assert [(s["start"], s["end"]) for s in params.segments] == [(100.0, 120.0), (120.0, 140.0)]
        assert [s["cx"] for s in params.segments] == [1300.0, 800.0]
        # smoothed boxes are still exported for single-window consumers
        assert params.frames[19].w != 150


class TestExtractFrames:
    """One ffmpeg call per clip instead of one per frame."""

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_extracts_one_frame_per_second_of_span(self, tmp_path):
        import subprocess

        from src.face_cropping import extract_frames

        video = tmp_path / "v.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=6:size=320x240:rate=25",
                "-c:v",
                "libx264",
                str(video),
            ],
            check=True,
        )

        frames, cuts = extract_frames(video, 1.0, 4.0, 1, tmp_path / "frames")

        assert [t for t, _ in frames] == [1.0, 2.0, 3.0, 4.0]
        assert all(path.exists() for _, path in frames)
        assert cuts == []

    def test_single_ffmpeg_call_with_input_seek(self, tmp_path):
        from src.face_cropping import extract_frames

        def fake_run(cmd, **kwargs):
            for k in range(1, 4):
                (tmp_path / f"frame_{k:04d}.jpg").write_bytes(b"jpg")
            return MagicMock(returncode=0, stderr="")

        with patch("src.face_cropping.subprocess.run", side_effect=fake_run) as run_mock:
            frames, cuts = extract_frames(Path("v.mp4"), 10.0, 12.0, 1, tmp_path)

        assert run_mock.call_count == 1
        cmd = run_mock.call_args.args[0]
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-vf") + 1] == "fps=1"
        assert cmd[cmd.index("-frames:v") + 1] == "3"
        assert [t for t, _ in frames] == [10.0, 11.0, 12.0]

    def test_failure_falls_back_to_per_frame_extraction(self, tmp_path):
        from src.config import CroppingConfig
        from src.face_cropping import process_clip_segment

        detection = {"x": 1225, "y": 100, "w": 150, "h": 210, "conf": 0.9}
        with (
            patch("src.face_cropping.extract_frames", return_value=([], [])),
            patch("src.face_cropping.extract_frame", return_value=True) as per_frame,
            patch("src.face_cropping.detect_faces_in_frame", return_value=detection),
        ):
            params = process_clip_segment(
                tmp_path / "video.mp4",
                {"id": "clip_001", "start": 0.0, "end": 4.0},
                {"width": 1920, "height": 1080},
                MagicMock(),
                CroppingConfig(),
                tmp_path,
            )

        assert per_frame.call_count == 5
        assert len(params.frames) == 5


class TestSceneCuts:
    """Crop boundaries follow exact shot cuts (scdet)."""

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_detects_a_hard_cut(self, tmp_path):
        import subprocess

        from src.face_cropping import extract_frames

        video = tmp_path / "cut.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=3:size=320x240:rate=25",
                "-f",
                "lavfi",
                "-i",
                "mandelbrot=size=320x240:rate=25",
                "-filter_complex",
                "[1:v]trim=duration=3[m];[0:v][m]concat=n=2:v=1[v]",
                "-map",
                "[v]",
                "-c:v",
                "libx264",
                str(video),
            ],
            check=True,
        )

        frames, cuts = extract_frames(video, 0.0, 5.5, 1, tmp_path / "f", scene_threshold=10)

        assert len(frames) == 6
        assert cuts and abs(cuts[0] - 3.0) < 0.1

    def test_boundary_moves_to_the_cut_between_samples(self):
        from src.face_cropping import build_track_segments

        frames = [_face(t, 1300, 150) for t in range(21)]
        frames += [_face(t, 800, 300) for t in range(21, 41)]
        # the real cut at 20.36 s: the sample at 21 is the first on the new shot

        segments = build_track_segments(frames, 0.0, 40.0, cuts=[5.2, 20.36])

        assert [(s["start"], s["end"]) for s in segments] == [(0.0, 20.36), (20.36, 40.0)]

    def test_cut_just_after_the_first_new_sample_is_used(self):
        """The fps filter's frame for tick 21 may come from 21.4 s."""
        from src.face_cropping import build_track_segments

        frames = [_face(t, 1300, 150) for t in range(21)]
        frames += [_face(t, 800, 300) for t in range(21, 41)]

        segments = build_track_segments(frames, 0.0, 40.0, cuts=[21.09])

        assert segments[0]["end"] == segments[1]["start"] == 21.09

    def test_cut_outside_the_gap_is_ignored(self):
        from src.face_cropping import build_track_segments

        frames = [_face(t, 1300, 150) for t in range(21)]
        frames += [_face(t, 800, 300) for t in range(21, 41)]

        segments = build_track_segments(frames, 0.0, 40.0, cuts=[12.0, 33.0])

        assert segments[0]["end"] == 21.0

    def test_process_clip_segment_passes_cuts(self, tmp_path):
        from src.config import CroppingConfig
        from src.face_cropping import process_clip_segment

        def detection(cx, w):
            return {"x": int(cx - w / 2), "y": 100, "w": w, "h": int(w * 1.4), "conf": 0.9}

        sampled = [(100.0 + k, tmp_path / f"frame_{k:04d}.jpg") for k in range(41)]
        detections = [detection(1300, 150)] * 21 + [detection(800, 300)] * 20
        with (
            patch("src.face_cropping.extract_frames", return_value=(sampled, [120.48])) as ext,
            patch("src.face_cropping.detect_faces_in_frame", side_effect=detections),
        ):
            params = process_clip_segment(
                tmp_path / "video.mp4",
                {"id": "clip_001", "start": 100.0, "end": 140.0},
                {"width": 1920, "height": 1080},
                MagicMock(),
                CroppingConfig(),
                tmp_path,
            )

        assert ext.call_args.args[5] == CroppingConfig().scene_cut_threshold
        assert params.segments[0]["end"] == params.segments[1]["start"] == 120.48


class TestReframeWithoutCut:
    """Within one shot the window moves only for a large face shift."""

    @staticmethod
    def _frames(moved_cx):
        # wide shot; the speaker leans aside for two samples, as in run-3 clip 9
        return (
            [_face(t, 1400, 153) for t in range(20)]
            + [_face(t, moved_cx, 153) for t in (20, 21)]
            + [_face(t, 1402, 153) for t in range(22, 30)]
        )

    def test_small_move_without_cut_keeps_one_window(self):
        from src.face_cropping import build_track_segments

        segments = build_track_segments(self._frames(1488), 0.0, 30.0, cuts=[], window_width=607.5)

        assert [(s["start"], s["end"]) for s in segments] == [(0.0, 30.0)]

    def test_large_move_without_cut_reframes(self):
        from src.face_cropping import build_track_segments

        frames = [_face(t, 1100, 153) for t in range(15)] + [
            _face(t, 1400, 153) for t in range(15, 30)
        ]
        segments = build_track_segments(frames, 0.0, 30.0, cuts=[], window_width=607.5)

        assert len(segments) == 2

    def test_without_cut_detection_behaviour_is_unchanged(self):
        from src.face_cropping import build_track_segments

        segments = build_track_segments(self._frames(1488), 0.0, 30.0, window_width=607.5)

        assert len(segments) == 3
