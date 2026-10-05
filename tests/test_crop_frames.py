"""
Faster frame extraction for the face crop — GPU decoding (VAAPI),
shot cuts looked for at SCENE_FPS, frames saved at the detector's width.
"""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import CroppingConfig
from src.face_cropping import (
    SOFTWARE_DECODER,
    FrameDecoder,
    _frames_filter,
    detect_faces_in_frame,
    extract_frames,
    process_clip_segment,
    select_frame_decoder,
)

VAAPI = FrameDecoder("vaapi", "/dev/dri/renderD128")


class TestFilterGraph:
    def test_cpu_scene_analysis_rate_and_detector_width(self):
        graph, complex_graph = _frames_filter(SOFTWARE_DECODER, 1, 10.0, 640)
        assert complex_graph
        assert graph.startswith("[0:v]fps=12.5,scale=640:-2:flags=area,split=2[a][b]")
        assert "[a]fps=1[f]" in graph
        assert "[b]scale=320:-2,scdet=threshold=10.0[s]" in graph

    def test_vaapi_scales_on_the_gpu_and_downloads(self):
        graph, _ = _frames_filter(VAAPI, 1, 10.0, 640)
        assert graph.startswith("[0:v]fps=12.5,scale_vaapi=w=640:h=-2,hwdownload,format=nv12,split")

    def test_without_cut_detection(self):
        assert _frames_filter(SOFTWARE_DECODER, 1, 0.0, None) == ("fps=1", False)
        assert _frames_filter(VAAPI, 1, 0.0, 640) == (
            "fps=1,scale_vaapi=w=640:h=-2,hwdownload,format=nv12",
            False,
        )

    def test_sampling_faster_than_scene_analysis(self):
        graph, _ = _frames_filter(SOFTWARE_DECODER, 25, 10.0, None)
        assert graph.startswith("[0:v]fps=25,split")


class TestExtractCommand:
    def test_vaapi_arguments_before_the_input(self, tmp_path):
        def fake_run(cmd, **kwargs):
            (tmp_path / "frame_0001.jpg").write_bytes(b"jpg")
            return MagicMock(returncode=0, stderr="")

        with patch("src.face_cropping.subprocess.run", side_effect=fake_run) as run:
            extract_frames(Path("v.mp4"), 10.0, 10.5, 1, tmp_path, 10.0, decoder=VAAPI)
        cmd = run.call_args.args[0]
        assert cmd[:3] == ["ffmpeg", "-nostdin", "-nostats"]
        assert cmd.index("-hwaccel") < cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-hwaccel_device") + 1] == "/dev/dri/renderD128"

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
    def test_frames_saved_at_the_given_width(self, tmp_path):
        import cv2

        video = tmp_path / "v.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=3:size=320x240:rate=25",
            ]
            + ["-c:v", "libx264", str(video)],
            check=True,
        )
        frames, _ = extract_frames(video, 0.0, 2.0, 1, tmp_path / "f", 10.0, frame_width=160)
        assert len(frames) == 3
        assert cv2.imread(str(frames[0][1])).shape[1] == 160


class TestFaceBoxScaling:
    def test_box_in_video_pixels(self, tmp_path):
        import cv2
        import numpy as np

        frame = tmp_path / "frame.jpg"
        cv2.imwrite(str(frame), np.zeros((360, 640, 3), dtype=np.uint8))
        model = MagicMock()
        model.detect.return_value = [(100.0, 50.0, 200.0, 150.0, 0.9)]

        face = detect_faces_in_frame(model, frame, 0.6, video_width=1920)
        assert (face["x"], face["y"], face["w"], face["h"]) == (300, 150, 300, 300)
        assert detect_faces_in_frame(model, frame, 0.6)["x"] == 100  # frame pixels


class TestSelectDecoder:
    def _patch(self, monkeypatch, nodes=(), error=None):
        probes = []

        def fake_probe(decoder, video):
            probes.append(decoder.device)
            return error

        monkeypatch.setattr("src.rendering._render_nodes", lambda: list(nodes))
        monkeypatch.setattr("src.face_cropping.probe_frame_decoder", fake_probe)
        return probes

    def test_software_setting_never_probes(self, monkeypatch):
        probes = self._patch(monkeypatch, nodes=["/dev/dri/renderD128"])
        config = CroppingConfig(frame_decoder="software")
        assert select_frame_decoder(config, Path("v.mp4")) == SOFTWARE_DECODER
        assert probes == []

    def test_no_gpu(self, monkeypatch):
        self._patch(monkeypatch)
        assert select_frame_decoder(CroppingConfig(), Path("v.mp4")) == SOFTWARE_DECODER

    def test_vaapi_when_the_test_decode_works(self, monkeypatch):
        self._patch(monkeypatch, nodes=["/dev/dri/renderD128"])
        assert select_frame_decoder(CroppingConfig(), Path("v.mp4")) == VAAPI

    def test_cpu_when_the_test_decode_fails(self, monkeypatch):
        probes = self._patch(
            monkeypatch, nodes=["/dev/dri/renderD128", "/dev/dri/renderD129"], error="x"
        )
        assert select_frame_decoder(CroppingConfig(), Path("v.mp4")) == SOFTWARE_DECODER
        assert probes == ["/dev/dri/renderD128", "/dev/dri/renderD129"]


class TestProcessClip:
    def _run(self, tmp_path, decoder, results, width=1920):
        detection = {"x": 900, "y": 100, "w": 150, "h": 210, "conf": 0.9}
        with (
            patch("src.face_cropping.extract_frames", side_effect=results) as ext,
            patch("src.face_cropping.detect_faces_in_frame", return_value=detection) as det,
        ):
            process_clip_segment(
                tmp_path / "video.mp4",
                {"id": "clip_001", "start": 0.0, "end": 2.0},
                {"width": width, "height": 1080},
                MagicMock(),
                CroppingConfig(),
                tmp_path,
                decoder,
            )
        return ext, det

    def test_gpu_failure_retries_on_the_cpu(self, tmp_path):
        sampled = [(float(k), tmp_path / f"frame_{k:04d}.jpg") for k in range(3)]
        ext, _ = self._run(tmp_path, VAAPI, [([], []), (sampled, [])])
        assert [c.kwargs.get("decoder", SOFTWARE_DECODER) for c in ext.call_args_list] == [
            VAAPI,
            SOFTWARE_DECODER,
        ]

    def test_detector_width_and_video_width_passed(self, tmp_path):
        sampled = [(0.0, tmp_path / "frame_0001.jpg")]
        ext, det = self._run(tmp_path, SOFTWARE_DECODER, [(sampled, [])])
        assert ext.call_args.kwargs["frame_width"] == 640
        assert det.call_args.kwargs["video_width"] == 1920

    def test_small_video_is_not_upscaled(self, tmp_path):
        sampled = [(0.0, tmp_path / "frame_0001.jpg")]
        ext, _ = self._run(tmp_path, SOFTWARE_DECODER, [(sampled, [])], width=480)
        assert ext.call_args.kwargs["frame_width"] is None
