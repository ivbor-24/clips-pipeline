"""
GPU video encoders and per-clip resume in rendering.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.notices import read_notices
from src.rendering import (
    RenderingError,
    VideoEncoder,
    build_render_command,
    probe_encoder,
    render_and_export,
    select_video_encoder,
)

INFO = {"width": 1920, "height": 1080, "fps": "25/1"}


def _config(tmp_path, **rendering):
    config = Config()
    config.work_dir = tmp_path
    for key, value in rendering.items():
        setattr(config.rendering, key, value)
    return config


class TestEncoderArguments:
    def test_software_keeps_libx264_crf(self):
        args = VideoEncoder("libx264").output_args("medium", "25/1")
        assert args == ["-c:v", "libx264", "-preset", "medium", "-crf", "23"]

    def test_qsv_gets_frame_rate_and_its_own_preset_names(self):
        args = VideoEncoder("h264_qsv", "qsv").output_args("ultrafast", "25/1")
        assert args[:4] == ["-c:v", "h264_qsv", "-preset", "veryfast"]
        assert "-global_quality" in args
        assert args[-2:] == ["-r", "25/1"]

    @pytest.mark.parametrize("fps", [None, "0/0", "abc"])
    def test_qsv_without_usable_frame_rate(self, fps):
        assert "-r" not in VideoEncoder("h264_qsv", "qsv").output_args("medium", fps)

    def test_vaapi_opens_device_and_uploads_frames(self):
        encoder = VideoEncoder("h264_vaapi", "vaapi", "/dev/dri/renderD129")
        assert encoder.input_args() == ["-vaapi_device", "/dev/dri/renderD129"]
        assert encoder.filter_suffix() == "format=nv12,hwupload"
        assert encoder.output_args("medium")[:2] == ["-c:v", "h264_vaapi"]

    def test_nvenc_constant_quality(self):
        args = VideoEncoder("h264_nvenc", "nvenc").output_args("medium")
        assert args[:2] == ["-c:v", "h264_nvenc"]
        assert "-cq" in args and args[args.index("-b:v") + 1] == "0"


class TestBuildRenderCommand:
    def _cmd(self, tmp_path, encoder, segments):
        subtitle = tmp_path / "clip_001.ass"
        subtitle.write_text("[Script Info]\n", encoding="utf-8")
        clip = {"start": 10.0, "end": 70.0, "segments": segments, "crop_frames": []}
        return build_render_command(
            tmp_path / "prep.mp4",
            clip,
            subtitle,
            tmp_path / "clip_001.part.mp4",
            _config(tmp_path),
            encoder,
            INFO,
            subtitle_format="ass",
        )

    def test_vaapi_single_window_uploads_after_subtitles(self, tmp_path):
        encoder = VideoEncoder("h264_vaapi", "vaapi", "/dev/dri/renderD128")
        cmd = self._cmd(tmp_path, encoder, [])
        assert cmd.index("-vaapi_device") < cmd.index("-i")
        vf = cmd[cmd.index("-vf") + 1]
        assert vf.endswith(",format=nv12,hwupload")
        assert vf.index("ass=") < vf.index("hwupload")

    def test_vaapi_multi_segment_maps_uploaded_stream(self, tmp_path):
        encoder = VideoEncoder("h264_vaapi", "vaapi", "/dev/dri/renderD128")
        segments = [
            {"start": 10.0, "end": 40.0, "cx": 1300.0, "cy": 400.0, "has_face": True},
            {"start": 40.0, "end": 70.0, "cx": 800.0, "cy": 300.0, "has_face": True},
        ]
        cmd = self._cmd(tmp_path, encoder, segments)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "[vout]format=nv12,hwupload[venc]" in graph
        maps = [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "-map"]
        assert maps == ["[venc]", "[aout]"]

    def test_software_has_no_upload(self, tmp_path):
        cmd = self._cmd(tmp_path, VideoEncoder("libx264"), [])
        assert "hwupload" not in cmd[cmd.index("-vf") + 1]
        assert "-vaapi_device" not in cmd
        assert cmd[-1].endswith("clip_001.part.mp4")


class TestProbeEncoder:
    def test_working_encoder(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr("src.rendering.subprocess.run", fake_run)
        assert probe_encoder(VideoEncoder("h264_qsv", "qsv"), 1080, 1920) is None
        assert calls[0][-3:] == ["-f", "null", "-"]
        assert "color=c=black:s=1080x1920:r=25:d=0.5" in calls[0]

    def test_failing_encoder_returns_last_lines(self, monkeypatch):
        stderr = "line 1\nline 2\nline 3\nError while opening encoder\n"
        monkeypatch.setattr(
            "src.rendering.subprocess.run",
            lambda cmd, **kw: MagicMock(returncode=1, stderr=stderr),
        )
        error = probe_encoder(VideoEncoder("h264_vaapi", "vaapi", "/dev/dri/renderD128"), 8, 8)
        assert error == "line 2 | line 3 | Error while opening encoder"


class TestSelectVideoEncoder:
    def _patch(self, monkeypatch, backend="cpu", nodes=(), works=()):
        tried = []

        def fake_probe(encoder, width, height):
            tried.append((encoder.name, encoder.device))
            return None if encoder.name in works else f"{encoder.name} failed"

        monkeypatch.setattr("src.rendering._gpu_backend", lambda config: backend)
        monkeypatch.setattr("src.rendering._render_nodes", lambda: list(nodes))
        monkeypatch.setattr("src.rendering.probe_encoder", fake_probe)
        return tried

    def test_software_never_probes(self, tmp_path, monkeypatch):
        tried = self._patch(monkeypatch, nodes=["/dev/dri/renderD128"], works={"h264_qsv"})
        encoder = select_video_encoder(_config(tmp_path, video_encoder="software"))
        assert encoder == VideoEncoder("libx264")
        assert tried == []

    def test_auto_without_gpu_is_libx264_without_notice(self, tmp_path, monkeypatch):
        tried = self._patch(monkeypatch)
        assert select_video_encoder(_config(tmp_path)).name == "libx264"
        assert tried == []
        assert read_notices(tmp_path) == []

    def test_auto_intel_falls_through_to_vaapi(self, tmp_path, monkeypatch):
        tried = self._patch(
            monkeypatch,
            backend="openvino",
            nodes=["/dev/dri/renderD128", "/dev/dri/renderD129"],
            works={"h264_vaapi"},
        )
        encoder = select_video_encoder(_config(tmp_path))
        assert encoder == VideoEncoder("h264_vaapi", "vaapi", "/dev/dri/renderD128")
        assert tried == [("h264_qsv", None), ("h264_vaapi", "/dev/dri/renderD128")]

    def test_auto_nvidia_tries_nvenc_first(self, tmp_path, monkeypatch):
        tried = self._patch(monkeypatch, backend="cuda", works={"h264_nvenc"})
        assert select_video_encoder(_config(tmp_path)).kind == "nvenc"
        assert tried == [("h264_nvenc", None)]

    def test_hevc_codec_uses_hevc_encoders(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, nodes=["/dev/dri/renderD128"], works={"hevc_qsv"})
        assert select_video_encoder(_config(tmp_path, video_codec="h265")).name == "hevc_qsv"

    def test_auto_gpu_backend_falls_back_with_notice(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, backend="openvino", nodes=["/dev/dri/renderD128"])
        assert select_video_encoder(_config(tmp_path)).name == "libx264"
        notices = read_notices(tmp_path)
        assert [n["code"] for n in notices] == ["video_encoded_on_cpu"]
        assert notices[0]["stage"] == "rendering"
        assert "h264_qsv" in notices[0]["errors"]

    def test_explicit_encoder_that_fails_stops_the_job(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, nodes=["/dev/dri/renderD128"])
        with pytest.raises(RenderingError, match="'qsv' does not work: h264_qsv: h264_qsv failed"):
            select_video_encoder(_config(tmp_path, video_encoder="qsv"))

    def test_explicit_vaapi_without_render_node(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        with pytest.raises(RenderingError, match="no GPU render node"):
            select_video_encoder(_config(tmp_path, video_encoder="vaapi"))

    def test_codec_without_gpu_encoder(self, tmp_path, monkeypatch):
        tried = self._patch(monkeypatch, nodes=["/dev/dri/renderD128"], works={"h264_qsv"})
        assert select_video_encoder(_config(tmp_path, video_codec="copy")).name == "copy"
        assert tried == []


class FakeFfmpeg:
    """ffprobe sees a valid clip; ffmpeg writes its output unless told to fail."""

    def __init__(self, fail_encoders=()):
        self.renders = []
        self.fail_encoders = set(fail_encoders)

    def __call__(self, cmd, **kwargs):
        if cmd[0] == "ffprobe":
            probe = {
                "streams": [
                    {"codec_type": "video", "width": 1920, "height": 1080, "r_frame_rate": "25/1"},
                    {"codec_type": "audio"},
                ],
                "format": {"duration": "30.0"},
            }
            return MagicMock(returncode=0, stdout=json.dumps(probe), stderr="")
        encoder = cmd[cmd.index("-c:v") + 1]
        self.renders.append((Path(cmd[-1]).name, encoder))
        if encoder in self.fail_encoders:
            import subprocess

            raise subprocess.CalledProcessError(1, cmd, stderr="encoder failed")
        Path(cmd[-1]).write_bytes(b"mp4")
        return MagicMock(returncode=0, stdout="", stderr="")


def _job(tmp_path, clips=2):
    artifacts = tmp_path / "artifacts"
    (artifacts / "video").mkdir(parents=True)
    (artifacts / "video" / "prep.mp4").write_bytes(b"video")
    segments = [
        {"start": 30.0 * i, "end": 30.0 * i + 25, "score": 0.8, "title": f"T{i}", "tags": []}
        for i in range(clips)
    ]
    (artifacts / "scored_segments.json").write_text(json.dumps(segments))
    (artifacts / "crop_params.json").write_text(
        json.dumps([{"start": s["start"], "end": s["end"], "frames": []} for s in segments])
    )
    (artifacts / "transcript.json").write_text(
        json.dumps(
            [{"start": 30.0 * i + 1, "end": 30.0 * i + 5, "text": "привет"} for i in range(clips)]
        )
    )
    (artifacts / "meta.json").write_text(
        json.dumps({"source_hash": "sha256:x", "duration_sec": 100})
    )
    return tmp_path / "output" / "clips"


class TestPerClipResume:
    def test_second_run_reuses_every_clip(self, tmp_path, monkeypatch):
        clips_dir = _job(tmp_path)
        config = _config(tmp_path)
        first = FakeFfmpeg()
        monkeypatch.setattr("src.rendering.subprocess.run", first)
        render_and_export(config)
        assert [name for name, _ in first.renders] == ["clip_001.part.mp4", "clip_002.part.mp4"]
        meta = json.loads((clips_dir / "clip_001.meta.json").read_text())
        assert meta["render_key"] and meta["encoder"] == "libx264"

        second = FakeFfmpeg()
        monkeypatch.setattr("src.rendering.subprocess.run", second)
        monkeypatch.setattr(
            "src.rendering.select_video_encoder",
            lambda config: pytest.fail("nothing to render: the encoder is not tested"),
        )
        result = render_and_export(config)
        assert second.renders == []
        assert [m["clip_id"] for m in result] == ["clip_001", "clip_002"]
        manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
        assert manifest["total_clips"] == 2

    def test_force_renders_everything_again(self, tmp_path, monkeypatch):
        _job(tmp_path)
        monkeypatch.setattr("src.rendering.subprocess.run", FakeFfmpeg())
        render_and_export(_config(tmp_path))
        again = FakeFfmpeg()
        monkeypatch.setattr("src.rendering.subprocess.run", again)
        render_and_export(_config(tmp_path), reuse=False)
        assert len(again.renders) == 2

    def test_changed_clip_is_rendered_again(self, tmp_path, monkeypatch):
        _job(tmp_path)
        monkeypatch.setattr("src.rendering.subprocess.run", FakeFfmpeg())
        render_and_export(_config(tmp_path))

        segments_path = tmp_path / "artifacts" / "scored_segments.json"
        segments = json.loads(segments_path.read_text())
        segments[1]["title"] = "new title"  # metadata only: video reused
        segments[0]["end"] = 27.0  # clip 1 changes
        segments_path.write_text(json.dumps(segments))
        crop_path = tmp_path / "artifacts" / "crop_params.json"
        crops = json.loads(crop_path.read_text())
        crops[0]["end"] = 27.0
        crop_path.write_text(json.dumps(crops))

        again = FakeFfmpeg()
        monkeypatch.setattr("src.rendering.subprocess.run", again)
        result = render_and_export(_config(tmp_path))
        assert [name for name, _ in again.renders] == ["clip_001.part.mp4"]
        assert result[1]["title"] == "new title"

    def test_subtitle_style_change_rerenders(self, tmp_path, monkeypatch):
        _job(tmp_path, clips=1)
        monkeypatch.setattr("src.rendering.subprocess.run", FakeFfmpeg())
        render_and_export(_config(tmp_path))
        again = FakeFfmpeg()
        monkeypatch.setattr("src.rendering.subprocess.run", again)
        render_and_export(_config(tmp_path, subtitle_style="minimal"))
        assert len(again.renders) == 1

    def test_encoder_or_sidecar_format_change_does_not_rerender(self, tmp_path, monkeypatch):
        _job(tmp_path, clips=1)
        monkeypatch.setattr("src.rendering.subprocess.run", FakeFfmpeg())
        render_and_export(_config(tmp_path))
        again = FakeFfmpeg()
        monkeypatch.setattr("src.rendering.subprocess.run", again)
        render_and_export(_config(tmp_path, video_encoder="software", subtitle_format="ass"))
        assert again.renders == []

    def test_clip_cut_short_is_rendered_again(self, tmp_path, monkeypatch):
        clips_dir = _job(tmp_path)
        monkeypatch.setattr("src.rendering.subprocess.run", FakeFfmpeg())
        render_and_export(_config(tmp_path))
        # A stop during clip 2: ffmpeg left a partial file, no metadata yet.
        (clips_dir / "clip_002.meta.json").unlink()
        (clips_dir / "clip_002.part.mp4").write_bytes(b"half")

        again = FakeFfmpeg()
        monkeypatch.setattr("src.rendering.subprocess.run", again)
        render_and_export(_config(tmp_path))
        assert [name for name, _ in again.renders] == ["clip_002.part.mp4"]
        assert not (clips_dir / "clip_002.part.mp4").exists()
        assert (clips_dir / "clip_002.mp4").read_bytes() == b"mp4"

    def test_clips_beyond_the_new_count_are_removed(self, tmp_path, monkeypatch):
        clips_dir = _job(tmp_path, clips=3)
        monkeypatch.setattr("src.rendering.subprocess.run", FakeFfmpeg())
        render_and_export(_config(tmp_path))
        assert (clips_dir / "clip_003.mp4").exists()

        segments_path = tmp_path / "artifacts" / "scored_segments.json"
        segments_path.write_text(json.dumps(json.loads(segments_path.read_text())[:2]))
        render_and_export(_config(tmp_path))
        assert sorted(p.name for p in clips_dir.glob("clip_003*")) == []
        assert (clips_dir / "clip_002.mp4").exists()


class TestGpuEncoderFailsMidJob:
    def test_falls_back_to_libx264_for_the_rest(self, tmp_path, monkeypatch):
        clips_dir = _job(tmp_path)
        fake = FakeFfmpeg(fail_encoders={"h264_qsv"})
        monkeypatch.setattr("src.rendering.subprocess.run", fake)
        monkeypatch.setattr(
            "src.rendering.select_video_encoder",
            lambda config: VideoEncoder("h264_qsv", "qsv"),
        )
        result = render_and_export(_config(tmp_path))

        assert [encoder for _, encoder in fake.renders] == ["h264_qsv", "libx264", "libx264"]
        assert len(result) == 2
        assert json.loads((clips_dir / "clip_002.meta.json").read_text())["encoder"] == "libx264"
        codes = [n["code"] for n in read_notices(tmp_path)]
        assert codes == ["video_encoded_on_cpu"]

    def test_explicit_gpu_encoder_does_not_fall_back(self, tmp_path, monkeypatch):
        _job(tmp_path, clips=1)
        fake = FakeFfmpeg(fail_encoders={"h264_qsv"})
        monkeypatch.setattr("src.rendering.subprocess.run", fake)
        monkeypatch.setattr(
            "src.rendering.select_video_encoder",
            lambda config: VideoEncoder("h264_qsv", "qsv"),
        )
        result = render_and_export(_config(tmp_path, video_encoder="qsv"))
        assert [encoder for _, encoder in fake.renders] == ["h264_qsv"]
        assert result == []
