"""
Tests for src/cli/image.py — image build/save/load/verify/release utilities.

Strategy: all docker / git / shell interactions are mocked. Tests focus on:
- argument validation
- tag formatting and git SHA validation
- sha256 / manifest read/write semantics
- prune logic (keep-stable / keep-latest)
- error propagation when docker is missing or commands fail
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import image_tool as image_mod
from src.image_tool import (
    ImageError,
    _read_manifest,
    _write_manifest,
    build_image,
    build_image_tag,
    image_size_mb,
    list_images,
    load_image,
    prune_local,
    record_deployment,
    release_image,
    save_image,
    verify_image,
)

# ----------------------------- fixtures ----------------------------------- #


@pytest.fixture
def fake_completed_proc():
    """Factory for a CompletedProcess-like MagicMock with stdout/stderr/rc."""

    def _factory(rc=0, stdout="", stderr=""):
        m = MagicMock(spec=subprocess.CompletedProcess)
        m.returncode = rc
        m.stdout = stdout
        m.stderr = stderr
        return m

    return _factory


@pytest.fixture
def manifest_only(tmp_path, monkeypatch):
    """Redirect MANIFEST_PATH to a tmp file for the duration of one test."""
    target = tmp_path / "release-manifest.json"
    monkeypatch.setattr(image_mod, "MANIFEST_PATH", target)
    return target


# ----------------------------- build_image_tag ---------------------------- #


class TestBuildImageTag:
    def test_valid_backend_with_sha(self):
        tag = build_image_tag("openvino", git_sha="a1b2c3d")
        # Format: pipeline:<backend>-<sha>-<YYYYMMDD>
        # (note that "pipeline:openvino" sticks together when split on "-")
        assert tag.startswith("pipeline:openvino-a1b2c3d-")
        # Verify the structure end-to-end
        repo, rest = tag.split(":", 1)
        assert repo == "pipeline"
        parts = rest.split("-")
        assert parts[0] == "openvino"
        assert parts[1] == "a1b2c3d"
        assert len(parts[2]) == 8 and parts[2].isdigit()  # YYYYMMDD

    def test_invalid_backend(self):
        with pytest.raises(ImageError, match="invalid backend"):
            build_image_tag("vulkan")

    def test_invalid_sha(self):
        with pytest.raises(ImageError, match="invalid git sha"):
            build_image_tag("openvino", git_sha="not-a-sha")

    def test_unknown_sha_falls_back(self):
        tag = build_image_tag("cpu", git_sha="unknown")
        assert "cpu-unknown-" in tag

    def test_sha_must_be_full_or_short_hex(self):
        # 7-char short sha is the project convention
        tag = build_image_tag("cuda", git_sha="abcdef0")
        assert "cuda-abcdef0-" in tag


# ----------------------------- build_image -------------------------------- #


class TestBuildImage:
    def test_build_uses_correct_args(self, fake_completed_proc):
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(image_mod, "_run") as mock_run,
            patch.object(image_mod, "DOCKERFILE", Path("Dockerfile.backend")),
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_run.return_value = fake_completed_proc()
            tag = build_image("openvino")
        assert tag.startswith("pipeline:openvino-")
        # First call should be `docker build`; check the args
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "/usr/bin/docker"
        assert "build" in cmd
        assert "--build-arg" in cmd and "GPU_BACKEND=openvino" in cmd
        assert "-t" in cmd
        # latest tag is appended by default
        assert any(t.endswith("-latest") for t in cmd if t.startswith("pipeline:"))

    def test_build_no_cache(self, fake_completed_proc):
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(image_mod, "_run") as mock_run,
            patch("pathlib.Path.exists", return_value=True),
        ):
            mock_run.return_value = fake_completed_proc()
            build_image("cpu", no_cache=True)
        cmd = mock_run.call_args[0][0]
        assert "--no-cache" in cmd

    def test_build_no_docker_raises(self):
        with patch("shutil.which", return_value=None):
            with pytest.raises(ImageError, match="docker CLI not found"):
                build_image("cpu")

    def test_build_missing_dockerfile(self):
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("pathlib.Path.exists", return_value=False),
        ):
            with pytest.raises(ImageError, match="dockerfile not found"):
                build_image("cpu", dockerfile=Path("/nope/Dockerfile"))


# ----------------------------- image_size_mb ------------------------------ #


class TestImageSizeMb:
    def test_parses_gb(self):
        with patch.object(image_mod, "_run") as r:
            r.return_value = MagicMock(stdout="1.5GB", returncode=0)
            assert image_size_mb("pipeline:openvino-latest") == pytest.approx(1536.0)

    def test_parses_mb(self):
        with patch.object(image_mod, "_run") as r:
            r.return_value = MagicMock(stdout="850MB", returncode=0)
            assert image_size_mb("pipeline:cpu-latest") == pytest.approx(850.0)

    def test_empty_returns_zero(self):
        with patch.object(image_mod, "_run") as r:
            r.return_value = MagicMock(stdout="", returncode=0)
            assert image_size_mb("nope") == 0.0

    def test_unparseable_returns_zero(self):
        with patch.object(image_mod, "_run") as r:
            r.return_value = MagicMock(stdout="weird", returncode=0)
            assert image_size_mb("nope") == 0.0


# ----------------------------- save_image --------------------------------- #


class TestSaveImage:
    def test_save_produces_archive_and_sha(self, tmp_path, monkeypatch):
        """Run save_image with docker/gzip replaced by deterministic stubs."""
        # `docker save` writes deterministic bytes; `gzip -c` wraps them.
        monkeypatch.setattr(image_mod.shutil, "which", lambda _: "/bin/true")
        # Replace the internal Popen calls by patching subprocess.Popen so that
        # `docker save` emits a known payload and `gzip -c` echoes it back.
        original_popen = image_mod.subprocess.Popen

        def fake_popen(cmd, *args, **kwargs):
            cmd0 = cmd[0] if isinstance(cmd, list) else cmd
            stdout = kwargs.get("stdout", None)
            if cmd0 == "/bin/true":  # docker save stub
                payload = b"FAKE-IMAGE-BYTES"
                if stdout is not None and stdout is not subprocess.PIPE:
                    # `with open(out, 'wb')` — write directly
                    stdout.write(payload)
                else:
                    proc = original_popen(
                        ["python3", "-c", f"import sys; sys.stdout.buffer.write({payload!r})"],
                        stdout=kwargs.get("stdout"),
                        stderr=subprocess.PIPE,
                    )
                    return proc
            if cmd0 == "gzip":
                # echo stdin to stdout
                return original_popen(
                    ["cat"],
                    stdin=kwargs.get("stdin"),
                    stdout=kwargs.get("stdout"),
                    stderr=subprocess.PIPE,
                )
            return original_popen(cmd, *args, **kwargs)

        monkeypatch.setattr(image_mod.subprocess, "Popen", fake_popen)
        out = save_image("pipeline:openvino-latest", dist_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".gz"
        assert out.name.startswith("pipeline-openvino-latest")
        sha_file = out.with_suffix(out.suffix + ".sha256")
        assert sha_file.exists()
        # sha256 is consistent with the on-disk bytes
        expected = hashlib.sha256(out.read_bytes()).hexdigest()
        assert sha_file.read_text().split()[0] == expected

    def test_save_docker_failure(self, tmp_path, monkeypatch):
        """If `docker save` exits non-zero, ImageError must surface."""
        monkeypatch.setattr(image_mod.shutil, "which", lambda _: "/bin/true")
        original_popen = image_mod.subprocess.Popen

        def fake_popen(cmd, *args, **kwargs):
            cmd0 = cmd[0] if isinstance(cmd, list) else cmd
            if cmd0 == "/bin/true":  # docker save stub
                proc = MagicMock()
                proc.stdout = MagicMock()
                proc.wait.return_value = 1
                proc.communicate.return_value = (b"", b"boom")
                return proc
            if cmd0 == "gzip":
                proc = MagicMock()
                proc.communicate.return_value = (b"", b"")
                return proc
            return original_popen(cmd, *args, **kwargs)

        monkeypatch.setattr(image_mod.subprocess, "Popen", fake_popen)
        with pytest.raises(ImageError, match="docker save failed"):
            save_image("pipeline:openvino-latest", dist_dir=tmp_path)

    def test_save_no_docker_raises(self, monkeypatch):
        monkeypatch.setattr(image_mod.shutil, "which", lambda _: None)
        with pytest.raises(ImageError, match="docker CLI not found"):
            save_image("pipeline:openvino-latest", dist_dir=Path("/tmp"))


# ----------------------------- load_image --------------------------------- #


class TestLoadImage:
    def test_load_tar_gz(self, tmp_path, monkeypatch):
        archive = tmp_path / "img.tar.gz"
        archive.write_bytes(b"fake-bytes")
        sha = hashlib.sha256(b"fake-bytes").hexdigest()
        (tmp_path / "img.tar.gz.sha256").write_text(f"{sha}  img.tar.gz\n")

        fake_cp = MagicMock(
            returncode=0, stdout="Loaded image: pipeline:openvino-latest\n", stderr=""
        )
        with patch("subprocess.Popen") as popen, patch("subprocess.run", return_value=fake_cp):
            gunzip = MagicMock()
            gunzip.stdout = MagicMock()
            popen.return_value = gunzip
            tags = load_image(archive)
        assert "pipeline:openvino-latest" in tags

    def test_load_sha_mismatch(self, tmp_path):
        archive = tmp_path / "img.tar.gz"
        archive.write_bytes(b"real")
        (tmp_path / "img.tar.gz.sha256").write_text("0" * 64 + "  img.tar.gz\n")
        with pytest.raises(ImageError, match="sha256 mismatch"):
            load_image(archive)

    def test_load_no_sha_file_ok(self, tmp_path):
        archive = tmp_path / "img.tar"
        archive.write_bytes(b"data")
        fake_cp = MagicMock(returncode=0, stdout="Loaded image: x:y\n", stderr="")
        with patch("subprocess.run", return_value=fake_cp), patch("builtins.open", MagicMock()):
            tags = load_image(archive)
        assert tags == ["x:y"]

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(ImageError, match="archive not found"):
            load_image(tmp_path / "nope.tar.gz")

    def test_load_docker_failure(self, tmp_path):
        archive = tmp_path / "img.tar"
        archive.write_bytes(b"data")
        fake_cp = MagicMock(returncode=1, stdout="", stderr="nope")
        with patch("subprocess.run", return_value=fake_cp), patch("builtins.open", MagicMock()):
            with pytest.raises(ImageError, match="docker load failed"):
                load_image(archive)


# ----------------------------- verify_image ------------------------------- #


class TestVerifyImage:
    def test_verify_success(self):
        fake = MagicMock(returncode=0, stdout="=== all checks passed ===\n", stderr="")
        with (
            patch.object(image_mod, "_run", return_value=fake),
            patch("shutil.which", return_value="/usr/bin/docker"),
        ):
            ok, summary = verify_image("pipeline:openvino-latest")
        assert ok is True
        assert "all checks passed" in summary

    def test_verify_failure(self):
        fake = MagicMock(returncode=1, stdout="some output", stderr="boom")
        with (
            patch.object(image_mod, "_run", return_value=fake),
            patch("shutil.which", return_value="/usr/bin/docker"),
        ):
            ok, summary = verify_image("pipeline:openvino-latest")
        assert ok is False

    def test_verify_exception(self):
        with (
            patch.object(image_mod, "_run", side_effect=ImageError("boom")),
            patch("shutil.which", return_value="/usr/bin/docker"),
        ):
            ok, summary = verify_image("pipeline:openvino-latest")
        assert ok is False
        assert "boom" in summary


# ----------------------------- release_image ------------------------------ #


class TestReleaseImage:
    def test_release_tags_and_writes_manifest(self, manifest_only):
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(image_mod, "_run") as run,
            patch.object(image_mod, "image_size_mb", return_value=1100.0),
        ):
            tag = release_image("pipeline:openvino-a1b2c3d-20260610")
        assert tag == "pipeline:openvino-stable"
        # tag was applied
        cmd = run.call_args_list[0][0][0]
        assert cmd[:3] == ["/usr/bin/docker", "tag", "pipeline:openvino-a1b2c3d-20260610"]
        assert cmd[-1] == "pipeline:openvino-stable"
        # manifest was written
        data = json.loads(manifest_only.read_text())
        assert data["last_stable"]["image"] == "pipeline:openvino-stable"
        assert data["last_stable"]["source"] == "pipeline:openvino-a1b2c3d-20260610"
        assert data["last_stable"]["size_mb"] == 1100.0
        assert "released_at" in data["last_stable"]

    def test_release_explicit_stable_tag(self, manifest_only):
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(image_mod, "_run"),
            patch.object(image_mod, "image_size_mb", return_value=900.0),
        ):
            tag = release_image("pipeline:cpu-latest", stable_tag="pipeline:cpu-stable")
        assert tag == "pipeline:cpu-stable"


# ----------------------------- record_deployment -------------------------- #


class TestRecordDeployment:
    def test_appends_entry(self, manifest_only):
        record_deployment("cachyos-local", "pipeline:openvino-stable", status="ok", note="first")
        record_deployment("vds-prod-01", "pipeline:openvino-stable", status="ok")
        data = json.loads(manifest_only.read_text())
        assert len(data["deployments"]) == 2
        hosts = [d["host"] for d in data["deployments"]]
        assert hosts == ["cachyos-local", "vds-prod-01"]
        assert data["deployments"][0]["note"] == "first"


# ----------------------------- list_images / prune ------------------------ #


class TestListImages:
    def test_parses_format(self):
        out = "pipeline\topenvino-a1b2c3d-20260610\t1.05GB\t2026-06-10 12:00:00 +0000 UTC\n"
        with (
            patch.object(image_mod, "_run") as r,
            patch("shutil.which", return_value="/usr/bin/docker"),
        ):
            r.return_value = MagicMock(stdout=out, returncode=0, stderr="")
            rows = list_images(backend="openvino")
        assert len(rows) == 1
        assert rows[0]["tag"] == "openvino-a1b2c3d-20260610"
        assert rows[0]["size"] == "1.05GB"

    def test_filter_excludes_other_backends(self):
        out = "pipeline\topenvino-a\t1GB\tnow\n" "pipeline\tcpu-b\t800MB\tnow\n"
        with (
            patch.object(image_mod, "_run") as r,
            patch("shutil.which", return_value="/usr/bin/docker"),
        ):
            r.return_value = MagicMock(stdout=out, returncode=0, stderr="")
            rows = list_images(backend="openvino")
        assert len(rows) == 1
        assert rows[0]["tag"] == "openvino-a"


class TestPruneLocal:
    def test_keeps_stable_and_latest(self):
        rows = [
            {
                "repository": "pipeline",
                "tag": "openvino-a1b2c3d-20260610",
                "size": "1GB",
                "created": "now",
            },
            {"repository": "pipeline", "tag": "openvino-stable", "size": "1GB", "created": "now"},
            {"repository": "pipeline", "tag": "openvino-latest", "size": "1GB", "created": "now"},
            {
                "repository": "pipeline",
                "tag": "openvino-x9y8z7-20260501",
                "size": "1GB",
                "created": "old",
            },
        ]
        with (
            patch.object(image_mod, "list_images", return_value=rows),
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch.object(image_mod, "_run") as run,
        ):
            n = prune_local("openvino")
        assert n == 2
        removed_refs = [c[0][0][-1] for c in run.call_args_list]
        assert "pipeline:openvino-a1b2c3d-20260610" in removed_refs
        assert "pipeline:openvino-x9y8z7-20260501" in removed_refs
        assert "pipeline:openvino-stable" not in removed_refs
        assert "pipeline:openvino-latest" not in removed_refs


# ----------------------------- manifest I/O ------------------------------- #


class TestManifestIO:
    def test_read_missing_returns_empty_dict(self, manifest_only):
        assert _read_manifest() == {}

    def test_write_and_read_roundtrip(self, manifest_only):
        _write_manifest({"last_stable": {"image": "x:y"}})
        assert _read_manifest() == {"last_stable": {"image": "x:y"}}

    def test_read_invalid_json_raises(self, manifest_only):
        manifest_only.write_text("{not json")
        with pytest.raises(ImageError, match="not valid JSON"):
            _read_manifest()

    def test_write_is_atomic(self, manifest_only):
        _write_manifest({"a": 1})
        assert not (manifest_only.parent / (manifest_only.name + ".tmp")).exists()
        assert manifest_only.exists()
