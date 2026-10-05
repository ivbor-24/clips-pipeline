"""Tests for src/model_registry.py (pinned model revisions and checksums)."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from src import model_registry
from src.model_registry import (
    ModelIntegrityError,
    PinnedFile,
    find_pinned_file,
    find_pinned_file_by_name,
    find_pinned_snapshot,
    link_into_place,
    verify_file,
)


def _pin_for(content: bytes, **kwargs) -> PinnedFile:
    fields = dict(
        repo="org/repo",
        file="m.bin",
        revision="rev",
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    fields.update(kwargs)
    return PinnedFile(**fields)


@pytest.fixture
def lock_file(tmp_path):
    path = tmp_path / "models.lock.yaml"
    path.write_text(
        "files:\n"
        "  - repo: org/repo\n"
        "    file: m.bin\n"
        "    revision: abc\n"
        "    sha256: ABCDEF\n"
        "    size: 3\n"
        "faster_whisper:\n"
        "  tiny:\n"
        "    repo: org/whisper-tiny\n"
        "    revision: def\n"
    )
    model_registry._load_lock.cache_clear()
    yield path
    model_registry._load_lock.cache_clear()


class TestLookup:
    def test_find_pinned_file(self, lock_file):
        pin = find_pinned_file("org/repo", "m.bin", lock_file)
        assert pin == PinnedFile("org/repo", "m.bin", "abc", "abcdef", 3)

    def test_unknown_file_is_none(self, lock_file):
        assert find_pinned_file("org/repo", "other.bin", lock_file) is None

    def test_find_by_name(self, lock_file):
        assert find_pinned_file_by_name("m.bin", lock_file).repo == "org/repo"

    def test_find_snapshot(self, lock_file):
        pin = find_pinned_snapshot("tiny", lock_file)
        assert (pin.repo, pin.revision) == ("org/whisper-tiny", "def")
        assert find_pinned_snapshot("large", lock_file) is None

    def test_missing_lock_file(self, tmp_path):
        model_registry._load_lock.cache_clear()
        assert find_pinned_file("org/repo", "m.bin", tmp_path / "absent.yaml") is None

    def test_repo_lock_is_valid(self):
        """The real config/models.lock.yaml parses and has sane entries."""
        model_registry._load_lock.cache_clear()
        lock = model_registry._load_lock()
        assert lock["files"], "no pinned files"
        for entry in lock["files"]:
            pin = find_pinned_file(entry["repo"], entry["file"])
            assert len(pin.revision) == 40 and len(pin.sha256) == 64 and pin.size > 0
        for name in lock["faster_whisper"]:
            assert len(find_pinned_snapshot(name).revision) == 40


class TestVerifyFile:
    def test_ok_writes_sidecar(self, tmp_path):
        f = tmp_path / "m.bin"
        f.write_bytes(b"weights")
        verify_file(f, _pin_for(b"weights"))
        assert (tmp_path / "m.bin.sha256").read_text().startswith(_pin_for(b"weights").sha256)

    def test_sidecar_skips_rehash(self, tmp_path):
        f = tmp_path / "m.bin"
        f.write_bytes(b"weights")
        pin = _pin_for(b"weights")
        verify_file(f, pin)
        with patch.object(model_registry, "sha256_file") as mock_hash:
            verify_file(f, pin)
        mock_hash.assert_not_called()

    def test_modified_file_is_rehashed(self, tmp_path):
        f = tmp_path / "m.bin"
        f.write_bytes(b"weights")
        pin = _pin_for(b"weights")
        verify_file(f, pin)
        f.write_bytes(b"WEIGHTS")  # same size, new mtime and content
        with pytest.raises(ModelIntegrityError, match="sha256"):
            verify_file(f, pin)

    def test_size_mismatch(self, tmp_path):
        f = tmp_path / "m.bin"
        f.write_bytes(b"short")
        with pytest.raises(ModelIntegrityError, match="size"):
            verify_file(f, _pin_for(b"weights"))

    def test_symlink_verifies_target(self, tmp_path):
        real = tmp_path / "store" / "m.bin"
        real.parent.mkdir()
        real.write_bytes(b"weights")
        link = tmp_path / "models" / "m.bin"
        link.parent.mkdir()
        link.symlink_to(real)
        verify_file(link, _pin_for(b"weights"))
        assert (real.parent / "m.bin.sha256").exists()


class TestDownloadAndLink:
    def test_download_pinned_file(self, tmp_path):
        f = tmp_path / "m.bin"
        f.write_bytes(b"weights")
        pin = _pin_for(b"weights")
        with patch("huggingface_hub.hf_hub_download", return_value=str(f)) as mock_dl:
            assert model_registry.download_pinned_file(pin) == f
        assert mock_dl.call_args.kwargs["revision"] == "rev"

    def test_link_into_place_replaces_old_link(self, tmp_path):
        src = tmp_path / "cache" / "m.bin"
        src.parent.mkdir()
        src.write_bytes(b"x")
        target = tmp_path / "models" / "m.bin"
        target.parent.mkdir()
        target.symlink_to(tmp_path / "gone.bin")
        link_into_place(target, src)
        assert Path(target).resolve() == src.resolve()
