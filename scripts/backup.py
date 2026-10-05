#!/usr/bin/env python3
"""
Back up and restore the user's data: `just backup`, `just restore`.

What a backup holds:
- the job database (data/api.db, a consistent snapshot even while the API runs);
- the user settings (data/settings.yaml, what the Settings page saves) and .env;
- every job directory: clips, subtitles, manifest, JSON artifacts, logs.
Source videos (uploads/, the job's copy in artifacts/video and its audio) only
with --with-sources: without them a restored job keeps its clips but cannot be
resumed. Model weights are never included: setup.sh downloads them again.

Restore replaces the database and the user settings (data/settings.yaml;
backups made before the Settings page carried config/config.yaml, which is
still restored) and writes the job directories (same names are overwritten).
.env is not replaced, because it describes this machine (GPU, user, model
directory): the backup's copy goes to .env.from-backup. Before anything is
written, the current data is backed up to backups/before-restore-<time>.tar.
The services must be stopped (scripts/backup.sh checks).

Only the standard library: it runs with the host's python3, also next to a
Docker installation that has no Python environment on the host.

Usage:
    python3 scripts/backup.py create [--with-sources] [--output DIR]
    python3 scripts/backup.py restore ARCHIVE [--yes]
"""

import argparse
import io
import json
import os
import re
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Optional, Tuple

ROOT = Path(os.environ.get("BACKUP_ROOT") or Path(__file__).resolve().parent.parent)
FORMAT = 1
MANIFEST = "backup.json"
DB_MEMBER = "data/api.db"
# The user settings (the Settings page); may hold an API key.
SETTINGS_MEMBER = "data/settings.yaml"
# Backups made before the Settings page carried the base config.
CONFIG_MEMBER = "config/config.yaml"
ENV_MEMBER = ".env"
# Parts of a job directory that only the source video needs (as in
# src/api/services/cleanup.py), and files that are never worth keeping.
SOURCE_PARTS = ("artifacts/video", "artifacts/audio", "artifacts/temp_download")
SKIPPED_PARTS = ("artifacts/tmp", "artifacts/cache", "audio/temp_chunks")
ACTIVE_STATUSES = ("QUEUED", "RUNNING", "queued", "running")


class BackupError(Exception):
    """A backup or restore that cannot go on; the message says why."""


def _env_value(name: str) -> Optional[str]:
    """``name`` from the environment, else from the project's .env."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = re.match(rf"\s*{re.escape(name)}\s*=\s*(.*)$", line)
        if match:
            return match.group(1).strip().strip("'\"") or None
    return None


def database_path() -> Path:
    """The SQLite file of the job queue (API_DATABASE_URL, as the API reads it)."""
    url = _env_value("API_DATABASE_URL") or "sqlite+aiosqlite:///./data/api.db"
    match = re.match(r"sqlite(\+\w+)?:///(.+)$", url)
    if not match:
        raise BackupError(f"Only SQLite databases can be backed up, not {url}")
    path = Path(match.group(2))
    return path if path.is_absolute() else ROOT / path


def _snapshot(db: Path, target: Path) -> List[Tuple[str, str, str, str]]:
    """Copy the database consistently; return (id, status, work_dir, input_source) of jobs."""
    # Not mode=ro: a read-only connection cannot open a WAL database whose
    # -shm file is gone (the API stopped cleanly). Reading changes nothing.
    source = sqlite3.connect(db, timeout=30)
    try:
        copy = sqlite3.connect(target)
        with copy:
            source.backup(copy)
        copy.close()
        try:
            rows = source.execute(
                "SELECT id, status, work_dir, input_source FROM jobs ORDER BY id"
            ).fetchall()
        except sqlite3.OperationalError:  # a database without jobs yet
            rows = []
    finally:
        source.close()
    return [tuple(str(v) for v in row) for row in rows]


def _job_files(job_dir: Path, with_sources: bool) -> Iterator[Path]:
    for path in sorted(job_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(job_dir).as_posix()
        if path.name.endswith(".part.mp4") or relative.startswith(SKIPPED_PARTS):
            continue
        if not with_sources and relative.startswith(SOURCE_PARTS):
            continue
        yield path


def _below(path: Path, root: Path) -> bool:
    """``path`` inside ``root`` (Path.is_relative_to needs Python 3.9)."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _app_version() -> str:
    init = ROOT / "src" / "__init__.py"
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init.read_text()) if init.exists() else None
    return match.group(1) if match else "unknown"


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = int(datetime.now().timestamp())
    tar.addfile(info, io.BytesIO(data))


def create_backup(
    output_dir: Optional[Path] = None, with_sources: bool = False, name: str = "backup"
) -> Path:
    """Write backups/<name>-<time>.tar; return its path."""
    db = database_path()
    if not db.is_file():
        raise BackupError(f"No job database at {db}: nothing to back up yet")
    output_dir = output_dir or ROOT / "backups"
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = output_dir / f"{name}-{stamp}.tar"
    partial = archive.with_name(archive.name + ".part")

    with tempfile.TemporaryDirectory(dir=output_dir) as tmp:
        snapshot = Path(tmp) / "api.db"
        jobs = _snapshot(db, snapshot)
        active = [job_id for job_id, status, _, _ in jobs if status in ACTIVE_STATUSES]
        job_dirs = sorted(p for p in (ROOT / "jobs").glob("*") if p.is_dir())
        uploads = []
        if with_sources:
            for _, _, _, source in jobs:
                path = (ROOT / source) if not Path(source).is_absolute() else Path(source)
                if path.is_file() and _below(path, ROOT / "uploads"):
                    uploads.append(path)
        files = 0
        size = 0
        with tarfile.open(partial, "w") as tar:
            tar.add(snapshot, arcname=DB_MEMBER)
            for path, member, mode in (
                (ROOT / SETTINGS_MEMBER, SETTINGS_MEMBER, 0o600),
                (ROOT / ".env", ENV_MEMBER, 0o600),
            ):
                if path.is_file():
                    info = tar.gettarinfo(path, arcname=member)
                    if mode is not None:
                        info.mode = mode
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
            for job_dir in job_dirs:
                for path in _job_files(job_dir, with_sources):
                    tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
                    files += 1
                    size += path.stat().st_size
            for path in sorted(set(uploads)):
                arcname = path.resolve().relative_to(ROOT.resolve()).as_posix()
                tar.add(path, arcname=arcname, recursive=False)
                files += 1
                size += path.stat().st_size
            manifest = {
                "format": FORMAT,
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "app_version": _app_version(),
                "with_sources": with_sources,
                "jobs": len(jobs),
                "job_dirs": len(job_dirs),
                "files": files,
                "bytes": size,
                "active_jobs": active,
            }
            _add_bytes(tar, MANIFEST, json.dumps(manifest, indent=2).encode())
    os.chmod(partial, 0o600)  # .env holds the password and the session secret
    partial.replace(archive)
    if active:
        print(
            f"Note: jobs {', '.join(active)} were queued or running; their files may be "
            "incomplete in this backup.",
            file=sys.stderr,
        )
    return archive


def _allowed(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    return name in (
        MANIFEST,
        DB_MEMBER,
        SETTINGS_MEMBER,
        CONFIG_MEMBER,
        ENV_MEMBER,
    ) or path.parts[0] in (
        "jobs",
        "uploads",
    )


def read_manifest(archive: Path) -> Dict:
    """The archive's backup.json, after checking that every member is safe to extract."""
    try:
        with tarfile.open(archive, "r") as tar:
            members = tar.getmembers()
            for member in members:
                if not (member.isfile() or member.isdir()) or not _allowed(member.name):
                    raise BackupError(f"Not a backup of this project: unexpected {member.name!r}")
            names = {m.name for m in members}
            if MANIFEST not in names or DB_MEMBER not in names:
                raise BackupError("Not a backup of this project: no backup.json or database")
            manifest = json.load(tar.extractfile(MANIFEST))
    except (tarfile.TarError, OSError, json.JSONDecodeError) as e:
        raise BackupError(f"Cannot read {archive}: {e}") from e
    if manifest.get("format") != FORMAT:
        raise BackupError(f"Unknown backup format {manifest.get('format')!r}")
    return manifest


def restore_backup(archive: Path, assume_yes: bool = False) -> Dict:
    """Restore ``archive`` over the current data, after a safety backup of it."""
    manifest = read_manifest(archive)
    db = database_path()
    print(
        f"Backup of {manifest['created_at']} (version {manifest['app_version']}): "
        f"{manifest['jobs']} jobs, {manifest['bytes'] / 1e9:.1f} GB, "
        f"{'with' if manifest['with_sources'] else 'without'} source videos."
    )
    print(
        "This replaces the job database and the user settings and writes the job "
        "directories (same names are overwritten). .env stays; the backup's copy "
        "goes to .env.from-backup."
    )
    if not assume_yes:
        answer = input("Restore? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "д", "да"):
            raise BackupError("Cancelled; nothing was changed")

    safety = None
    if db.is_file():
        safety = create_backup(name="before-restore")
        print(f"Current data saved to {safety}")

    with tarfile.open(archive, "r") as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if member.name == MANIFEST:
                continue
            if member.name == DB_MEMBER:
                target = db
            elif member.name == ENV_MEMBER:
                target = ROOT / ".env.from-backup"
            else:
                target = ROOT / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".restoring")
            with tar.extractfile(member) as source, partial.open("wb") as handle:
                while chunk := source.read(1 << 20):
                    handle.write(chunk)
            private = member.name in (ENV_MEMBER, SETTINGS_MEMBER)
            os.chmod(partial, 0o600 if private else 0o644)
            if member.name == DB_MEMBER:
                # A write-ahead log of the old database would be replayed
                # into the restored one.
                for suffix in ("-wal", "-shm"):
                    Path(str(db) + suffix).unlink(missing_ok=True)
            partial.replace(target)
    return {"manifest": manifest, "safety_backup": str(safety) if safety else None}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="write backups/backup-<time>.tar")
    create.add_argument(
        "--with-sources",
        action="store_true",
        help="also the source videos (uploads and the jobs' copies)",
    )
    create.add_argument("--output", type=Path, help="directory for the archive (default: backups/)")
    restore = sub.add_parser("restore", help="restore an archive over the current data")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--yes", action="store_true", help="do not ask")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            archive = create_backup(args.output, with_sources=args.with_sources)
            manifest = read_manifest(archive)
            print(
                f"Backup: {archive} ({archive.stat().st_size / 1e9:.2f} GB, "
                f"{manifest['jobs']} jobs, {manifest['files']} files"
                f"{', with source videos' if manifest['with_sources'] else ''})"
            )
        else:
            result = restore_backup(args.archive, assume_yes=args.yes)
            print(f"Restored {result['manifest']['jobs']} jobs. Start again: just up")
    except BackupError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
