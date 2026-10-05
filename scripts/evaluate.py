#!/usr/bin/env python3
"""
Evaluate the pipeline against a reference set of hand-made shorts.

The reference set lives outside the repository and is described by a
``manifest.json`` (see docs/EVALUATION.md). This script talks to the running
service over HTTP exactly like the web UI does (login, upload, jobs, review),
so it runs with the host's ``python3`` while the Docker stack is up.

Three subcommands:

    python3 scripts/evaluate.py run --set PATH/manifest.json --label NAME
        [--only EPISODE ...] [--overrides FILE.json] [--verify-sha] [--url URL]
    python3 scripts/evaluate.py report --set PATH/manifest.json NAME
        [--against PREV] [--hit 0.5] [--hook-sec 3] [--url URL]
    python3 scripts/evaluate.py carry-over NAME --from PREV [--iou 0.8] [--url URL]

Output lives under ``eval/``: run files in ``eval/runs/<label>.json``, reports
in ``eval/reports/<label>.md``. Only the standard library is used, so it works
next to a Docker installation that has no Python environment on the host.

Metrics are time overlaps in the source: the pipeline cuts one continuous
45-90 s window per clip, hand-made shorts are often 2-7 pieces with parts cut out,
so exact boundaries never match.
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ROOT = Path(os.environ.get("EVAL_ROOT") or Path(__file__).resolve().parent.parent)
EVAL_DIR = ROOT / "eval"
RUNS_DIR = EVAL_DIR / "runs"
REPORTS_DIR = EVAL_DIR / "reports"

FORMAT = 1

# Job statuses that mean "the job will not change anymore".
TERMINAL_STATUSES = ("completed", "failed", "cancelled")
# Review statuses that count as a human decision (not "pending").
RATED_STATUSES = ("keep", "edit", "reject")

DEFAULT_HIT_THRESHOLD = 0.5
DEFAULT_HOOK_SEC = 3.0
DEFAULT_IOU = 0.8
# IoU for matching a pipeline clip to keep/edit/reject labels from earlier reviews.
DEFAULT_LABEL_IOU = 0.5
DEFAULT_POLL_SEC = 5.0


class EvaluateError(Exception):
    """An evaluation that cannot go on; the message says what to do."""


class ApiError(Exception):
    """A non-success HTTP answer from the service."""

    def __init__(self, status: int, path: str, detail: str):
        self.status = status
        self.path = path
        self.detail = detail
        super().__init__(f"{status} {path}: {detail}")


def _env_value(name: str) -> Optional[str]:
    """``name`` from the environment, else from the project's ``.env``."""
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


def default_url() -> str:
    """The web UI's address, as the user opens it: WEB_PORT from .env else 8080."""
    port = _env_value("WEB_PORT") or "8080"
    return f"http://127.0.0.1:{port}"


def _write_json(path: Path, data: Any) -> None:
    """Atomically write a JSON file (a crash never leaves a half-written one)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".tmp")
    partial.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def _git_commit() -> Optional[str]:
    """The checkout's HEAD, or None when git is not available (e.g. in tests)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


class EvalApi:
    """The one small HTTP client the evaluation needs; tests replace it with a fake."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None

    def _connection(self) -> http.client.HTTPConnection:
        parts = urllib.parse.urlsplit(self.base_url)
        if parts.scheme == "https":
            return http.client.HTTPSConnection(parts.hostname, parts.port, timeout=60)
        return http.client.HTTPConnection(parts.hostname, parts.port, timeout=60)

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[bytes] = None,
        content_type: str = "application/json",
    ) -> Tuple[int, bytes]:
        conn = self._connection()
        headers = {}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if body is not None:
            headers["Content-Type"] = content_type
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        data = response.read()
        status = response.status
        conn.close()
        return status, data

    def _json(self, method: str, path: str, payload: Any = None) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        status, data = self._request(method, path, body=body)
        if status == 204:
            return None
        if status >= 400:
            detail = ""
            try:
                detail = json.loads(data).get("detail", data.decode("utf-8", "replace"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                detail = data.decode("utf-8", "replace")
            raise ApiError(status, path, str(detail))
        return json.loads(data) if data else None

    # --- Endpoints used by the evaluation -----------------------------------

    def health(self) -> Dict[str, Any]:
        return self._json("GET", "/api/v1/health")

    def password_required(self) -> bool:
        """Whether the service asks for the shared password (API_PASSWORD)."""
        try:
            me = self._json("GET", "/api/v1/auth/me")
        except ApiError as e:
            if e.status == 401:
                return True
            raise
        return bool(me.get("password_required"))

    def login(self, password: str) -> None:
        status, data = self._request(
            "POST",
            "/api/v1/auth/login",
            body=json.dumps({"password": password}).encode("utf-8"),
        )
        if status == 400:
            raise EvaluateError("Login is off: API_PASSWORD is not set")
        if status == 401:
            raise EvaluateError("Wrong API_PASSWORD")
        if status >= 400:
            raise ApiError(status, "/api/v1/auth/login", data.decode("utf-8", "replace"))
        self.token = json.loads(data)["access_token"]

    def config(self) -> Dict[str, Any]:
        return self._json("GET", "/api/v1/config/")

    def upload(self, filename: str, source_path: Path, size: int) -> Dict[str, Any]:
        """Stream ``source_path`` to the upload endpoint; returns its JSON answer."""
        path = "/api/v1/upload/?filename=" + urllib.parse.quote(filename)
        conn = self._connection()
        headers = {"Content-Type": "application/octet-stream", "Content-Length": str(size)}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        try:
            conn.putrequest("POST", path)
            for name, value in headers.items():
                conn.putheader(name, value)
            conn.endheaders()
            with open(source_path, "rb") as handle:
                while chunk := handle.read(1 << 20):
                    conn.send(chunk)
            response = conn.getresponse()
            data = response.read()
            status = response.status
        finally:
            conn.close()
        if status >= 400:
            raise ApiError(status, path, data.decode("utf-8", "replace"))
        return json.loads(data)

    def create_job(
        self, input_source: str, config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"input_source": input_source, "job_type": "clips"}
        if config_overrides:
            payload["config_overrides"] = config_overrides
        return self._json("POST", "/api/v1/jobs/", payload)

    def get_job(self, job_id: int) -> Dict[str, Any]:
        return self._json("GET", f"/api/v1/jobs/{job_id}")

    def manifest(self, job_id: int) -> Dict[str, Any]:
        return self._json("GET", f"/api/v1/jobs/{job_id}/manifest")

    def clips(self, job_id: int) -> Dict[str, Any]:
        return self._json("GET", f"/api/v1/jobs/{job_id}/clips")

    def set_review(
        self, job_id: int, clip_id: str, status: str, notes: Optional[str] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"review_status": status}
        if notes is not None:
            payload["notes"] = notes
        return self._json("PUT", f"/api/v1/jobs/{job_id}/clips/{clip_id}", payload)


def connect(api: EvalApi, password: Optional[str]) -> None:
    """Log in when the service has a shared password, as the web UI does."""
    if api.password_required():
        if not password:
            raise EvaluateError(
                "The service requires a password. Set API_PASSWORD in the "
                "environment or in .env (the same password the web UI asks for)."
            )
        api.login(password)


# ---------------------------------------------------------------------------
# Pure metric functions (time overlap in the source, never exact boundaries)
# ---------------------------------------------------------------------------


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the intersection of two time intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def piece_duration(piece: Dict[str, Any]) -> float:
    return float(piece["source_end"]) - float(piece["source_start"])


def coverage(clip_start: float, clip_end: float, pieces: Sequence[Dict[str, Any]]) -> float:
    """Share of a short's source material (sum of piece durations) that a clip covers."""
    total = sum(piece_duration(p) for p in pieces)
    if total <= 0:
        return 0.0
    covered = sum(
        overlap(clip_start, clip_end, float(p["source_start"]), float(p["source_end"]))
        for p in pieces
    )
    return covered / total


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Intersection over union of two time intervals."""
    inter = overlap(a_start, a_end, b_start, b_end)
    union = (a_end - a_start) + (b_end - b_start) - inter
    if union <= 0:
        return 0.0
    return inter / union


def is_montage(short: Dict[str, Any]) -> bool:
    """Teasers stitched from far-apart places; the pipeline cannot make them."""
    return bool((short.get("structure") or {}).get("montage"))


def non_montage_shorts(shorts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [s for s in shorts if not is_montage(s)]


def opening_piece(short: Dict[str, Any]) -> Dict[str, Any]:
    """The piece the short starts with (smallest position in the short)."""
    return min(short["pieces"], key=lambda p: p["short_start"])


def best_covering_clip(
    clips: Sequence[Dict[str, Any]], short: Dict[str, Any]
) -> Tuple[Optional[Dict[str, Any]], float]:
    """The clip that covers the most of a short, and its coverage."""
    pieces = short["pieces"]
    best: Optional[Dict[str, Any]] = None
    best_cov = 0.0
    for clip in clips:
        cov = coverage(float(clip["start"]), float(clip["end"]), pieces)
        if cov > best_cov:
            best_cov = cov
            best = clip
    return best, best_cov


def find_shorts(
    clips: Sequence[Dict[str, Any]], shorts: Sequence[Dict[str, Any]], hit_threshold: float
) -> Dict[str, Dict[str, Any]]:
    """Per (non-montage) short: the best clip, its coverage and whether it is a hit."""
    result: Dict[str, Dict[str, Any]] = {}
    for short in shorts:
        clip, cov = best_covering_clip(clips, short)
        found = clip is not None and cov >= hit_threshold
        result[short["id"]] = {"clip": clip, "coverage": cov, "found": found}
    return result


def hook_ok(clip: Dict[str, Any], short: Dict[str, Any], hook_sec: float) -> bool:
    """Whether a matched clip starts within +-hook_sec of the short's opening piece."""
    target = float(opening_piece(short)["source_start"])
    return abs(float(clip["start"]) - target) <= hook_sec


def source_span_start(short: Dict[str, Any]) -> Optional[float]:
    span = (short.get("structure") or {}).get("source_span")
    if not span:
        return None
    return float(span[0])


def clip_finds_short(
    clip: Dict[str, Any], shorts: Sequence[Dict[str, Any]], hit_threshold: float
) -> bool:
    return any(
        coverage(float(clip["start"]), float(clip["end"]), s["pieces"]) >= hit_threshold
        for s in shorts
    )


def clip_matches_label(clip: Dict[str, Any], label: Dict[str, Any], iou_threshold: float) -> bool:
    return (
        temporal_iou(
            float(clip["start"]),
            float(clip["end"]),
            float(label["source_start"]),
            float(label["source_end"]),
        )
        >= iou_threshold
    )


def greedy_iou_matching(
    pending: Sequence[Dict[str, Any]],
    rated: Sequence[Dict[str, Any]],
    iou_threshold: float,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """One-to-one matching of pending clips to rated clips by best IoU, per episode.

    Greedy: pending clips are matched in order, each to the rated clip of the
    same episode with the highest IoU not below the threshold (each rated clip
    is used at most once).
    """
    rated_by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for clip in rated:
        rated_by_episode[clip["episode"]].append(clip)

    pending_by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for clip in pending:
        pending_by_episode[clip["episode"]].append(clip)

    matches: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    for episode, candidates in pending_by_episode.items():
        available = list(rated_by_episode.get(episode, []))
        for clip in candidates:
            best: Optional[Dict[str, Any]] = None
            best_iou = float("-inf")
            for other in available:
                iou = temporal_iou(
                    float(clip["start"]),
                    float(clip["end"]),
                    float(other["start"]),
                    float(other["end"]),
                )
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best = other
            if best is not None:
                available.remove(best)
                matches.append((clip, best, best_iou))
    return matches


def median_duration(clips: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not clips:
        return None
    return statistics.median(float(c["end"]) - float(c["start"]) for c in clips)


# ---------------------------------------------------------------------------
# run / report / carry-over
# ---------------------------------------------------------------------------


def load_set(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise EvaluateError(f"Reference set not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvaluateError(f"Cannot read {path}: {e}") from e


def run_path(label: str) -> Path:
    return RUNS_DIR / f"{label}.json"


def load_run(label: str) -> Dict[str, Any]:
    path = run_path(label)
    if not path.is_file():
        raise EvaluateError(f"No run '{label}' at {path}: run it first with `run --label {label}`")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvaluateError(f"Cannot read {path}: {e}") from e
    if data.get("format") != FORMAT:
        raise EvaluateError(f"Unknown run format {data.get('format')!r}")
    return data


def _load_overrides(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        raise EvaluateError(f"Overrides file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvaluateError(f"Cannot read {path}: {e}") from e
    if not isinstance(data, dict):
        raise EvaluateError(f"Overrides file must hold a JSON object: {path}")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _check_source(episode_id: str, episode: Dict[str, Any], verify_sha: bool) -> Optional[str]:
    """Return an error message when the source file does not match the set, else None."""
    path = Path(episode["path"])
    if not path.is_file():
        return f"source video not found: {episode['path']}"
    actual = path.stat().st_size
    expected = int(episode["size_bytes"])
    if actual != expected:
        return f"source size mismatch: {actual} != {expected} bytes"
    if verify_sha and episode.get("sha256"):
        digest = _sha256(path)
        if digest != episode["sha256"]:
            return f"source sha256 mismatch: {digest} != {episode['sha256']}"
    return None


def _record_clip(clip: Dict[str, Any]) -> Dict[str, Any]:
    """A clip from the manifest endpoint, reduced to what the metrics need."""
    return {
        "id": clip.get("clip_id") or clip.get("id"),
        "start": float(clip.get("start_time", clip.get("start"))),
        "end": float(clip.get("end_time", clip.get("end"))),
        "title": clip.get("title", ""),
        "score": clip.get("score", 0.0),
    }


def run_evaluation(
    api: EvalApi,
    set_path: Path,
    set_data: Dict[str, Any],
    episodes: Sequence[str],
    label: str,
    overrides: Optional[Dict[str, Any]],
    verify_sha: bool,
    progress: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    poll_interval: float = DEFAULT_POLL_SEC,
) -> Dict[str, Any]:
    """Run the pipeline for every episode; write and return ``eval/runs/<label>.json``.

    Run again with the same label and episodes whose job already finished are
    skipped; a job left running by Ctrl+C is polled again instead of restarted.
    """
    path = run_path(label)
    run: Dict[str, Any] = {}
    if path.is_file():
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress(f"warning: ignoring unreadable run file {path}")
            run = {}
        if run.get("label") != label:
            raise EvaluateError(f"{path} already holds run '{run.get('label')}'")

    run.setdefault("format", FORMAT)
    run["label"] = label
    run["created_at"] = run.get("created_at") or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    run["set"] = str(set_path)
    run["app_version"] = run.get("app_version") or api.health().get("version")
    run["git_commit"] = run.get("git_commit") or _git_commit()
    run["config"] = run.get("config") or api.config().get("config", {})
    run["config_overrides"] = overrides
    run.setdefault("episodes", {})

    for episode_id in episodes:
        episode = set_data["episodes"][episode_id]
        existing = run["episodes"].get(episode_id, {})

        # Resume: a finished job is done; a queued/running one is polled again.
        if existing.get("status") in TERMINAL_STATUSES:
            progress(f"{episode_id}: skipped (job {existing.get('job_id')} already finished)")
            continue

        entry: Dict[str, Any] = existing
        job_id = existing.get("job_id")
        if job_id is None:
            error = _check_source(episode_id, episode, verify_sha)
            if error is not None:
                entry["status"] = "failed"
                entry["error"] = error
                entry.setdefault("clips", [])
                run["episodes"][episode_id] = entry
                _write_json(path, run)
                progress(f"{episode_id}: {error}")
                continue
            uploaded = api.upload(
                Path(episode["path"]).name,
                Path(episode["path"]),
                Path(episode["path"]).stat().st_size,
            )
            job = api.create_job(uploaded["path"], overrides)
            job_id = job["id"]
            entry["job_id"] = job_id
            entry["status"] = "queued"
            entry["error"] = None
            entry.setdefault("clips", [])
            run["episodes"][episode_id] = entry
            _write_json(path, run)
            progress(f"{episode_id}: queued (job {job_id})")

        status = entry.get("status")
        while status not in TERMINAL_STATUSES:
            sleep(poll_interval)
            job = api.get_job(job_id)
            new_status = job["status"]
            entry["status"] = new_status
            entry["error"] = job.get("error_message")
            if new_status != status:
                status = new_status
                progress(f"{episode_id}: {status} (job {job_id})")

        if status == "completed":
            manifest = api.manifest(job_id)
            entry["clips"] = [_record_clip(c) for c in manifest.get("clips", [])]
            progress(f"{episode_id}: {status}, {len(entry['clips'])} clips")
        run["episodes"][episode_id] = entry
        _write_json(path, run)

    return run


def _fetch_reviews(api: EvalApi, run: Dict[str, Any]) -> None:
    """Fetch the current Review statuses of the run's completed jobs into the run file."""
    for episode_id, entry in run.get("episodes", {}).items():
        if entry.get("status") != "completed" or entry.get("job_id") is None:
            continue
        clips = api.clips(entry["job_id"]).get("clips", [])
        entry["reviews"] = {c["id"]: c.get("review_status", "pending") for c in clips}


def _partial_warnings(run: Dict[str, Any], set_data: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for episode_id, entry in run.get("episodes", {}).items():
        if entry.get("status") in ("failed", "cancelled"):
            warnings.append(f"job for {episode_id} is {entry['status']}: {entry.get('error')}")
    run_episodes = run.get("episodes", {})
    for episode_id in set_data.get("episodes", {}):
        if episode_id not in run_episodes:
            warnings.append(f"episode {episode_id} is missing from this run (--only?)")
    pending = sum(
        1
        for entry in run.get("episodes", {}).values()
        for status in entry.get("reviews", {}).values()
        if status == "pending"
    )
    if pending:
        warnings.append(f"{pending} clips still have Review status 'pending'")
    return warnings


def compute_metrics(
    run: Dict[str, Any],
    set_data: Dict[str, Any],
    hit_threshold: float = DEFAULT_HIT_THRESHOLD,
    hook_sec: float = DEFAULT_HOOK_SEC,
    iou_threshold: float = DEFAULT_LABEL_IOU,
) -> Dict[str, Any]:
    """Assemble every metric the report shows, from a run file and the set."""
    episodes = set_data["episodes"]
    shorts = set_data.get("shorts", [])
    labels = set_data.get("pipeline_clip_labels", [])

    shorts_by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for short in shorts:
        shorts_by_episode[short["episode"]].append(short)
    labels_by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for label in labels:
        labels_by_episode[label["episode"]].append(label)

    found_total = 0
    hit_total = 0
    missing: List[Dict[str, Any]] = []
    hooks: List[Dict[str, Any]] = []
    short_coverage: Dict[str, float] = {}
    precise_total = 0
    clip_total = 0
    per_episode: Dict[str, Dict[str, Any]] = {}
    labels_report: Dict[str, Dict[str, Any]] = {}

    run_episodes = run.get("episodes", {})
    # Only episodes present in the run count: `run --only EP_1` must not count the
    # other set episodes as missed. Set order is kept for a stable report.
    ordered_ids = [e for e in episodes if e in run_episodes]
    ordered_ids += [e for e in run_episodes if e not in episodes]

    for episode_id in ordered_ids:
        entry = run_episodes.get(episode_id, {})
        clips = entry.get("clips", [])
        reviews = entry.get("reviews", {})
        ep_shorts = non_montage_shorts(shorts_by_episode.get(episode_id, []))
        ep_labels = labels_by_episode.get(episode_id, [])

        found = find_shorts(clips, ep_shorts, hit_threshold)
        found_count = sum(1 for r in found.values() if r["found"])
        found_total += found_count
        hit_total += len(ep_shorts)
        for short_id, result in found.items():
            short_coverage[short_id] = result["coverage"]
            if not result["found"]:
                missing.append(
                    {
                        "episode": episode_id,
                        "short": short_id,
                        "coverage": result["coverage"],
                    }
                )
                continue
            short = next(s for s in ep_shorts if s["id"] == short_id)
            clip = result["clip"]
            hook_entry: Dict[str, Any] = {
                "episode": episode_id,
                "short": short_id,
                "hook": hook_ok(clip, short, hook_sec),
                "clip_start": clip["start"],
            }
            if (short.get("structure") or {}).get("reordered"):
                span = source_span_start(short)
                hook_entry["span_hook"] = (
                    abs(clip["start"] - span) <= hook_sec if span is not None else None
                )
                hook_entry["source_span_start"] = span
            hooks.append(hook_entry)

        precise = sum(
            1
            for clip in clips
            if clip_finds_short(clip, ep_shorts, hit_threshold)
            or any(
                label["label"] in ("keep", "edit")
                and clip_matches_label(clip, label, iou_threshold)
                for label in ep_labels
            )
        )
        precise_total += precise
        clip_total += len(clips)

        review_counts = Counter(reviews.values())
        rated = sum(review_counts[s] for s in RATED_STATUSES)
        keep_share = (review_counts["keep"] / rated) if rated else None

        per_episode[episode_id] = {
            "clip_count": len(clips),
            "median_duration": median_duration(clips),
            "review_counts": {s: review_counts[s] for s in ("keep", "edit", "reject", "pending")},
            "rated": rated,
            "keep_share": keep_share,
            "found": found_count,
            "total": len(ep_shorts),
        }

        if ep_labels:
            labels_report[episode_id] = {
                "reject": [
                    clip["id"]
                    for clip in clips
                    if any(
                        label["label"] == "reject"
                        and clip_matches_label(clip, label, iou_threshold)
                        for label in ep_labels
                    )
                ],
                "keep": [
                    clip["id"]
                    for clip in clips
                    if any(
                        label["label"] == "keep" and clip_matches_label(clip, label, iou_threshold)
                        for label in ep_labels
                    )
                ],
                "edit": [
                    clip["id"]
                    for clip in clips
                    if any(
                        label["label"] == "edit" and clip_matches_label(clip, label, iou_threshold)
                        for label in ep_labels
                    )
                ],
            }

    return {
        "hit": {"found": found_total, "total": hit_total},
        "missing": missing,
        "short_coverage": short_coverage,
        "hooks": hooks,
        "precision": {"precise": precise_total, "total": clip_total},
        "labels": labels_report,
        "episodes": per_episode,
        "warnings": _partial_warnings(run, set_data),
    }


def _pct(part: float, whole: float) -> str:
    if not whole:
        return "—"
    return f"{100 * part / whole:.0f}%"


def _pct_ratio(part: float, whole: float) -> Optional[float]:
    """The percentage as a number, or None when the total is zero."""
    if not whole:
        return None
    return 100 * part / whole


def _ids_cell(ids: Sequence[str]) -> str:
    """A count followed by the ids, or an em dash when empty."""
    if not ids:
        return "—"
    return f"{len(ids)}: {', '.join(ids)}"


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    return f"{seconds:.1f}s"


def render_report(
    run: Dict[str, Any],
    set_data: Dict[str, Any],
    metrics: Dict[str, Any],
    against_run: Optional[Dict[str, Any]] = None,
    against_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """The Markdown report; printed and written to ``eval/reports/<label>.md``."""
    label = run["label"]
    lines: List[str] = []
    lines.append(f"# Оценка прогона `{label}`\n")
    lines.append(f"- Набор: `{run.get('set')}`")
    lines.append(f"- Версия приложения: {run.get('app_version')}")
    if run.get("git_commit"):
        lines.append(f"- Git commit: `{run['git_commit']}`")
    lines.append(f"- Записан: {run.get('created_at')}")

    for warning in metrics["warnings"]:
        lines.append(f"\n> ⚠ Прогон неполный: {warning}")

    hit = metrics["hit"]
    precision = metrics["precision"]
    lines.append("\n## Hit (найденные шорты)\n")
    lines.append("| Эпизод | Найдено | Всего | Доля |")
    lines.append("|---|---|---|---|")
    for episode_id, ep in metrics["episodes"].items():
        lines.append(
            f"| {episode_id} | {ep['found']} | {ep['total']} | "
            f"{_pct(ep['found'], ep['total'])} |"
        )
    lines.append(
        f"| **Всего** | **{hit['found']}** | **{hit['total']}** | "
        f"**{_pct(hit['found'], hit['total'])}** |"
    )

    if metrics["missing"]:
        lines.append("\n### Шорты, которые не найдены\n")
        lines.append("| Эпизод | Шорт | Лучшее покрытие |")
        lines.append("|---|---|---|")
        for m in sorted(metrics["missing"], key=lambda m: m["coverage"]):
            lines.append(f"| {m['episode']} | {m['short']} | {m['coverage']:.2f} |")

    if metrics["hooks"]:
        lines.append("\n## Hook (начало найденного клипа)\n")
        lines.append(
            "| Эпизод | Шорт | В пределах ±hook-sec начала | "
            "В пределах ±hook-sec `source_span[0]` (для reordered) |"
        )
        lines.append("|---|---|---|---|")
        for h in metrics["hooks"]:
            span = "—"
            if "span_hook" in h:
                span = "да" if h["span_hook"] else "нет"
            lines.append(
                f"| {h['episode']} | {h['short']} | " f"{'да' if h['hook'] else 'нет'} | {span} |"
            )

    lines.append("\n## Precision (доля клипов, которые что-то нашли)\n")
    lines.append(
        f"{precision['precise']} из {precision['total']} клипов "
        f"({_pct(precision['precise'], precision['total'])}) находят шорт "
        f"или совпадают с оценкой keep/edit."
    )

    if metrics["labels"]:
        lines.append("\n## Оценки (Labels)\n")
        lines.append(
            "| Эпизод | Совпали с reject (ошибки) | Совпали с keep (хорошие) | " "Совпали с edit |"
        )
        lines.append("|---|---|---|---|")
        for episode_id, ep_labels in metrics["labels"].items():
            lines.append(
                f"| {episode_id} | {_ids_cell(ep_labels['reject'])} | "
                f"{_ids_cell(ep_labels['keep'])} | "
                f"{_ids_cell(ep_labels['edit'])} |"
            )

    lines.append("\n## По эпизодам\n")
    lines.append(
        "| Эпизод | Клипов | Медиана длительности | keep | edit | reject | pending | "
        "keep среди оценённых |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for episode_id, ep in metrics["episodes"].items():
        counts = ep["review_counts"]
        keep_share = _pct(ep["keep_share"], 1) if ep["keep_share"] is not None else "—"
        lines.append(
            f"| {episode_id} | {ep['clip_count']} | {_fmt_duration(ep['median_duration'])} | "
            f"{counts['keep']} | {counts['edit']} | {counts['reject']} | {counts['pending']} | "
            f"{keep_share} |"
        )

    if against_run is not None and against_metrics is not None:
        lines.append("\n## Сравнение с прогоном `%s`\n" % against_run["label"])
        a_hit = against_metrics["hit"]
        a_prec = against_metrics["precision"]
        before_pp = _pct_ratio(a_prec["precise"], a_prec["total"])
        after_pp = _pct_ratio(precision["precise"], precision["total"])
        prec_delta = (
            f"{after_pp - before_pp:+.0f} п.п."
            if before_pp is not None and after_pp is not None
            else "—"
        )
        lines.append("| Метрика | %s | %s | Δ |" % (against_run["label"], label))
        lines.append("|---|---|---|---|")
        rows = [
            (
                "Hit (найдено/всего)",
                f"{a_hit['found']}/{a_hit['total']}",
                f"{hit['found']}/{hit['total']}",
                f"{hit['found'] - a_hit['found']:+d}",
            ),
            (
                "Precision",
                f"{_pct(a_prec['precise'], a_prec['total'])}",
                f"{_pct(precision['precise'], precision['total'])}",
                prec_delta,
            ),
            (
                "Всего клипов",
                str(a_prec["total"]),
                str(precision["total"]),
                f"{precision['total'] - a_prec['total']:+d}",
            ),
        ]
        for name, before, after, delta in rows:
            lines.append(f"| {name} | {before} | {after} | {delta} |")

        lines.append("\n| Эпизод | Шорт | Покрытие до | Покрытие после |")
        lines.append("|---|---|---|---|")
        before_by_short = against_metrics["short_coverage"]
        after_by_short = metrics["short_coverage"]
        for key in sorted(set(before_by_short) | set(after_by_short)):
            episode_id, short_id = key.split("/", 1)
            lines.append(
                f"| {episode_id} | {short_id} | "
                f"{before_by_short.get(key, 0.0):.2f} | {after_by_short.get(key, 0.0):.2f} |"
            )

    return "\n".join(lines) + "\n"


def report_evaluation(
    api: EvalApi,
    set_data: Dict[str, Any],
    label: str,
    against: Optional[str],
    hit_threshold: float,
    hook_sec: float,
) -> str:
    """Fetch review statuses, compute metrics and write ``eval/reports/<label>.md``."""
    run = load_run(label)
    _fetch_reviews(api, run)
    _write_json(run_path(label), run)

    against_run = None
    against_metrics = None
    if against:
        against_run = load_run(against)
        against_metrics = compute_metrics(against_run, set_data, hit_threshold, hook_sec)

    metrics = compute_metrics(run, set_data, hit_threshold, hook_sec)
    report = render_report(run, set_data, metrics, against_run, against_metrics)
    path = REPORTS_DIR / f"{label}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report


def _rated_clips(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    rated: List[Dict[str, Any]] = []
    for episode_id, entry in run.get("episodes", {}).items():
        reviews = entry.get("reviews", {})
        for clip in entry.get("clips", []):
            if reviews.get(clip["id"]) in RATED_STATUSES:
                rated.append(
                    {
                        "episode": episode_id,
                        "id": clip["id"],
                        "start": clip["start"],
                        "end": clip["end"],
                        "status": reviews[clip["id"]],
                    }
                )
    return rated


def carry_over(
    api: EvalApi,
    label: str,
    prev: str,
    iou_threshold: float,
    progress: Callable[[str], None] = print,
) -> None:
    """Copy the Review status of the best-matching rated clip of ``prev`` onto pending clips."""
    run = load_run(label)
    prev_run = load_run(prev)
    _fetch_reviews(api, run)

    pending: List[Dict[str, Any]] = []
    for episode_id, entry in run.get("episodes", {}).items():
        reviews = entry.get("reviews", {})
        for clip in entry.get("clips", []):
            if reviews.get(clip["id"], "pending") == "pending":
                pending.append({"episode": episode_id, **clip})

    rated = _rated_clips(prev_run)
    if not rated:
        raise EvaluateError(
            f"Run '{prev}' has no rated clips: run `report --set ... {prev}` first "
            "and rate its clips in Review."
        )

    matches = greedy_iou_matching(pending, rated, iou_threshold)
    for clip, other, iou in matches:
        episode_id = clip["episode"]
        job_id = run["episodes"][episode_id]["job_id"]
        api.set_review(
            job_id,
            clip["id"],
            other["status"],
            notes=f"carried over from {prev} (IoU {iou:.2f})",
        )
        run["episodes"][episode_id].setdefault("reviews", {})[clip["id"]] = other["status"]
        progress(f"{episode_id}/{clip['id']}: {other['status']} (IoU {iou:.2f})")

    _write_json(run_path(label), run)
    carried = len(matches)
    left = len(pending) - carried
    progress(f"Carried over {carried} review(s); {left} clip(s) left to rate.")


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


def _add_url(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help=f"base URL of the service (default: {default_url()})",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="process every episode and record the run")
    run_p.add_argument("--set", required=True, type=Path, metavar="PATH/manifest.json")
    run_p.add_argument("--label", required=True, help="name of the run, e.g. baseline")
    run_p.add_argument("--only", nargs="*", metavar="EPISODE", help="only these episodes")
    run_p.add_argument("--overrides", type=Path, metavar="FILE.json", help="config overrides")
    run_p.add_argument("--verify-sha", action="store_true", help="also check sha256 (slow)")
    _add_url(run_p)

    report_p = sub.add_parser("report", help="compute metrics and write a report")
    report_p.add_argument("--set", required=True, type=Path, metavar="PATH/manifest.json")
    report_p.add_argument("name", help="label of the run to report")
    report_p.add_argument("--against", type=str, help="label of a previous run to compare")
    report_p.add_argument("--hit", type=float, default=DEFAULT_HIT_THRESHOLD)
    report_p.add_argument("--hook-sec", type=float, default=DEFAULT_HOOK_SEC)
    _add_url(report_p)

    co_p = sub.add_parser("carry-over", help="copy review statuses from a previous run")
    co_p.add_argument("name", help="label of the run whose pending clips get rated")
    co_p.add_argument("--from", dest="prev", required=True, help="label of the rated run")
    co_p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    _add_url(co_p)

    args = parser.parse_args(argv)
    try:
        if getattr(args, "hit", None) is not None and not 0.0 <= args.hit <= 1.0:
            raise EvaluateError(f"--hit must be between 0 and 1, got {args.hit}")
        if getattr(args, "iou", None) is not None and not 0.0 <= args.iou <= 1.0:
            raise EvaluateError(f"--iou must be between 0 and 1, got {args.iou}")
        if getattr(args, "hook_sec", None) is not None and args.hook_sec < 0:
            raise EvaluateError(f"--hook-sec must be >= 0, got {args.hook_sec}")
        url = args.url or default_url()
        api = EvalApi(url)
        password = _env_value("API_PASSWORD")

        if args.command == "run":
            set_data = load_set(args.set)
            episodes = set_data.get("episodes") or {}
            if args.only:
                missing = set(args.only) - set(episodes)
                if missing:
                    raise EvaluateError(f"Unknown episode(s): {sorted(missing)}")
                ordered = [e for e in episodes if e in set(args.only)]
            else:
                ordered = list(episodes)
            if not ordered:
                raise EvaluateError(f"No episodes in {args.set}")
            connect(api, password)
            overrides = _load_overrides(args.overrides) if args.overrides else None
            run_evaluation(
                api,
                args.set,
                set_data,
                ordered,
                args.label,
                overrides,
                args.verify_sha,
            )
        elif args.command == "report":
            connect(api, password)
            print(
                report_evaluation(
                    api, load_set(args.set), args.name, args.against, args.hit, args.hook_sec
                )
            )
        elif args.command == "carry-over":
            connect(api, password)
            carry_over(api, args.name, args.prev, args.iou)
    except KeyboardInterrupt:
        print(
            "\nInterrupted; jobs are left running and the run file is resumable.", file=sys.stderr
        )
        return 130
    except (EvaluateError, ApiError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
