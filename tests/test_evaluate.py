"""Tests for scripts/evaluate.py: evaluation against the reference set."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.api.models.job import JobStatus, JobType
from src.api.schemas.job import JobResponse
from src.api.schemas.review import ClipResponse, ReviewStatus

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def evaluate(tmp_path, monkeypatch):
    """Load scripts/evaluate.py with EVAL_ROOT pointed at a temp directory."""
    monkeypatch.setenv("EVAL_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        "evaluate_under_test", REPO / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeApi:
    """A stand-in for the running service; its payloads mirror the real API."""

    def __init__(self, module, password=None):
        self.module = module
        self.password = password
        self.token = None
        self.jobs = {}
        self._next_id = 1
        self.uploads = []
        self.created = []
        self.reviews = {}
        self.notes = {}
        self.final_status = "completed"
        self.error_message = None
        self.manifest_clips = []
        self.review_clips = []
        self.version = "0.0.0-test"
        self.config_data = {"scoring": {"strategy": "llm_windows"}}

    def health(self):
        return {"status": "ok", "version": self.version}

    def password_required(self):
        return self.password is not None

    def login(self, password):
        if self.password is None:
            raise self.module.EvaluateError("Login is off: API_PASSWORD is not set")
        if password != self.password:
            raise self.module.EvaluateError("Wrong API_PASSWORD")
        self.token = "session-token"

    def config(self):
        return {"config": self.config_data}

    def upload(self, filename, source_path, size):
        self.uploads.append((filename, str(source_path), size))
        return {
            "path": f"uploads/1/{len(self.uploads)}_{filename}",
            "filename": filename,
            "size": size,
        }

    def create_job(self, input_source, config_overrides=None):
        job_id = self._next_id
        self._next_id += 1
        self.created.append((input_source, config_overrides))
        self.jobs[job_id] = self._job(job_id, self.final_status)
        return self.jobs[job_id]

    def get_job(self, job_id):
        return self.jobs[job_id]

    def _job(self, job_id, status):
        return JobResponse(
            id=job_id,
            user_id=1,
            status=JobStatus(status),
            job_type=JobType.CLIPS,
            input_source="uploads/1/x.mp4",
            work_dir=f"jobs/job_{job_id}",
            created_at=datetime.now(timezone.utc),
            started_at=None,
            completed_at=None,
            error_message=self.error_message if status == "failed" else None,
        ).model_dump(mode="json")

    def manifest(self, job_id):
        return {"clips": self.manifest_clips, "total_clips": len(self.manifest_clips)}

    def clips(self, job_id):
        return {"clips": self.review_clips, "total": len(self.review_clips)}

    def set_review(self, job_id, clip_id, status, notes=None):
        self.reviews[(job_id, clip_id)] = status
        if notes is not None:
            self.notes[(job_id, clip_id)] = notes
        return {"id": clip_id, "review_status": status}


def _make_fake(module, **kwargs):
    return FakeApi(module, **kwargs)


def _manifest_clip(clip_id, start, end, title="t", score=0.8):
    return {
        "clip_id": clip_id,
        "start_time": start,
        "end_time": end,
        "duration_sec": end - start,
        "score": score,
        "title": title,
    }


def _review_clip(clip_id, start, end, status="pending"):
    return ClipResponse(
        id=clip_id,
        video_path=f"output/clips/{clip_id}.mp4",
        srt_path=None,
        score=0.8,
        duration=end - start,
        tags=[],
        review_status=ReviewStatus(status),
        review_notes=None,
        metadata={"clip_id": clip_id, "start_time": start, "end_time": end},
    ).model_dump(mode="json")


def _write_refset(tmp_path, size_ep1=1000, size_ep2=2000, path_ep1=None):
    ep1 = path_ep1 or tmp_path / "ep1.mp4"
    ep2 = tmp_path / "ep2.mp4"
    ep1.write_bytes(b"\x00" * size_ep1)
    ep2.write_bytes(b"\x00" * size_ep2)
    data = {
        "name": "synthetic",
        "schema_version": 1,
        "episodes": {
            "EP_1": {
                "path": str(ep1),
                "title": "Ep 1",
                "duration_sec": 100.0,
                "size_bytes": size_ep1,
                "sha256": "",
            },
            "EP_2": {
                "path": str(ep2),
                "title": "Ep 2",
                "duration_sec": 200.0,
                "size_bytes": size_ep2,
                "sha256": "",
            },
        },
        "shorts": [
            {
                "id": "EP_1/S_1",
                "episode": "EP_1",
                "pieces": [
                    {"source_start": 100, "source_end": 110, "short_start": 0, "short_end": 10},
                    {"source_start": 120, "source_end": 130, "short_start": 10, "short_end": 20},
                ],
                "structure": {"montage": False, "reordered": False, "source_span": [100, 130]},
            },
            {
                "id": "EP_1/S_2",
                "episode": "EP_1",
                "pieces": [
                    {"source_start": 200, "source_end": 210, "short_start": 10, "short_end": 20},
                    {"source_start": 150, "source_end": 160, "short_start": 0, "short_end": 10},
                ],
                "structure": {"montage": False, "reordered": True, "source_span": [150, 210]},
            },
            {
                "id": "EP_1/S_mont",
                "episode": "EP_1",
                "pieces": [
                    {"source_start": 10, "source_end": 20, "short_start": 0, "short_end": 10},
                    {"source_start": 5000, "source_end": 5010, "short_start": 10, "short_end": 20},
                ],
                "structure": {"montage": True, "reordered": True, "source_span": [10, 5010]},
            },
        ],
        "pipeline_clip_labels": [
            {
                "episode": "EP_2",
                "clip_id": "clip_001",
                "source_start": 100,
                "source_end": 160,
                "label": "keep",
            },
            {
                "episode": "EP_2",
                "clip_id": "clip_002",
                "source_start": 300,
                "source_end": 360,
                "label": "reject",
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_coverage_with_gaps(self, evaluate):
        pieces = [
            {"source_start": 0, "source_end": 10},
            {"source_start": 20, "source_end": 30},
        ]
        assert evaluate.coverage(0, 30, pieces) == 1.0
        assert evaluate.coverage(5, 25, pieces) == pytest.approx(0.5)
        assert evaluate.coverage(0, 5, pieces) == pytest.approx(0.25)
        assert evaluate.coverage(11, 19, pieces) == 0.0

    def test_coverage_zero_total(self, evaluate):
        assert evaluate.coverage(0, 10, [{"source_start": 5, "source_end": 5}]) == 0.0


class TestIoU:
    def test_iou(self, evaluate):
        assert evaluate.temporal_iou(0, 10, 5, 15) == pytest.approx(5 / 15)
        assert evaluate.temporal_iou(0, 10, 20, 30) == 0.0
        assert evaluate.temporal_iou(0, 10, 0, 10) == 1.0

    def test_iou_zero_union(self, evaluate):
        assert evaluate.temporal_iou(5, 5, 5, 5) == 0.0


class TestHitAndHook:
    def test_hit_threshold(self, evaluate):
        shorts = [{"id": "S", "episode": "E", "pieces": [{"source_start": 0, "source_end": 100}]}]
        clips = [{"id": "c", "start": 0, "end": 60}]
        found = evaluate.find_shorts(clips, shorts, hit_threshold=0.5)
        assert found["S"]["found"] is True
        found = evaluate.find_shorts(clips, shorts, hit_threshold=0.61)
        assert found["S"]["found"] is False

    def test_hit_zero_with_no_clips_does_not_find(self, evaluate):
        shorts = [{"id": "S", "episode": "E", "pieces": [{"source_start": 0, "source_end": 10}]}]
        found = evaluate.find_shorts([], shorts, hit_threshold=0.0)
        assert found["S"]["found"] is False
        assert found["S"]["clip"] is None

    def test_montage_excluded(self, evaluate):
        shorts = [
            {
                "id": "M",
                "episode": "E",
                "pieces": [{"source_start": 0, "source_end": 10}],
                "structure": {"montage": True},
            },
            {
                "id": "N",
                "episode": "E",
                "pieces": [{"source_start": 0, "source_end": 10}],
                "structure": {"montage": False},
            },
        ]
        assert [s["id"] for s in evaluate.non_montage_shorts(shorts)] == ["N"]

    def test_montage_excluded_from_metrics(self, evaluate):
        run = {
            "episodes": {
                "EP_1": {
                    "status": "completed",
                    "clips": [{"id": "c", "start": 0, "end": 10, "title": "t", "score": 0.8}],
                    "reviews": {"c": "keep"},
                }
            }
        }
        set_data = {
            "episodes": {"EP_1": {}},
            "shorts": [
                {
                    "id": "M",
                    "episode": "EP_1",
                    "pieces": [
                        {"source_start": 0, "source_end": 10, "short_start": 0, "short_end": 10}
                    ],
                    "structure": {"montage": True},
                },
                {
                    "id": "N",
                    "episode": "EP_1",
                    "pieces": [
                        {"source_start": 0, "source_end": 10, "short_start": 0, "short_end": 10}
                    ],
                    "structure": {"montage": False},
                },
            ],
            "pipeline_clip_labels": [],
        }
        metrics = evaluate.compute_metrics(run, set_data)
        assert metrics["hit"]["total"] == 1  # only the non-montage short counts

    def test_hook_reordered(self, evaluate):
        short = {
            "id": "S",
            "pieces": [
                {"source_start": 200, "source_end": 210, "short_start": 10},
                {"source_start": 150, "source_end": 160, "short_start": 0},
            ],
            "structure": {"reordered": True, "source_span": [150, 210]},
        }
        # Opening piece (smallest short_start) starts at 150; span[0] is also 150.
        assert evaluate.opening_piece(short)["source_start"] == 150
        assert evaluate.hook_ok({"start": 152}, short, hook_sec=3.0) is True
        assert evaluate.hook_ok({"start": 160}, short, hook_sec=3.0) is False
        assert evaluate.source_span_start(short) == 150


class TestPrecision:
    def test_clip_matches_label(self, evaluate):
        clip = {"start": 0, "end": 60}
        assert (
            evaluate.clip_matches_label(
                clip, {"source_start": 0, "source_end": 60, "label": "keep"}, 0.5
            )
            is True
        )
        assert (
            evaluate.clip_matches_label(
                clip, {"source_start": 61, "source_end": 100, "label": "keep"}, 0.5
            )
            is False
        )

    def test_precision_with_labels(self, evaluate):
        clips = [{"id": "c1", "start": 0, "end": 60}, {"id": "c2", "start": 100, "end": 160}]
        shorts = [
            {
                "id": "S",
                "episode": "EP_2",
                "pieces": [{"source_start": 0, "source_end": 60}],
                "structure": {"montage": False},
            }
        ]
        labels = [
            {
                "episode": "EP_2",
                "clip_id": "x",
                "source_start": 100,
                "source_end": 160,
                "label": "keep",
            }
        ]
        assert evaluate.clip_finds_short(clips[0], shorts, 0.5) is True
        assert evaluate.clip_finds_short(clips[1], shorts, 0.5) is False
        assert any(
            label["label"] in ("keep", "edit") and evaluate.clip_matches_label(clips[1], label, 0.5)
            for label in labels
        )

    def test_label_match_uses_iou_0_5_by_default(self, evaluate):
        run = {
            "episodes": {
                "EP_2": {
                    "status": "completed",
                    "clips": [{"id": "c1", "start": 100, "end": 160, "title": "t", "score": 0.8}],
                    "reviews": {},
                }
            }
        }
        set_data = {
            "episodes": {"EP_2": {}},
            "shorts": [],
            "pipeline_clip_labels": [
                {
                    "episode": "EP_2",
                    "clip_id": "x",
                    "source_start": 85,
                    "source_end": 145,
                    "label": "keep",
                }
            ],
        }
        # IoU([100,160], [85,145]) == 0.6: counts at the 0.5 label threshold,
        # but would be dropped at the 0.8 carry-over threshold.
        metrics = evaluate.compute_metrics(run, set_data)
        assert metrics["precision"]["precise"] == 1
        assert metrics["labels"]["EP_2"]["keep"] == ["c1"]


class TestGreedyMatching:
    def test_greedy_matching_one_to_one(self, evaluate):
        pending = [
            {"episode": "E", "id": "p1", "start": 0, "end": 60},
            {"episode": "E", "id": "p2", "start": 100, "end": 160},
        ]
        rated = [
            {"episode": "E", "id": "r1", "start": 0, "end": 60},
            {"episode": "E", "id": "r2", "start": 100, "end": 160},
        ]
        matches = evaluate.greedy_iou_matching(pending, rated, 0.5)
        assert [(p["id"], r["id"]) for p, r, _ in matches] == [("p1", "r1"), ("p2", "r2")]

    def test_greedy_matching_respects_episode_and_threshold(self, evaluate):
        pending = [{"episode": "A", "id": "p1", "start": 0, "end": 60}]
        rated = [{"episode": "B", "id": "r1", "start": 0, "end": 60}]
        assert evaluate.greedy_iou_matching(pending, rated, 0.5) == []
        assert (
            evaluate.greedy_iou_matching(
                [{"episode": "A", "id": "p1", "start": 0, "end": 60}],
                [{"episode": "A", "id": "r1", "start": 30, "end": 90}],
                0.99,
            )
            == []
        )

    def test_greedy_matching_accepts_iou_equal_to_threshold(self, evaluate):
        pending = [{"episode": "E", "id": "p1", "start": 0, "end": 60}]
        rated = [{"episode": "E", "id": "r1", "start": 0, "end": 60}]
        matches = evaluate.greedy_iou_matching(pending, rated, 1.0)
        assert [(p["id"], r["id"]) for p, r, _ in matches] == [("p1", "r1")]


# ---------------------------------------------------------------------------
# run / report / carry-over through the fake service
# ---------------------------------------------------------------------------


class TestConnect:
    def test_no_password(self, evaluate):
        api = _make_fake(evaluate)
        evaluate.connect(api, None)
        assert api.token is None

    def test_password_logs_in(self, evaluate):
        api = _make_fake(evaluate, password="secret")
        evaluate.connect(api, "secret")
        assert api.token == "session-token"

    def test_password_missing_raises(self, evaluate):
        api = _make_fake(evaluate, password="secret")
        with pytest.raises(evaluate.EvaluateError, match="API_PASSWORD"):
            evaluate.connect(api, None)

    def test_wrong_password_raises(self, evaluate):
        api = _make_fake(evaluate, password="secret")
        with pytest.raises(evaluate.EvaluateError, match="Wrong API_PASSWORD"):
            evaluate.connect(api, "nope")


class TestEvalApiLogin:
    def test_login_wrong_password(self, evaluate):
        api = evaluate.EvalApi("http://x")
        api._request = lambda *a, **k: (401, b'{"detail": "Wrong password"}')
        with pytest.raises(evaluate.EvaluateError, match="Wrong API_PASSWORD"):
            api.login("bad")

    def test_login_off(self, evaluate):
        api = evaluate.EvalApi("http://x")
        api._request = lambda *a, **k: (400, b'{"detail": "Login is off"}')
        with pytest.raises(evaluate.EvaluateError, match="Login is off"):
            api.login("x")


class TestRun:
    def test_happy_path(self, evaluate, tmp_path):
        _, data = _write_refset(tmp_path)
        api = _make_fake(evaluate)
        api.manifest_clips = [_manifest_clip("clip_001", 100, 130)]
        lines = []
        run = evaluate.run_evaluation(
            api,
            tmp_path / "manifest.json",
            data,
            ["EP_1"],
            "l1",
            None,
            False,
            progress=lines.append,
            sleep=lambda s: None,
            poll_interval=0,
        )
        assert run["episodes"]["EP_1"]["status"] == "completed"
        assert run["episodes"]["EP_1"]["clips"][0]["id"] == "clip_001"
        assert len(api.uploads) == 1 and len(api.created) == 1
        assert (evaluate.RUNS_DIR / "l1.json").is_file()

    def test_resume_skips_finished_jobs(self, evaluate, tmp_path):
        _, data = _write_refset(tmp_path)
        api = _make_fake(evaluate)
        api.manifest_clips = [_manifest_clip("clip_001", 100, 130)]
        evaluate.run_evaluation(
            api,
            tmp_path / "manifest.json",
            data,
            ["EP_1"],
            "l1",
            None,
            False,
            progress=lambda s: None,
            sleep=lambda s: None,
            poll_interval=0,
        )
        evaluate.run_evaluation(
            api,
            tmp_path / "manifest.json",
            data,
            ["EP_1"],
            "l1",
            None,
            False,
            progress=lambda s: None,
            sleep=lambda s: None,
            poll_interval=0,
        )
        assert len(api.created) == 1  # the second run skipped the finished episode

    def test_size_mismatch_fails_without_upload(self, evaluate, tmp_path):
        _, data = _write_refset(tmp_path)
        data["episodes"]["EP_1"]["size_bytes"] = 999  # file is 1000 bytes
        api = _make_fake(evaluate)
        lines = []
        run = evaluate.run_evaluation(
            api,
            tmp_path / "manifest.json",
            data,
            ["EP_1"],
            "l1",
            None,
            False,
            progress=lines.append,
            sleep=lambda s: None,
            poll_interval=0,
        )
        assert run["episodes"]["EP_1"]["status"] == "failed"
        assert "size mismatch" in run["episodes"]["EP_1"]["error"]
        assert api.uploads == [] and api.created == []

    def test_missing_source_fails(self, evaluate, tmp_path):
        _, data = _write_refset(tmp_path)
        data["episodes"]["EP_1"]["path"] = str(tmp_path / "gone.mp4")
        api = _make_fake(evaluate)
        run = evaluate.run_evaluation(
            api,
            tmp_path / "manifest.json",
            data,
            ["EP_1"],
            "l1",
            None,
            False,
            progress=lambda s: None,
            sleep=lambda s: None,
            poll_interval=0,
        )
        assert "not found" in run["episodes"]["EP_1"]["error"]
        assert api.uploads == []

    def test_verify_sha(self, evaluate, tmp_path):
        ep = tmp_path / "ep.mp4"
        ep.write_bytes(b"hello world")
        data = {
            "episodes": {
                "EP_1": {"path": str(ep), "size_bytes": 11, "sha256": evaluate._sha256(ep)}
            },
            "shorts": [],
            "pipeline_clip_labels": [],
        }
        api = _make_fake(evaluate)
        run = evaluate.run_evaluation(
            api,
            tmp_path / "manifest.json",
            data,
            ["EP_1"],
            "l1",
            None,
            True,
            progress=lambda s: None,
            sleep=lambda s: None,
            poll_interval=0,
        )
        assert run["episodes"]["EP_1"]["status"] == "completed"


class TestReport:
    def test_partial_run_warning(self, evaluate, tmp_path):
        run = {
            "format": 1,
            "label": "p",
            "created_at": "2026-01-01T00:00:00+00:00",
            "app_version": "0.0.0",
            "git_commit": None,
            "set": "manifest.json",
            "config": {},
            "episodes": {
                "EP_1": {
                    "job_id": 1,
                    "status": "failed",
                    "error": "boom",
                    "clips": [],
                    "reviews": {},
                }
            },
        }
        (evaluate.RUNS_DIR / "p.json").parent.mkdir(parents=True, exist_ok=True)
        (evaluate.RUNS_DIR / "p.json").write_text(json.dumps(run))
        set_data = {"episodes": {"EP_1": {}}, "shorts": [], "pipeline_clip_labels": []}
        api = _make_fake(evaluate)
        report = evaluate.report_evaluation(api, set_data, "p", None, 0.5, 3.0)
        assert "неполный" in report
        assert "boom" in report

    def test_report_counts_reviews(self, evaluate, tmp_path):
        run = {
            "format": 1,
            "label": "r",
            "created_at": "2026-01-01T00:00:00+00:00",
            "app_version": "0.0.0",
            "git_commit": None,
            "set": "manifest.json",
            "config": {},
            "episodes": {
                "EP_1": {
                    "job_id": 1,
                    "status": "completed",
                    "error": None,
                    "clips": [
                        {"id": "clip_001", "start": 100, "end": 130, "title": "t", "score": 0.8}
                    ],
                }
            },
        }
        (evaluate.RUNS_DIR / "r.json").parent.mkdir(parents=True, exist_ok=True)
        (evaluate.RUNS_DIR / "r.json").write_text(json.dumps(run))
        set_data = {
            "episodes": {"EP_1": {}},
            "shorts": [
                {
                    "id": "EP_1/S_1",
                    "episode": "EP_1",
                    "pieces": [
                        {"source_start": 100, "source_end": 130, "short_start": 0, "short_end": 30}
                    ],
                    "structure": {"montage": False, "reordered": False, "source_span": [100, 130]},
                }
            ],
            "pipeline_clip_labels": [],
        }
        api = _make_fake(evaluate)
        api.review_clips = [_review_clip("clip_001", 100, 130, "keep")]
        report = evaluate.report_evaluation(api, set_data, "r", None, 0.5, 3.0)
        assert "| **Всего** | **1** | **1** |" in report  # hit: 1 of 1 shorts found
        assert "| EP_1 | 1 | 30.0s | 1 |" in report  # 1 clip, 30 s, 1 keep review
        assert "keep" in report
        # Review statuses were stored back into the run file.
        saved = json.loads((evaluate.RUNS_DIR / "r.json").read_text())
        assert saved["episodes"]["EP_1"]["reviews"] == {"clip_001": "keep"}

    def test_report_against_compares_coverage(self, evaluate, tmp_path):
        run = {
            "format": 1,
            "label": "after",
            "created_at": "2026-01-01T00:00:00+00:00",
            "app_version": "0.0.0",
            "git_commit": None,
            "set": "manifest.json",
            "config": {},
            "episodes": {
                "EP_1": {
                    "job_id": 1,
                    "status": "completed",
                    "error": None,
                    "clips": [{"id": "c", "start": 100, "end": 160, "title": "t", "score": 0.8}],
                }
            },
        }
        prev = dict(run)
        prev["label"] = "before"
        prev["episodes"]["EP_1"]["clips"] = [
            {"id": "c", "start": 100, "end": 115, "title": "t", "score": 0.8}
        ]
        for label, data in (("after", run), ("before", prev)):
            (evaluate.RUNS_DIR / f"{label}.json").parent.mkdir(parents=True, exist_ok=True)
            (evaluate.RUNS_DIR / f"{label}.json").write_text(json.dumps(data))
        set_data = {
            "episodes": {"EP_1": {}},
            "shorts": [
                {
                    "id": "EP_1/S_1",
                    "episode": "EP_1",
                    "pieces": [
                        {"source_start": 100, "source_end": 160, "short_start": 0, "short_end": 60}
                    ],
                    "structure": {"montage": False, "reordered": False, "source_span": [100, 160]},
                }
            ],
            "pipeline_clip_labels": [],
        }
        api = _make_fake(evaluate)
        report = evaluate.report_evaluation(api, set_data, "after", "before", 0.5, 3.0)
        assert "Сравнение с прогоном `before`" in report
        assert "| EP_1 | S_1 |" in report  # episode and short id, not double-prefixed

    def test_only_episode_does_not_count_absent_set_episodes(self, evaluate):
        run = {
            "episodes": {
                "EP_1": {
                    "status": "completed",
                    "clips": [{"id": "c", "start": 100, "end": 130, "title": "t", "score": 0.8}],
                    "reviews": {},
                }
            }
        }
        set_data = {
            "episodes": {"EP_1": {}, "EP_2": {}},
            "shorts": [
                {
                    "id": "EP_1/S_1",
                    "episode": "EP_1",
                    "pieces": [
                        {"source_start": 100, "source_end": 130, "short_start": 0, "short_end": 30}
                    ],
                    "structure": {"montage": False},
                },
                {
                    "id": "EP_2/S_1",
                    "episode": "EP_2",
                    "pieces": [
                        {"source_start": 0, "source_end": 10, "short_start": 0, "short_end": 10}
                    ],
                    "structure": {"montage": False},
                },
            ],
            "pipeline_clip_labels": [],
        }
        metrics = evaluate.compute_metrics(run, set_data)
        assert metrics["hit"]["total"] == 1  # EP_2 is not in the run
        assert any("EP_2" in w for w in metrics["warnings"])


class TestCarryOver:
    def _write_run(self, evaluate, label, episodes):
        run = {
            "format": 1,
            "label": label,
            "created_at": "2026-01-01T00:00:00+00:00",
            "app_version": "0.0.0",
            "git_commit": None,
            "set": "manifest.json",
            "config": {},
            "episodes": episodes,
        }
        (evaluate.RUNS_DIR / f"{label}.json").parent.mkdir(parents=True, exist_ok=True)
        (evaluate.RUNS_DIR / f"{label}.json").write_text(json.dumps(run))

    def test_carry_over_pending_clips(self, evaluate, tmp_path):
        self._write_run(
            evaluate,
            "name",
            {
                "EP_1": {
                    "job_id": 1,
                    "status": "completed",
                    "error": None,
                    "clips": [
                        {"id": "clip_A", "start": 100, "end": 130, "title": "t", "score": 0.8},
                        {"id": "clip_B", "start": 300, "end": 360, "title": "t", "score": 0.8},
                    ],
                    "reviews": {"clip_A": "pending", "clip_B": "pending"},
                }
            },
        )
        self._write_run(
            evaluate,
            "prev",
            {
                "EP_1": {
                    "job_id": 9,
                    "status": "completed",
                    "error": None,
                    "clips": [
                        {"id": "clip_X", "start": 105, "end": 135, "title": "t", "score": 0.8},
                        {"id": "clip_Y", "start": 310, "end": 360, "title": "t", "score": 0.8},
                    ],
                    "reviews": {"clip_X": "keep", "clip_Y": "reject"},
                }
            },
        )
        api = _make_fake(evaluate)
        api.review_clips = [
            _review_clip("clip_A", 100, 130, "pending"),
            _review_clip("clip_B", 300, 360, "pending"),
        ]
        lines = []
        evaluate.carry_over(api, "name", "prev", 0.5, progress=lines.append)
        assert api.reviews[(1, "clip_A")] == "keep"
        assert api.reviews[(1, "clip_B")] == "reject"
        assert "carried over from prev (IoU 0.71)" in api.notes[(1, "clip_A")]
        assert any("Carried over 2" in line for line in lines)

    def test_carry_over_respects_threshold(self, evaluate, tmp_path):
        self._write_run(
            evaluate,
            "name",
            {
                "EP_1": {
                    "job_id": 1,
                    "status": "completed",
                    "error": None,
                    "clips": [
                        {"id": "clip_A", "start": 100, "end": 130, "title": "t", "score": 0.8}
                    ],
                    "reviews": {"clip_A": "pending"},
                }
            },
        )
        self._write_run(
            evaluate,
            "prev",
            {
                "EP_1": {
                    "job_id": 9,
                    "status": "completed",
                    "error": None,
                    "clips": [
                        {"id": "clip_X", "start": 105, "end": 135, "title": "t", "score": 0.8}
                    ],
                    "reviews": {"clip_X": "keep"},
                }
            },
        )
        api = _make_fake(evaluate)
        api.review_clips = [_review_clip("clip_A", 100, 130, "pending")]
        lines = []
        evaluate.carry_over(api, "name", "prev", 0.9, progress=lines.append)
        assert (1, "clip_A") not in api.reviews
        assert any("Carried over 0" in line for line in lines)

    def test_carry_over_only_pending(self, evaluate, tmp_path):
        self._write_run(
            evaluate,
            "name",
            {
                "EP_1": {
                    "job_id": 1,
                    "status": "completed",
                    "error": None,
                    "clips": [
                        {"id": "clip_A", "start": 100, "end": 130, "title": "t", "score": 0.8}
                    ],
                    "reviews": {"clip_A": "reject"},
                }
            },
        )
        self._write_run(
            evaluate,
            "prev",
            {
                "EP_1": {
                    "job_id": 9,
                    "status": "completed",
                    "error": None,
                    "clips": [
                        {"id": "clip_X", "start": 100, "end": 130, "title": "t", "score": 0.8}
                    ],
                    "reviews": {"clip_X": "keep"},
                }
            },
        )
        api = _make_fake(evaluate)
        api.review_clips = [_review_clip("clip_A", 100, 130, "reject")]
        lines = []
        evaluate.carry_over(api, "name", "prev", 0.5, progress=lines.append)
        assert (1, "clip_A") not in api.reviews  # already rated, not carried


class TestCli:
    def test_default_url(self, evaluate, tmp_path):
        assert evaluate.default_url() == "http://127.0.0.1:8080"
        (tmp_path / ".env").write_text("WEB_PORT=9000\n")
        assert evaluate.default_url() == "http://127.0.0.1:9000"

    def test_env_value_from_file(self, evaluate, tmp_path):
        (tmp_path / ".env").write_text("API_PASSWORD='hunter2'\n")
        assert evaluate._env_value("API_PASSWORD") == "hunter2"

    def test_main_run(self, evaluate, tmp_path, monkeypatch):
        set_path, _ = _write_refset(tmp_path)
        api = _make_fake(evaluate)
        api.manifest_clips = [_manifest_clip("clip_001", 100, 130)]
        monkeypatch.setattr(evaluate, "EvalApi", lambda url: api)
        monkeypatch.setenv("API_PASSWORD", "")
        code = evaluate.main(
            ["run", "--set", str(set_path), "--label", "l1", "--url", "http://fake"]
        )
        assert code == 0
        assert (evaluate.RUNS_DIR / "l1.json").is_file()
