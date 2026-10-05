"""The web client calls only paths the API serves.

The chapters and b-roll panels once asked for /api/v1/artifacts/... while the
router lives under /api/v1/jobs, and nothing failed until someone opened the page.
"""

import re
from pathlib import Path

import pytest

from src.api.app import app

WEB_SRC = Path(__file__).resolve().parent.parent / "web" / "src"
API_PREFIX = "/api/v1"  # API_BASE_URL in web/src/lib/api.ts

# api.get('/jobs/'), api.put(`/jobs/${jobId}/clips/${clipId}`, ...)
AXIOS_CALL = re.compile(r"\bapi\.(get|post|put|patch|delete)(?:<[^>(]*>)?\(\s*[`'\"]([^`'\"?]+)")
# EventSource and <video> URLs: `${API_BASE_URL}/jobs/${jobId}/progress`
BASE_URL_GET = re.compile(r"\$\{API_BASE_URL\}(/[^`'\"?]*)")


def web_calls() -> list[tuple[str, str, str]]:
    """(method, API path, file:line) for every literal API path in web/src."""
    calls = []
    for source in sorted(WEB_SRC.rglob("*.ts*")):
        for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            where = f"{source.relative_to(WEB_SRC)}:{lineno}"
            for method, path in AXIOS_CALL.findall(line):
                calls.append((method.upper(), API_PREFIX + path, where))
            for path in BASE_URL_GET.findall(line):
                calls.append(("GET", API_PREFIX + path, where))
    return calls


def served(method: str, path: str) -> bool:
    concrete = re.sub(r"\$\{[^}]+\}", "1", path)
    # The OpenAPI schema lists every route; app.routes hides included routers.
    for route_path, operations in app.openapi()["paths"].items():
        pattern = re.sub(r"\{[^}]+\}", "[^/]+", route_path)
        if method.lower() in operations and re.fullmatch(pattern, concrete):
            return True
    return False


def test_web_client_calls_are_found():
    """Guard the scanner itself: an empty list would pass the test below vacuously."""
    paths = {path for _, path, _ in web_calls()}
    assert "/api/v1/jobs/${jobId}/chapters" in paths
    assert "/api/v1/jobs/${jobId}/progress" in paths


@pytest.mark.parametrize("method,path,where", web_calls())
def test_web_client_path_is_served(method, path, where):
    assert served(method, path), f"{where}: {method} {path} has no API route"
