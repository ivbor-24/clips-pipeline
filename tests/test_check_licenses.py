"""Tests for scripts/check_licenses.py: THIRD_PARTY_LICENSES.md against the lock files."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO / "scripts" / "check_licenses.py"

DOC = """# Сторонние лицензии

## Пакеты Python

| Пакет | Версия | Лицензия |
|---|---|---|
| fastapi | 0.115.0 | MIT |
| llama_cpp_python | 0.3.34 | MIT |

Пакеты `nvidia-*`: `nvidia-cublas-cu12`.

## Веб-интерфейс
"""


def uv_lock(*names: str) -> str:
    return "".join(f'[[package]]\nname = "{n}"\nversion = "1.0"\n\n' for n in names)


@pytest.fixture
def check():
    spec = importlib.util.spec_from_file_location("check_licenses_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPython:
    def test_documented_packages_pass(self, check):
        lock = uv_lock("fastapi", "llama-cpp-python", "nvidia-cublas-cu12", check.PROJECT_NAME)

        assert check.check_python(lock, DOC) == []

    def test_new_dependency_needs_a_row(self, check):
        lock = uv_lock("fastapi", "llama-cpp-python", "left-pad")

        problems = check.check_python(lock, DOC)

        assert len(problems) == 1
        assert "left-pad" in problems[0] and "no row" in problems[0]

    def test_removed_dependency_leaves_no_row(self, check):
        problems = check.check_python(uv_lock("fastapi"), DOC)

        assert len(problems) == 1
        assert "llama-cpp-python" in problems[0] and "no longer in uv.lock" in problems[0]

    @pytest.mark.parametrize("license_", ["GPL-3.0", "LGPL-3.0", "AGPL-3.0-only", "LGPLv3"])
    def test_copyleft_is_refused(self, check, license_):
        doc = DOC.replace("| fastapi | 0.115.0 | MIT |", f"| fastapi | 0.115.0 | {license_} |")

        problems = check.check_python(uv_lock("fastapi", "llama-cpp-python"), doc)

        assert len(problems) == 1
        assert "copyleft" in problems[0]


class TestWeb:
    def lock(self, **packages):
        return {"packages": {"": {"name": "web"}, **packages}}

    def test_permissive_licenses_pass(self, check):
        lock = self.lock(
            **{
                "node_modules/react": {"license": "MIT"},
                "node_modules/caniuse-lite": {"license": "CC-BY-4.0"},
                "node_modules/a/node_modules/b": {"license": "(MIT OR CC0-1.0)"},
                "node_modules/lightningcss": {"license": "MPL-2.0", "dev": True},
            }
        )

        assert check.check_web(lock) == []

    def test_mpl_is_not_allowed_in_the_bundle(self, check):
        lock = self.lock(**{"node_modules/x": {"license": "MPL-2.0"}})

        assert check.check_web(lock) == [
            "Web: x (bundled) has license 'MPL-2.0', not in the allowed list"
        ]

    @pytest.mark.parametrize("info", [{}, {"license": "GPL-3.0"}, {"license": "SEE LICENSE IN x"}])
    def test_missing_or_copyleft_license_fails(self, check, info):
        assert len(check.check_web(self.lock(**{"node_modules/x": {**info, "dev": True}}))) == 1

    def test_and_needs_every_part(self, check):
        lock = self.lock(**{"node_modules/x": {"license": "MIT AND GPL-2.0"}})

        assert len(check.check_web(lock)) == 1


def test_the_repository_is_consistent(check):
    """THIRD_PARTY_LICENSES.md matches uv.lock and web/package-lock.json."""
    assert check.check_python((REPO / "uv.lock").read_text(), (REPO / check.DOC).read_text()) == []
    assert check.check_web(json.loads((REPO / "web/package-lock.json").read_text())) == []
