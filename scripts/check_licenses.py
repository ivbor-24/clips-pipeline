#!/usr/bin/env python3
"""
Check that THIRD_PARTY_LICENSES.md covers every dependency and that no
dependency brings a copyleft license into the project (MIT).

- Python: every package in uv.lock (all extras: cpu, cuda, openvino, test,
  dev) has a row in the "Пакеты Python" table, or is one of the NVIDIA
  packages named in the file; every row is still in uv.lock; no row's license
  is a GPL variant.
- Web: every package in web/package-lock.json has a license from the allowed
  list (MPL-2.0 is allowed for build tools only: dev packages never reach the
  bundle).

Standard library only (CI runs it before installing anything).

Usage:
    python3 scripts/check_licenses.py [--root DIR]
    just licenses
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

PROJECT_NAME = "scientific-clips-pipeline"
DOC = "THIRD_PARTY_LICENSES.md"
PYTHON_SECTION = "## Пакеты Python"
WEB_SECTION = "## Веб-интерфейс"
# Licenses of web packages that may end up in the bundle.
WEB_ALLOWED = {
    "MIT",
    "ISC",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "Apache-2.0",
    "0BSD",
    "BlueOak-1.0.0",
    "CC0-1.0",
    "CC-BY-4.0",
    "OFL-1.1",
    "Python-2.0",
    "Unlicense",
}
WEB_DEV_ALLOWED = WEB_ALLOWED | {"MPL-2.0"}
COPYLEFT = re.compile(r"\b(A|L)?GPL", re.IGNORECASE)


def normalize(name: str) -> str:
    """PEP 503 name: lower case, runs of -_. become one -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_packages(uv_lock: str) -> Set[str]:
    """Package names in uv.lock (without tomllib: CI lints on Python 3.10)."""
    names = re.findall(r'^\[\[package\]\]\nname = "([^"]+)"', uv_lock, re.MULTILINE)
    return {normalize(n) for n in names} - {PROJECT_NAME}


def documented_python(doc: str) -> Dict[str, str]:
    """Rows of the Python table (name -> license), plus the NVIDIA packages."""
    start = doc.index(PYTHON_SECTION)
    end = doc.index(WEB_SECTION, start)
    rows = {
        normalize(m.group(1)): m.group(2).strip()
        for m in re.finditer(
            r"^\| ([A-Za-z0-9_.-]+) \| [^|]+ \| ([^|]+) \|$", doc[start:end], re.MULTILINE
        )
    }
    # The CUDA libraries are listed in a paragraph (proprietary, NVIDIA terms).
    for name in re.findall(r"`(nvidia-[a-z0-9-]+)`", doc):
        rows.setdefault(normalize(name), "NVIDIA")
    return rows


def check_python(uv_lock: str, doc: str) -> List[str]:
    locked = lock_packages(uv_lock)
    rows = documented_python(doc)
    problems = [
        f"Python: {name} (uv.lock) has no row in {DOC}: check its license and add it"
        for name in sorted(locked - set(rows))
    ]
    problems += [
        f"Python: {name} is in {DOC} but no longer in uv.lock: remove the row"
        for name in sorted(set(rows) - locked)
        if not name.startswith("nvidia-")
    ]
    problems += [
        f"Python: {name} is {license_} — copyleft does not fit the MIT project"
        for name, license_ in sorted(rows.items())
        if name in locked and COPYLEFT.search(license_)
    ]
    return problems


def allowed(expression: str, permitted: Set[str]) -> bool:
    """An SPDX expression such as "(MIT OR CC0-1.0)" fits the permitted set."""
    expression = expression.strip().strip("()")
    if " OR " in expression:
        return any(allowed(part, permitted) for part in expression.split(" OR "))
    if " AND " in expression:
        return all(allowed(part, permitted) for part in expression.split(" AND "))
    return expression in permitted


def check_web(package_lock: dict) -> List[str]:
    problems = []
    for path, info in sorted(package_lock.get("packages", {}).items()):
        if not path:  # the web interface itself
            continue
        name = path.rsplit("node_modules/", 1)[-1]
        license_ = info.get("license")
        permitted = WEB_DEV_ALLOWED if info.get("dev") else WEB_ALLOWED
        if not isinstance(license_, str) or not allowed(license_, permitted):
            kind = "build tool" if info.get("dev") else "bundled"
            problems.append(
                f"Web: {name} ({kind}) has license {license_!r}, not in the allowed list"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    root = Path(parser.parse_args().root)
    doc = (root / DOC).read_text(encoding="utf-8")
    problems = check_python((root / "uv.lock").read_text(encoding="utf-8"), doc)
    problems += check_web(json.loads((root / "web/package-lock.json").read_text(encoding="utf-8")))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} license problem(s); see {DOC}, 'Как обновлять этот файл'.")
        return 1
    print("Licenses: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
