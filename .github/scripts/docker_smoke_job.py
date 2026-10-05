#!/usr/bin/env python3
"""Run one small job through the web UI of a Docker installation (CI smoke test).

Uploads a video through nginx, creates a job with the given config profile,
waits for it to finish and checks that it rendered a vertical clip, owned by
the user who runs the check (the containers run as HOST_UID).

Usage: docker_smoke_job.py VIDEO [--url http://127.0.0.1:8080] [--profile ci_smoke]
Only the standard library: it runs on the CI host, outside the project's venv.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DONE = {"completed", "failed", "cancelled"}


def request(url: str, data: bytes = None, content_type: str = "application/json") -> dict:
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--profile", default="ci_smoke", help='"" for the default settings')
    parser.add_argument("--timeout", type=int, default=900, help="seconds to wait for the job")
    args = parser.parse_args()
    api = f"{args.url}/api/v1"

    upload = request(
        f"{api}/upload/?filename={args.video.name}",
        args.video.read_bytes(),
        "application/octet-stream",
    )
    print(f"uploaded: {upload['path']} ({upload['size']} bytes)")

    job = request(
        f"{api}/jobs/",
        json.dumps(
            {
                "input_source": upload["path"],
                "config_overrides": {"profile": args.profile} if args.profile else {},
            }
        ).encode(),
    )
    print(f"job {job['id']}: {job['status']}")

    deadline = time.monotonic() + args.timeout
    status = job["status"]
    while status not in DONE:
        if time.monotonic() > deadline:
            print(f"job {job['id']} did not finish in {args.timeout} s")
            return 1
        time.sleep(5)
        job = request(f"{api}/jobs/{job['id']}")
        if job["status"] != status:
            status = job["status"]
            print(f"job {job['id']}: {status}")
    if status != "completed":
        print(f"job {job['id']} {status}: {job.get('error_message')}")
        return 1

    manifest = request(f"{api}/jobs/{job['id']}/manifest")
    print(f"manifest: {json.dumps(manifest)[:500]}")
    clips = sorted(Path(job["work_dir"]).glob("output/clips/clip_*.mp4"))
    if not clips:
        print(f"no clips in {job['work_dir']}/output/clips")
        return 1
    for clip in clips:
        if clip.stat().st_uid != os.getuid():
            print(f"{clip} belongs to uid {clip.stat().st_uid}, not {os.getuid()}")
            return 1
        size = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", str(clip)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        print(f"{clip}: {size}")
        if size != "1080,1920":
            print(f"{clip} is not 1080x1920")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
