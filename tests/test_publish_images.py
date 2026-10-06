"""Tests for scripts/publish_images.sh and scripts/image_sources.sh.

Both run in a throwaway git repository with fake docker and curl first in
PATH; git is real (a bare repository stands in for origin).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")

# What `dpkg-query` in an image prints through the script's loop: package,
# version, source package, source version, "copyleft" or empty.
PACKAGES = (
    "libc6\t2.41-12\tglibc\t2.41-12\tcopyleft\n"
    "libc-bin\t2.41-12\tglibc\t2.41-12\tcopyleft\n"
    "ffmpeg\t7:7.1.2-1\tffmpeg\t7:7.1.2-1\tcopyleft\n"
    "libssl3t64\t3.5.7-1\topenssl\t3.5.7-1\t\n"
)

FAKE_DOCKER = r"""#!/usr/bin/env bash
# One line per call: the scripts pass multi-line shell code to `docker run`.
echo "docker $*" | tr '\n' ' ' >>"$STATE/docker.log"
echo >>"$STATE/docker.log"
case " $* " in
    *" run "*" -v "*":/out "*)
        # image_sources.sh's download container: one file per source package,
        # or MISSING for those listed in $STATE/missing.
        out=$(echo "$*" | sed -n 's/.* -v \([^ ]*\):\/out .*/\1/p')
        missing=0
        while read -r sv; do
            if grep -qx "$sv" "$STATE/missing" 2>/dev/null; then
                echo "MISSING $sv" >&2; missing=1
            else
                touch "$out/sources/${sv%%=*}.dsc"
            fi
        done <"$out/sources.txt"
        exit "$missing" ;;
    *" run "*"dpkg-query"*) cat "$STATE/packages" ;;
    *" run "*) exit "${FAKE_RUN_RC:-0}" ;;
    *" build "*|*" tag "*|*" push "*|*" logout "*|*" image rm "*) ;;
    *" login "*) cat >"$STATE/login-password" ;;
    *) echo "unexpected docker call: $*" >&2; exit 99 ;;
esac
"""

FAKE_CURL = r"""#!/usr/bin/env bash
echo "curl $*" >>"$STATE/curl.log"
case "$*" in
    *"/releases?per_page=100") echo '[{"id": 41, "tag_name": "v1.0.0"}, {"id": 42, "tag_name": "v1.2.3", "draft": true}]' ;;
    *"/releases/42/assets") echo '[{"id": 7, "name": "other.tar"}]' ;;
    *) ;;
esac
"""


def write(path: Path, text: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def env(tmp_path):
    project = tmp_path / "project"
    for name in ("scripts/publish_images.sh", "scripts/image_sources.sh"):
        (project / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, project / name)
    write(project / "src/__init__.py", '__version__ = "1.2.3"\n')
    write(project / "Dockerfile.backend", "FROM scratch\n")
    write(project / "Dockerfile.frontend", "FROM scratch\n")
    write(project / ".gitignore", "dist/\n")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    git(project, "init", "-q", "-b", "main")
    git(project, "config", "user.email", "test@example.com")
    git(project, "config", "user.name", "Test")
    git(project, "add", "-A")
    git(project, "commit", "-q", "-m", "release")
    git(project, "tag", "v1.2.3")
    git(project, "remote", "add", "origin", str(origin))

    state = tmp_path / "state"
    bin_dir = tmp_path / "bin"
    write(state / "packages", PACKAGES)
    write(bin_dir / "docker", FAKE_DOCKER, 0o755)
    write(bin_dir / "curl", FAKE_CURL, 0o755)
    variables = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "STATE": str(state),
        # The token comes from git's credential helper.
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "credential.helper",
        "GIT_CONFIG_VALUE_0": "!f() { echo username=ivbor-24; echo password=tok-123; }; f",
    }
    return {"project": project, "state": state, "origin": origin, "vars": variables}


def publish(env, *args, **extra_vars):
    return subprocess.run(
        [BASH, "scripts/publish_images.sh", *args],
        cwd=env["project"],
        env={**env["vars"], **extra_vars},
        capture_output=True,
        text=True,
        timeout=60,
    )


def log(env, name="docker.log") -> str:
    path = env["state"] / name
    return path.read_text() if path.exists() else ""


class TestRefuses:
    def test_a_checkout_with_changes(self, env):
        (env["project"] / "README.md").write_text("draft\n")

        result = publish(env)

        assert result.returncode == 1
        assert "has changes" in result.stderr
        assert log(env) == ""

    def test_head_without_a_release_tag(self, env):
        (env["project"] / "notes.txt").write_text("x\n")
        git(env["project"], "add", "-A")
        git(env["project"], "commit", "-q", "-m", "after the release")

        result = publish(env)

        assert result.returncode == 1
        assert "not a release tag" in result.stderr

    def test_tag_and_version_disagree(self, env):
        git(env["project"], "tag", "-d", "v1.2.3")
        git(env["project"], "tag", "v1.2.4")

        result = publish(env)

        assert result.returncode == 1
        assert "src/__init__.py says 1.2.3" in result.stderr


class TestBuild:
    def test_portable_images_with_labels_and_the_sources_archive(self, env):
        result = publish(env, "--backends", "openvino cuda")

        assert result.returncode == 0, result.stdout + result.stderr
        calls = log(env)
        sha = git(env["project"], "rev-parse", "HEAD")
        for backend in ("openvino", "cuda"):
            assert f"--build-arg GPU_BACKEND={backend} --build-arg CPU_TARGET=portable" in calls
            assert f"-t ghcr.io/ivbor-24/clips-pipeline:1.2.3-{backend}" in calls
        assert "-t ghcr.io/ivbor-24/clips-pipeline-web:1.2.3" in calls
        assert f"org.opencontainers.image.revision={sha}" in calls
        assert "org.opencontainers.image.version=1.2.3" in calls
        assert "check_gpu.py --backend cuda --libraries-only" in calls
        assert "login" not in calls and "push" not in calls

        out = env["project"] / "dist/release-1.2.3"
        assert (out / "sources.txt").read_text().split() == [
            "ffmpeg=7:7.1.2-1",
            "glibc=2.41-12",
        ]
        archive = out / "clips-pipeline-1.2.3-sources.tar"
        names = subprocess.run(
            ["tar", "-tf", str(archive)], check=True, capture_output=True, text=True
        ).stdout.split()
        assert "README.txt" in names
        assert "sources/glibc.dsc" in names
        assert "sources/openssl.dsc" not in names  # not copyleft
        assert "Built and checked; nothing published" in result.stdout

    def test_a_failed_check_stops_before_publishing(self, env):
        result = publish(env, "--push", "--backends", "cpu", FAKE_RUN_RC="1")

        assert result.returncode != 0
        assert "push" not in log(env)


class TestPush:
    def test_pushes_one_image_at_a_time_and_uploads_the_sources(self, env):
        result = publish(env, "--push", "--backends", "openvino cuda")

        assert result.returncode == 0, result.stdout + result.stderr
        calls = log(env).splitlines()
        login = next(i for i, c in enumerate(calls) if " login " in c)
        pushes = [c for c in calls if c.startswith("docker push ")]
        assert [c.split()[-1] for c in pushes] == [
            "ghcr.io/ivbor-24/clips-pipeline-web:1.2.3",
            "ghcr.io/ivbor-24/clips-pipeline-web:latest",
            "ghcr.io/ivbor-24/clips-pipeline:1.2.3-openvino",
            "ghcr.io/ivbor-24/clips-pipeline:openvino",
            "ghcr.io/ivbor-24/clips-pipeline:1.2.3-cuda",
            "ghcr.io/ivbor-24/clips-pipeline:cuda",
        ]
        assert login < calls.index(pushes[0])
        # Each image leaves the disk before the next one is built.
        removed = next(
            i
            for i, c in enumerate(calls)
            if "image rm ghcr.io/ivbor-24/clips-pipeline:1.2.3-openvino" in c
        )
        cuda_build = next(i for i, c in enumerate(calls) if "GPU_BACKEND=cuda" in c)
        assert calls.index(pushes[3]) < removed < cuda_build
        assert calls[-1].startswith("docker logout ghcr.io")
        # The token reaches docker on stdin, never on the command line.
        assert (env["state"] / "login-password").read_text().strip() == "tok-123"
        assert "tok-123" not in log(env)

        curl = log(env, "curl.log")
        assert "releases/42/assets?name=clips-pipeline-1.2.3-sources.tar" in curl
        assert "-X DELETE" not in curl  # no asset of that name yet
        # Moving the release branch is a separate step after the release is out.
        assert (
            subprocess.run(
                ["git", "rev-parse", "--verify", "-q", "release"],
                cwd=env["origin"],
                capture_output=True,
            ).returncode
            != 0
        )
        assert "Next: publish the release" in result.stdout

    def test_a_draft_release_is_found(self, env):
        result = publish(env, "--push", "--backends", "cpu")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "releases/42/assets?name=" in log(env, "curl.log")

    def test_checkout_of_another_directory(self, env, tmp_path):
        tools = tmp_path / "tools/scripts"
        tools.mkdir(parents=True)
        for name in ("publish_images.sh", "image_sources.sh"):
            shutil.copy2(env["project"] / "scripts" / name, tools / name)

        result = subprocess.run(
            [
                BASH,
                str(tools / "publish_images.sh"),
                "--backends",
                "cpu",
                "--checkout",
                str(env["project"]),
            ],
            cwd=tmp_path,
            env=env["vars"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert (env["project"] / "dist/release-1.2.3/clips-pipeline-1.2.3-sources.tar").exists()

    def test_logs_out_when_a_push_fails(self, env):
        fake = (env["state"].parent / "bin/docker").read_text()
        (env["state"].parent / "bin/docker").write_text(
            fake.replace('*" push "*|', "").replace(
                '*" login "*)', '*" push "*) exit 1 ;;\n    *" login "*)'
            )
        )

        result = publish(env, "--push", "--backends", "cpu")

        assert result.returncode != 0
        assert log(env).splitlines()[-1].startswith("docker logout ghcr.io")
        assert (
            subprocess.run(
                ["git", "rev-parse", "--verify", "-q", "release"],
                cwd=env["origin"],
                capture_output=True,
            ).returncode
            != 0
        )


class TestImageSources:
    def run(self, env, *args):
        return subprocess.run(
            [BASH, "scripts/image_sources.sh", *map(str, args)],
            cwd=env["project"],
            env=env["vars"],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_lists_packages_and_downloads_the_copyleft_sources(self, env, tmp_path):
        out = tmp_path / "out"

        listed = self.run(env, "list", out, "repo/backend:1")
        listed_too = self.run(env, "list", out, "repo/web:1")
        result = self.run(env, "fetch", out)

        assert listed.returncode == listed_too.returncode == result.returncode == 0, result.stderr
        assert (out / "packages-repo_backend_1.txt").read_text() == PACKAGES
        assert (out / "sources.txt").read_text() == "ffmpeg=7:7.1.2-1\nglibc=2.41-12\n"
        assert sorted(p.name for p in (out / "sources").iterdir()) == ["ffmpeg.dsc", "glibc.dsc"]
        assert "repo/backend:1: 4 packages, 3 copyleft" in listed.stdout

    def test_a_missing_source_package_fails(self, env, tmp_path):
        write(env["state"] / "missing", "glibc=2.41-12\n")

        self.run(env, "list", tmp_path / "out", "repo/backend:1")
        result = self.run(env, "fetch", tmp_path / "out")

        assert result.returncode == 1
        assert "MISSING glibc=2.41-12" in result.stderr
