"""A fake `docker` command for the tests of setup.sh and scripts/stack.sh.

It logs every call to $FAKE_STATE/log and answers from files in $FAKE_STATE,
so a test sets up the "system" by writing files:

- ``running``: services of this copy that run, one per line (`compose up -d`
  fills it, `compose stop` empties it);
- ``other_copies``: "project<TAB>dir" lines of other copies' containers;
- ``check_gpu``: output of scripts/check_gpu.py in the worker container;
- ``llm_status``: output of ``scripts/llm_provider.py status`` (default: the
  local model, not downloaded yet);
- ``logs``: what `compose logs` prints;
- ``restarting``: containers in a restart loop (`ps --status restarting`);
- ``published``: published images, "<ref> <commit>" per line
  (`buildx imagetools inspect`).

Environment knobs: FAKE_DOCKER_INFO_ERR (docker info fails with it),
FAKE_RUNTIMES, FAKE_COMPOSE_VERSION, FAKE_NO_BUILDX, FAKE_IMAGE_MISSING,
FAKE_NET_BRIDGE / FAKE_NET_HOST (exit code of the internet probe),
FAKE_CHECK_GPU_RC, FAKE_WEB_PORT (`compose up -d` starts a fake web UI there),
FAKE_RECREATE (`compose up -d` recreates running containers, as after a rebuild),
FAKE_LLM_CHECK_FAIL (`llm_provider.py check` fails with this message),
FAKE_PULL_FAIL (pulling a ghcr.io image fails).
"""

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

FAKE_DOCKER = r"""#!/usr/bin/env bash
state=$FAKE_STATE
echo "docker $*" >>"$state/log"
args=" $* "
case "$args" in
    " info "*"--format"*"DockerRootDir"*) echo "$state" ;;
    " info "*"--format"*"Runtimes"*) echo "${FAKE_RUNTIMES:-runc}" ;;
    " info "*)
        if [[ -n "${FAKE_DOCKER_INFO_ERR:-}" ]]; then echo "$FAKE_DOCKER_INFO_ERR" >&2; exit 1; fi ;;
    " version "*) echo "27.0.3" ;;
    " compose version "*) echo "${FAKE_COMPOSE_VERSION:-2.29.1}" ;;
    " buildx version "*) [[ -z "${FAKE_NO_BUILDX:-}" ]] ;;
    " buildx imagetools inspect "*)
        # Published images: "<ref> <commit>" lines in $FAKE_STATE/published.
        ref=$(echo "$*" | awk '{print $4}')
        sha=$(awk -v r="$ref" '$1 == r {print $2}' "$state/published" 2>/dev/null)
        [[ -n "$sha" ]] || { echo "ERROR: not found: $ref" >&2; exit 1; }
        # Indented like the real output ("key": "value").
        printf '{\n  "config": {\n    "Labels": {\n      "org.opencontainers.image.revision": "%s"\n    }\n  }\n}\n' "$sha" ;;
    " pull "*ghcr.io*) [[ -z "${FAKE_PULL_FAIL:-}" ]] ;;
    " tag "*) ;;
    " image inspect "*) [[ -z "${FAKE_IMAGE_MISSING:-}" ]] ;;
    " pull "*) ;;
    " run "*"--network host"*) exit "${FAKE_NET_HOST:-0}" ;;
    " run "*) exit "${FAKE_NET_BRIDGE:-0}" ;;
    *" compose"*" run "*"check_gpu.py"*)
        cat "$state/check_gpu" 2>/dev/null
        exit "${FAKE_CHECK_GPU_RC:-0}" ;;
    *" compose"*" run "*"llm_provider.py status"*)
        if [[ -f "$state/llm_status" ]]; then
            cat "$state/llm_status"
        else
            printf 'LLM_ENABLED=yes\nLLM_PROVIDER=llama_cpp\nLLM_MODEL=Qwen/Qwen3-14B\nLLM_API_BASE=\nLLM_READY=no\n'
        fi ;;
    *" compose"*" run "*"llm_provider.py check"*)
        if [[ -n "${FAKE_LLM_CHECK_FAIL:-}" ]]; then
            echo "$FAKE_LLM_CHECK_FAIL" >&2
            exit 1
        fi
        echo "the model answered in 0.4 s" ;;
    *" compose"*" run "*) ;;
    *" compose"*" build "*)
        # FAKE_BUILD_FAIL: fail every build with a non-OOM error.
        if [[ -n "${FAKE_BUILD_FAIL:-}" ]]; then
            echo "target backend: failed to solve: some compiler error" >&2
            exit 1
        fi
        # FAKE_BUILD_137=N: fail the next N builds with an OOM kill (exit 137
        # printed by buildx; `docker compose build` itself exits non-zero).
        if [[ -n "${FAKE_BUILD_137:-}" ]]; then
            n=$(cat "$state/build_attempts" 2>/dev/null || echo 0)
            n=$((n + 1))
            echo "$n" >"$state/build_attempts"
            if [[ "$n" -le "${FAKE_BUILD_137}" ]]; then
                echo "target backend: failed to solve: process did not complete successfully: exit code: 137" >&2
                exit 1
            fi
        fi ;;
    *" compose"*" up -d "*)
        # New containers: none ran yet, or FAKE_RECREATE (a rebuilt image).
        if [[ ! -s "$state/running" || -n "${FAKE_RECREATE:-}" ]]; then
            echo $(( $(cat "$state/generation" 2>/dev/null || echo 0) + 1 )) >"$state/generation"
        fi
        printf 'backend\nfrontend\nworker-%s\n' "${COMPOSE_PROFILES:-cpu}" >"$state/running"
        if [[ -n "${FAKE_WEB_PORT:-}" ]]; then
            python3 "$state/web.py" "$FAKE_WEB_PORT" >/dev/null 2>&1 &
            echo $! >"$state/web.pid"
            sleep 0.5
        fi ;;
    *" compose"*" stop"*) : >"$state/running" ;;
    *" compose"*" logs "*) cat "$state/logs" 2>/dev/null ;;
    *" compose"*" ps "*"--status running"*"{{.ID}}"*)
        sed "s/\$/ id$(cat "$state/generation" 2>/dev/null)/" "$state/running" 2>/dev/null ;;
    *" compose"*" ps "*"--status running"*) cat "$state/running" 2>/dev/null ;;
    *" compose"*" ps "*"--status restarting"*) cat "$state/restarting" 2>/dev/null ;;
    *" compose"*" ps "*"-q"*) [[ -s "$state/running" ]] && echo 0123456789ab ;;
    *" compose"*" ps "*) sed 's/$/  Up 1 minute/' "$state/running" 2>/dev/null ;;
    " ps "*"com.docker.compose.project.working_dir"*) cat "$state/other_copies" 2>/dev/null ;;
    " ps "*"{{.Ports}}"*) echo "127.0.0.1:8099->80/tcp" ;;
    " ps "*) echo 0123456789ab ;;
    " stop "*) ;;
    *) echo "fake docker: unexpected call: $*" >&2; exit 99 ;;
esac
"""

# The API's health endpoint, as the web UI (nginx) serves it.
FAKE_WEB = r"""
import http.server, sys

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"status":"ok","version":"9.9.9"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Health).serve_forever()
"""


# What the scripts may run besides docker. PATH holds only these (linked from
# the real system) and the fakes, so a real docker, nvidia-smi or just is
# never reached.
SYSTEM_TOOLS = (
    "bash cat grep sed awk cut tail head tr base64 stat df find mkdir chmod mv cp rm mktemp "
    "tee sort comm paste id printenv readlink basename dirname sleep timeout ss pgrep ps python3 "
    "env"
).split()


def make_fake_docker(tmp_path: Path) -> tuple:
    """Create the fake docker in tmp_path/bin; returns (bin_dir, state_dir).

    Use PATH=bin_dir: it also holds links to SYSTEM_TOOLS.
    """
    bin_dir = tmp_path / "bin"
    state = tmp_path / "state"
    bin_dir.mkdir()
    state.mkdir()
    for tool in SYSTEM_TOOLS:
        real = shutil.which(tool)
        if real:
            # Absolute: PATH may hold relative entries (.venv/bin).
            (bin_dir / tool).symlink_to(Path(real).absolute())
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    (state / "web.py").write_text(FAKE_WEB)
    (state / "log").write_text("")
    return bin_dir, state


def add_command(bin_dir: Path, name: str, script: str) -> None:
    path = bin_dir / name
    path.write_text("#!/usr/bin/env bash\n" + script)
    path.chmod(0o755)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stop_fake_web(state: Path) -> None:
    pid_file = state / "web.pid"
    if pid_file.exists():
        try:
            os.kill(int(pid_file.read_text()), 15)
        except (ProcessLookupError, ValueError):
            pass


def start_fake_web(state: Path, port: int) -> subprocess.Popen:
    """A web UI that already runs (for the "already running" cases)."""
    import time

    proc = subprocess.Popen([sys.executable, str(state / "web.py"), str(port)])
    for _ in range(50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    return proc
