# shellcheck shell=bash
# Helpers shared by setup.sh and scripts/stack.sh (the Docker installation).
# Source it; it defines functions and variables only.
#
# Environment (used by the tests):
#   STACK_ROOT  project directory (default: the parent of this script's directory)

ROOT="${STACK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="$ROOT/.env"

# Every worker profile of docker-compose.yml: `ps` and `stop` must see the
# containers of all of them, not only the one COMPOSE_PROFILES enables now.
ALL_PROFILES=(--profile cpu --profile cuda --profile openvino)
# The label Dockerfile.backend puts on the image, inherited by the API and
# worker containers of every copy of the project.
APP_LABEL="org.opencontainers.image.title=scientific-clips-pipeline"

# shellcheck disable=SC2034  # C_CYAN is used by setup.sh
if [[ -t 1 ]]; then
    C_RED=$'\033[0;31m' C_GREEN=$'\033[0;32m' C_YELLOW=$'\033[1;33m' C_CYAN=$'\033[0;36m'
    C_OFF=$'\033[0m'
else
    C_RED="" C_GREEN="" C_YELLOW="" C_CYAN="" C_OFF=""
fi

info() { echo "${C_GREEN}[INFO]${C_OFF} $*"; }
warn() { echo "${C_YELLOW}[WARN]${C_OFF} $*"; }
error() { echo "${C_RED}[ERROR]${C_OFF} $*" >&2; }
die() {
    error "$@"
    exit 1
}

# confirm QUESTION Y|N: ask a yes/no question. With ASSUME_YES=true or without
# a terminal the default answer is taken without asking.
ASSUME_YES=${ASSUME_YES:-false}
confirm() {
    local question=$1 default=$2 answer choices="y/N"
    [[ "$default" == Y ]] && choices="Y/n"
    if [[ "$ASSUME_YES" == true || ! -t 0 ]]; then
        [[ "$default" == Y ]]
        return
    fi
    read -rp "$question [$choices] " answer
    answer=${answer:-$default}
    [[ "${answer,,}" =~ ^(y|yes)$ ]]
}

# ---- Docker -----------------------------------------------------------------

# docker_available: docker is installed and answers (dev mode works without it).
docker_available() {
    command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

# check_docker: stop with what to do when docker is missing or does not answer.
check_docker() {
    local err
    command -v docker >/dev/null 2>&1 || die "Docker is not installed. $(docker_install_hint)"
    if ! err=$(docker info 2>&1 >/dev/null); then
        case "$err" in
            *"permission denied"*)
                die "Your user may not use Docker. Add it to the docker group:" \
                    "sudo usermod -aG docker \$USER, then log out and back in." ;;
            *"Cannot connect"* | *"Is the docker daemon running"*)
                die "The Docker service is not running. Start it: sudo systemctl enable --now docker" ;;
            *)
                die "Docker does not answer: $err" ;;
        esac
    fi
}

docker_install_hint() {
    if command -v apt-get >/dev/null 2>&1; then
        echo "Install it: sudo apt install docker.io docker-compose-v2 docker-buildx"
    elif command -v pacman >/dev/null 2>&1; then
        echo "Install it: sudo pacman -S docker docker-compose docker-buildx"
    elif command -v dnf >/dev/null 2>&1; then
        echo "Install it: sudo dnf install moby-engine docker-compose docker-buildx"
    else
        echo "Install Docker Engine: https://docs.docker.com/engine/install/"
    fi
}

# running_services: services of this copy whose containers run now.
running_services() {
    (cd "$ROOT" && docker compose "${ALL_PROFILES[@]}" ps --status running --format '{{.Service}}') \
        2>/dev/null
}

# other_copies: "project<TAB>directory<TAB>port" of every other copy of the
# project (a compose project in another directory) that has running
# containers. The port is the one its web UI is published on, empty if its
# web UI does not run.
other_copies() {
    local project dir port
    docker ps --filter "label=$APP_LABEL" --filter "label=com.docker.compose.project" \
        --format '{{.Label "com.docker.compose.project"}}\t{{.Label "com.docker.compose.project.working_dir"}}' \
        2>/dev/null | sort -u | while IFS=$'\t' read -r project dir; do
        [[ "$dir" == "$ROOT" ]] && continue
        port=$(docker ps --filter "label=com.docker.compose.project=$project" \
            --format '{{.Ports}}' | grep -o '[0-9]*->80/tcp' | head -n 1 | cut -d- -f1)
        printf '%s\t%s\t%s\n' "$project" "$dir" "$port"
    done
}

# native_processes: dev processes of this copy running on the host rather than
# in Docker (just dev, just serve, just worker, just dev-web), as
# "PID SERVICE command" lines; SERVICE is api, worker (with its jobs) or web.
native_processes() {
    local pid cwd self_ns argv i kind
    self_ns=$(readlink /proc/self/ns/mnt)
    for pid in $(pgrep -f 'src\.worker|src\.api\.app|vite' 2>/dev/null); do
        # Processes in containers show up here too; they live in another
        # mount namespace.
        [[ "$(readlink "/proc/$pid/ns/mnt" 2>/dev/null)" == "$self_ns" ]] || continue
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
        [[ "$cwd" == "$ROOT" || "$cwd" == "$ROOT/"* ]] || continue
        # Whole arguments, not a substring: a shell or an editor whose command
        # line merely mentions src.worker is not a worker.
        mapfile -d '' -t argv <"/proc/$pid/cmdline" 2>/dev/null || continue
        kind=""
        for ((i = 1; i < ${#argv[@]}; i++)); do
            case "${argv[i]}" in
                src.worker) [[ "${argv[i - 1]}" == "-m" ]] && kind=worker ;;
                src.api.app:app) kind=api ;;
                */.bin/vite | */vite/bin/vite.js) kind=web ;;
            esac
        done
        [[ -n "$kind" ]] && echo "$pid $kind ${argv[*]}"
    done
    return 0
}

# ---- Dev processes (scripts/dev.sh) -----------------------------------------

# dev.sh runs each service in its own process group and records it here.
RUN_DIR="$ROOT/.run"

# service_grace SERVICE: seconds to wait after SIGTERM. The worker puts its
# running job back in the queue first (like stop_grace_period in compose).
service_grace() {
    if [[ "$1" == worker ]]; then echo 30; else echo 10; fi
}

group_alive() { kill -0 -- "-$1" 2>/dev/null; }

# group_here PGID: a process of the group runs in this copy. A PGID from a
# stale .run file may belong to someone else's processes after a reboot.
group_here() {
    local pid cwd
    for pid in $(pgrep -g "$1" 2>/dev/null); do
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null) || continue
        [[ "$cwd" == "$ROOT" || "$cwd" == "$ROOT/"* ]] && return 0
    done
    return 1
}

# stop_group PGID GRACE: SIGTERM to every process of the group (uvicorn's
# reloader and server, npm and vite), SIGKILL to what is left after GRACE s.
stop_group() {
    local pgid=$1 grace=$2 waited=0
    kill -TERM -- "-$pgid" 2>/dev/null || return 0
    while group_alive "$pgid" && ((waited < grace * 10)); do
        sleep 0.1
        waited=$((waited + 1))
    done
    if group_alive "$pgid"; then kill -KILL -- "-$pgid" 2>/dev/null; fi
    return 0
}

# stop_process PID GRACE: the same for a process started some other way. A
# job process of the worker leads its own group (ffmpeg, whisper-cli in it).
stop_process() {
    local pid=$1 grace=$2 target=$1 waited=0
    [[ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$pid" ]] && target="-$pid"
    kill -TERM -- "$target" 2>/dev/null || return 0
    while kill -0 "$pid" 2>/dev/null && ((waited < grace * 10)); do
        sleep 0.1
        waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then kill -KILL -- "$target" 2>/dev/null; fi
    return 0
}

# live_dev_groups: "service pgid" of the groups dev.sh recorded that still run
# here; stale records are removed.
live_dev_groups() {
    local file service pgid
    for file in "$RUN_DIR"/*.pgid; do
        [[ -e "$file" ]] || continue
        service=$(basename "$file" .pgid)
        pgid=$(cat "$file" 2>/dev/null)
        if [[ "$pgid" =~ ^[0-9]+$ ]] && group_alive "$pgid" && group_here "$pgid"; then
            echo "$service $pgid"
        else
            rm -f "$file"
        fi
    done
    return 0
}

# stop_dev_processes: stop this copy's dev processes: the groups dev.sh
# started, children included, then any started by hand (just worker of an
# older version, python -m src.worker).
stop_dev_processes() {
    local service pgid pid rest groups
    # The whole list first: once one service stops, dev.sh stops the others.
    groups=$(live_dev_groups)
    while read -r service pgid; do
        [[ -n "$service" ]] || continue
        info "Stopping $service (dev mode, process group $pgid)..."
        stop_group "$pgid" "$(service_grace "$service")"
        rm -f "$RUN_DIR/$service.pgid"
    done <<<"$groups"
    while read -r pid service rest; do
        [[ -n "$pid" ]] || continue
        info "Stopping $service (PID $pid: $rest)..."
        stop_process "$pid" "$(service_grace "$service")"
    done < <(native_processes)
}

# ---- .env -------------------------------------------------------------------

# env_value KEY [DEFAULT]: the value docker compose would use. A real
# environment variable wins over .env, as in compose; quotes are removed.
env_value() {
    local key=$1 default=${2:-} value=""
    # printenv: only exported variables, as compose sees them (not the
    # scripts' own shell variables).
    if value=$(printenv "$key"); then
        :
    elif [[ -f "$ENV_FILE" ]]; then
        value=$(grep -E "^${key}=" "$ENV_FILE" | tail -n 1 | cut -d= -f2-) || true
        if [[ "$value" =~ ^\'(.*)\'$ || "$value" =~ ^\"(.*)\"$ ]]; then
            value=${BASH_REMATCH[1]}
        fi
    fi
    echo "${value:-$default}"
}

# set_env KEY VALUE: replace the KEY= line in .env (keeping its comments) or
# append one. Values are written single-quoted when they contain anything but
# plain characters, so compose does not expand `$` in a password.
set_env() {
    local key=$1 value=$2 line tmp
    if [[ "$value" =~ ^[A-Za-z0-9_./:@+-]*$ ]]; then
        line="$key=$value"
    else
        line="$key='$value'"
    fi
    tmp="$ENV_FILE.tmp.$$"
    if grep -qE "^${key}=" "$ENV_FILE"; then
        # awk keeps the value verbatim (sed would interpret & and \ in it).
        LINE=$line awk -v key="$key" 'index($0, key "=") == 1 { print ENVIRON["LINE"]; next } 1' \
            "$ENV_FILE" >"$tmp"
    else
        cat "$ENV_FILE" >"$tmp"
        # A last line without a newline would swallow the new one.
        [[ -s "$ENV_FILE" && -n "$(tail -c 1 "$ENV_FILE")" ]] && echo >>"$tmp"
        printf '%s\n' "$line" >>"$tmp"
    fi
    # Keep the file's permissions (setup.sh makes it private with a password).
    chmod --reference="$ENV_FILE" "$tmp" 2>/dev/null || true
    mv "$tmp" "$ENV_FILE"
}

# ---- Network ----------------------------------------------------------------

# port_in_use PORT: something on this computer listens on the TCP port.
port_in_use() {
    local port=$1
    if command -v ss >/dev/null 2>&1; then
        [[ -n "$(ss -Hltn "sport = :$port" 2>/dev/null)" ]]
    else
        timeout 2 bash -c "exec 3<>/dev/tcp/127.0.0.1/$port" 2>/dev/null
    fi
}

# port_owner PORT: the listening process as ss shows it (the process name only
# for processes of this user), empty if unknown.
port_owner() {
    command -v ss >/dev/null 2>&1 || return 0
    ss -Hltnp "sport = :$1" 2>/dev/null | head -n 1 | sed -nE 's/.*users:\(\((.*)\)\).*/\1/p'
    return 0
}

# http_get HOST PORT PATH: the response body of a plain HTTP GET (no curl
# needed; the host may not have it).
http_get() {
    # shellcheck disable=SC2016  # expanded by the inner bash
    timeout 5 bash -c '
        exec 3<>"/dev/tcp/$1/$2" || exit 1
        printf "GET %s HTTP/1.0\r\nHost: localhost\r\n\r\n" "$3" >&3
        sed "1,/^\r\{0,1\}$/d" <&3
    ' _ "$1" "$2" "$3" 2>/dev/null
}

# app_version PORT: the version the web UI on this port reports, empty when
# no copy of this project answers there.
app_version() {
    http_get 127.0.0.1 "$1" /api/v1/health | sed -n 's/.*"version" *: *"\([^"]*\)".*/\1/p'
    return 0
}

# web_url: where the web UI of this copy is reached from this computer.
web_url() {
    local bind port
    bind=$(env_value BIND_ADDRESS 127.0.0.1)
    port=$(env_value WEB_PORT 8080)
    [[ "$bind" == "0.0.0.0" ]] && bind=127.0.0.1
    echo "http://$bind:$port"
}
