#!/usr/bin/env bash
# Start, stop and inspect the Docker installation (docker-compose.yml).
#
# Usage: scripts/stack.sh up|stop|status [--yes]
#   up      start the web UI, API and job worker. Says so when this copy
#           already runs; offers to stop another copy; refuses while this
#           copy's worker runs outside Docker or another program has the port
#   stop    stop the containers (a running job goes back to the queue and
#           continues on the next start) and this copy's dev processes
#           (scripts/dev.sh, just worker), then report what still runs
#   status  the containers, address and version; dev processes; other copies
#   --yes   take the default answer instead of asking
#
# `just up`, `just stop` and `just status` run this script.
set -uo pipefail

# shellcheck source=scripts/docker_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"
cd "$ROOT" || exit 1

usage() {
    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# The worker service this copy starts (docker-compose.yml: worker-<profile>).
worker_service() {
    echo "worker-$(env_value COMPOSE_PROFILES "$(env_value GPU_BACKEND cpu)")"
}

print_other_copies() {
    local copies=$1 project dir port version where
    while IFS=$'\t' read -r project dir port; do
        where="web UI not running"
        if [[ -n "$port" ]]; then
            version=$(app_version "$port")
            where="http://127.0.0.1:$port, version ${version:-unknown}"
        fi
        echo "    $dir ($where)"
    done <<<"$copies"
}

# stop_other_copies: another copy would run jobs on the same GPU at the same
# time (and usually wants the same port). Offer to stop it.
stop_other_copies() {
    local copies project dir port ids
    copies=$(other_copies)
    [[ -z "$copies" ]] && return 0
    warn "Another copy of AI Video Clips Pipeline is running:"
    print_other_copies "$copies"
    echo "  Two copies would run jobs on the same GPU at once."
    if ! confirm "Stop it and start this copy?" N; then
        die "Not started. Stop the other copy first (in its directory: just stop)."
    fi
    while IFS=$'\t' read -r project dir port; do
        ids=$(docker ps -q --filter "label=com.docker.compose.project=$project")
        # 30 s like stop_grace_period: the worker puts its job back in the queue.
        # shellcheck disable=SC2086
        [[ -n "$ids" ]] && docker stop -t 30 $ids >/dev/null
        info "Stopped the copy in $dir."
    done <<<"$copies"
}

port_busy_error() {
    local port=$1 owner version
    version=$(app_version "$port")
    owner=$(port_owner "$port")
    if [[ -n "$version" ]]; then
        error "Port $port already serves AI Video Clips Pipeline $version, not from this copy's containers."
    else
        error "Port $port is taken by another program${owner:+ ($owner)}."
    fi
    die "Stop it, or set a free WEB_PORT in $ENV_FILE and run just up again."
}

refuse_native_worker() {
    local native
    native=$(native_processes | awk '$2 == "worker"')
    [[ -z "$native" ]] && return 0
    error "A job worker of this copy runs outside Docker (just dev or just worker):"
    sed 's/^/    PID /' <<<"$native" >&2
    die "Two workers would take jobs from the same database and share the GPU." \
        "Stop it (Ctrl+C in its terminal, or: just stop) and run just up again."
}

# service_ids: "service container-id" of this copy's running containers.
service_ids() {
    docker compose "${ALL_PROFILES[@]}" ps --status running --format '{{.Service}} {{.ID}}' \
        2>/dev/null | sort
}

wait_for_web() {
    local port=$1 timeout_sec=${STACK_START_TIMEOUT:-180} version="" waited=0
    while ((waited < timeout_sec)); do
        version=$(app_version "$port")
        [[ -n "$version" ]] && break
        sleep 2
        waited=$((waited + 2))
    done
    echo "$version"
}

# update_running: this copy already runs. compose recreates only what changed
# since the start (an image rebuilt by ./setup.sh after git pull, settings in
# .env). Returns 1 when nothing changed.
update_running() {
    local before output changed
    before=$(service_ids)
    output=$(docker compose up -d 2>&1) || die "docker compose up failed: $output"
    changed=$(comm -13 <(echo "$before") <(service_ids) | cut -d' ' -f1 | paste -sd' ')
    [[ -n "$changed" ]] || return 1
    info "Restarted with the new image or settings: $changed" \
        "(a running job continues from its last stage)."
}

# start_stack: this copy does not run (or only partly).
start_stack() {
    local backend=$1 worker=$2 port=$3 running=$4 service workers
    stop_other_copies
    # A worker of another GPU backend (COMPOSE_PROFILES changed since the last
    # start) would take jobs too.
    mapfile -t workers < <(grep '^worker-' <<<"$running")
    for service in "${workers[@]}"; do
        if [[ "$service" != "$worker" ]]; then
            info "Stopping $service (GPU backend changed to $backend)"
            docker compose "${ALL_PROFILES[@]}" stop "$service" >/dev/null
        fi
    done
    if ! grep -qx frontend <<<"$running" && port_in_use "$port"; then
        port_busy_error "$port"
    fi
    info "Starting the web UI, API and job worker ($worker)..."
    docker compose up -d || die "docker compose up failed (see above)."
}

cmd_up() {
    check_docker
    [[ -f "$ENV_FILE" ]] || die "No .env yet: run ./setup.sh first."
    local backend worker port running version
    backend=$(env_value GPU_BACKEND cpu)
    worker=$(worker_service)
    port=$(env_value WEB_PORT 8080)
    docker image inspect "clips-pipeline:$backend" >/dev/null 2>&1 ||
        die "The image clips-pipeline:$backend is not built yet: run ./setup.sh"

    refuse_native_worker
    running=$(running_services)
    if grep -qx frontend <<<"$running" && grep -qx backend <<<"$running" &&
        grep -qx "$worker" <<<"$running"; then
        if ! update_running; then
            version=$(app_version "$port")
            info "Already running: $(web_url) (version ${version:-unknown}). Stop it with: just stop"
            return 0
        fi
    else
        start_stack "$backend" "$worker" "$port" "$running"
    fi

    version=$(wait_for_web "$port")
    if [[ -z "$version" ]]; then
        docker compose ps
        show_startup_errors "$worker"
        die "The web UI does not answer on $(web_url). See: docker compose logs backend frontend"
    fi
    if ! grep -qx "$worker" <<<"$(running_services)"; then
        die "The job worker stopped right after the start. See: docker compose logs $worker"
    fi
    info "Running: $(web_url) (version $version, GPU backend $backend)"
    if [[ "$(env_value BIND_ADDRESS 127.0.0.1)" == "0.0.0.0" ]]; then
        info "Other devices on your network: http://<this computer's address>:$port"
    fi
    info "Stop it with: just stop"
}

# show_startup_errors WORKER: the last errors the API and the worker logged,
# so a failed start says why (a job database from a newer version, a broken
# config) instead of only pointing at the logs.
show_startup_errors() {
    local errors
    # Unique lines: the API and the worker often fail on the same thing.
    errors=$(docker compose logs --no-log-prefix --tail 200 backend "$1" 2>/dev/null |
        grep -E 'Error|error' | grep -vE '^[[:space:]]*raise ' | awk '!seen[$0]++' | tail -n 3)
    if [[ -n "$errors" ]]; then
        error "Last errors in the logs:"
        echo "$errors" >&2
    fi
    # restart: on-failure would restart a service that fails at start forever.
    if [[ -n "$(docker compose "${ALL_PROFILES[@]}" ps --status restarting -q 2>/dev/null)" ]]; then
        docker compose "${ALL_PROFILES[@]}" stop >/dev/null 2>&1
        warn "The services kept failing at start; they are stopped. Fix the error, then: just up"
    fi
}

report_leftovers() {
    local copies native
    if docker_available; then
        copies=$(other_copies)
        if [[ -n "$copies" ]]; then
            warn "Another copy is running:"
            print_other_copies "$copies"
            echo "  Stop it in its directory: just stop"
        fi
    fi
    native=$(native_processes)
    if [[ -n "$native" ]]; then
        warn "Dev processes of this copy still run:"
        sed 's/^/    PID /' <<<"$native"
        echo "  Stop them with: just stop"
    fi
}

stop_containers() {
    if [[ -z "$(running_services)" ]]; then
        info "The Docker installation of this copy is not running."
        return
    fi
    info "Stopping the containers (a running job goes back to the queue)..."
    docker compose "${ALL_PROFILES[@]}" stop || die "docker compose stop failed (see above)."
    [[ -z "$(running_services)" ]] || die "Some containers still run: docker compose ps"
    info "Stopped. Unfinished jobs continue from their last stage on the next start (just up)."
}

cmd_stop() {
    # Dev mode (install.sh) works without Docker; stop what there is.
    if docker_available; then
        stop_containers
    elif [[ -f "$ENV_FILE" ]] && command -v docker >/dev/null 2>&1; then
        warn "Docker does not answer: containers not checked."
    fi
    stop_dev_processes
    report_leftovers
}

print_containers() {
    local running version port
    port=$(env_value WEB_PORT 8080)
    if [[ -z "$(docker compose "${ALL_PROFILES[@]}" ps --all -q 2>/dev/null)" ]]; then
        echo "  no containers yet (start: just up)"
    else
        docker compose "${ALL_PROFILES[@]}" ps --all --format 'table {{.Service}}\t{{.Status}}' |
            sed 's/^/  /'
    fi
    running=$(running_services)
    if grep -qx frontend <<<"$running"; then
        version=$(app_version "$port")
        echo "Web UI: $(web_url) (version ${version:-not answering yet})"
    else
        echo "Web UI: not running (start: just up)"
    fi
}

cmd_status() {
    local native copies
    echo "Copy: $ROOT"
    if docker_available; then
        print_containers
    else
        echo "  Docker: not available"
    fi
    native=$(native_processes)
    if [[ -n "$native" ]]; then
        echo "Dev mode (just dev, stop with just stop):"
        sed 's/^/    PID /' <<<"$native"
    fi
    if docker_available; then
        copies=$(other_copies)
        if [[ -n "$copies" ]]; then
            warn "Another copy is running:"
            print_other_copies "$copies"
        fi
    fi
}

main() {
    local command=${1:-}
    shift || true
    for arg in "$@"; do
        case "$arg" in
            --yes | -y) ASSUME_YES=true ;;
            *)
                usage >&2
                exit 2 ;;
        esac
    done
    case "$command" in
        up) cmd_up ;;
        stop) cmd_stop ;;
        status) cmd_status ;;
        -h | --help) usage ;;
        *)
            usage >&2
            exit 2 ;;
    esac
}

main "$@"
