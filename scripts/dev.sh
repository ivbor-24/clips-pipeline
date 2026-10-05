#!/usr/bin/env bash
# Development mode on this computer: the API (uvicorn --reload), the job worker
# and the web dev server (vite) of this copy. Ctrl+C stops all of them.
#
# Usage: scripts/dev.sh [api] [worker] [web]     (default: all three)
#   just dev = all three; just serve = api; just worker = worker; just dev-web = web
#
# Each service runs in its own process group, recorded in .run/<service>.pgid.
# Ctrl+C, closing the terminal or one of the services exiting stops every
# service this script started, children included (uvicorn's reloader and
# server, vite under npm). `just stop` stops them from another terminal.
#
# It does not start a service that already runs in this copy, a worker while
# this copy's Docker installation runs (two workers on one database), or a
# service whose port is taken.
#
# Environment:
#   DEV_API_HOST, DEV_API_PORT  where the API listens (default 127.0.0.1:8000)
#   DEV_API_CMD, DEV_WORKER_CMD, DEV_WEB_CMD, DEV_WEB_PORT  other commands and
#                               the web port to check (used by the tests)
set -uo pipefail

# shellcheck source=scripts/docker_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"
cd "$ROOT" || exit 1
# Run directly (not through just) the project's venv is not in PATH yet.
[[ -d "$ROOT/.venv/bin" ]] && PATH="$ROOT/.venv/bin:$PATH"

API_HOST=${DEV_API_HOST:-127.0.0.1}
API_PORT=${DEV_API_PORT:-8000}
# web/vite.config.ts (strictPort: a taken port is an error, not port 5174).
WEB_DEV_PORT=${DEV_WEB_PORT:-5173}
ALL_SERVICES=(api worker web)

usage() {
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

service_command() {
    case "$1" in
        api) echo "${DEV_API_CMD:-uvicorn src.api.app:app --host $API_HOST --port $API_PORT --reload}" ;;
        worker) echo "${DEV_WORKER_CMD:-python3 -m src.worker}" ;;
        web) echo "${DEV_WEB_CMD:-npm run dev}" ;;
    esac
}

service_port() {
    case "$1" in
        api) echo "$API_PORT" ;;
        web) echo "$WEB_DEV_PORT" ;;
    esac
}

# refuse SERVICE: stop with the reason when SERVICE must not start now.
refuse() {
    local service=$1 port running owner
    running=$(native_processes | awk -v s="$service" '$2 == s')
    if [[ -z "$running" ]] && grep -q "^$service " <<<"$(live_dev_groups)"; then
        running="process group $(cat "$RUN_DIR/$service.pgid")"
    fi
    if [[ -n "$running" ]]; then
        error "The $service of this copy already runs:"
        sed 's/^/    PID /' <<<"$running" >&2
        die "Stop it with Ctrl+C in its terminal or: just stop"
    fi
    if [[ "$service" == worker ]] && docker_available &&
        grep -q '^worker-' <<<"$(running_services)"; then
        die "This copy's Docker installation runs (just up): two workers would take" \
            "jobs from the same database and share the GPU. Stop it with: just stop"
    fi
    port=$(service_port "$service")
    if [[ -n "$port" ]] && port_in_use "$port"; then
        owner=$(port_owner "$port")
        die "Port $port ($service) is taken${owner:+ by $owner}: stop that program first."
    fi
    if [[ "$service" == web && -z "${DEV_WEB_CMD:-}" && ! -d "$ROOT/web/node_modules" ]]; then
        die "The web dependencies are not installed: cd web && npm ci"
    fi
}

declare -A GROUPS_STARTED=()

cleanup() {
    local service
    trap - EXIT INT TERM HUP
    ((${#GROUPS_STARTED[@]} > 0)) || return
    echo
    info "Stopping ${!GROUPS_STARTED[*]}..."
    for service in "${!GROUPS_STARTED[@]}"; do
        stop_group "${GROUPS_STARTED[$service]}" "$(service_grace "$service")"
        # Only our own record: `just stop` may have removed it already.
        if [[ "$(cat "$RUN_DIR/$service.pgid" 2>/dev/null)" == "${GROUPS_STARTED[$service]}" ]]; then
            rm -f "$RUN_DIR/$service.pgid"
        fi
    done
    info "Stopped."
}

start() {
    local service=$1 dir=$ROOT command pgid
    [[ "$service" == web ]] && dir="$ROOT/web"
    command=$(service_command "$service")
    # Job control gives every background job its own process group (PGID =
    # PID), so the whole service stops with one signal. No terminal input: a
    # background group that reads it (vite's shortcuts) is stopped by SIGTTIN.
    (cd "$dir" && exec bash -c "exec $command") </dev/null &
    pgid=$!
    GROUPS_STARTED[$service]=$pgid
    echo "$pgid" >"$RUN_DIR/$service.pgid"
    info "Started $service (process group $pgid): $command"
}

main() {
    local services=() service code
    for service in "$@"; do
        case "$service" in
            api | worker | web) services+=("$service") ;;
            -h | --help)
                usage
                exit 0 ;;
            *)
                usage >&2
                exit 2 ;;
        esac
    done
    ((${#services[@]} > 0)) || services=("${ALL_SERVICES[@]}")

    for service in "${services[@]}"; do refuse "$service"; done

    mkdir -p "$RUN_DIR"
    set -m
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM HUP
    for service in "${services[@]}"; do start "$service"; done
    [[ -n "${GROUPS_STARTED[api]:-}" ]] && info "API: http://$API_HOST:$API_PORT/api/docs"
    [[ -n "${GROUPS_STARTED[web]:-}" ]] && info "Web UI: http://127.0.0.1:$WEB_DEV_PORT"
    info "Ctrl+C stops everything started here."

    # Back as soon as one service exits: the EXIT trap stops the others.
    wait -n "${GROUPS_STARTED[@]}"
    code=$?
    for service in "${!GROUPS_STARTED[@]}"; do
        if ! group_alive "${GROUPS_STARTED[$service]}"; then
            warn "$service exited (code $code)."
        fi
    done
    exit "$code"
}

main "$@"
