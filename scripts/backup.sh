#!/usr/bin/env bash
# just backup / just restore: scripts/backup.py with the host's python3 (it
# needs only the standard library, so it works next to a Docker installation).
# A restore replaces the job database: it refuses while the API, the worker or
# dev processes of this copy run.
#
# Usage: scripts/backup.sh create [--with-sources] [--output DIR]
#        scripts/backup.sh restore ARCHIVE [--yes]
set -euo pipefail

# shellcheck source=scripts/docker_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/docker_common.sh"

command -v python3 >/dev/null 2>&1 ||
    die "python3 is needed for backups (Debian/Ubuntu: sudo apt install python3)."

if [[ "${1:-}" == restore ]]; then
    busy=""
    if docker_available; then
        busy+=$(running_services)
    fi
    busy+=$(live_dev_groups)$(native_processes)
    if [[ -n "$busy" ]]; then
        die "The services of this copy are running; a restore replaces their database. Stop them first: just stop"
    fi
fi

exec python3 "$ROOT/scripts/backup.py" "$@"
