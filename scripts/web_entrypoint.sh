#!/bin/sh
# The web UI container's command (Dockerfile.frontend): nginx, which stops
# by itself after a shutdown from the web UI.
#
# The API reports "shutting_down" on its health endpoint, then exits; once
# this script has seen that and the API stays unreachable, nginx quits and
# the container ends with code 0 (restart: on-failure leaves it stopped).
# Without "shutting_down" first nothing happens: the API restarting for an
# update, or not up yet, does not take the web UI down.
#
# Environment (used by the tests):
#   BACKEND_HEALTH_URL  default http://backend:8000/api/v1/health
#   WATCH_INTERVAL_SEC  default 3
set -u

URL=${BACKEND_HEALTH_URL:-http://backend:8000/api/v1/health}
INTERVAL=${WATCH_INTERVAL_SEC:-3}

nginx -g 'daemon off;' &
NGINX=$!
# docker stop: let nginx finish its requests.
trap 'nginx -s quit; wait "$NGINX"; exit 0' TERM INT

seen=0
gone=0
while kill -0 "$NGINX" 2>/dev/null; do
    sleep "$INTERVAL" &
    wait $!
    if body=$(wget -q --tries=1 -O - -T 3 "$URL" 2>/dev/null); then
        gone=0
        case "$body" in *shutting_down*) seen=1 ;; esac
    elif [ "$seen" = 1 ]; then
        gone=$((gone + 1))
        if [ "$gone" -ge 2 ]; then
            echo "web_entrypoint: the API has shut down, stopping nginx"
            nginx -s quit
            wait "$NGINX"
            exit 0
        fi
    fi
done
wait "$NGINX"
