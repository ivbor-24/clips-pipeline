#!/usr/bin/env bash
# Docker smoke test: install from scratch the way docs/INSTALL.md describes it, run
# one job through the web UI, stop. Run it in a fresh clone: it writes .env,
# builds the image and leaves the job in jobs/.
#
# Usage: scripts/docker_smoke.sh [options]   (or: just smoke [options])
#   --backend cpu|cuda|openvino  image to build (default: cpu)
#   --models-dir DIR             the real models in DIR and the default
#                                settings, LLM included; without it: tiny
#                                whisper and no LLM (profile ci_smoke), as on
#                                a GitHub runner, which has no room for them
#   --video FILE                 the job's input (default: a short lecture
#                                spoken by espeak-ng)
#
# Run it for changes to the Docker installation; .github/workflows/docker-smoke.yml
# runs it by hand on a GitHub runner. The stack is stopped at the end, also
# when a step fails.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BACKEND=cpu
MODELS_DIR=""
VIDEO=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND=${2:-} && shift ;;
        --models-dir) MODELS_DIR=${2:-} && shift ;;
        --video) VIDEO=${2:-} && shift ;;
        *)
            sed -n '6,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
            exit 2 ;;
    esac
    shift
done
[[ -z "$VIDEO" ]] || VIDEO=$(realpath "$VIDEO")
WORK=$(mktemp -d)

step() { echo -e "\n━━━ $* ━━━"; }

finish() {
    local rc=$?
    if ((rc != 0)); then
        step "Logs (the smoke test failed)"
        docker compose --profile "$BACKEND" logs --no-color --tail 300 || true
        find jobs -name job.log -exec tail -n 200 {} \; 2>/dev/null || true
    fi
    just stop >/dev/null 2>&1 || true
    rm -rf "$WORK"
    exit "$rc"
}
trap finish EXIT

step "./setup.sh (backend $BACKEND)"
if [[ -n "$MODELS_DIR" ]]; then
    ./setup.sh --yes --backend "$BACKEND" --models-dir "$MODELS_DIR" --no-start
    PROFILE=""
else
    ./setup.sh --yes --backend "$BACKEND" --no-models --no-start
    cat >config/profiles/ci_smoke.yaml <<'EOF'
# Written by scripts/docker_smoke.sh.
transcription:
  model: tiny
  language: en
scoring:
  llm:
    enabled: false
  term_extraction_method: statistical
  min_duration: 5
  max_duration: 30
  min_score_threshold: 0.0
  max_clips_per_video: 1
rendering:
  video_preset: ultrafast
EOF
    docker compose run --rm prefetch-models \
        python scripts/prefetch_models.py --profile ci_smoke --skip-llm
    PROFILE=ci_smoke
fi

step "just up, twice"
just up
just up | tee "$WORK/up-again.log"
grep -q "Already running" "$WORK/up-again.log"

step "A job through the web UI"
if [[ -z "$VIDEO" ]]; then
    VIDEO=$WORK/lecture.mp4
    espeak-ng -w "$WORK/speech.wav" "Welcome to this short lecture about the water cycle. \
        Water evaporates from the ocean, rises into the air and cools down. \
        The vapour condenses into clouds, and the clouds bring rain and snow. \
        Rivers carry the water back to the sea, and the cycle starts again. \
        This simple loop moves heat around the planet and shapes our weather."
    ffmpeg -loglevel error -f lavfi -i testsrc2=size=1280x720:rate=25 -i "$WORK/speech.wav" \
        -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$VIDEO"
fi
port=$(sed -n 's/^WEB_PORT=//p' .env | tail -n 1)
python3 .github/scripts/docker_smoke_job.py "$VIDEO" --url "http://127.0.0.1:${port:-8080}" \
    --profile "$PROFILE"

step "just status, just stop"
just status
just stop
test -z "$(docker compose --profile "$BACKEND" ps -q --status running)"
echo "Docker smoke test: OK"
