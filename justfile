# Justfile for Scientific Clips Pipeline MVP

# Load .env (if present) into every recipe: API keys, HF_TOKEN and API_* settings
# reach the CLI, the API and the job worker. Real environment variables win.
set dotenv-load

# Use venv executables. An absolute path: a relative PATH entry is resolved
# against whatever directory a tool runs in (tests link tools found on PATH).
export PATH := justfile_directory() + "/.venv/bin:" + env_var_or_default("PATH", "/usr/bin")

# Tests run in a clean environment, as on a CI runner: the .env loaded into
# every recipe (dotenv-load) must not leak into them (WEB_PORT, API_*), and
# they never download models.
clean_env := 'env -i HOME="$HOME" PATH="$PATH" LANG="${LANG:-C.UTF-8}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1'

# Shell scripts that shellcheck checks (the CI lint job lists the same).
shell_scripts := "setup.sh scripts/stack.sh scripts/docker_common.sh scripts/dev.sh scripts/web_entrypoint.sh scripts/backup.sh scripts/docker_smoke.sh scripts/publish_images.sh scripts/image_sources.sh"

# Keep the HuggingFace/transformers model caches inside the project,
# matching the layout Dockerfile.backend uses (artifacts/cache/...). This way
# a native `just process` run and a `docker compose` run share the same
# downloaded weights instead of one using ~/.cache and the other a volume.
# Respects an HF_HOME/etc already set in the environment.
export HF_HOME := env_var_or_default("HF_HOME", "artifacts/cache/huggingface")
export TRANSFORMERS_CACHE := env_var_or_default("TRANSFORMERS_CACHE", "artifacts/cache/huggingface")

# Default target
default:
	@echo "Scientific Clips Pipeline MVP"
	@echo ""
	@echo "Available commands:"
	@echo "  just up | stop | status     - Start, stop, check the web UI (Docker, after ./setup.sh)"
	@echo "  just process <input> [config] [dry_run] [resume] [force] - Process a video file or URL"
	@echo "  just validate-config        - Validate configuration file"
	@echo "  just prefetch-models        - Pre-download Whisper/LLM models"
	@echo "  just test                   - Run tests"
	@echo "  just install                - Install dependencies"
	@echo "  just clean                  - Clean build artifacts"
	@echo "  just clean-cache            - Clean cache directories"
	@echo ""

# Process a video through the pipeline
# Usage: just process test.mp4 [config] [dry_run] [resume] [force] [profile]
# Example: just process video.mp4 config/config.yaml True False False low_vram
process input='' config='config/config.yaml' dry_run='False' resume='False' force='False' profile='':
	@if [ -z "{{input}}" ]; then \
		echo "Error: input parameter is required"; \
		echo "Usage: just process <input> [config] [dry_run] [resume] [force] [profile]"; \
		exit 1; \
	fi
	@DRY_RUN=$(echo "{{dry_run}}" | tr '[:upper:]' '[:lower:]'); \
	RESUME=$(echo "{{resume}}" | tr '[:upper:]' '[:lower:]'); \
	FORCE=$(echo "{{force}}" | tr '[:upper:]' '[:lower:]'); \
	DRY_FLAG=""; \
	RESUME_FLAG=""; \
	FORCE_FLAG=""; \
	if [ "$DRY_RUN" = "true" ] || [ "$DRY_RUN" = "1" ] || [ "$DRY_RUN" = "yes" ]; then DRY_FLAG="--dry-run"; fi; \
	if [ "$RESUME" = "true" ] || [ "$RESUME" = "1" ] || [ "$RESUME" = "yes" ]; then RESUME_FLAG="--resume"; fi; \
	if [ "$FORCE" = "true" ] || [ "$FORCE" = "1" ] || [ "$FORCE" = "yes" ]; then FORCE_FLAG="--force"; fi; \
	PROFILE_FLAG=""; \
	if [ -n "{{profile}}" ]; then PROFILE_FLAG="--profile {{profile}}"; fi; \
	python3 -m src.cli process \
		--input "{{input}}" \
		--config "{{config}}" \
		$DRY_FLAG $RESUME_FLAG $FORCE_FLAG $PROFILE_FLAG

# Validate configuration file
validate-config config='config/config.yaml':
	python3 -m src.cli validate-config --config "{{config}}"

# Pre-download the Whisper and LLM models config.yaml is configured to use,
# without loading them into VRAM. Run this once after install so the first
# `just process` doesn't silently block on a multi-GB download.
prefetch-models config='config/config.yaml' profile='':
	@PROFILE_FLAG=""; \
	if [ -n "{{profile}}" ]; then PROFILE_FLAG="--profile {{profile}}"; fi; \
	python3 scripts/prefetch_models.py --config "{{config}}" $PROFILE_FLAG

# Check that llama.cpp, whisper.cpp and CTranslate2 really use the GPU.
# backend: cpu | cuda | openvino | rocm (default: detected from the llama.cpp build)
check-gpu backend='':
	@BACKEND_FLAG=""; \
	if [ -n "{{backend}}" ]; then BACKEND_FLAG="--backend {{backend}}"; fi; \
	python3 scripts/check_gpu.py $BACKEND_FLAG

# Build whisper.cpp's whisper-cli (engine whisper_cpp): cpu | cuda | openvino
install-whisper-cpp backend="openvino" *flags="":
	./scripts/install_whisper_cpp.sh {{backend}} {{flags}}

# Show version
version:
	python3 -m src.cli version

# Install Python dependencies from uv.lock with the test and dev extras
# (backend: cpu | cuda | openvino); stops first if a build tool is missing
install backend="cpu":
	./scripts/check_system_deps.sh {{backend}}
	uv sync --frozen --inexact --extra {{backend}} --extra test --extra dev --no-install-package llama-cpp-python
	./scripts/install_llama_cpp_backend.sh {{backend}}

# Run tests
test:
	{{clean_env}} pytest tests/ -v

# Run tests with coverage
test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Lint exactly as CI does (install with: pip install -e ".[dev]")
lint:
	ruff check src tests scripts
	black --check src tests scripts
	python3 scripts/check_licenses.py
	@if command -v shellcheck >/dev/null; then \
		shellcheck -x -S warning {{shell_scripts}}; \
	elif command -v uv >/dev/null; then \
		uv tool run --from shellcheck-py shellcheck -x -S warning {{shell_scripts}}; \
	else echo "shellcheck not installed: skipped (CI runs it)"; fi

# Full check of a change: lint, uv.lock, all tests (as in CI)
ci: lint
	uv lock --check
	{{clean_env}} pytest tests/ -q -p no:cacheprovider

# Docker install from scratch + one job via the web UI, in a fresh clone (--backend B, --models-dir DIR, --video FILE)
smoke *args:
	scripts/docker_smoke.sh {{args}}

# For changes to the image's build (see docker-cuda.yml).
# Release images to GHCR and their copyleft sources to the GitHub release (docs/RELEASING.md)
publish-images *args:
	scripts/publish_images.sh {{args}}

# Build the NVIDIA image and check its CUDA libraries (no GPU needed, ~15 GB of disk)
check-cuda-image:
	docker build -f Dockerfile.backend -t clips-pipeline:cuda \
		--build-arg GPU_BACKEND=cuda --build-arg CUDA_ARCHITECTURES=75 .
	docker run --rm clips-pipeline:cuda python scripts/check_gpu.py --backend cuda --libraries-only

# Auto-format and apply safe lint fixes
fmt:
	ruff check src tests scripts --fix
	black src tests scripts

# Clean build artifacts
clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf src/api/**/__pycache__
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type f -name "*.pyc" -delete

# Clean cache directories
clean-cache:
	rm -rf artifacts/cache
	rm -rf output/clips
	rm -f output/manifest.json
	@echo "Cache cleaned"

# Create necessary directories
setup-dirs:
	mkdir -p artifacts/logs
	mkdir -p artifacts/cache
	mkdir -p artifacts/cache/huggingface
	mkdir -p artifacts/cache/models
	mkdir -p output/clips
	mkdir -p input_videos
	mkdir -p data
	@echo "Directories created"

# Batch process multiple videos
batch dir='./input_videos' config='config/config.yaml' dry_run='False' resume='False' force='False' profile='':
	@if [ ! -d "{{dir}}" ]; then \
		echo "Error: directory {{dir}} does not exist"; \
		exit 1; \
	fi
	@for file in {{dir}}/*; do \
		if [ -f "$file" ]; then \
			echo "Processing: $file"; \
			just process "$file" "{{config}}" "{{dry_run}}" "{{resume}}" "{{force}}" "{{profile}}"; \
		fi; \
	done

# Review rendered clips interactively
review clips='output/clips' output='artifacts/review.json' auto='False':
	@AUTO=$(echo "{{auto}}" | tr '[:upper:]' '[:lower:]'); \
	if [ "$AUTO" = "true" ] || [ "$AUTO" = "1" ] || [ "$AUTO" = "yes" ]; then \
		python3 -m src.cli review --clips-dir "{{clips}}" --output "{{output}}" --auto; \
	else \
		python3 -m src.cli review --clips-dir "{{clips}}" --output "{{output}}"; \
	fi

# Start API server with reload (jobs run in the worker: start it with `just worker`).
# Listens on this computer only; for the local network pass host=0.0.0.0 and set
# API_PASSWORD in .env. Ctrl+C or `just stop` stops it.
serve host="127.0.0.1" port="8000":
	@DEV_API_HOST={{host}} DEV_API_PORT={{port}} scripts/dev.sh api

# Start the job worker: runs queued jobs one at a time, each in its own process
worker:
	@scripts/dev.sh worker

# Start API server (production)
serve-prod host="127.0.0.1" port="8000":
	uvicorn src.api.app:app --host {{host}} --port {{port}} --workers 4

# Help command
help:
	@just --list

# Start frontend dev server (http://127.0.0.1:5173)
dev-web:
	@scripts/dev.sh web

# Build frontend
build-web:
	cd web && npm run build

# Start API, job worker and web dev server (development); Ctrl+C stops all three
dev:
	@scripts/dev.sh

# ---- Docker installation (docs/INSTALL.md) ----

# Install: check Docker, detect the GPU, write .env, build, download models, start
setup *flags="":
	@./setup.sh {{flags}}

# Start the web UI, API and job worker in Docker (says so if already running)
up *flags="":
	@scripts/stack.sh up {{flags}}

# Stop the containers and dev processes; a running job goes back to the queue
stop:
	@scripts/stack.sh stop

# Containers, address and version; dev processes; other copies
status:
	@scripts/stack.sh status

# Back up the job database, settings and clips to backups/ (source videos too: --with-sources)
backup *flags="":
	@scripts/backup.sh create {{flags}}

# Evaluate against a reference set of hand-made shorts (docs/EVALUATION.md)
eval *args="":
	python3 scripts/evaluate.py {{args}}

# New migration of the job database after a model change (review it before committing)
db-revision message:
	uv run alembic revision --autogenerate -m "{{message}}"

# Restore a backup made by `just backup`; stop the services first (just stop)
restore archive *flags="":
	@scripts/backup.sh restore "{{archive}}" {{flags}}

# Run E2E tests
test-e2e:
	cd web && npm run test:e2e

# Run E2E tests with UI
test-e2e-ui:
	cd web && npm run test:e2e:ui

# View Docker logs
docker-logs:
	docker compose logs -f

# ---- Image build / save / load / verify / release (docs/INSTALL.md, "Образ для другой машины") ----

# Build the pipeline Docker image for a given GPU backend.
#   just image-build backend=openvino
image-build backend='openvino' no_cache='':
	@if [ "{{no_cache}}" = "True" ]; then \
		python3 -m src.cli image build --backend "{{backend}}" --no-cache; \
	else \
		python3 -m src.cli image build --backend "{{backend}}"; \
	fi

# Export an image to dist/<name>.tar.gz + sidecar sha256.
#   just image-save image=pipeline:openvino-latest
image-save image dist='dist':
	python3 -m src.cli image save --image "{{image}}" --dist "{{dist}}"

# Load a previously exported archive on this host.
image-load archive:
	python3 -m src.cli image load "{{archive}}"

# Smoke-check GPU/onnx/whisper inside the image.
#   just image-verify image=pipeline:openvino-latest model=tiny
image-verify image model='tiny':
	python3 -m src.cli image verify --image "{{image}}" --model "{{model}}"

# Promote a verified image to pipeline:<backend>-stable and update the manifest.
image-release image stable_tag='':
	@if [ -z "{{stable_tag}}" ]; then \
		python3 -m src.cli image release --image "{{image}}"; \
	else \
		python3 -m src.cli image release --image "{{image}}" --stable-tag "{{stable_tag}}"; \
	fi

# List local pipeline images, optionally filtered by backend.
image-list backend='':
	@if [ -z "{{backend}}" ]; then \
		python3 -m src.cli image list; \
	else \
		python3 -m src.cli image list --backend "{{backend}}"; \
	fi

# Remove local pipeline images for a backend except stable/latest.
image-prune backend:
	python3 -m src.cli image prune --backend "{{backend}}"

# Convenience: full local cycle (build → save) for the current backend.
image-cycle backend='openvino':
	just image-build backend="{{backend}}"
	just image-save image="pipeline:{{backend}}-latest"
