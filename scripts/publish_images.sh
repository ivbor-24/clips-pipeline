#!/usr/bin/env bash
# Build the release images on this machine and publish them: the backend image
# for each GPU backend and the web UI image to GHCR, the source code of their
# copyleft packages to the GitHub release. Publishing the release and moving
# the release branch to the tag stay the last, separate steps
# (docs/RELEASING.md).
#
# Usage: scripts/publish_images.sh [--push] [--backends "openvino cpu cuda"] [--checkout DIR]
#   without --push  build and check every image, collect the sources into
#                   dist/release-X.Y.Z/ of the checkout
#   --push          also publish: each image is pushed right after its check
#                   and removed here (the disk holds one image at a time),
#                   then the sources go to the release (a draft is fine)
#   --checkout DIR  the release checkout to build (default: this repository),
#                   e.g. a tag whose own copy of this script is older
#
# Run it on a clean checkout of a release tag vX.Y.Z (X.Y.Z = src/__init__.py).
# The images are portable (CPU_TARGET=portable: any x86-64 CPU with AVX); the
# NVIDIA one is compiled for all GPUs from RTX 20 to RTX 50.
#
# Environment:
#   IMAGE_REPO     default ghcr.io/ivbor-24/clips-pipeline (the web UI: IMAGE_REPO-web)
#   GH_REPO        default ivbor-24/clips-pipeline
#   BUILD_NETWORK  default "default"; host where Docker's network has no internet
#   BUILD_JOBS     parallel compile jobs (empty: automatic)
# Needs docker with buildx, git, curl, python3, and a GitHub token with
# write:packages in git's credential helper (git credential fill).
set -euo pipefail
SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

IMAGE_REPO=${IMAGE_REPO:-ghcr.io/ivbor-24/clips-pipeline}
GH_REPO=${GH_REPO:-ivbor-24/clips-pipeline}
BUILD_NETWORK=${BUILD_NETWORK:-default}
BUILD_JOBS=${BUILD_JOBS:-}
BACKENDS="openvino cpu cuda"
PUSH=false
CHECKOUT="$SCRIPTS/.."

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) PUSH=true ;;
        --backends) BACKENDS=${2:?} && shift ;;
        --checkout) CHECKOUT=${2:?} && shift ;;
        *)
            sed -n '8,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
            exit 2 ;;
    esac
    shift
done
cd "$CHECKOUT"

die() {
    echo "ERROR: $*" >&2
    exit 1
}
step() { echo -e "\n━━━ $* ━━━"; }

# ---- What is being released ---------------------------------------------------

[[ -z "$(git status --porcelain)" ]] || die "The checkout has changes: publish a clean release tag."
VERSION=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' src/__init__.py)
[[ -n "$VERSION" ]] || die "No __version__ in src/__init__.py."
TAG=$(git describe --exact-match --tags HEAD 2>/dev/null) ||
    die "HEAD is not a release tag: check out v$VERSION."
[[ "$TAG" == "v$VERSION" ]] || die "HEAD is tagged $TAG, but src/__init__.py says $VERSION."
SHA=$(git rev-parse HEAD)
OUT=dist/release-$VERSION
WEB_REPO=$IMAGE_REPO-web
LABELS=(
    --label "org.opencontainers.image.version=$VERSION"
    --label "org.opencontainers.image.revision=$SHA"
    --label "org.opencontainers.image.source=https://github.com/$GH_REPO"
    --label "org.opencontainers.image.licenses=MIT"
)
echo "Release $TAG ($SHA), backends: $BACKENDS, push: $PUSH"
rm -rf "$OUT"
mkdir -p "$OUT"

if [[ "$PUSH" == true ]]; then
    TOKEN=$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill 2>/dev/null |
        sed -n 's/^password=//p')
    [[ -n "$TOKEN" ]] || die "No GitHub token in git's credential helper."
    REGISTRY=${IMAGE_REPO%%/*}
    # Logged in only while publishing: the token must not stay in
    # ~/.docker/config.json.
    trap 'docker logout "$REGISTRY" >/dev/null 2>&1 || true' EXIT
    echo "$TOKEN" | docker login "$REGISTRY" -u "${GH_REPO%%/*}" --password-stdin >/dev/null
fi

# ---- One image at a time: build, check, list its packages, push, remove ------

PUBLISHED=()
# finish IMAGE MOVING_TAG: list the packages of a checked image; with --push,
# publish it under both tags and remove it here.
finish() {
    local image=$1 moving=$2
    BUILD_NETWORK=$BUILD_NETWORK "$SCRIPTS/image_sources.sh" list "$OUT" "$image"
    PUBLISHED+=("$image")
    [[ "$PUSH" == true ]] || return 0
    docker tag "$image" "$moving"
    docker push "$image"
    docker push "$moving"
    docker image rm "$image" "$moving" >/dev/null
}

step "Web UI: $WEB_REPO:$VERSION"
docker build --network "$BUILD_NETWORK" -f Dockerfile.frontend "${LABELS[@]}" \
    -t "$WEB_REPO:$VERSION" .
docker run --rm --entrypoint nginx "$WEB_REPO:$VERSION" -t
finish "$WEB_REPO:$VERSION" "$WEB_REPO:latest"

for backend in $BACKENDS; do
    image=$IMAGE_REPO:$VERSION-$backend
    step "Backend $backend: $image"
    docker build --network "$BUILD_NETWORK" -f Dockerfile.backend \
        --build-arg GPU_BACKEND="$backend" --build-arg CPU_TARGET=portable \
        --build-arg BUILD_JOBS="$BUILD_JOBS" "${LABELS[@]}" -t "$image" .
    docker run --rm --entrypoint python "$image" -c \
        "import src; assert src.__version__ == '$VERSION', src.__version__"
    case "$backend" in
        # llama_cpp of the cuda image loads only where the NVIDIA driver is
        # (libcuda.so.1); check_gpu.py checks its libraries without it.
        cuda) docker run --rm --entrypoint python "$image" scripts/check_gpu.py --backend cuda --libraries-only ;;
        openvino)
            docker run --rm --entrypoint python "$image" -c "import llama_cpp"
            docker run --rm --entrypoint sh "$image" -c "command -v whisper-cli >/dev/null" ;;
        *) docker run --rm --entrypoint python "$image" -c "import llama_cpp" ;;
    esac
    finish "$image" "$IMAGE_REPO:$backend"
done

# ---- Sources of the copyleft packages -----------------------------------------

step "Source code of the copyleft packages"
BUILD_NETWORK=$BUILD_NETWORK "$SCRIPTS/image_sources.sh" fetch "$OUT"
cat >"$OUT/README.txt" <<EOF
Source code of the copyleft (GPL, LGPL, AGPL) Debian packages in the Docker
images of Clips Pipeline $VERSION:

$(printf '  %s\n' "${PUBLISHED[@]}")

packages-*.txt  every Debian package of each image (package, version, source
                package, source version, "copyleft" where its copyright file
                names a GPL variant)
sources.txt     the copyleft source packages, exactly the versions in the images
sources/        their source files as Debian distributes them (.dsc, .orig.tar.*,
                .debian.tar.*); unpack one with: dpkg-source -x <name>.dsc

The project's own code is MIT-licensed: https://github.com/$GH_REPO/tree/$TAG
EOF
ARCHIVE=$OUT/clips-pipeline-$VERSION-sources.tar
(cd "$OUT" && tar -cf "$(basename "$ARCHIVE")" README.txt sources.txt packages-*.txt sources)
# The archive holds them now; ~1.2 GB less on the disk.
rm -rf "$OUT/sources"
echo "Archive: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

if [[ "$PUSH" != true ]]; then
    echo -e "\nBuilt and checked; nothing published (add --push)."
    exit 0
fi

# ---- The release ----------------------------------------------------------------

step "Upload the sources to the release $TAG"
api() { curl -fsS -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "$@"; }
# The release list, not releases/tags/<tag>: the latter skips drafts, and a
# release stays a draft until its images and sources are in place.
RELEASE_ID=$(api "https://api.github.com/repos/$GH_REPO/releases?per_page=100" |
    python3 -c "import json, sys; print(''.join(str(r['id']) for r in json.load(sys.stdin) if r['tag_name'] == '$TAG'))")
[[ -n "$RELEASE_ID" ]] || die "No GitHub release (or draft) for $TAG: create it first."
NAME=$(basename "$ARCHIVE")
OLD=$(api "https://api.github.com/repos/$GH_REPO/releases/$RELEASE_ID/assets" |
    python3 -c "import json, sys; print(''.join(str(a['id']) for a in json.load(sys.stdin) if a['name'] == '$NAME'))")
[[ -z "$OLD" ]] || api -X DELETE "https://api.github.com/repos/$GH_REPO/releases/assets/$OLD"
api -X POST -H "Content-Type: application/x-tar" --data-binary "@$ARCHIVE" \
    "https://uploads.github.com/repos/$GH_REPO/releases/$RELEASE_ID/assets?name=$NAME" >/dev/null
echo "Uploaded $NAME"

echo -e "\nPublished $TAG:"
printf '  %s\n' "${PUBLISHED[@]}"
echo "  sources: $NAME on the release $TAG"
echo "Next: publish the release, then move the release branch: git push origin $TAG^{commit}:refs/heads/release"
