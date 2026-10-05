#!/usr/bin/env bash
# Build the release images on this machine and publish them: the backend image
# for each GPU backend and the web UI image to GHCR, the source code of their
# copyleft packages to the GitHub release, and the release branch to the tag.
#
# Run it in a clean checkout of a release tag vX.Y.Z (X.Y.Z = src/__init__.py).
#
# Usage: scripts/publish_images.sh [--push] [--backends "openvino cpu cuda"]
#   without --push  build, check and collect the sources into dist/release-X.Y.Z/
#   --push          then push the images, upload the sources to the release
#                   and move the release branch to the tag
#
# The images are portable (CPU_TARGET=portable: any x86-64 CPU with AVX); the
# NVIDIA one is compiled for all GPUs from RTX 20 to RTX 50.
#
# Environment:
#   IMAGE_REPO     default ghcr.io/ivbor-24/clips-pipeline (the web UI: IMAGE_REPO-web)
#   GH_REPO        default ivbor-24/clips-pipeline
#   BUILD_NETWORK  default "default"; host where Docker's network has no internet
#   BUILD_JOBS     parallel compile jobs (empty: automatic)
# Needs docker with buildx, git, curl, python3, and a GitHub token with
# write:packages in git's credential helper (git credential fill; git push to
# origin uses it too).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE_REPO=${IMAGE_REPO:-ghcr.io/ivbor-24/clips-pipeline}
GH_REPO=${GH_REPO:-ivbor-24/clips-pipeline}
BUILD_NETWORK=${BUILD_NETWORK:-default}
BUILD_JOBS=${BUILD_JOBS:-}
BACKENDS="openvino cpu cuda"
PUSH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push) PUSH=true ;;
        --backends) BACKENDS=${2:?} && shift ;;
        *)
            sed -n '8,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
            exit 2 ;;
    esac
    shift
done

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

# ---- Build ---------------------------------------------------------------------

IMAGES=()
for backend in $BACKENDS; do
    step "Build $IMAGE_REPO:$VERSION-$backend"
    docker build --network "$BUILD_NETWORK" -f Dockerfile.backend \
        --build-arg GPU_BACKEND="$backend" --build-arg CPU_TARGET=portable \
        --build-arg BUILD_JOBS="$BUILD_JOBS" "${LABELS[@]}" \
        -t "$IMAGE_REPO:$VERSION-$backend" .
    IMAGES+=("$IMAGE_REPO:$VERSION-$backend")
done
step "Build $WEB_REPO:$VERSION"
docker build --network "$BUILD_NETWORK" -f Dockerfile.frontend "${LABELS[@]}" \
    -t "$WEB_REPO:$VERSION" .
IMAGES+=("$WEB_REPO:$VERSION")

# ---- Check ---------------------------------------------------------------------

step "Check the images"
for backend in $BACKENDS; do
    image=$IMAGE_REPO:$VERSION-$backend
    docker run --rm --entrypoint python "$image" -c \
        "import llama_cpp, src; assert src.__version__ == '$VERSION', src.__version__"
    case "$backend" in
        openvino) docker run --rm --entrypoint sh "$image" -c "command -v whisper-cli >/dev/null" ;;
        cuda) docker run --rm --entrypoint python "$image" scripts/check_gpu.py --backend cuda --libraries-only ;;
    esac
    echo "$image: OK"
done
docker run --rm --entrypoint nginx "$WEB_REPO:$VERSION" -t
echo "$WEB_REPO:$VERSION: OK"

# ---- Sources of the copyleft packages -----------------------------------------

step "Source code of the copyleft packages"
rm -rf "$OUT"
BUILD_NETWORK=$BUILD_NETWORK scripts/image_sources.sh "$OUT" "${IMAGES[@]}"
cat >"$OUT/README.txt" <<EOF
Source code of the copyleft (GPL, LGPL, AGPL) Debian packages in the Docker
images of Clips Pipeline $VERSION:

$(printf '  %s\n' "${IMAGES[@]}")

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

# ---- Publish -------------------------------------------------------------------

TOKEN=$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill 2>/dev/null |
    sed -n 's/^password=//p')
[[ -n "$TOKEN" ]] || die "No GitHub token in git's credential helper."
OWNER=${GH_REPO%%/*}
REGISTRY=${IMAGE_REPO%%/*}
step "Push to $REGISTRY"
# Logged in only while pushing: the token must not stay in ~/.docker/config.json.
trap 'docker logout "$REGISTRY" >/dev/null 2>&1 || true' EXIT
echo "$TOKEN" | docker login "$REGISTRY" -u "$OWNER" --password-stdin >/dev/null
for backend in $BACKENDS; do
    docker tag "$IMAGE_REPO:$VERSION-$backend" "$IMAGE_REPO:$backend"
    docker push "$IMAGE_REPO:$VERSION-$backend"
    docker push "$IMAGE_REPO:$backend"
done
docker tag "$WEB_REPO:$VERSION" "$WEB_REPO:latest"
docker push "$WEB_REPO:$VERSION"
docker push "$WEB_REPO:latest"
docker logout "$REGISTRY" >/dev/null

step "Upload the sources to the release $TAG"
api() { curl -fsS -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.github+json" "$@"; }
RELEASE_ID=$(api "https://api.github.com/repos/$GH_REPO/releases/tags/$TAG" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])') ||
    die "No GitHub release for $TAG: create it first."
NAME=$(basename "$ARCHIVE")
OLD=$(api "https://api.github.com/repos/$GH_REPO/releases/$RELEASE_ID/assets" |
    python3 -c "import json, sys; print(''.join(str(a['id']) for a in json.load(sys.stdin) if a['name'] == '$NAME'))")
[[ -z "$OLD" ]] || api -X DELETE "https://api.github.com/repos/$GH_REPO/releases/assets/$OLD"
api -X POST -H "Content-Type: application/x-tar" --data-binary "@$ARCHIVE" \
    "https://uploads.github.com/repos/$GH_REPO/releases/$RELEASE_ID/assets?name=$NAME" >/dev/null
echo "Uploaded $NAME"

step "Move the release branch to $TAG"
git push origin "$SHA:refs/heads/release"

echo -e "\nPublished $TAG:"
printf '  %s\n' "${IMAGES[@]}"
echo "  sources: $NAME on https://github.com/$GH_REPO/releases/tag/$TAG"
