#!/usr/bin/env bash
# Collect the source code of the copyleft Debian packages in the given images,
# for publishing next to them: GPL, LGPL and AGPL require it.
#
# Usage: scripts/image_sources.sh OUT_DIR IMAGE...
#
# Writes into OUT_DIR:
#   packages-<image>.txt  every Debian package of the image: package, version,
#                         source package, source version, "copyleft" when its
#                         copyright file names a GPL variant;
#   sources.txt           the copyleft source packages of all images (unique);
#   sources/              their source files, exactly those versions
#                         (apt-get source --download-only).
# Fails when a source package cannot be downloaded: an archive that misses
# one is not complete.
#
# Environment:
#   SOURCES_IMAGE   Debian image that downloads the sources (default debian:trixie)
#   BUILD_NETWORK   its network (default "default"; host where Docker's network
#                   has no internet)
set -euo pipefail

OUT=${1:?Usage: $0 OUT_DIR IMAGE...}
shift
(($# > 0)) || { echo "Usage: $0 OUT_DIR IMAGE..." >&2; exit 2; }
SOURCES_IMAGE=${SOURCES_IMAGE:-debian:trixie}
BUILD_NETWORK=${BUILD_NETWORK:-default}

mkdir -p "$OUT/sources"
OUT=$(cd "$OUT" && pwd)

# Inside each image: one line per package, copyleft decided by its copyright
# file (Debian keeps those even in slim images).
# shellcheck disable=SC2016  # expanded inside the container
LIST='dpkg-query -W -f="\${Package}\t\${Version}\t\${source:Package}\t\${source:Version}\n" |
while IFS="	" read -r p v sp sv; do
    kind=""
    if grep -q -E "GPL" "/usr/share/doc/$p/copyright" 2>/dev/null; then kind=copyleft; fi
    if [ ! -f "/usr/share/doc/$p/copyright" ]; then kind=copyleft; fi
    printf "%s\t%s\t%s\t%s\t%s\n" "$p" "$v" "$sp" "$sv" "$kind"
done'

: >"$OUT/sources.all"
for image in "$@"; do
    name=$(echo "$image" | tr '/:' '__')
    docker run --rm --entrypoint sh "$image" -c "$LIST" >"$OUT/packages-$name.txt"
    awk -F'\t' '$5 == "copyleft" { print $3 "=" $4 }' "$OUT/packages-$name.txt" >>"$OUT/sources.all"
    echo "$image: $(wc -l <"$OUT/packages-$name.txt") packages," \
        "$(awk -F'\t' '$5 == "copyleft"' "$OUT/packages-$name.txt" | wc -l) copyleft"
done
sort -u "$OUT/sources.all" >"$OUT/sources.txt"
rm "$OUT/sources.all"
echo "Source packages to download: $(wc -l <"$OUT/sources.txt")"

# deb-src for the suites the images install from (trixie, its updates and
# security fixes, backports for Mesa).
# shellcheck disable=SC2016  # expanded inside the container
FETCH='set -e
sed -i "s/^Types: deb$/Types: deb deb-src/" /etc/apt/sources.list.d/debian.sources
cat >>/etc/apt/sources.list.d/debian.sources <<EOF

Types: deb deb-src
URIs: http://deb.debian.org/debian
Suites: trixie-backports
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.pgp
EOF
apt-get update -qq
cd /out/sources
missing=0
while read -r sv; do
    if ! apt-get source --download-only -qq "$sv" >/dev/null 2>&1; then
        echo "MISSING $sv" >&2
        missing=$((missing + 1))
    fi
done </out/sources.txt
chown -R "$OWNER" /out/sources
exit "$missing"'

if ! docker run --rm --network "$BUILD_NETWORK" -e OWNER="$(id -u):$(id -g)" \
    -v "$OUT":/out "$SOURCES_IMAGE" sh -c "$FETCH"; then
    echo "Some source packages could not be downloaded (see MISSING above)." >&2
    exit 1
fi
echo "Sources: $(du -sh "$OUT/sources" | cut -f1) in $OUT/sources"
