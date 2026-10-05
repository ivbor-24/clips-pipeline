#!/usr/bin/env bash
# CMake flags that pick the CPU instructions llama.cpp and whisper.cpp use.
#
#   native    the build machine's CPU (-march=native). Right when the image
#             or install runs where it was built: `docker compose build`,
#             install.sh.
#   portable  any x86-64 CPU with AVX (2011+), for an image that moves to
#             another machine (just image-build / image-save). ggml's plain
#             -DGGML_NATIVE=OFF is not portable: it turns on AVX2 and FMA, and
#             the result dies with SIGILL on CPUs without them (Xeon E5 v2,
#             Ivy Bridge and older).
#
# Usage: ggml_cpu_flags.sh <native|portable>
set -euo pipefail

case "${1:?Usage: $0 <native|portable>}" in
    native)   echo "-DGGML_NATIVE=ON" ;;
    portable) echo "-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_F16C=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_BMI2=OFF -DGGML_AVX512=OFF" ;;
    *)
        echo "Unknown CPU target: $1 (native | portable)" >&2
        exit 1 ;;
esac
