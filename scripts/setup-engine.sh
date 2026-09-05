#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="$(cat "$REPO/engine/llama-revision")"
ENGINE_DIR="${LATTICE_ENGINE_DIR:-$HOME/.local/share/lattice/engine}"
ENGINE_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/lattice/engine"
CHECK_ONLY=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --engine-dir)
            [ "$#" -ge 2 ] || { printf 'Missing path after --engine-dir\n' >&2; exit 2; }
            ENGINE_DIR="$2"
            shift 2
            ;;
        --check) CHECK_ONLY=true; shift ;;
        -h|--help)
            printf '%s\n' 'Usage: setup-engine.sh [--engine-dir PATH] [--check]' \
                'Builds the pinned Metal (macOS) or Vulkan (Linux) inference runtime.' \
                'LATTICE_ENGINE_DIR changes the install directory.' \
                'LATTICE_LLAMA_SOURCE reuses a clean checkout at the pinned revision.' \
                'LATTICE_BUILD_JOBS sets parallel build jobs.'
            exit 0
            ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[ -n "$ENGINE_DIR" ] || { printf 'Engine directory must not be empty\n' >&2; exit 2; }
PLATFORM="$(uname -s)"
install_hint() {
    if [ "$PLATFORM" = Darwin ]; then
        printf '%s\n' 'Install the Xcode command line tools: xcode-select --install' \
            'Install build tools: brew install cmake ninja' \
            'Or install cmake and ninja in a Python virtual environment, then add its bin directory to PATH.'
    elif command -v pacman >/dev/null; then
        printf '%s\n' 'sudo pacman -S --needed base-devel git cmake ninja vulkan-headers vulkan-icd-loader shaderc spirv-headers'
    elif command -v apt-get >/dev/null; then
        printf '%s\n' 'sudo apt-get install build-essential git cmake ninja-build libvulkan-dev glslc spirv-headers'
    elif command -v dnf >/dev/null; then
        printf '%s\n' 'sudo dnf install gcc-c++ git cmake ninja-build vulkan-loader-devel vulkan-headers glslc spirv-headers'
    elif command -v zypper >/dev/null; then
        printf '%s\n' 'sudo zypper install gcc-c++ git cmake ninja vulkan-devel shaderc spirv-headers-devel'
    else
        printf '%s\n' 'Install a C++ compiler, git, cmake, ninja, Vulkan development headers/loader, glslc, and SPIRV-Headers.'
    fi
}

case "$PLATFORM" in
    Darwin) BACKEND=(-DGGML_METAL=ON -DGGML_VULKAN=OFF) ;;
    Linux) BACKEND=(-DGGML_METAL=OFF -DGGML_VULKAN=ON) ;;
    *) printf 'Unsupported engine platform: %s\n' "$PLATFORM" >&2; exit 1 ;;
esac

MISSING=0
for tool in git cmake ninja c++; do
    if ! command -v "$tool" >/dev/null; then
        printf 'Missing build tool: %s\n' "$tool" >&2
        MISSING=1
    fi
done
if [ "$PLATFORM" = Linux ] && ! command -v glslc >/dev/null; then
    printf 'Missing Vulkan shader compiler: glslc\n' >&2
    MISSING=1
fi
if [ "$PLATFORM" = Darwin ] && ! xcrun --find metal >/dev/null 2>&1; then
    printf 'Missing Metal compiler. Install Xcode and its Metal Toolchain component.\n' >&2
    MISSING=1
fi
if [ "$MISSING" -ne 0 ]; then
    install_hint >&2
    exit 1
fi
if "$CHECK_ONLY"; then
    printf 'Engine build tools are available for %s. CMake will verify backend headers and libraries.\n' "$PLATFORM"
    exit 0
fi

SOURCE="${LATTICE_LLAMA_SOURCE:-$ENGINE_CACHE/llama.cpp-$REVISION}"
if [ ! -d "$SOURCE" ]; then
    if [ -n "${LATTICE_LLAMA_SOURCE:-}" ]; then
        printf 'LATTICE_LLAMA_SOURCE does not exist: %s\n' "$SOURCE" >&2
        exit 1
    fi
    mkdir -p "$SOURCE"
    git -C "$SOURCE" init -q
    git -C "$SOURCE" remote add origin https://github.com/ggml-org/llama.cpp.git
fi
if [ -z "${LATTICE_LLAMA_SOURCE:-}" ] && ! git -C "$SOURCE" rev-parse --verify HEAD >/dev/null 2>&1; then
    git -C "$SOURCE" fetch --depth 1 origin "$REVISION"
    git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
if [ "$(git -C "$SOURCE" rev-parse HEAD)" != "$REVISION" ]; then
    printf 'Engine source must be at revision %s: %s\n' "$REVISION" "$SOURCE" >&2
    exit 1
fi
if ! git -C "$SOURCE" diff --quiet HEAD --; then
    printf 'Engine source has tracked changes; use a clean checkout: %s\n' "$SOURCE" >&2
    exit 1
fi
SOURCE="$(cd "$SOURCE" && pwd)"
BUILD="$ENGINE_CACHE/build-$REVISION-$PLATFORM-$(uname -m)"
JOBS="${LATTICE_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"
printf 'Building inference runtime %s for %s\n' "$REVISION" "$PLATFORM"
if ! cmake -S "$REPO/engine" -B "$BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DLATTICE_LLAMA_SOURCE="$SOURCE" "${BACKEND[@]}"; then
    install_hint >&2
    exit 1
fi
cmake --build "$BUILD" --target llama-cli ggml-rpc-server lattice-engine-info --parallel "$JOBS"
mkdir -p "$ENGINE_DIR/bin"
for binary in llama-cli ggml-rpc-server lattice-engine-info; do
    install -m 0755 "$BUILD/bin/$binary" "$ENGINE_DIR/bin/$binary"
done
printf '%s\n' "$REVISION" > "$ENGINE_DIR/engine-revision"
printf 'Installed inference runtime: %s\n' "$ENGINE_DIR"
"$ENGINE_DIR/bin/lattice-engine-info"
