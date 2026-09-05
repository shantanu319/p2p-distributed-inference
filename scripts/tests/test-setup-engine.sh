#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
mkdir -p "$TEST_DIR/bin" "$TEST_DIR/source"
export ENGINE_TEST_CALLS="$TEST_DIR/calls"
export ENGINE_TEST_REVISION="$(cat "$REPO/engine/llama-revision")"
export XDG_CACHE_HOME="$TEST_DIR/cache" LATTICE_LLAMA_SOURCE="$TEST_DIR/source"
export LATTICE_ENGINE_DIR="$TEST_DIR/installed engine"
export LATTICE_BUILD_JOBS=2
cat > "$TEST_DIR/bin/git" <<'EOF'
#!/usr/bin/env bash
case "$3" in
    rev-parse) printf '%s\n' "$ENGINE_TEST_REVISION" ;;
    diff) exit "${ENGINE_TEST_DIRTY:-0}" ;;
    *) exit 1 ;;
esac
EOF
cat > "$TEST_DIR/bin/cmake" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" >> "$ENGINE_TEST_CALLS"
if [ "$1" = --build ]; then
    mkdir -p "$2/bin"
    for binary in llama-cli ggml-rpc-server lattice-engine-info; do
        printf '#!/usr/bin/env bash\nprintf "[]\\n"\n' > "$2/bin/$binary"
    done
fi
EOF
cat > "$TEST_DIR/bin/uname" <<'EOF'
#!/usr/bin/env bash
if [ "$1" = -s ]; then printf 'Linux\n'; else printf 'x86_64\n'; fi
EOF
for tool in ninja c++ glslc; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "$TEST_DIR/bin/$tool"
done
chmod +x "$TEST_DIR/bin/"*
export PATH="$TEST_DIR/bin:/usr/bin:/bin"

bash "$REPO/scripts/setup-engine.sh" > "$TEST_DIR/output"
[ "$(cat "$LATTICE_ENGINE_DIR/engine-revision")" = "$ENGINE_TEST_REVISION" ]
for binary in llama-cli ggml-rpc-server lattice-engine-info; do
    [ -x "$LATTICE_ENGINE_DIR/bin/$binary" ]
done
grep -Fxq -- -DGGML_VULKAN=ON "$ENGINE_TEST_CALLS"
grep -Fxq -- -DGGML_METAL=OFF "$ENGINE_TEST_CALLS"
grep -Fxq -- lattice-engine-info "$ENGINE_TEST_CALLS"

rm "$ENGINE_TEST_CALLS"
if ENGINE_TEST_REVISION=wrong bash "$REPO/scripts/setup-engine.sh" > "$TEST_DIR/output" 2>&1; then
    printf 'Wrong source revision was accepted\n' >&2
    exit 1
fi
[ ! -e "$ENGINE_TEST_CALLS" ]
if ENGINE_TEST_DIRTY=1 bash "$REPO/scripts/setup-engine.sh" > "$TEST_DIR/output" 2>&1; then
    printf 'Modified source was accepted\n' >&2
    exit 1
fi
[ ! -e "$ENGINE_TEST_CALLS" ]
bash "$REPO/scripts/setup-engine.sh" --check > "$TEST_DIR/output"
[ ! -e "$ENGINE_TEST_CALLS" ]
if bash "$REPO/scripts/setup-engine.sh" --engine-dir '' > "$TEST_DIR/output" 2>&1; then
    printf 'Empty install directory was accepted\n' >&2
    exit 1
fi
printf 'Engine installation tests passed\n'
