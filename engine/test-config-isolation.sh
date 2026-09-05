#!/usr/bin/env bash
set -euo pipefail

ENGINE_DIR="${1:-${LATTICE_ENGINE_DIR:-$HOME/.local/share/lattice/engine}}"
TEST_DIR="$(mktemp -d)"
trap 'rm -rf "$TEST_DIR"' EXIT
mkdir -p "$TEST_DIR/llama.cpp"
printf '[*]\ndevice = lattice-config-test-invalid\n' > "$TEST_DIR/llama.cpp/config.ini"

XDG_CONFIG_HOME="$TEST_DIR" "$ENGINE_DIR/bin/llama-cli" --list-devices \
    > "$TEST_DIR/devices" 2> "$TEST_DIR/log"
grep -Fq 'Available devices:' "$TEST_DIR/devices"
if XDG_CONFIG_HOME="$TEST_DIR" "$ENGINE_DIR/bin/llama-cli" \
    --device lattice-config-test-invalid --list-devices > "$TEST_DIR/output" 2>&1; then
    printf 'Explicit invalid device was accepted\n' >&2
    exit 1
fi
grep -Fq 'invalid device: lattice-config-test-invalid' "$TEST_DIR/output"
printf 'Managed engine ignores host configuration and still validates explicit devices\n'
