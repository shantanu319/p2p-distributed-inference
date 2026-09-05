#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 1 ]; then
    printf 'Usage: fetch-test-model.sh [DESTINATION_DIRECTORY]\n' >&2
    exit 2
fi
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    printf '%s\n' 'Usage: fetch-test-model.sh [DESTINATION_DIRECTORY]' \
        'Downloads and verifies SmolLM2-135M-Instruct Q4_K_M (135M parameters, 105 MB).' \
        'Prints the verified model path to stdout; reuses a matching cached file.'
    exit 0
fi

DESTINATION="${1:-${XDG_CACHE_HOME:-$HOME/.cache}/lattice/models}"
FILENAME=SmolLM2-135M-Instruct-Q4_K_M.gguf
EXPECTED=ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747
URL="https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/resolve/main/$FILENAME"
if command -v sha256sum >/dev/null; then
    HASH=(sha256sum)
elif command -v shasum >/dev/null; then
    HASH=(shasum -a 256)
else
    printf 'Install sha256sum or shasum to verify the model download.\n' >&2
    exit 1
fi
checksum() { "${HASH[@]}" < "$1" | awk '{print $1}'; }

mkdir -p -- "$DESTINATION"
DESTINATION="$(cd "$DESTINATION" && pwd)"
MODEL="$DESTINATION/$FILENAME"
if [ -f "$MODEL" ] && [ "$(checksum "$MODEL")" = "$EXPECTED" ]; then
    printf '%s\n' "$MODEL"
    exit 0
fi
if ! command -v curl >/dev/null; then
    printf 'Install curl to download the test model.\n' >&2
    exit 1
fi
TEMP="$(mktemp "$DESTINATION/.$FILENAME.XXXXXX")"
trap 'rm -f -- "$TEMP"' EXIT
printf 'Downloading SmolLM2-135M-Instruct Q4_K_M (105 MB).\n' >&2
curl --fail --location --retry 3 --proto '=https' --proto-redir '=https' \
    --output "$TEMP" "$URL"
if [ "$(checksum "$TEMP")" != "$EXPECTED" ]; then
    printf 'Model SHA-256 did not match; the download was discarded.\n' >&2
    exit 1
fi
mv -f -- "$TEMP" "$MODEL"
printf '%s\n' "$MODEL"
