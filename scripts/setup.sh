#!/usr/bin/env bash
#
# Builds and installs latticed, checking the things that actually go wrong
# on a fresh machine. Safe to re-run.
#
#   ./scripts/setup.sh            build and install to ~/.local/bin
#   ./scripts/setup.sh --serve    ...then start serving on the LAN
#   ./scripts/setup.sh --check    only report what is missing
#
# Never installs system packages: it prints the command for your distro and
# stops, so nothing runs as root that you did not type yourself.

set -euo pipefail

MIN_RUST="1.85.0"          # edition 2024
PREFIX="${PREFIX:-$HOME/.local/bin}"
MODE="install"

for arg in "$@"; do
    case "$arg" in
        --serve) MODE="serve" ;;
        --check) MODE="check" ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ok()   { printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad()  { printf '  \033[31mmissing\033[0m %s\n' "$1"; }
note() { printf '\n%s\n' "$1"; }
MISSING=0

# --- what package manager should we point at? -------------------------------
install_hint() {
    local pkgs="$1"
    if   command -v apt-get >/dev/null; then echo "sudo apt-get install -y $pkgs"
    elif command -v dnf     >/dev/null; then echo "sudo dnf install -y $pkgs"
    elif command -v pacman  >/dev/null; then echo "sudo pacman -S --needed $pkgs"
    elif command -v zypper  >/dev/null; then echo "sudo zypper install -y $pkgs"
    elif command -v apk     >/dev/null; then echo "sudo apk add $pkgs"
    else echo "install: $pkgs"
    fi
}

echo "Checking prerequisites on $(uname -s)/$(uname -m)"

# --- a C compiler, for ring's assembly ---------------------------------------
if command -v cc >/dev/null || command -v gcc >/dev/null || command -v clang >/dev/null; then
    ok "C compiler (ring compiles C and assembly)"
else
    bad "C compiler — ring cannot build without one"
    note "  $(install_hint "build-essential")"
    MISSING=1
fi

# --- rust, new enough for edition 2024 ---------------------------------------
if command -v cargo >/dev/null && command -v rustc >/dev/null; then
    HAVE="$(rustc --version | awk '{print $2}' | cut -d- -f1)"
    OLDEST="$(printf '%s\n%s\n' "$MIN_RUST" "$HAVE" | sort -V | head -1)"
    if [ "$OLDEST" = "$MIN_RUST" ]; then
        ok "rustc $HAVE (need >= $MIN_RUST)"
    else
        bad "rustc $HAVE is too old — edition 2024 needs >= $MIN_RUST"
        note "  rustup update stable    # or install rustup: https://rustup.rs"
        MISSING=1
    fi
else
    bad "cargo/rustc"
    note "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    note "  Distro rust packages are often too old for edition 2024; rustup is safer."
    MISSING=1
fi

if [ "$MISSING" -ne 0 ]; then
    note "Install the above, then run this script again."
    exit 1
fi
[ "$MODE" = "check" ] && { note "All prerequisites present."; exit 0; }

# --- build -------------------------------------------------------------------
note "Building (first build fetches crates and takes a few minutes)"
cargo build --release --locked --manifest-path "$REPO/Cargo.toml" -p latticed

# Respect CARGO_TARGET_DIR; the binary is not always under the repo.
BUILT="${CARGO_TARGET_DIR:-$REPO/target}/release/latticed"
mkdir -p "$PREFIX"
install -m 0755 "$BUILT" "$PREFIX/latticed"
install -m 0755 "$REPO/scripts/latticed-master" "$PREFIX/latticed-master"
install -m 0755 "$REPO/scripts/latticed-worker" "$PREFIX/latticed-worker"
ok "installed $PREFIX/latticed"

case ":$PATH:" in
    *":$PREFIX:"*) ;;
    *) note "$PREFIX is not on your PATH. Add it:
  echo 'export PATH=\"$PREFIX:\$PATH\"' >> ~/.bashrc && exec bash" ;;
esac

"$PREFIX/latticed" id

note "On your main machine:
  latticed master

On each other machine:
  latticed worker

The commands discover peers and configure supported Linux firewalls for your LAN.
For the first connection, enter the code shown by the master on the worker.
The latticed-master and latticed-worker shell launchers work the same way."

if [ "$MODE" = "serve" ]; then
    note "Starting: latticed serve --port 47900   (ctrl-c to stop)"
    exec "$PREFIX/latticed" serve --port 47900
fi
