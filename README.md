# Lattice

Run bigger local models by pooling the machines you already own. See
[PLAN.md](PLAN.md) for the design and the reasoning behind it.

## Status

The prototype supports paired networking and distributed terminal chat:

- Ed25519 device identity, one keypair per install
- mDNS discovery on the LAN
- SPAKE2 pairing over a six-digit code, then pinned mutual TLS
- QUIC transport with typed streams, so any device can open any number of
  channels to any other
- a master that introduces devices to each other, so followers talk directly
  rather than relaying (see [docs/streams.md](docs/streams.md))
- link measurement: RTT, throughput, and the per-boundary cost from PLAN.md §1
- GPU chat using Metal on macOS and Vulkan on Linux, with automatic tensor
  transfer over authenticated QUIC and placement based on reported GPU memory

Verified on macOS/aarch64 and Linux/aarch64.

## Setup

```
git clone https://github.com/shantanu319/p2p-distributed-inference.git
cd p2p-distributed-inference
./scripts/setup.sh
```

The script checks what is missing, builds, and installs `latticed` to
`~/.local/bin`. It never installs system packages — it prints the command for
your distro and stops. `--check` reports prerequisites without building;
`--serve` starts serving once installed.

Requirements: Rust >= 1.85 (edition 2024) and a C compiler, which `ring` needs
for its assembly. Distro Rust packages are often too old; rustup is safer.

## Connecting two machines

On the machine you want others to join:

```bash
latticed master
```

On each other machine:

```bash
latticed worker
```

The worker discovers the master and asks for its six-digit code on first use.
Existing pairings are reused. Leave both commands running; the worker remembers
its master and reconnects after either process restarts. The master accepts
multiple workers, introduces them to each other, and tests each worker's inbound
listener. A successful check prints `verified both ways` with RTT and throughput.

These commands use fixed UDP ports: **47900** for serving, **47901** for master
pairing, and **5353** for mDNS discovery. On Linux, they configure an active UFW
or firewalld through sudo, limiting access to directly connected IPv4 subnets.
They do not enable disabled firewalls. Use `--firewall print` to see the rules
without changing them, or `--firewall off` if you manage the firewall yourself.

If discovery is blocked, provide the master's IPv4 address. If there are multiple
masters, choose an address or a device ID from `latticed discover`:

```bash
latticed worker --master 192.168.7.40
```

Use `--port` to change the serving port and `--pairing-port` to change the master's
pairing port. For an explicit address with custom ports:

```bash
latticed master --port 4242 --pairing-port 4243
latticed worker --master 192.168.7.40:4242 --pairing-port 4243
```

The pairing code changes after each attempt. Join new workers one at a time using
the latest code. `--code 123456` supplies it for a noninteractive first start.
Both commands also accept `--model path/to/model.gguf`.

The setup script installs `latticed-master` and `latticed-worker` shell launchers
alongside the binary. They forward the same options; in a checkout you can also
run `./scripts/latticed-master` and `./scripts/latticed-worker` after building.

The lower-level `host`, `pair`, `serve`, `discover`, `peers`, `probe`, and
`provision` commands remain available for manual operation. Stop old `serve`
processes before starting a master or worker, even if they use different ports:
one device identity should have one running daemon.

## GPU chat

Build the inference runtime on both machines:

```bash
./scripts/setup.sh --inference
latticed devices
```

This builds the same pinned llama.cpp revision with Metal on macOS or Vulkan
on Linux. The installer prints any missing system dependencies. Gaming's RX
7600 uses its existing RADV driver; ROCm is not required for this backend.

Restart the worker with GPU inference required:

```bash
latticed worker --require-gpu
```

On the master, leave `latticed master` running and open another terminal:

```bash
latticed chat --model /path/to/model.gguf
```

For a small test model, download SmolLM2-135M-Instruct Q4_K_M (135M
parameters, 105 MB) on the master. The script verifies its pinned SHA-256 and
reuses a matching cached file:

```bash
MODEL="$(./scripts/fetch-test-model.sh)"
latticed chat --model "$MODEL" --context 2048 --tokens 64
```

The GGUF only needs to exist on the master. Chat discovers paired workers,
checks engine versions and live GPU capabilities, assigns contiguous layers,
and transfers their tensors automatically. Each selected GPU receives transformer
work. The discrete GPU is preferred over a Ryzen integrated GPU; `--device`
selects an exact name from `latticed devices` when needed.

For a single response or explicit worker selection:

```bash
latticed chat --model /path/to/model.gguf --worker 192.168.7.62:47900 \
  --context 2048 --tokens 128 --prompt "Explain why the sky is blue."
```

Repeat `--worker` for multiple devices. `--local` explicitly runs on the master's
GPU alone. A missing worker or GPU is an error; chat does not silently switch to
local CPU inference. `/exit` or Ctrl-C stops chat.

The first milestone supports a single-file Llama-family GGUF and one active
chat per worker. It uses backend graph RPC over a dedicated QUIC stream.
The existing Candle activation protocol is a separate path. See
[docs/inference.md](docs/inference.md) for placement, cache, and validation details.

## Manual activation pipeline

The older `generate` command needs the same GGUF on both machines, so
copy it across yourself. The hash is checked before anything loads, so two
machines holding different weights is refused rather than discovered later as
fluent nonsense.

On the machine that will hold the far half:

```
latticed serve --model llama.gguf
```

On the machine you are sitting at — which is the master, whatever it holds:

```
latticed generate --model llama.gguf --prompt 1,15043,29892 \
    --place 0-11@local --place 11-22@<device-id>
```

`--place first-last@where` is repeated so the ranges tile the model exactly;
`where` is `local` or a paired device's id. A gap would drop layers and an
overlap would run them twice, so both are refused. Omit `--place` entirely to
run everything here.

For this older command, prompts and output are token IDs and sampling is greedy.
Use `chat` above for tokenization, conversation templates, and readable text.

**Expect the token stream to diverge across backends.** Metal and CUDA and CPU
agree for a few dozen tokens at temperature 0 and then part ways — measured at
token 39 on TinyLlama between this machine's own CPU and GPU. A Mac-plus-Linux
split crosses backends by construction, so a prefix match is the most that can
be asserted. See PLAN.md §12 and bench/RESULTS.md.

## Firewall and network

Discovery needs multicast (UDP 5353) and the QUIC port you serve on. The master
and worker commands configure supported active Linux firewalls. Networks that block
multicast — many corporate and guest networks do — will not discover anything;
every command also accepts a plain `host:port`.

Listeners currently bind IPv4 only.

## Measuring

[bench/README.md](bench/README.md) has the two-machine procedure and how to
read the numbers against PLAN.md §1. Measure with a release build and take
several samples; run-to-run spread is wide.
