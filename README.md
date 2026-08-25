# Lattice

Run bigger local models by pooling the machines you already own. See
[PLAN.md](PLAN.md) for the design and the reasoning behind it.

## Status

**No inference yet.** What exists is the network layer the rest will sit on:

- Ed25519 device identity, one keypair per install
- mDNS discovery on the LAN
- SPAKE2 pairing over a six-digit code, then pinned mutual TLS
- QUIC transport with typed streams, so any device can open any number of
  channels to any other
- a master that introduces devices to each other, so followers talk directly
  rather than relaying (see [docs/streams.md](docs/streams.md))
- link measurement: RTT, throughput, and the per-boundary cost from PLAN.md §1

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

```
latticed host              # prints a six-digit code and the command to run
```

On the other machine, run the command it printed. Then:

```
latticed serve             # advertise on the LAN and serve paired devices
latticed discover          # list devices, with the address a dial would use
latticed peers             # paired devices, and how each came to be trusted
latticed probe <device-id> # measure the link
```

With three or more machines, pair each one to the *same* machine, then run
`latticed provision` there. Every device receives the others' keys and pins
them, so no one has to type six codes for four machines.

## Firewall and network

Discovery needs multicast (UDP 5353) and the QUIC port you serve on. The setup
script detects `ufw` and `firewalld` and prints the rules. Networks that block
multicast — many corporate and guest networks do — will not discover anything;
every command also accepts a plain `host:port`.

Listeners currently bind IPv4 only.

## Measuring

[bench/README.md](bench/README.md) has the two-machine procedure and how to
read the numbers against PLAN.md §1. Measure with a release build and take
several samples; run-to-run spread is wide.
