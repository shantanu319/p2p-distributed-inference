# Measuring a real link

What this measures is §7 and the network half of §1: whether two machines find
each other, pair, and what the link between them actually costs. It does not
touch a model, so it cannot answer M0's decision gate — that still needs
weights on both machines.

## Build

Rust toolchain only. The rustls provider is `ring` rather than `aws-lc-rs`
specifically so the Linux box needs no cmake or C toolchain:

```
cargo build --release -p latticed
```

## Pair the two machines

On the machine you want to reach (say the Linux box):

```
latticed host
```

It prints a six-digit code and the exact command to run on the other machine.
On the Mac, run that command. Both sides print `paired with ...` and write the
peer into `trusted_devices.json`, which you can read.

The code is single-use and one guess per attempt — a wrong digit fails the
exchange rather than pairing weakly.

## Measure

Leave the Linux box serving:

```
latticed serve
```

From the Mac:

```
latticed discover          # confirms mDNS works on this network
latticed probe <device-id> --mib 32
```

`probe` reports RTT, throughput, and the derived per-boundary decode cost from
§1. Run it on Wi-Fi and again on ethernet.

## What the numbers mean

§1 assumes 3–5 ms per boundary crossing per token on a decent 5 GHz link, and
calls the result a 10–25% tax rather than a wall. The `decode hop` line is that
number for your actual network.

Record, for both Wi-Fi and ethernet:

| link | rtt | throughput | decode hop (8192) |
| --- | --- | --- | --- |

- **decode hop under ~5 ms** — §1 holds, network is a tax.
- **decode hop 10–30 ms** — the 15% tax is closer to 100%. Worth checking
  whether the laptop is on 2.4 GHz or far from the AP.
- **throughput well under 50 MB/s** — prefill is where this bites, not decode:
  a 4K-token prompt moves 67 MB across each boundary (§1).

Compare ethernet against Wi-Fi before concluding anything about the design.
"Plug in a cable" is legitimate advice the product should be willing to give.

## Known limits of this harness

- Listeners bind IPv4 only. Peers are dialed on every address they advertise,
  routable before loopback and IPv4 before IPv6, but an IPv6-only peer is not
  yet reachable.
- `discover` assumes multicast is not blocked. On networks that filter mDNS,
  pass `host:port` directly to `pair` and `probe` — every command accepts one.
