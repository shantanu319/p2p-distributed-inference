# Channels between devices — a model

Answering two questions: what carries bytes between devices, and which devices
are allowed to talk to each other. No code yet.

---

## 1. WebSockets would be a downgrade, because QUIC already does this

The ask — "n channels for bytestreaming between two devices" — is the thing
QUIC streams *are*. We already hold one authenticated QUIC connection per peer.
Opening another channel on it is one call and no handshake.

| | QUIC streams (have) | WebSockets (proposed) |
| --- | --- | --- |
| n channels | n streams on one connection | n sockets, each its own TCP + TLS + HTTP upgrade |
| loss on channel A | only A stalls | **all n channels stall** — one TCP bytestream |
| congestion control | one controller for the peer | n controllers competing with each other |
| auth | pinned Ed25519, already done | would have to be rebuilt on top |
| channel setup | ~0, no round trip | full handshake per channel |

The head-of-line row is the one that decides it. §7 chose QUIC because Wi-Fi
gives packet loss and AP roaming. On WebSockets, a single lost packet belonging
to a 40 GB model transfer would stall every activation frame behind it, because
they share one TCP sequence space. That is the exact failure QUIC's per-stream
recovery exists to prevent, and on the network §12 warns about it would be a
routine event, not an edge case.

**Where WebSockets are still correct: the UI.** Browsers cannot speak raw QUIC,
and §4.3 already specifies a WebSocket from the TS frontend to `latticed` on
localhost. That stays. It is a different problem — localhost, no loss, one
consumer — and nothing about it should leak into the device-to-device path.

(If we ever wanted a browser to stream *directly* from a device, the answer
would be WebTransport over HTTP/3, not WebSockets. Not needed for anything on
the roadmap.)

---

## 2. Star vs mesh: followers must talk to followers

Your instinct that follower↔follower might be needed is right, and it is
stronger than "in case" — for pipeline parallelism it is the **normal** path,
not an exception.

§1 fixes the dataflow: contiguous layer ranges, batch size 1, activations
flowing along a chain.

```
master ──tokens──▶ shard A ──16 KB──▶ shard B ──16 KB──▶ shard C ──logits──▶ master
```

Relaying that through the master turns every hop into two:

```
master ──▶ A ──▶ master ──▶ B ──▶ master ──▶ C ──▶ master
```

For `n` devices holding layers, counting boundary crossings per token:

| followers | mesh (`n+1`) | star (`2n`) | extra cost at 4 ms/crossing |
| --- | --- | --- | --- |
| 2 | 3 | 4 | +4 ms/token |
| 3 | 4 | 6 | +8 ms/token |
| 4 | 5 | 8 | +12 ms/token |

Against §1's ~40 ms of per-token compute, the star turns a ~25% network tax
into ~45% at three followers. It is worse than the table suggests, because §3
says the master may be the weakest device and may hold zero layers — a star
routes `2n` copies of every activation through exactly the machine least able
to serve them, and serialises them behind one uplink.

**So: mesh for the data plane, star for the control plane.** Those are
different questions and they get different answers.

- **Control plane is a star.** The master owns the plan, the generation
  counter, and the session (§3). Followers report to it and take instructions
  from it. Small messages, latency-insensitive.
- **Data plane is a mesh.** Activations go directly between adjacent shards.
  The master is on this path only at the two ends (tokens in, logits out).

The mesh is small: with the ≤4 devices §5 targets, at most 6 connections. This
is not a distributed-systems problem, it is a lookup table.

---

## 3. Connection layer

One long-lived QUIC connection per *pair* of devices that the current plan
makes adjacent, plus master↔every follower for control.

- **Who dials.** Both sides discovering each other simultaneously would open
  two connections. Rule: **the lower `DeviceId` dials the higher.** Total
  order, no negotiation, no race.
- **Lifetime.** Established when the plan makes two devices adjacent, kept warm
  across sessions. Re-planning may add or drop pairs; it should not tear down
  connections that survive the change.
- **Keepalive is not optional.** Measured quinn defaults: `max_idle_timeout` is
  30 s and `keep_alive_interval` is `None`. An idle mesh connection therefore
  dies 30 s after the last request, and the next request pays a full handshake.
  Set a keepalive well under the idle timeout.

---

## 4. Stream layer: a header, and a demultiplexer

Today `Connection::open_stream` dispatches on dialer/listener role. That works
for a one-shot handshake and does not generalise: in a mesh either side may
need to originate a channel.

Replace it with a rule that every stream opens by declaring itself:

```
u32  magic
u16  version
u16  kind             // control | activation | bulk
u64  session_id
u32  shard_generation
```

The accepting side reads the header and routes the stream. That *is* the "n
channels" mechanism — the peer opens as many as it likes, each self-describing.
It generalises §6's activation frame rather than replacing it; §6's per-frame
header keeps describing payloads within an activation stream.

Three kinds, and they want different treatment:

| kind | lifetime | carries |
| --- | --- | --- |
| **control** | one per connection, permanent | heartbeat, plan push, generation bump, load/unload shard, drain notice |
| **activation** | one per session per boundary | §6 frames, prefill and decode |
| **bulk** | ephemeral | GGUF transfer between peers (§8), probe payloads |

**One activation stream per session, not per token.** A session is inherently
sequential, so in-order delivery is what we want and head-of-line blocking
*within* one session costs nothing. Per-token streams would pay setup on every
token for no benefit.

---

## 5. Three measured defaults that will bite

These are quinn's actual defaults, not guesses:

**`stream_receive_window` is 1.25 MB.** Sustaining a given rate needs a window
of at least bandwidth × RTT. §12's bad case — a congested 2.4 GHz link at
30–50 ms — needs 50 MB/s × 50 ms = 2.5 MB. The default caps that stream at
~25 MB/s, so *the transport becomes the bottleneck before the link does*, and
we would misdiagnose it as the network. This shows up on prefill, where §1
already flags 67 MB crossing a boundary as the one place bandwidth genuinely
hurts.

**`send_fairness` is `true`.** By default a 40 GB model transfer gets an equal
share against a 16 KB activation frame. Peer-to-peer model sync (§8) would
wreck inference latency while it ran. Streams need explicit priority:
activation above control above bulk.

**`max_concurrent_bidi_streams` is 100.** A ceiling on
concurrent sessions × boundaries, which matters once M4 adds microbatching.

---

## 6. Fencing and failure

`shard_generation` (§6) goes in the stream header and is checked *at stream
open*, so a stale stream is rejected before any payload moves — cheaper than
per-frame checks and it kills the whole channel at once.

Two failures that must not be conflated:

- **stream reset** — one session's channel died. Fail that session.
- **connection lost** — the device is gone. This is what triggers §9's
  re-plan-and-re-prefill.

---

## 7. The open question this forces: who may talk to whom

A mesh means every adjacent pair needs mutual trust, and trust today is
strictly pairwise — pinned Ed25519 keys from a SPAKE2 exchange. With 4 devices
a full mesh is 6 pairings, so the user types six codes.

Two ways out, and this is a real decision, not a detail:

- **Pairwise.** Every pair pairs. Trust stays exactly as strong as it is now:
  compromising one device gives an attacker only that device's peers. Cost is
  `n(n-1)/2` code exchanges, one time — 1 for two devices, 3 for three, 6 for
  four.
- **Cluster roster.** Devices pair once into a cluster and vouch for each
  other; a device trusts keys signed into the roster by a device it already
  trusts. One pairing per new device. Cost: trust becomes transitive, so
  compromising one device admits the attacker to all of them, and it needs a
  stable signing identity — which the current "master is whoever received the
  request" model deliberately does not have.

§2 says peers are your own machines and trusted once paired, which argues for
the roster. §7 says coffee-shop Wi-Fi is a hostile LAN, which argues for
pairwise. They are in tension and the answer should be chosen, not defaulted
into.
