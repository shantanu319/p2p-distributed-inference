# Lattice — run bigger local models by pooling the machines you already own

> Working name. `lattice` / `latticed` throughout; swap it before the first public commit.

## The pitch, in one paragraph

You have a MacBook and a desktop. Or two Macs. Or a laptop and a Linux box with a
3090 in it. Individually, none of them can hold a 70B model in memory, so you either
run a smaller model or you let llama.cpp mmap the weights off SSD and watch it emit
one token per second. Lattice is an app you install on both machines. They find each
other on your Wi-Fi, split the model between them, and serve it at an OpenAI-compatible
endpoint. No account, no cloud, no config file.

**The win we are selling is memory capacity, not speed.** Two 32 GB Macs running a
70B at Q4 should land somewhere around 8–12 tok/s. One 32 GB Mac swapping to disk
lands near 1 tok/s. That ~10x is the entire product, and Milestone 0 exists to
confirm it before we build anything else.

---

## 1. The physics of this problem (and why it is a different problem from the WAN version)

Everything below follows from four numbers. They are worth internalising because they
kill several "obvious" design choices.

**Decode activation size.** One token crossing one shard boundary for a 70B-class
model (hidden dim 8192, fp16) is `8192 × 2 = 16 KB`. On Wi-Fi 6 at a realistic
50 MB/s that is 0.3 ms of serialisation plus ~2–3 ms of round-trip. Call it **3–5 ms
per boundary crossing per token**. Per-token compute on a shard is tens of
milliseconds. So on a LAN, **network overhead is a 10–25% tax, not a wall.** This is
the single fact that makes the project viable.

**Prefill activation size.** A 4K-token prompt crossing one boundary is
`4096 × 8192 × 2 = 67 MB`. At 50 MB/s that is **1.3 seconds** of pure transfer added
to time-to-first-token. On gigabit ethernet, 0.5 s. This is the one place where
bandwidth genuinely hurts, and it is why fp8 wire dtype and chunked-prefill overlap
are on the roadmap and lossy activation compression is not.

**Pipeline parallelism gives a single user zero throughput gain.** With batch size 1,
exactly one device is computing at any instant; the others are idle waiting for the
activation to arrive. Total per-token latency is the *sum* of every shard's compute
plus every transfer:

```
TPOT ≈ Σ_devices compute_d  +  Σ_boundaries transfer_b
```

There is no pipelining to exploit until there are concurrent requests. **This inverts
the standard layer-splitting objective.** Datacenter pipeline schedulers minimise the
*bottleneck* stage to maximise throughput. We must minimise the *sum*, which means:

> Put as many layers as possible on the fastest device, fill it to its memory limit,
> then spill the remainder to the next-fastest. Do not balance the stages.

Balancing stages is actively wrong here and would make Lattice slower than necessary
on the very common asymmetric case (fast desktop + slow laptop). See §5.

**Tensor parallelism is off the table.** It requires an all-reduce inside every layer.
80 layers × 2 collectives × even 2 ms of LAN round-trip is 320 ms per token before any
compute happens. Pipeline (layer-range) sharding only. Mac-to-Mac over a Thunderbolt
bridge could eventually justify TP; it is explicitly out of scope.

---

## 2. Prior art, honestly

| Project | What it does | Why we are still building this |
| --- | --- | --- |
| `llama.cpp --rpc` / `rpc-server` | Ships GGML graph ops to remote backends | Works, but upstream labels it proof-of-concept with **no authentication or input validation** — it is a documented RCE surface on your LAN. Chatty per-graph round trips. No discovery, no UI, no model sync. |
| **exo** | Python P2P cluster for local LLMs, auto-discovery, ring pipeline | Closest prior art and the clearest proof the demand exists. Python runtime, heavy install, maintenance has stalled. |
| **distributed-llama** | C++ TCP tensor+pipeline parallel over ethernet | CLI-only, manual topology, no model management. |
| **MLX `mx.distributed`** | Ring/MPI collectives across Apple Silicon | Apple-only, so it cannot bridge a Mac and a CUDA box. |
| **Petals** | Layer-sharded inference over the public internet | Different problem (WAN, untrusted, multi-tenant). We deliberately keep none of its recovery machinery. |

The gap Lattice fills: **a signed, one-click, cross-platform app with zero-config
discovery, authenticated transport, automatic layer planning, and an OpenAI-compatible
endpoint — that works across Metal and CUDA at the same time.** Nobody has shipped that
combination.

Because every device here is *your own device*, three enormous problem categories from
the WAN design evaporate: no verification/fraud detection, no accounting, no
byzantine-fault assumptions. Peers are trusted once paired. Keep it that way.

---

## 3. System shape

Every install is the same binary. There is no server. Roles are per-session and
negotiated, not configured.

```text
┌──────────────────────────── Device A (master for this session) ───────────┐
│                                                                            │
│   TS UI  ──HTTP/localhost──▶  latticed                                      │
│   (Tauri shell or browser)         │                                        │
│                                    ├── OpenAI-compatible API :11434-ish     │
│   Any OpenAI client ───────────────┤                                        │
│   (Zed, Continue, Open WebUI)      │                                        │
│                                    ├── mDNS discovery + pairing             │
│                                    ├── planner (layer split)                │
│                                    ├── session driver (sampling, tokens)    │
│                                    └── shard executor  [layers 0..47]       │
│                                              │                              │
└──────────────────────────────────────────────┼──────────────────────────────┘
                                               │ QUIC (mTLS, Ed25519 device keys)
                                               │ 16 KB activation frames
┌──────────────────────────────────────────────▼──────────────────────────────┐
│  Device B                          latticed                                  │
│                                    └── shard executor  [layers 48..79 + head]│
└──────────────────────────────────────────────────────────────────────────────┘
```

### Master and followers

**Master** = whichever device received the request. Every device can be a master; the
role is claimed per session, never configured. In practice this means the machine you
are sitting at is always the master, which conveniently puts the latency-sensitive
sampling step at zero network distance.

The master owns everything the user can observe:

- the HTTP surface the user or their OpenAI client talks to, and the UI
- tokenization, sampling, and the token history for the session
- the plan for the session, and the `shard_generation` counter that fences it (§6)
- the decode loop: it drives the shard chain and aggregates what comes back
- failure handling — it is the only device with enough state to re-plan (§9)

**Follower** = holds a contiguous layer range and its KV cache for that range, executes
`Prefill`/`Decode` for the sessions it has been given, and reports telemetry upstream.
Stateless with respect to everything except KV. A follower never talks to the user, never
samples, and never decides the plan.

**The master does not have to be the fastest device, and may hold zero layers.** These
are separate questions: who the user is talking to, and where the weights fit. §5's
planner fills the fastest device first, so on a fast-desktop/slow-laptop pair, a request
issued from the laptop makes the laptop the master while the desktop holds most or all
of the model. That is the intended behaviour, not a degenerate case — the master's job
is kilobytes of state and one sampling step per token, which even a weak machine does in
well under the time a boundary crossing costs.

The corollary is that there is **no cluster-wide leader and no shared cluster state to
keep consistent.** Two devices can each be master of their own concurrent session
against overlapping followers. Nothing needs to agree, because everything a master owns
is scoped to its own session.

---

## 4. Components

### 4.1 `latticed` — the Rust daemon

One binary, one process, no privileged install. Runs as a user-level service
(launchd agent on macOS, systemd user unit on Linux).

Responsibilities:
- device identity (Ed25519 keypair in the OS keychain / `~/.local/share/lattice`)
- LAN discovery and pairing
- peer connection management over QUIC
- model catalogue, download, and LAN-local model transfer
- the layer planner
- session driving (tokenize → shard chain → sample → repeat)
- the shard executor (via FFI into the inference core)
- OpenAI-compatible HTTP API
- serving the TS UI on localhost
- resource policy (memory ceiling, "pause while on battery", "pause while I'm gaming")

Crate picks:

| Concern | Crate |
| --- | --- |
| async runtime | `tokio` |
| QUIC | `quinn` + `rustls` |
| HTTP / API / static UI | `axum` |
| mDNS-SD | `mdns-sd` |
| control-message codec | `postcard` (compact, no schema compiler, `serde`-native) |
| identity | `ed25519-dalek`, `rcgen` for the self-signed cert wrapping the device key |
| pairing PAKE | `spake2` |
| GGUF parsing | `gguf-rs` or hand-rolled — the header format is simple |
| FFI build | `cc` / `cmake` crates |
| observability | `tracing` + `tracing-subscriber` |

### 4.2 The inference core

This is the highest-risk decision in the project, so it gets a measured decision
point rather than an upfront guess. See Milestone 0.

- **Option A — wrap `llama.cpp` `rpc-server`.** Fastest path to a working product;
  our value-add is discovery, auth, planning, UI, model sync. Inherits upstream's
  chattiness and its security posture (which we would firewall behind our own
  authenticated tunnel, so the RCE surface is closed by construction).
- **Option B — a thin C++ shim over `ggml`.** Expose exactly the six calls in §6 and
  nothing else. ~1500 lines. Full control of the wire boundary, fp8 activations,
  chunked prefill, proper KV ownership. This is where the project's real engineering
  value lives.

**Recommendation: A to ship, B to win.** Build A first because it makes Milestone 1
demoable in weeks and gives us a real performance baseline. Replace with B once we can
point at a specific measured gap. Do not start with B — you will spend three months in
`ggml` graph-building before ever seeing two machines talk.

Backends, in priority order: **Metal** (macOS), **CUDA** (Linux), **CPU** (fallback,
always built). Vulkan and ROCm later, community-contributed if possible.

### 4.3 The TS frontend

`latticed` serves the built UI from `axum` at `http://127.0.0.1:<port>`. Tauri v2 is a
thin shell around that same URL, giving the app icon, tray menu, and autostart.

This dual-mode is deliberate: the Tauri app is what a Mac user downloads, and the plain
browser URL is what you use on a headless Linux box — or from your phone, on the same
LAN, which is a genuinely nice demo.

Stack: Vite + React + TypeScript + Tailwind. State over a WebSocket to `latticed`.

Screens (keep it to four):
1. **Devices** — discovered peers, pair/unpair, per-device memory and measured speed,
   live "this device is holding layers 48–79" badge.
2. **Models** — installed GGUFs, size, whether the current device set can hold them,
   download progress, "pull from peer over LAN" when a peer already has the file.
3. **Chat** — a minimal chat pane so the product works with zero other software.
4. **Settings** — memory ceiling, battery/thermal policy, API port, log level.

Plus a persistent status strip: current plan, tok/s, TTFT, per-hop network time.
Surfacing the split live is what makes the product feel like magic instead of a config
file.

---

## 5. The planner

With 2–4 known devices this is small enough to solve exactly. No beam search, no
heuristics.

**Inputs per device `d`:** usable memory `M_d` (physical minus OS reserve minus the
user's ceiling), measured decode throughput, measured prefill throughput, measured
pairwise bandwidth and RTT.

**Memory feasibility** for device `d` holding layers `p..q`:

```
W[p..q] + KV[p..q](C) + workspace_d  ≤  ρ · M_d       (ρ ≈ 0.85)
```

`KV[p..q](C)` must be included at the *user's configured max context*, not at the
current context. Omitting it is the classic failure where a model loads fine and then
OOMs 6000 tokens into a conversation.

**Objective.** Minimise single-stream per-token latency:

```
minimise   Σ_d  layers_d · t_decode_per_layer(d)  +  Σ_boundaries  (16 KB / bw + rtt)
```

Because `t_decode_per_layer` is a per-device constant, the sum is minimised by
assigning layers greedily to the fastest device until memory runs out, then spilling.
Fewer boundaries is also strictly better, so the planner should **prefer using fewer
devices** — if the model fits on one machine, use one machine and take the network out
of the loop entirely.

The planner is indifferent to which device is the master. Layer placement is decided by
speed and memory alone; assigning the master zero layers is a normal outcome (§3).

Practical algorithm:

```
1. sort devices by measured decode throughput, descending
2. if model + KV fits on device[0] alone → single-device plan, done
3. otherwise fill devices in order, largest-share-to-fastest, until all layers placed
4. if the layers don't all fit → report the shortfall in the UI, suggest a smaller
   quant, and refuse to start (do NOT silently fall back to disk-swapping)
5. run the exact DP over contiguous splits as a check; with ≤4 devices and ≤128 layers
   this is trivially cheap and catches cases where transfer cost changes the ordering
```

Step 5's DP, for completeness:

```
dp[j][l] = min over p<l of ( dp[j-1][p] + compute(device_j, p..l) + transfer(j→j+1) )
```

subject to the memory constraint. Note the `+` where the throughput-oriented version
would have a `max` — that difference *is* the insight from §1.

**Re-planning.** Only on membership change (device joins, leaves, sleeps) or model
change. Never mid-session. Hysteresis: do not re-plan for a device that has been stable
for under 30 seconds.

---

## 6. The shard ABI

Deliberately tiny. Six calls.

```rust
LoadShard  { model_hash, layer_range, max_context, kv_dtype } -> ShardId
UnloadShard{ shard_id }

Prefill    { session, shard, pos_range, activations }        -> activations
Decode     { session, shard, pos, activations }              -> activations
DropSession{ session }
Stats      {}                                                -> ShardStats
```

The first shard's `Prefill`/`Decode` input is token IDs rather than activations (it
owns the embedding); the last shard's output is logits or a sampled token (it owns the
output head). Everything between is hidden states in, hidden states out.

**Activation frame** — raw binary, not JSON, not protobuf:

```
u32  magic
u16  version
u16  flags            (dtype, compression)
u64  session_id
u32  shard_generation  // bumped on re-plan; stale frames rejected
u32  pos_start, pos_end
u32  n_tokens, hidden_dim
u32  payload_len
u32  crc32
[payload bytes]
```

`shard_generation` is the entire replacement for the WAN plan's placement-epoch
machinery. It is a `u32` counter **owned by that session's master** and scoped to that
session — a follower tracks `(session_id, generation)` and drops any frame whose
generation is behind what it has already seen for that session. Because the counter never
crosses session boundaries, two masters re-planning concurrently cannot race: each bumps
only its own. That is all the fencing a trusted LAN needs.

Wire dtype: **fp16 in v1, fp8 behind a flag once we can A/B it.** fp8 halves prefill
transfer time, which is the only place bandwidth genuinely bites — but it needs a
quality check per model family before it goes on by default.

---

## 7. Discovery, pairing, transport

**Discovery.** mDNS-SD, service type `_lattice._tcp`. TXT records advertise device
name, platform, backend, total memory, protocol version. Manual `host:port` entry
covers VLANs and machines where mDNS is blocked.

**Pairing.** First contact shows a 6-digit code on both devices. SPAKE2 over that code
establishes a shared secret; each side records the other's Ed25519 public key in a
trusted-devices file. Subsequent connections are mutual-TLS with those keys pinned.

This matters more than it looks. The alternative — an open port that accepts model
graph execution requests from anyone on the network — is precisely the hole upstream
`rpc-server` warns about. Coffee-shop Wi-Fi is a hostile LAN. **Never bind the shard
port to `0.0.0.0` without pairing being enforced.**

**Transport.** QUIC via `quinn`. On a LAN, TCP would mostly do — but Wi-Fi gives us
packet loss and AP roaming, where QUIC's per-stream loss recovery and connection
migration are worth the modest extra complexity. One long-lived connection per peer,
one QUIC stream per concurrent session, plus a control stream.

---

## 8. Model management

Models are GGUF files, content-addressed by SHA-256, stored in a shared cache dir.

Acquisition order:
1. already in the local cache
2. **from a paired peer over the LAN** — gigabit ethernet moves a 40 GB GGUF in ~6
   minutes versus an hour on a typical home downlink. This is a real feature, not a
   nicety, and users will notice it.
3. from Hugging Face over the internet

Each device downloads only the layer ranges it will execute where the format allows
slicing; otherwise it takes the whole file and mmaps its range. Start with whole-file —
slicing GGUF is a phase-2 optimisation.

The UI must show, per model: total size, per-device share under the current plan, and
a clear "your devices can/cannot hold this" verdict *before* the download starts.

---

## 9. Failure handling — deliberately minimal

The WAN plan had boundary journals, KV checkpoints, warm backups, and fenced epochs.
On a LAN serving one user, **all of it is replaced by: re-plan and re-prefill.**

The master already holds the prompt and every token generated so far. When a follower
drops:

```
1. detect (QUIC connection loss, or a Decode that misses its deadline)
2. bump shard_generation
3. re-plan over the remaining devices
4. if the model no longer fits → surface the error, offer a smaller quant
5. otherwise re-prefill (prompt + tokens_so_far) on the new plan
6. resume streaming
```

For a 2K prompt and 400 generated tokens, re-prefill costs a couple of seconds. The
journal machinery would save maybe one of those seconds in exchange for several
thousand lines of code and a class of bugs that only reproduce under partial failure.
Not worth it.

The one thing worth doing properly is **graceful drain**: catch macOS sleep
(`NSWorkspaceWillSleepNotification`) and Linux `PrepareForSleep`, and tell peers before
going away so each master re-plans without a timeout stall.

---

## 10. Repo layout

```text
lattice/
├── crates/
│   ├── latticed/          # the daemon binary: wiring, config, service install
│   ├── lattice-net/       # QUIC transport, discovery, pairing, frame codec
│   ├── lattice-plan/      # planner + device profiles (pure, heavily unit-tested)
│   ├── lattice-model/     # GGUF parsing, catalogue, download, peer transfer
│   ├── lattice-engine/    # shard executor: Rust side of the FFI
│   └── lattice-api/       # OpenAI-compatible HTTP surface
├── engine/                # C/C++ shim over ggml (Milestone 2+)
├── ui/                    # Vite + React + TS
├── app/                   # Tauri v2 shell
├── bench/                 # the measurement harness from Milestone 0
└── dist/                  # packaging: .dmg, .deb, .rpm, AppImage
```

Languages: **Rust** (everything), **TypeScript** (UI), **C/C++** (engine shim only).
No Python anywhere — the whole point of not being exo is that this installs as one
binary.

---

## 11. Milestones

Each one ends with something you can actually use.

### M0 — Measure before you build (a weekend)

Do not write a line of Lattice until this is done. On your actual two machines:

- single-node baseline: tok/s for a model that *fits*, and for one that doesn't
  (the disk-swapping case) — this establishes the 10x claim, or kills it
- raw LAN bandwidth and RTT between the machines (`iperf3`), Wi-Fi and ethernet
- `llama.cpp --rpc` across the two machines: measured tok/s and TTFT
- theoretical best from §1's model, for the same split

**Decision gate:** if RPC lands within ~30% of theoretical, Option A (wrap it) is the
path. If it is 3x off, Option B (own shim) is justified and we know why. Write the
numbers into `bench/RESULTS.md` and commit them.

### M1 — Two machines, one model, hardcoded split

- `latticed` runs on both, discovers over mDNS, pairs with a code
- QUIC connection, activation frames, generation counter
- manually specified layer split via config
- prefill + decode, streaming out
- **correctness harness:** at temperature 0, the sharded output must be token-identical
  to single-node output for the same prompt on the same backend. This is a cheap,
  brutal test and it should run in CI against a tiny model on localhost.
- CLI only, no UI yet

### M2 — It's a product

- the planner from §5, replacing the hardcoded split
- device profiling (a short benchmark on first run and after hardware change)
- OpenAI-compatible `/v1/chat/completions` with SSE streaming
- model catalogue, HF download, peer-to-peer LAN model transfer
- the TS UI: all four screens
- Tauri shell, `.dmg` and `.deb`
- **this is the first public release**

### M3 — Make it fast

- Option-B engine shim if M0 said so
- fp8 wire dtype behind a flag, with a quality A/B
- chunked prefill overlapped with transfer (hides most of the 1.3 s TTFT hit)
- prompt-prefix KV reuse across turns in a conversation — huge for chat, since turn N+1
  re-sends the whole history
- 3+ device support

### M4 — Make it fast when there's more than one request

- concurrent sessions microbatched through the pipeline (this is where pipeline
  parallelism finally earns its keep)
- speculative decoding with the draft model held entirely on the master: verifying
  k tokens per pipeline traversal divides the per-token network tax by k
- Vulkan / ROCm backends

### Explicitly not building

1. Tensor parallelism across the network.
2. Boundary journals, KV checkpoints, warm standby shards.
3. Any central server, account, or telemetry-by-default.
4. Verification, reputation, or accounting — peers are your own machines.
5. Windows, until Mac + Linux are genuinely good.
6. Automatic device *discovery-and-join* without explicit pairing.
7. Lossy activation compression before fp8 has been quality-checked.
8. Running unsigned or arbitrary model code — GGUF and our own signed binaries only.

---

## 12. The risks I would actually worry about

**The 10x claim might not hold.** If two-machine 70B lands at 3 tok/s instead of 10,
the product is a curiosity. M0 answers this before any commitment.

**Wi-Fi variance.** The 3–5 ms per hop assumes a decent 5 GHz link. On a congested
2.4 GHz network with a laptop three rooms away, per-hop latency can hit 30–50 ms and
the tax goes from 15% to 200%. Mitigation: measure the link continuously, show it in
the UI, and warn loudly when the network is the bottleneck. "Plug in an ethernet cable"
is legitimate advice that we should be willing to give the user.

**Cross-backend numerics.** Metal and CUDA will not produce bit-identical activations.
This does not affect correctness (there is no consensus to break), but it does mean the
temperature-0 golden test can only be exact within a backend. Cross-backend, assert
"same tokens for the first N" with a tolerance, and expect eventual divergence.

**Memory estimation on Apple Silicon.** Unified memory makes "how much can I actually
use" genuinely hard — the OS will let you allocate past the point of usefulness and
then start compressing and swapping. Get `ρ` wrong and users see beachballs and blame
Lattice. Needs empirical calibration per machine class, and a conservative default.

**Scope drift back toward the cloud version.** The moment "let me use my friend's
machine across the internet" enters the roadmap, every problem from the deleted plan
comes back at once. The LAN assumption is load-bearing. Guard it.

---

## First commit

`bench/` — the M0 harness and its results. The numbers decide the engine, and the
engine decides everything else.
