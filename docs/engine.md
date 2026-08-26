# Choosing the inference core

PLAN.md §4.2 offered two options and recommended "A to ship, B to win". Two
facts about the actual test hardware — a 16 GB Mac and a Linux box with an
8 GB **AMD** card, not the CUDA card originally assumed — change that answer.

## The wire format is the decision

`llama.cpp`'s `rpc-server` is backend offload, not layer-range pipelining. The
master builds the graph and ships **ops** to remote backends over its own TCP
protocol. Wrapping it means:

- §6's activation frame carries nothing
- `shard_generation` fences nothing
- the QUIC Activation streams become a tunnel around someone else's protocol
- the planner still works, because layer placement is our decision either way

So the `Shard` trait's payload type settles A versus B/C before any engine code
exists. It takes and returns **hidden states**. That rules out A.

## What can actually run on both machines

| | Mac (Metal) | AMD 8 GB |
| --- | --- | --- |
| `candle` | yes | **no** — backends are CPU, CUDA, Metal only |
| `llama.cpp` / `ggml` | yes (Metal) | yes (Vulkan, and ROCm) |
| Triton | **no** — no Metal backend exists | ROCm only |

Two things worth knowing:

**Triton cannot span these machines.** There is no Metal backend and no public
Apple GPU ISA to write one against; it remains an open feature request. Triton
would cover the AMD card and nothing else.

**Vulkan is no longer the consolation prize on AMD.** On RDNA4, RADV Vulkan
runs token generation roughly 20–23% *faster* than ROCm in `llama.cpp`, and
reaches 77–79% of theoretical memory bandwidth against ROCm's 63–72%. If we
ever need one kernel set for both machines, Vulkan is the candidate — not
Triton, and not CUDA.

`candle` has no ROCm backend. There is a work-in-progress PR and a fork; neither
is something to build a product on.

## Decision

**Implement §6 against `candle` first, keep the engine behind the trait.**

- Metal on the Mac today, which is where the interactive work happens.
- CPU on the AMD box. Correct, and slow enough to be useless for speed work.
  We are trading the AMD card away for now, deliberately and with a way back.
- The way back is a second implementation of the same trait: a ggml/Vulkan
  shim, which is PLAN.md's Option B arriving through the side door, or
  `candle`'s ROCm support maturing.

Nothing above is a bet on `candle` being fast. It is a bet that `Shard` is a
narrow enough interface that replacing what sits behind it stays a contained
job — which is the same bet §4.2 made, minus the detour through a protocol
that would have made our own transport redundant.

## What the hardware means for the demo

16 GB + 8 GB is about 20 GB usable after §5's `ρ`. A 70B is not happening on
this pair, so the run that proves the pitch is a ~32B at Q4 (≈18 GB): fits
neither machine alone, fits both together. That is the target the correctness
work should aim at, and the number to put in `bench/RESULTS.md`.
