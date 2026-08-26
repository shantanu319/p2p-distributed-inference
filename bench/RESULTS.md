# Measurements

## Single-node, candle engine — 2026-08-26

**This is not the M0 gate.** M0 needs a model that does not fit on one machine,
and both machines. These are the single-node reference numbers the sharded
output will be compared against, taken on one device.

Machine: Apple M3, 17 GB.
Model: TinyLlama 1.1B Chat v1.0, Q4_K_M (638 MB, 22 layers, hidden 2048,
32 heads / 4 KV).
Prompt: 6 tokens. Greedy, temperature 0. Release build.

| Backend | TTFT | Decode | Per decode |
| --- | --- | --- | --- |
| Metal | 59 ms | **104.9 tok/s** | 9.5 ms |
| CPU | 207 ms | 51.9 tok/s | 19.2 ms |

Measured over 128 generated tokens. Over 24 tokens Metal reports 8.5 tok/s
instead — that is candle compiling its Metal kernels on the first call, which
costs roughly ten decodes and is why `ShardStats` drops the first sample.

### Cross-backend determinism

CPU and Metal produce **identical tokens for the first 38 steps, then diverge**
(token 39: `450` vs `306`) from the same prompt at temperature 0.

This is §12's predicted behaviour, now measured rather than assumed. It sets
what M1's correctness harness can assert: token-identical **within** a backend,
and a prefix match with expected eventual divergence **across** backends. A
Mac-plus-Linux split will cross backends by construction, so the harness cannot
demand exact equality there.

## Layer splits — 2026-08-26

Same machine and model, shards chained in one process. No network involved.

| Split | Tokens | Decode |
| --- | --- | --- |
| whole model | reference | 105.3 tok/s |
| at layer 11, f32 wire | identical | 103.0 tok/s |
| at layer 11, f16 wire | identical | 100.6 tok/s |
| at 1 / at 21 (the extremes) | identical | ~104 tok/s |
| at 5, 11, 17 (four shards) | identical | 93.2 tok/s |

`splitting_a_model_does_not_change_what_it_says` cuts at all 21 interior layers
and asserts token equality for each. Rounding activations to f16 at the
boundary does not change the token stream on this model — which is evidence
for §6's fp16 wire dtype, on one model, not a general result.

The tok/s figures are not a cost model. Every shard is on the same GPU, so a
four-way split loses ~11% to per-shard overhead with no transfer to pay for.
§1's `Σ transfer` term is still entirely unmeasured.

### Memory, per shard

Weights held resident, TinyLlama Q4_K_M:

| Split | Shard sizes | Total |
| --- | --- | --- |
| whole | 0.89 | 0.89 GB |
| at 11 | 0.55 + 0.34 | 0.89 GB |
| at 5, 11, 17 | 0.40 + 0.16 + 0.15 + 0.18 | 0.89 GB |

The first shard is the fat one because candle dequantizes the token embedding
to f32 on load: 40 MB on disk becomes 262 MB in memory. **The planner must
charge whoever holds layer 0 for that**, or it will place layers by a number
that is wrong by a quarter of a gigabyte on a model this small.

## Two processes, one link — 2026-08-26

Same Mac, two `latticed` processes with separate identities, paired over the
LAN and talking QUIC to each other. Not two machines, but a real socket, real
serialisation, and two independent Metal contexts. TinyLlama Q4_K_M, 64 tokens.

Link, measured by `latticed probe`: **0.11 ms RTT, 90.5 MB/s**.

| Placement | Decode | Waiting on the peer |
| --- | --- | --- |
| all 22 layers in one process | 105.3 tok/s | — |
| all 22 layers on the peer | 81.9 tok/s | 100% |
| 0–11 here, 11–22 on the peer | 73.9 tok/s | 26% |

Both remote configurations produce **token-identical output** to the
single-process reference.

That 26% is §1's model meeting reality for the first time: the prediction was
a 10–25% tax per boundary and the measurement is 26% on a link with almost no
latency. It is a floor, not a result — real Wi-Fi adds the RTT this link does
not have.

### Thread affinity costs more than the network does

The first working version ran at 21.6 tok/s and spent 94% of the run waiting.
The link was never the problem; `probe` reported 0.11 ms throughout. The cause
was that candle's Metal backend is roughly three times slower when its work
migrates between threads, and tokio's worker pool migrates it on nearly every
step:

| Where the shard runs | Decode |
| --- | --- |
| tokio worker pool | 21.6 tok/s |
| `spawn_blocking` pool | 21.6 tok/s, and 6x worse time-to-first-token |
| one dedicated thread | 71.0 tok/s |

Worth recording because the obvious instinct — keep compute off the async
runtime with `spawn_blocking` — makes it worse, and because the symptom looked
exactly like a slow network.

### What this does not tell us

- nothing about a model that exceeds one device's memory, which is the product
- nothing about network cost, which is the next slice
- 1.1B is small enough that CPU stays within 2x of Metal; the gap widens with
  model size, so do not read the ratio as general
