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

### What this does not tell us

- nothing about a model that exceeds one device's memory, which is the product
- nothing about network cost, which is the next slice
- 1.1B is small enough that CPU stays within 2x of Metal; the gap widens with
  model size, so do not read the ratio as general
