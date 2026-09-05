# Distributed GPU chat

Install with `./scripts/setup.sh --inference` on each host. The optional engine
is installed under `~/.local/share/lattice/engine`; `LATTICE_ENGINE_DIR` or
`--engine-dir` changes that location. The native revision is pinned in
`engine/llama-revision` and must match the Rust runtime and every worker.
The managed engine ignores llama.cpp system and user configuration files so
unrelated host settings cannot change Lattice's device selection or placement.

On Arch Linux the build dependencies are:

```bash
sudo pacman -S --needed base-devel git cmake ninja vulkan-headers \
  vulkan-icd-loader shaderc spirv-headers
```

On macOS use Xcode with its Metal compiler, CMake, Ninja, and a C++ compiler.
CMake and Ninja can be installed in a Python virtual environment. The engine
installer prints commands for missing dependencies and does not install system
packages. A first build and first Metal initialization can take several minutes.

Run `latticed master` and `latticed worker --require-gpu` as usual. An existing
worker service must be restarted after installation. Run `latticed chat --model
/path/to/model.gguf` in another terminal on the master with the same data directory.
First-time pairing and firewall behavior are unchanged.

## Devices and placement

`latticed devices --json` reports the engine's device names, type, and memory.
The worker selects a discrete GPU in preference to an integrated GPU and never
selects a CPU automatically. Selection uses backend properties, so gaming's
RX 7600 is distinguished from its Ryzen iGPU. Names are backend-specific; this
Mac reports `MTL0`. Do not assume a particular Vulkan index across machines.

The master queries each selected worker before chat and refreshes local device
information. It verifies the engine revision and worker availability. Only one
chat can use a worker at a time; multiple workers can participate in one chat.

The planner reads tensor dimensions and quantization sizes from the GGUF header,
groups transformer tensors by layer, and adds f16 KV storage for the requested
context. Non-transformer tensors are conservatively charged to the final, local
GPU. It finds a contiguous partition in O(devices × layers) time and memory,
with at least one transformer layer per GPU and the output on the master.
Among feasible partitions it prefers shares proportional to available memory.

Each GPU keeps at least 1 GiB or 10% of reported free memory as headroom, whichever
is larger. `--reserve-mib` changes the fixed part of that reserve. System RAM and
integrated GPU memory are not added together. A plan that cannot fit is refused.
Backend memory figures are estimates: Metal reports a recommended working-set
budget, and another application can allocate memory after discovery. Compute
buffers, backend padding, and driver allocations still consume headroom. An
engine allocation failure is reported with its log; the planner does not promise
that an admitted model will fit under every workload.

Chat requests all transformer layers on the selected GPUs. Input embedding and
some supporting operations can still run on the master CPU, as part of the
engine's normal GPU execution. Missing workers or missing GPU backends fail
instead of producing a local-only response. Use `--local` to request local GPU
execution explicitly.

## Transfer and lifecycle

Workers need no GGUF file. The master loads it and sends assigned tensors through
loopback TCP proxies carried by authenticated QUIC streams. Both the proxy and
the native RPC server bind `127.0.0.1`; no extra LAN TCP firewall rule is needed.
RDMA is disabled to keep engine traffic inside the tunnel.

The worker's native RPC server starts on demand, uses its selected GPU, and is
restricted to its registered master. Its process is stopped when that master
disconnects or the worker receives Ctrl-C/SIGTERM. Chat monitors worker connection
loss and stops the model process when a selected worker disappears.

Native tensor caching is enabled under `<worker-data-dir>/rpc-cache/rpc`. Upstream
caches tensors larger than 10 MiB; smaller tensors transfer again. This is a
performance cache, not a GGUF inventory, and currently has no automatic size quota
or eviction. Stop the worker before deleting it to reclaim disk space.

Chat prints the engine log path under `<master-data-dir>/logs`. The worker keeps
a bounded `rpc-server.log` in its data directory. Worker errors from systemd are
also visible with `journalctl --user -u latticed-worker -f`.

## Scope and validation

This milestone supports single-file Llama-family GGUFs, f16 KV, terminal chat,
and a fixed set of GPUs for each conversation. It does not implement worker
replacement during generation, concurrent chats on one worker, HTTP serving,
Triton execution, or split-file GGUF loading. Distributed execution enables
capacity pooling; it does not imply a speedup on small models or slow links.

Use SmolLM2-135M-Instruct Q4_K_M for smoke tests: 135 million parameters and
roughly 105 MB. It is below the 0.5B test-model ceiling. The GGUF is published by
[Unsloth](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/blob/main/SmolLM2-135M-Instruct-Q4_K_M.gguf).

```bash
cargo test --workspace
bash scripts/tests/test-launchers.sh
bash scripts/tests/test-setup-engine.sh
bash engine/test-config-isolation.sh
python3 scripts/tests/test-cluster.py
python3 scripts/tests/test-chat.py --model /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf
python3 scripts/tests/test-chat.py --model /path/to/SmolLM2-135M-Instruct-Q4_K_M.gguf --workers 2
```

The real chat test uses isolated identities and processes on one physical host's
GPU. It checks text generation, remote model buffers, consecutive conversations,
and failure after a worker stops. This verifies the transport and engine wiring;
a separate Mac-to-Linux run is necessary to validate the actual Metal/Vulkan pair.
