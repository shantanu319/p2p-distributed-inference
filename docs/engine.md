# Choosing the inference core

The first GPU chat backend is a pinned llama.cpp build: Metal on the Mac and
Vulkan on gaming's Radeon RX 7600. The requirement to use the AMD GPU immediately
changes the earlier Candle-first decision. Candle's existing activation pipeline
remains available through `generate`.

Triton is a kernel compiler, not a complete model-serving engine. Its official
support covers Linux with NVIDIA GPUs or AMD through ROCm. Experimental Metal
projects exist, but the official backend does not cover this Mac. Gaming's working
RADV Vulkan installation also does not establish that ROCm/Triton works there.
Triton can become another backend after its runtime and model execution are
validated on the actual machines.

Sources: [Triton compatibility](https://github.com/triton-lang/triton#compatibility),
[llama.cpp backends](https://github.com/ggml-org/llama.cpp),
[AMD GPU target list](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#gfx-target-lookup-table).

## Two execution paths

`generate` implements Lattice's `Shard` trait: explicit layer ranges, hidden states
across Activation streams, and Candle execution. It needs a GGUF on each host.

`chat` implements an engine adapter around llama.cpp backend RPC. Lattice owns
identity, discovery, master authorization, device capabilities, placement, process
lifecycle, and QUIC transport. llama.cpp owns model loading, tokenization, chat
templates, KV caches, GPU kernels, graph execution, and tensor transfer.

The RPC adapter has its own stream kind. It does not implement the hidden-state
`Shard` ABI or claim its generation fencing. The master builds the graph and
remote GPUs execute the operations associated with their assigned layers.

The upstream RPC service is experimental and is not exposed to the LAN. Both
TCP endpoints bind loopback. Lattice tunnels to a fixed managed service only,
over a connection authenticated with the pairing identity. Introduced workers
cannot invoke another worker's engine; only its registered master can.

See [inference.md](inference.md) for installation, placement limits, and tests.
