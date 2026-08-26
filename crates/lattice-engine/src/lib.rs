//! The shard ABI from PLAN.md §6, and the engines that implement it.
//!
//! A shard is a contiguous range of transformer layers plus the KV cache for
//! that range. The interface is deliberately six calls wide so that swapping
//! the engine underneath it stays a contained decision.

mod abi;
#[cfg(feature = "candle")]
mod candle;
#[cfg(feature = "candle")]
mod facts;

#[cfg(feature = "candle")]
pub use candle::CandleShard;
#[cfg(feature = "candle")]
pub use facts::{ModelFacts, hash_file};
pub use abi::{Activation, KvDtype, Payload, Shard, ShardSpec, ShardStats, WireDtype};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("shard holds layers {held:?}, but was handed {got}")]
    WrongPayload { held: (u32, u32), got: &'static str },
    #[error("activation is {got} bytes, but {n_tokens}x{hidden_dim} {dtype:?} needs {want}")]
    Shape {
        got: usize,
        want: usize,
        n_tokens: u32,
        hidden_dim: u32,
        dtype: WireDtype,
    },
    #[error("shard was asked for layers {asked:?}, but this engine holds all {have}")]
    UnsupportedRange { asked: (u32, u32), have: u32 },
    #[error("model architecture {0:?} is not supported; this engine reads llama-family GGUF")]
    UnsupportedArch(String),
    #[error("no such session {0}")]
    UnknownSession(u64),
    #[error("{0}")]
    Engine(String),
}
