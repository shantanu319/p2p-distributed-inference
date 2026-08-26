use serde::{Deserialize, Serialize};

use crate::Error;

/// What crosses a shard boundary, on the wire and in memory alike.
///
/// The shard that owns the embedding takes tokens; the shard that owns the
/// output head yields logits; everything between is hidden states in, hidden
/// states out. A single shard holding every layer sees `Tokens` in and
/// `Logits` out, which is why a one-device plan needs no special case.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Payload {
    Tokens(Vec<u32>),
    Hidden(Activation),
    Logits(Vec<f32>),
}

impl Payload {
    /// For error messages, so a mis-wired pipeline says what it was handed.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Tokens(_) => "tokens",
            Self::Hidden(_) => "hidden states",
            Self::Logits(_) => "logits",
        }
    }
}

/// Hidden states as raw bytes, laid out `[n_tokens][hidden_dim]` row-major.
///
/// Bytes rather than floats because this is exactly §6's frame payload: the
/// wire and the ABI share one type so a hop costs no conversion.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Activation {
    pub n_tokens: u32,
    pub hidden_dim: u32,
    pub dtype: WireDtype,
    /// As bytes, not as a sequence of them. serde's default would encode a
    /// 67 MB prefill activation one element at a time (§1); `serde_bytes` makes
    /// it one length-prefixed copy.
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl Activation {
    pub fn new(n_tokens: u32, hidden_dim: u32, dtype: WireDtype, data: Vec<u8>) -> Result<Self, Error> {
        let want = n_tokens as usize * hidden_dim as usize * dtype.size();
        if data.len() != want {
            return Err(Error::Shape {
                got: data.len(),
                want,
                n_tokens,
                hidden_dim,
                dtype,
            });
        }
        Ok(Self { n_tokens, hidden_dim, dtype, data })
    }
}

/// fp16 in v1; fp8 lands behind a flag once we can A/B it per model family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WireDtype {
    F16,
    F32,
}

impl WireDtype {
    pub fn size(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvDtype {
    F16,
}

impl KvDtype {
    pub fn size(self) -> usize {
        match self {
            Self::F16 => 2,
        }
    }
}

/// What to load. `max_context` is the user's configured ceiling, not the
/// current context: the KV cache is sized from it, and sizing from anything
/// else is the classic OOM-6000-tokens-in bug (§5).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShardSpec {
    pub model_hash: String,
    pub first_layer: u32,
    /// Exclusive, so `first_layer == last_layer` is a legal empty shard.
    pub last_layer: u32,
    pub max_context: u32,
    pub kv_dtype: KvDtype,
    /// What this shard emits at its outgoing boundary. §6 says fp16 in v1;
    /// fp32 exists so the correctness harness can separate a wrong split from
    /// a lossy one.
    pub wire_dtype: WireDtype,
}

impl ShardSpec {
    pub fn layers(&self) -> u32 {
        self.last_layer - self.first_layer
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShardStats {
    pub sessions: u32,
    pub kv_bytes: u64,
    pub weight_bytes: u64,
    /// Mean wall time for one `decode`, which is what the planner's cost
    /// model needs and the only figure it cannot derive from hardware specs.
    pub decode_micros: u64,
}

/// §6's six calls. Load and unload are `new` and `Drop`, which Rust gives us.
pub trait Shard {
    fn prefill(&mut self, session: u64, pos_start: u32, input: Payload) -> Result<Payload, Error>;
    fn decode(&mut self, session: u64, pos: u32, input: Payload) -> Result<Payload, Error>;
    fn drop_session(&mut self, session: u64);
    fn stats(&self) -> ShardStats;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_rejects_a_buffer_that_does_not_match_its_shape() {
        let err = Activation::new(2, 8, WireDtype::F16, vec![0; 31]).expect_err("short buffer");
        assert!(matches!(err, Error::Shape { got: 31, want: 32, .. }));
    }

    #[test]
    fn activation_accepts_the_exact_size() {
        let a = Activation::new(2, 8, WireDtype::F16, vec![0; 32]).expect("exact");
        assert_eq!(a.data.len(), 32);
    }

    #[test]
    fn a_big_activation_encodes_as_bytes_not_as_a_sequence_of_them() {
        // 4096 tokens at hidden 8192 in f16: §1's prefill frame. Encoded as
        // bytes this is the payload plus a small header; encoded as a sequence
        // postcard would still emit one byte each, but at a per-element cost
        // that shows up as seconds on a frame this size.
        let payload = Payload::Hidden(
            Activation::new(4096, 8192, WireDtype::F16, vec![0xab; 4096 * 8192 * 2]).unwrap(),
        );
        let encoded = postcard::to_allocvec(&payload).unwrap();
        assert!(
            encoded.len() < 4096 * 8192 * 2 + 64,
            "{} bytes of overhead",
            encoded.len() - 4096 * 8192 * 2
        );
        assert_eq!(postcard::from_bytes::<Payload>(&encoded).unwrap(), payload);
    }

    #[test]
    fn an_empty_layer_range_is_legal() {
        let spec = ShardSpec {
            model_hash: "abc".into(),
            first_layer: 12,
            last_layer: 12,
            max_context: 4096,
            kv_dtype: KvDtype::F16,
            wire_dtype: WireDtype::F16,
        };
        assert_eq!(spec.layers(), 0);
    }
}
