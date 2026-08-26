//! §6's ABI over `candle`.
//!
//! A shard holds a contiguous layer range and reads only that range's tensors.
//! What it takes and returns follows from which end it owns: the shard with
//! layer 0 takes token ids, the one with the last layer returns logits, and
//! everything else is hidden states both ways. A shard holding every layer
//! sees tokens in and logits out, which is why §5's one-device plan needs no
//! special case.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use half::f16;

use crate::llama::ShardModel;
use crate::{Activation, Error, ModelFacts, Payload, Shard, ShardSpec, ShardStats, WireDtype};

pub struct CandleShard {
    model: ShardModel,
    device: Device,
    facts: ModelFacts,
    spec: ShardSpec,
    weight_bytes: u64,
    /// One KV cache means one session. A second concurrent session needs
    /// either a second instance or cache swapping; neither is built.
    active: Option<u64>,
    decodes: u64,
    decode_micros: u64,
}

impl CandleShard {
    pub fn open(path: &Path, spec: ShardSpec) -> Result<Self, Error> {
        let device = best_device();
        let mut file = File::open(path).map_err(|e| Error::Engine(format!("{}: {e}", path.display())))?;
        let content = gguf_file::Content::read(&mut file).map_err(engine)?;
        let facts = ModelFacts::read(&content.metadata)?;

        // candle's loader reads `llama.*` metadata keys by name, so anything
        // else fails deep inside with a confusing missing-key error.
        if facts.arch != "llama" {
            return Err(Error::UnsupportedArch(facts.arch));
        }
        if spec.last_layer > facts.layers || spec.first_layer > spec.last_layer {
            return Err(Error::UnsupportedRange {
                asked: (spec.first_layer, spec.last_layer),
                have: facts.layers,
            });
        }
        let weight_bytes = shard_weight_bytes(&content, &spec, facts.layers);
        let model = ShardModel::from_gguf(
            content,
            &mut file,
            &device,
            spec.first_layer as usize,
            spec.last_layer as usize,
        )
        .map_err(engine)?;

        Ok(Self {
            model,
            device,
            facts,
            spec,
            weight_bytes,
            active: None,
            decodes: 0,
            decode_micros: 0,
        })
    }

    pub fn facts(&self) -> &ModelFacts {
        &self.facts
    }

    pub fn spec(&self) -> &ShardSpec {
        &self.spec
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn owns_embedding(&self) -> bool {
        self.spec.first_layer == 0
    }

    fn owns_head(&self) -> bool {
        self.spec.last_layer == self.facts.layers
    }

    fn run(&mut self, input: Payload, pos: u32) -> Result<Payload, Error> {
        let wrong = |got: &'static str| Error::WrongPayload {
            held: (self.spec.first_layer, self.spec.last_layer),
            got,
        };
        let x = match (&input, self.owns_embedding()) {
            (Payload::Tokens(ids), true) => Tensor::new(ids.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(engine)?,
            (Payload::Hidden(activation), false) => to_tensor(activation, &self.device)?,
            (other, _) => return Err(wrong(other.kind())),
        };

        let out = self.model.forward(&x, pos as usize).map_err(engine)?;
        if !self.owns_head() {
            return Ok(Payload::Hidden(from_tensor(&out, self.spec.wire_dtype)?));
        }
        let logits = out
            .squeeze(0)
            .and_then(|l| l.to_dtype(DType::F32))
            .and_then(|l| l.to_vec1::<f32>())
            .map_err(engine)?;
        Ok(Payload::Logits(logits))
    }
}

/// Hidden states off the wire. §6's payload is row-major `[n_tokens][hidden]`,
/// and the layer stack works in f32 whatever the boundary carried.
fn to_tensor(activation: &Activation, device: &Device) -> Result<Tensor, Error> {
    let values: Vec<f32> = match activation.dtype {
        WireDtype::F16 => activation
            .data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        WireDtype::F32 => activation
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
    };
    let shape = (1, activation.n_tokens as usize, activation.hidden_dim as usize);
    Tensor::from_vec(values, shape, device).map_err(engine)
}

fn from_tensor(out: &Tensor, dtype: WireDtype) -> Result<Activation, Error> {
    let (_batch, n_tokens, hidden_dim) = out.dims3().map_err(engine)?;
    let flat = out.flatten_all().map_err(engine)?;
    let data = match dtype {
        WireDtype::F16 => flat
            .to_dtype(DType::F16)
            .and_then(|t| t.to_vec1::<f16>())
            .map_err(engine)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        WireDtype::F32 => flat
            .to_dtype(DType::F32)
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(engine)?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    };
    Activation::new(n_tokens as u32, hidden_dim as u32, dtype, data)
}

impl Shard for CandleShard {
    fn prefill(&mut self, session: u64, pos_start: u32, input: Payload) -> Result<Payload, Error> {
        if self.active != Some(session) || pos_start == 0 {
            self.model.clear_kv_cache();
            self.active = Some(session);
        }
        self.run(input, pos_start)
    }

    fn decode(&mut self, session: u64, pos: u32, input: Payload) -> Result<Payload, Error> {
        if self.active != Some(session) {
            return Err(Error::UnknownSession(session));
        }
        let started = Instant::now();
        let out = self.run(input, pos)?;
        // Skip the first: Metal compiles its kernels on the first call, which
        // costs ten decodes' worth and is not what the planner should cost on.
        self.decodes += 1;
        if self.decodes > 1 {
            self.decode_micros += started.elapsed().as_micros() as u64;
        }
        Ok(out)
    }

    fn drop_session(&mut self, session: u64) {
        if self.active == Some(session) {
            self.model.clear_kv_cache();
            self.active = None;
        }
    }

    fn stats(&self) -> ShardStats {
        ShardStats {
            sessions: self.active.is_some() as u32,
            kv_bytes: self
                .facts
                .kv_bytes(self.spec.layers(), self.spec.max_context, self.spec.kv_dtype),
            weight_bytes: self.weight_bytes,
            decode_micros: self.decode_micros.checked_div(self.decodes.saturating_sub(1)).unwrap_or(0),
        }
    }
}

/// Bytes this shard holds resident, which is what the planner's memory check
/// needs. The file's own size describes the whole model and says nothing about
/// a range; and the embedding is dequantized on load, so on disk it is a
/// fraction of what it costs in memory.
fn shard_weight_bytes(content: &gguf_file::Content, spec: &ShardSpec, layers: u32) -> u64 {
    let quantized = |name: &str| {
        content.tensor_infos.get(name).map_or(0, |info| {
            let dtype = info.ggml_dtype;
            (info.shape.elem_count() / dtype.block_size() * dtype.type_size()) as u64
        })
    };
    let owns_head = spec.last_layer == layers;
    let tied = !content.tensor_infos.contains_key("output.weight");

    let blocks: u64 = content
        .tensor_infos
        .keys()
        .filter(|name| in_range(name, spec.first_layer, spec.last_layer))
        .map(|name| quantized(name))
        .sum();

    // The embedding lands as f32, whatever it was quantized to.
    let embedding = match spec.first_layer {
        0 => content
            .tensor_infos
            .get("token_embd.weight")
            .map_or(0, |info| info.shape.elem_count() as u64 * 4),
        _ => 0,
    };
    let head = match (owns_head, tied) {
        (false, _) => 0,
        (true, false) => quantized("output_norm.weight") + quantized("output.weight"),
        // Tied: the head reuses the embedding tensor in its quantized form.
        (true, true) => quantized("output_norm.weight") + quantized("token_embd.weight"),
    };
    blocks + embedding + head
}

fn in_range(name: &str, first_layer: u32, last_layer: u32) -> bool {
    let Some(index) = name.strip_prefix("blk.").and_then(|r| r.split('.').next()) else {
        return false;
    };
    index
        .parse::<u32>()
        .is_ok_and(|index| index >= first_layer && index < last_layer)
}

fn engine(e: candle_core::Error) -> Error {
    Error::Engine(e.to_string())
}

/// Metal on the Mac, CUDA where it exists, CPU otherwise — including the AMD
/// card, which candle cannot reach (docs/engine.md).
fn best_device() -> Device {
    #[cfg(feature = "metal")]
    if let Ok(device) = Device::new_metal(0) {
        return device;
    }
    #[cfg(feature = "cuda")]
    if let Ok(device) = Device::new_cuda(0) {
        return device;
    }
    Device::Cpu
}

#[cfg(test)]
mod tests {
    use super::in_range;

    #[test]
    fn layer_tensors_belong_to_the_shard_holding_that_layer() {
        assert!(in_range("blk.11.attn_q.weight", 11, 22));
        assert!(!in_range("blk.10.attn_q.weight", 11, 22));
        assert!(!in_range("blk.22.attn_q.weight", 11, 22));
    }

    #[test]
    fn a_two_digit_index_is_not_confused_with_a_one_digit_one() {
        assert!(!in_range("blk.1.ffn_up.weight", 11, 22));
        assert!(in_range("blk.1.ffn_up.weight", 0, 11));
    }

    #[test]
    fn tensors_outside_the_blocks_belong_to_no_range() {
        assert!(!in_range("token_embd.weight", 0, 22));
        assert!(!in_range("output_norm.weight", 0, 22));
    }
}
