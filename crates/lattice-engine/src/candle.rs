//! §6's ABI over `candle`.
//!
//! This implementation holds every layer of the model, so it sees tokens in
//! and logits out — the one-device plan §5 prefers when the model fits. Layer
//! ranges need a loader that reads only the tensors for its own range, which
//! is the next piece; see docs/engine.md.

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use candle_core::Device;
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_llama::ModelWeights;

use crate::{Error, ModelFacts, Payload, Shard, ShardSpec, ShardStats};

pub struct CandleShard {
    model: ModelWeights,
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
        if spec.first_layer != 0 || spec.last_layer != facts.layers {
            return Err(Error::UnsupportedRange {
                asked: (spec.first_layer, spec.last_layer),
                have: facts.layers,
            });
        }
        let weight_bytes = file.metadata().map_err(|e| Error::Engine(e.to_string()))?.len();
        let model = ModelWeights::from_gguf(content, &mut file, &device).map_err(engine)?;

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

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn run(&mut self, tokens: &[u32], pos: u32) -> Result<Payload, Error> {
        let input = candle_core::Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(engine)?;
        let logits = self
            .model
            .forward(&input, pos as usize)
            .and_then(|l| l.squeeze(0))
            .and_then(|l| l.to_dtype(candle_core::DType::F32))
            .and_then(|l| l.to_vec1::<f32>())
            .map_err(engine)?;
        Ok(Payload::Logits(logits))
    }

    fn tokens<'a>(&self, input: &'a Payload) -> Result<&'a [u32], Error> {
        match input {
            Payload::Tokens(ids) => Ok(ids),
            other => Err(Error::WrongPayload {
                held: (self.spec.first_layer, self.spec.last_layer),
                got: other.kind(),
            }),
        }
    }
}

impl Shard for CandleShard {
    fn prefill(&mut self, session: u64, pos_start: u32, input: Payload) -> Result<Payload, Error> {
        let tokens = self.tokens(&input)?.to_vec();
        if self.active != Some(session) || pos_start == 0 {
            self.model.clear_kv_cache();
            self.active = Some(session);
        }
        self.run(&tokens, pos_start)
    }

    fn decode(&mut self, session: u64, pos: u32, input: Payload) -> Result<Payload, Error> {
        if self.active != Some(session) {
            return Err(Error::UnknownSession(session));
        }
        let tokens = self.tokens(&input)?.to_vec();
        let started = Instant::now();
        let out = self.run(&tokens, pos)?;
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
