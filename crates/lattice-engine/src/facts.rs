//! What the planner needs to know about a model, read from the GGUF header.
//!
//! Every figure here comes from metadata, so answering "can my devices hold
//! this?" costs a header read rather than a download (§8).

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use candle_core::quantized::gguf_file::Value;
use sha2::{Digest, Sha256};

use crate::{Error, KvDtype};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelFacts {
    pub arch: String,
    pub layers: u32,
    pub hidden_dim: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub train_context: u32,
}

impl ModelFacts {
    /// Read the shape of a model without loading a byte of its weights, which
    /// is what §8 needs to answer "can my devices hold this?" before a download.
    pub fn read_file(path: &Path) -> Result<Self, Error> {
        let mut file =
            File::open(path).map_err(|e| Error::Engine(format!("{}: {e}", path.display())))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| Error::Engine(e.to_string()))?;
        Self::read(&content.metadata)
    }

    pub fn read(metadata: &HashMap<String, Value>) -> Result<Self, Error> {
        let arch = string(metadata, "general.architecture")?.to_owned();
        let heads = u32_at(metadata, &format!("{arch}.attention.head_count"))?;
        Ok(Self {
            layers: u32_at(metadata, &format!("{arch}.block_count"))?,
            hidden_dim: u32_at(metadata, &format!("{arch}.embedding_length"))?,
            // Multi-head attention omits the kv key entirely; there it equals heads.
            kv_heads: u32_at(metadata, &format!("{arch}.attention.head_count_kv")).unwrap_or(heads),
            train_context: u32_at(metadata, &format!("{arch}.context_length"))?,
            heads,
            arch,
        })
    }

    pub fn head_dim(&self) -> u32 {
        self.hidden_dim / self.heads
    }

    /// Bytes of KV cache for `layers` layers at the configured context ceiling.
    ///
    /// Two tensors per layer, `kv_heads * head_dim` wide each. Sizing from the
    /// ceiling rather than the current context is what §5 insists on.
    pub fn kv_bytes(&self, layers: u32, max_context: u32, dtype: KvDtype) -> u64 {
        let per_token = 2 * u64::from(self.kv_heads) * u64::from(self.head_dim());
        per_token * u64::from(max_context) * u64::from(layers) * dtype.size() as u64
    }
}

/// §8's content address for a model file.
pub fn hash_file(path: &Path) -> Result<String, Error> {
    let mut file =
        File::open(path).map_err(|e| Error::Engine(format!("{}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| Error::Engine(e.to_string()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn string<'a>(metadata: &'a HashMap<String, Value>, key: &str) -> Result<&'a str, Error> {
    let value = metadata.get(key).ok_or_else(|| missing(key))?;
    value
        .to_string()
        .map(String::as_str)
        .map_err(|e| Error::Engine(e.to_string()))
}

/// GGUF writers disagree on integer width for the same key, so accept any.
fn u32_at(metadata: &HashMap<String, Value>, key: &str) -> Result<u32, Error> {
    let value = metadata.get(key).ok_or_else(|| missing(key))?;
    match value {
        Value::U8(v) => Ok(u32::from(*v)),
        Value::U16(v) => Ok(u32::from(*v)),
        Value::U32(v) => Ok(*v),
        Value::U64(v) => u32::try_from(*v).map_err(|_| Error::Engine(format!("{key} is {v}"))),
        other => Err(Error::Engine(format!(
            "{key} is {:?}, not an integer",
            other.value_type()
        ))),
    }
}

fn missing(key: &str) -> Error {
    Error::Engine(format!("GGUF metadata has no {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_metadata() -> HashMap<String, Value> {
        HashMap::from([
            ("general.architecture".into(), Value::String("llama".into())),
            ("llama.block_count".into(), Value::U32(22)),
            ("llama.embedding_length".into(), Value::U32(2048)),
            ("llama.attention.head_count".into(), Value::U32(32)),
            ("llama.attention.head_count_kv".into(), Value::U32(4)),
            ("llama.context_length".into(), Value::U64(2048)),
        ])
    }

    #[test]
    fn reads_a_grouped_query_model() {
        let facts = ModelFacts::read(&llama_metadata()).expect("well-formed");
        assert_eq!(facts.layers, 22);
        assert_eq!(facts.head_dim(), 64);
        assert_eq!(facts.kv_heads, 4);
    }

    #[test]
    fn kv_heads_defaults_to_heads_when_the_key_is_absent() {
        let mut metadata = llama_metadata();
        metadata.remove("llama.attention.head_count_kv");
        let facts = ModelFacts::read(&metadata).expect("well-formed");
        assert_eq!(facts.kv_heads, 32);
    }

    #[test]
    fn kv_grows_with_the_context_ceiling_not_the_current_context() {
        let facts = ModelFacts::read(&llama_metadata()).expect("well-formed");
        // 2 * 4 kv heads * 64 head dim * 2 bytes = 1024 bytes per token per layer.
        assert_eq!(facts.kv_bytes(22, 4096, KvDtype::F16), 1024 * 4096 * 22);
    }

    #[test]
    fn a_missing_layer_count_is_an_error_not_a_default() {
        let mut metadata = llama_metadata();
        metadata.remove("llama.block_count");
        assert!(ModelFacts::read(&metadata).is_err());
    }
}

#[derive(Clone, Debug)]
pub struct ModelInventory {
    pub facts: ModelFacts,
    pub layer_bytes: Vec<u64>,
    pub output_bytes: u64,
}

impl ModelInventory {
    pub fn read_file(path: &Path) -> Result<Self, Error> {
        let mut file =
            File::open(path).map_err(|e| Error::Engine(format!("{}: {e}", path.display())))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| Error::Engine(e.to_string()))?;
        Self::read(&content)
    }

    fn read(content: &candle_core::quantized::gguf_file::Content) -> Result<Self, Error> {
        if content.metadata.contains_key("split.count")
            && u32_at(&content.metadata, "split.count")? > 1
        {
            return Err(Error::Engine(
                "split GGUF files are not supported; use a single-file GGUF".into(),
            ));
        }
        let facts = ModelFacts::read(&content.metadata)?;
        if facts.arch != "llama" {
            return Err(Error::UnsupportedArch(facts.arch));
        }
        if facts.layers == 0 || facts.layers > 4096 {
            return Err(Error::Engine(
                "model must have between 1 and 4096 layers".into(),
            ));
        }
        let mut layer_bytes = vec![0u64; facts.layers as usize];
        let mut output_bytes = 0u64;
        for (name, tensor) in &content.tensor_infos {
            let invalid = || Error::Engine(format!("invalid tensor size or layer index: {name}"));
            let elements = tensor
                .shape
                .dims()
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(u64::try_from(d).ok()?))
                .ok_or_else(invalid)?;
            let block = tensor.ggml_dtype.block_size() as u64;
            if elements == 0 || elements % block != 0 {
                return Err(invalid());
            }
            let bytes = (elements / block)
                .checked_mul(tensor.ggml_dtype.type_size() as u64)
                .ok_or_else(invalid)?;
            let destination = if let Some(rest) = name.strip_prefix("blk.") {
                let index: usize = rest
                    .split('.')
                    .next()
                    .ok_or_else(invalid)?
                    .parse()
                    .map_err(|_| invalid())?;
                layer_bytes.get_mut(index).ok_or_else(invalid)?
            } else {
                &mut output_bytes
            };
            *destination = destination.checked_add(bytes).ok_or_else(invalid)?;
        }
        if layer_bytes.contains(&0) {
            return Err(Error::Engine(
                "GGUF is missing weights for a transformer layer".into(),
            ));
        }
        Ok(Self {
            facts,
            layer_bytes,
            output_bytes,
        })
    }
}

#[cfg(test)]
mod inventory_tests {
    use super::*;
    use candle_core::quantized::{
        GgmlDType,
        gguf_file::{Content, TensorInfo, VersionedMagic},
    };

    fn content() -> Content {
        Content {
            magic: VersionedMagic::GgufV2,
            metadata: HashMap::from([
                ("general.architecture".into(), Value::String("llama".into())),
                ("llama.block_count".into(), Value::U32(2)),
                ("llama.embedding_length".into(), Value::U32(32)),
                ("llama.attention.head_count".into(), Value::U32(1)),
                ("llama.context_length".into(), Value::U32(128)),
            ]),
            tensor_infos: HashMap::from([
                (
                    "blk.0.attn_q.weight".into(),
                    TensorInfo {
                        ggml_dtype: GgmlDType::Q4_0,
                        shape: vec![32, 2].into(),
                        offset: 0,
                    },
                ),
                (
                    "blk.1.attn_q.weight".into(),
                    TensorInfo {
                        ggml_dtype: GgmlDType::F16,
                        shape: vec![32, 2].into(),
                        offset: 0,
                    },
                ),
                (
                    "token_embd.weight".into(),
                    TensorInfo {
                        ggml_dtype: GgmlDType::F32,
                        shape: vec![32, 2].into(),
                        offset: 0,
                    },
                ),
            ]),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn quantized_blocks_and_tied_embeddings_are_counted() {
        let inventory = ModelInventory::read(&content()).unwrap();
        assert_eq!(inventory.layer_bytes, vec![36, 128]);
        assert_eq!(inventory.output_bytes, 256);
    }

    #[test]
    fn split_gguf_is_refused_before_weight_accounting() {
        let mut content = content();
        content.metadata.insert("split.count".into(), Value::U16(2));
        let error = ModelInventory::read(&content).unwrap_err();
        assert!(error.to_string().contains("split GGUF"));
    }

    #[test]
    fn malformed_tensor_dimensions_and_missing_layers_are_refused() {
        let mut invalid = content();
        invalid
            .tensor_infos
            .get_mut("blk.0.attn_q.weight")
            .unwrap()
            .shape = vec![31].into();
        assert!(ModelInventory::read(&invalid).is_err());
        let mut missing = content();
        missing.tensor_infos.remove("blk.1.attn_q.weight");
        assert!(ModelInventory::read(&missing).is_err());
    }
}
