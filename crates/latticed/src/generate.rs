//! The decode loop, on one device.
//!
//! §3 puts sampling on the master, so it lives here rather than in the engine.
//! Tokens go in as ids and come out as ids: tokenization belongs with the HTTP
//! surface, which does not exist yet.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use lattice_engine::{
    CandleShard, KvDtype, ModelFacts, Payload, Shard, ShardSpec, hash_file,
};

const SESSION: u64 = 1;

pub fn run(model: &Path, prompt: &[u32], max_new: u32, max_context: u32) -> Result<()> {
    let facts = ModelFacts::read_file(model).context("reading the GGUF header")?;
    println!(
        "{} {} layers, hidden {}, {} heads / {} kv, trained to {} tokens",
        facts.arch, facts.layers, facts.hidden_dim, facts.heads, facts.kv_heads, facts.train_context
    );

    let spec = ShardSpec {
        model_hash: hash_file(model).context("hashing the model")?,
        first_layer: 0,
        last_layer: facts.layers,
        max_context,
        kv_dtype: KvDtype::F16,
    };
    let loading = Instant::now();
    let mut shard = CandleShard::open(model, spec).context("loading the shard")?;
    println!(
        "loaded on {:?} in {:.2}s",
        shard.device(),
        loading.elapsed().as_secs_f64()
    );

    let started = Instant::now();
    let logits = shard.prefill(SESSION, 0, Payload::Tokens(prompt.to_vec()))?;
    let ttft = started.elapsed();
    let mut next = argmax(&logits)?;

    let mut out = vec![next];
    let decoding = Instant::now();
    for step in 0..max_new.saturating_sub(1) {
        let pos = prompt.len() as u32 + step;
        let logits = shard.decode(SESSION, pos, Payload::Tokens(vec![next]))?;
        next = argmax(&logits)?;
        out.push(next);
    }

    let elapsed = decoding.elapsed().as_secs_f64();
    println!("{out:?}");
    println!(
        "ttft {:.0} ms, {} tokens at {:.1} tok/s",
        ttft.as_secs_f64() * 1e3,
        out.len(),
        (out.len() - 1) as f64 / elapsed
    );
    println!("{:?}", shard.stats());
    Ok(())
}

/// Greedy sampling. Temperature 0 is what M1's token-identical test compares,
/// so it is the only mode worth having before there is a sampler to configure.
fn argmax(logits: &Payload) -> Result<u32> {
    let Payload::Logits(values) = logits else {
        anyhow::bail!("the last shard returned {}, not logits", logits.kind());
    };
    let best = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .context("empty logits")?;
    Ok(best.0 as u32)
}
