//! The decode loop, driving a chain of shards.
//!
//! §3 puts sampling on the master, so it lives here rather than in the engine.
//! Tokens go in as ids and come out as ids: tokenization belongs with the HTTP
//! surface, which does not exist yet.
//!
//! Every shard here is in this process. That is deliberate — a split that is
//! wrong locally is wrong over the network too, and much harder to see there.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use lattice_engine::{
    CandleShard, KvDtype, ModelFacts, Payload, Shard, ShardSpec, WireDtype, hash_file,
};

const SESSION: u64 = 1;

pub struct Options {
    pub max_new: u32,
    pub max_context: u32,
    /// Layer indices to cut at. Empty means one shard holding everything.
    pub cuts: Vec<u32>,
    pub wire_dtype: WireDtype,
}

/// A chain of shards covering every layer exactly once.
struct Pipeline {
    shards: Vec<CandleShard>,
}

impl Pipeline {
    fn open(model: &Path, facts: &ModelFacts, options: &Options) -> Result<Self> {
        let hash = hash_file(model).context("hashing the model")?;
        let mut bounds = vec![0];
        bounds.extend_from_slice(&options.cuts);
        bounds.push(facts.layers);
        if bounds.windows(2).any(|w| w[0] >= w[1]) {
            bail!("cuts {:?} are not strictly inside 0..{}", options.cuts, facts.layers);
        }

        let shards = bounds
            .windows(2)
            .map(|w| {
                let spec = ShardSpec {
                    model_hash: hash.clone(),
                    first_layer: w[0],
                    last_layer: w[1],
                    max_context: options.max_context,
                    kv_dtype: KvDtype::F16,
                    wire_dtype: options.wire_dtype,
                };
                CandleShard::open(model, spec)
            })
            .collect::<Result<Vec<_>, _>>()
            .context("loading shards")?;
        Ok(Self { shards })
    }

    /// One traversal of the chain. §1's `Σ compute` with no transfer term,
    /// because nothing crosses a machine yet.
    fn run(&mut self, pos: u32, input: Payload, prefill: bool) -> Result<Payload> {
        let mut payload = input;
        for shard in self.shards.iter_mut() {
            payload = match prefill {
                true => shard.prefill(SESSION, pos, payload)?,
                false => shard.decode(SESSION, pos, payload)?,
            };
        }
        Ok(payload)
    }
}

pub fn run(model: &Path, prompt: &[u32], options: &Options) -> Result<()> {
    let facts = ModelFacts::read_file(model).context("reading the GGUF header")?;
    println!(
        "{} {} layers, hidden {}, {} heads / {} kv, trained to {} tokens",
        facts.arch, facts.layers, facts.hidden_dim, facts.heads, facts.kv_heads, facts.train_context
    );

    let loading = Instant::now();
    let mut pipeline = Pipeline::open(model, &facts, options)?;
    for shard in &pipeline.shards {
        let stats = shard.stats();
        println!(
            "  layers {:>3}..{:<3} on {:?}, {:.2} GB weights, {:.2} GB kv at {} ctx",
            shard.spec().first_layer,
            shard.spec().last_layer,
            shard.device(),
            stats.weight_bytes as f64 / 1e9,
            stats.kv_bytes as f64 / 1e9,
            options.max_context,
        );
    }
    println!("loaded in {:.2}s", loading.elapsed().as_secs_f64());

    let started = Instant::now();
    let mut next = argmax(&pipeline.run(0, Payload::Tokens(prompt.to_vec()), true)?)?;
    let ttft = started.elapsed();

    let mut out = vec![next];
    let decoding = Instant::now();
    for step in 0..options.max_new.saturating_sub(1) {
        let pos = prompt.len() as u32 + step;
        next = argmax(&pipeline.run(pos, Payload::Tokens(vec![next]), false)?)?;
        out.push(next);
    }

    println!("{out:?}");
    println!(
        "ttft {:.0} ms, {} tokens at {:.1} tok/s",
        ttft.as_secs_f64() * 1e3,
        out.len(),
        (out.len() - 1) as f64 / decoding.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Greedy sampling. Temperature 0 is what M1's token-identical test compares,
/// so it is the only mode worth having before there is a sampler to configure.
fn argmax(logits: &Payload) -> Result<u32> {
    let Payload::Logits(values) = logits else {
        bail!("the last shard returned {}, not logits", logits.kind());
    };
    let best = values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .context("empty logits")?;
    Ok(best.0 as u32)
}
