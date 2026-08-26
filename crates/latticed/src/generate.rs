//! The decode loop, driving a chain of shards.
//!
//! §3 puts sampling on the master, so it lives here rather than in the engine.
//! Tokens go in as ids and come out as ids: tokenization belongs with the HTTP
//! surface, which does not exist yet.
//!
//! Every shard here is in this process. That is deliberate — a split that is
//! wrong locally is wrong over the network too, and much harder to see there.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use lattice_engine::{
    CandleShard, KvDtype, ModelFacts, Payload, Shard, ShardSpec, WireDtype, hash_file,
};
use lattice_net::{Connection, DeviceId, RemoteShard, Request, Response, Step, control};

const SESSION: u64 = 1;

pub struct Options {
    pub max_new: u32,
    pub max_context: u32,
    /// Where each layer range runs, in order, tiling the model.
    pub places: Vec<Placement>,
    pub wire_dtype: WireDtype,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub first_layer: u32,
    pub last_layer: u32,
    pub host: Host,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Host {
    Here,
    Peer(DeviceId),
}

/// `0-11@local`, or `11-22@` and a device id.
impl std::str::FromStr for Placement {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        let (range, host) = text
            .split_once('@')
            .with_context(|| format!("{text:?} is not first-last@where"))?;
        let (first, last) = range
            .split_once('-')
            .with_context(|| format!("{range:?} is not first-last"))?;
        Ok(Self {
            first_layer: first.parse().context("first layer")?,
            last_layer: last.parse().context("last layer")?,
            host: match host {
                "local" | "here" => Host::Here,
                id => Host::Peer(id.parse().context("device id")?),
            },
        })
    }
}

/// One stage of the chain. Exactly two kinds, so an enum rather than a trait
/// object: the difference between them is where the work happens, and that is
/// the one thing the decode loop needs to be able to see.
// A local shard is a few hundred bytes larger than a remote one, and there
// are at most a handful of stages.
#[allow(clippy::large_enum_variant)]
enum Stage {
    Here { spec: ShardSpec, shard: CandleShard },
    Peer { id: DeviceId, shard: RemoteShard },
}

impl Stage {
    async fn run(&mut self, pos: u32, input: Payload, prefill: bool) -> Result<Payload> {
        match self {
            Self::Here { shard, .. } => match prefill {
                true => shard.prefill(SESSION, pos, input),
                false => shard.decode(SESSION, pos, input),
            }
            .map_err(Into::into),
            Self::Peer { shard, id, .. } => {
                let step = match prefill {
                    true => Step::Prefill { pos, input },
                    false => Step::Decode { pos, input },
                };
                shard.step(&step).await.with_context(|| format!("shard on {id}"))
            }
        }
    }
}

/// A chain of shards covering every layer exactly once.
struct Pipeline {
    stages: Vec<Stage>,
    /// Time spent waiting on remote stages: §1's `Σ transfer` plus their
    /// compute, which is the number the planner has no way to guess.
    remote_wait: Duration,
}

impl Pipeline {
    async fn open(
        model: &Path,
        facts: &ModelFacts,
        options: &Options,
        peers: &HashMap<DeviceId, Connection>,
    ) -> Result<Self> {
        let hash = hash_file(model).context("hashing the model")?;
        check_tiling(&options.places, facts.layers)?;

        let mut stages = Vec::with_capacity(options.places.len());
        for place in &options.places {
            let spec = ShardSpec {
                model_hash: hash.clone(),
                first_layer: place.first_layer,
                last_layer: place.last_layer,
                max_context: options.max_context,
                kv_dtype: KvDtype::F16,
                wire_dtype: options.wire_dtype,
            };
            stages.push(match place.host {
                Host::Here => Stage::Here {
                    shard: CandleShard::open(model, spec.clone())
                        .with_context(|| format!("loading layers {}..{}", spec.first_layer, spec.last_layer))?,
                    spec,
                },
                Host::Peer(id) => {
                    let conn = peers
                        .get(&id)
                        .with_context(|| format!("no connection to {id}"))?;
                    open_remote(conn, id, spec).await?
                }
            });
        }
        Ok(Self { stages, remote_wait: Duration::ZERO })
    }

    /// One traversal of the chain: §1's `Σ compute + Σ transfer`.
    async fn run(&mut self, pos: u32, input: Payload, prefill: bool) -> Result<Payload> {
        let mut payload = input;
        for stage in self.stages.iter_mut() {
            let started = Instant::now();
            let remote = matches!(stage, Stage::Peer { .. });
            payload = stage.run(pos, payload, prefill).await?;
            if remote {
                self.remote_wait += started.elapsed();
            }
        }
        Ok(payload)
    }

    async fn finish(self) {
        for stage in self.stages {
            if let Stage::Peer { shard, .. } = stage {
                shard.finish().await;
            }
        }
    }
}

async fn open_remote(conn: &Connection, id: DeviceId, spec: ShardSpec) -> Result<Stage> {
    match control::request(conn, &Request::LoadShard(spec.clone())).await? {
        Response::ShardLoaded { weight_bytes, kv_bytes } => println!(
            "  layers {:>3}..{:<3} on {id}, {:.2} GB weights, {:.2} GB kv",
            spec.first_layer,
            spec.last_layer,
            weight_bytes as f64 / 1e9,
            kv_bytes as f64 / 1e9,
        ),
        Response::Refused(why) => bail!("{id} refused layers {}..{}: {why}", spec.first_layer, spec.last_layer),
        other => bail!("{id} answered a load request with {other:?}"),
    }
    let shard = RemoteShard::open(conn, SESSION, 0).await?;
    Ok(Stage::Peer { id, shard })
}

/// Every layer runs exactly once, in order. A gap silently drops layers and a
/// repeat silently runs them twice; both produce plausible-looking nonsense
/// rather than an error, so they are worth refusing up front.
fn check_tiling(places: &[Placement], layers: u32) -> Result<()> {
    let mut next = 0;
    for place in places {
        if place.first_layer != next {
            bail!("layers {next}..{} are unplaced", place.first_layer);
        }
        if place.last_layer <= place.first_layer {
            bail!("{}-{} is not a layer range", place.first_layer, place.last_layer);
        }
        next = place.last_layer;
    }
    match next == layers {
        true => Ok(()),
        false => bail!("placements cover 0..{next}, but the model has {layers} layers"),
    }
}

pub async fn run(
    model: &Path,
    prompt: &[u32],
    options: &Options,
    peers: &HashMap<DeviceId, Connection>,
) -> Result<()> {
    let facts = ModelFacts::read_file(model).context("reading the GGUF header")?;
    println!(
        "{} {} layers, hidden {}, {} heads / {} kv, trained to {} tokens",
        facts.arch, facts.layers, facts.hidden_dim, facts.heads, facts.kv_heads, facts.train_context
    );

    let loading = Instant::now();
    let mut pipeline = Pipeline::open(model, &facts, options, peers).await?;
    for stage in &pipeline.stages {
        if let Stage::Here { spec, shard } = stage {
            let stats = shard.stats();
            println!(
                "  layers {:>3}..{:<3} here on {:?}, {:.2} GB weights, {:.2} GB kv",
                spec.first_layer,
                spec.last_layer,
                shard.device(),
                stats.weight_bytes as f64 / 1e9,
                stats.kv_bytes as f64 / 1e9,
            );
        }
    }
    println!("loaded in {:.2}s", loading.elapsed().as_secs_f64());

    let started = Instant::now();
    let mut next = argmax(&pipeline.run(0, Payload::Tokens(prompt.to_vec()), true).await?)?;
    let ttft = started.elapsed();

    let mut out = vec![next];
    let decoding = Instant::now();
    for step in 0..options.max_new.saturating_sub(1) {
        let pos = prompt.len() as u32 + step;
        next = argmax(&pipeline.run(pos, Payload::Tokens(vec![next]), false).await?)?;
        out.push(next);
    }
    let elapsed = decoding.elapsed();

    println!("{out:?}");
    println!(
        "ttft {:.0} ms, {} tokens at {:.1} tok/s",
        ttft.as_secs_f64() * 1e3,
        out.len(),
        (out.len() - 1) as f64 / elapsed.as_secs_f64()
    );
    if !pipeline.remote_wait.is_zero() {
        let share = pipeline.remote_wait.as_secs_f64() / (ttft + elapsed).as_secs_f64();
        println!(
            "{:.0} ms waiting on remote shards, {:.0}% of the run",
            pipeline.remote_wait.as_secs_f64() * 1e3,
            share * 100.0
        );
    }
    pipeline.finish().await;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn here(first_layer: u32, last_layer: u32) -> Placement {
        Placement { first_layer, last_layer, host: Host::Here }
    }

    #[test]
    fn a_placement_parses_a_range_and_a_destination() {
        assert_eq!("0-11@local".parse::<Placement>().unwrap(), here(0, 11));
        let remote: Placement = "11-22@e19a64287ece5641".parse().unwrap();
        assert!(matches!(remote.host, Host::Peer(_)));
        assert!("0-11".parse::<Placement>().is_err());
        assert!("0-11@not-a-device-id".parse::<Placement>().is_err());
    }

    #[test]
    fn contiguous_placements_covering_every_layer_are_accepted() {
        assert!(check_tiling(&[here(0, 11), here(11, 22)], 22).is_ok());
        assert!(check_tiling(&[here(0, 22)], 22).is_ok());
    }

    #[test]
    fn a_gap_is_refused_rather_than_silently_dropping_layers() {
        let err = check_tiling(&[here(0, 10), here(11, 22)], 22).unwrap_err();
        assert!(err.to_string().contains("10..11 are unplaced"), "{err}");
    }

    #[test]
    fn an_overlap_is_refused_rather_than_running_layers_twice() {
        assert!(check_tiling(&[here(0, 12), here(11, 22)], 22).is_err());
    }

    #[test]
    fn placements_must_reach_the_end_of_the_model() {
        let err = check_tiling(&[here(0, 11)], 22).unwrap_err();
        assert!(err.to_string().contains("has 22 layers"), "{err}");
    }
}
