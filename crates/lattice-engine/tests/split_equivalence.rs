//! M1's correctness harness: a split model must decode exactly like a whole one.
//!
//! Needs a llama-family GGUF, so it is skipped unless `LATTICE_TEST_MODEL`
//! points at one. Within a single backend this asserts token-identical output;
//! §12 explains why it can only assert a prefix match across backends, which is
//! what a Mac-plus-Linux split will be.

use std::path::{Path, PathBuf};

use lattice_engine::{CandleShard, KvDtype, ModelFacts, Payload, Shard, ShardSpec, WireDtype};

const SESSION: u64 = 7;
const PROMPT: &[u32] = &[1, 15043, 29892, 590, 1024, 338];
const STEPS: u32 = 24;

fn model() -> Option<PathBuf> {
    std::env::var_os("LATTICE_TEST_MODEL").map(PathBuf::from)
}

fn generate(path: &Path, cuts: &[u32], wire: WireDtype) -> Vec<u32> {
    let facts = ModelFacts::read_file(path).expect("gguf header");
    let mut bounds = vec![0];
    bounds.extend_from_slice(cuts);
    bounds.push(facts.layers);

    let mut shards: Vec<CandleShard> = bounds
        .windows(2)
        .map(|w| {
            let spec = ShardSpec {
                // The engine never reads it; hashing 600 MB per shard would
                // dominate this test's runtime.
                model_hash: String::new(),
                first_layer: w[0],
                last_layer: w[1],
                max_context: 512,
                kv_dtype: KvDtype::F16,
                wire_dtype: wire,
            };
            CandleShard::open(path, spec).expect("shard loads")
        })
        .collect();

    let mut run = |pos: u32, input: Payload, prefill: bool| -> u32 {
        let mut payload = input;
        for shard in shards.iter_mut() {
            payload = match prefill {
                true => shard.prefill(SESSION, pos, payload),
                false => shard.decode(SESSION, pos, payload),
            }
            .expect("shard runs");
        }
        let Payload::Logits(values) = payload else {
            panic!("the last shard returned {}, not logits", payload.kind());
        };
        values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("non-empty logits")
            .0 as u32
    };

    let mut next = run(0, Payload::Tokens(PROMPT.to_vec()), true);
    let mut out = vec![next];
    for step in 0..STEPS - 1 {
        next = run(PROMPT.len() as u32 + step, Payload::Tokens(vec![next]), false);
        out.push(next);
    }
    out
}

#[test]
fn splitting_a_model_does_not_change_what_it_says() {
    let Some(path) = model() else {
        eprintln!("set LATTICE_TEST_MODEL to a llama GGUF to run this");
        return;
    };
    let whole = generate(&path, &[], WireDtype::F16);
    let layers = ModelFacts::read_file(&path).expect("gguf header").layers;

    // A cut at every legal position, so an off-by-one at either end shows up.
    for cut in 1..layers {
        let split = generate(&path, &[cut], WireDtype::F16);
        assert_eq!(whole, split, "cut at layer {cut} changed the output");
    }
}

#[test]
fn an_f16_wire_does_not_change_what_it_says() {
    let Some(path) = model() else { return };
    assert_eq!(
        generate(&path, &[], WireDtype::F16),
        generate(&path, &[1, 5, 11], WireDtype::F16),
    );
    assert_eq!(
        generate(&path, &[1, 5, 11], WireDtype::F32),
        generate(&path, &[1, 5, 11], WireDtype::F16),
        "rounding activations to f16 changed the token stream",
    );
}

#[test]
fn a_range_outside_the_model_is_refused() {
    let Some(path) = model() else { return };
    let facts = ModelFacts::read_file(&path).expect("gguf header");
    let spec = ShardSpec {
        model_hash: String::new(),
        first_layer: 0,
        last_layer: facts.layers + 1,
        max_context: 512,
        kv_dtype: KvDtype::F16,
        wire_dtype: WireDtype::F16,
    };
    assert!(CandleShard::open(&path, spec).is_err());
}
