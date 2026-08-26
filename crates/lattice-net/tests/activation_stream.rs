//! Hidden states across a real QUIC connection, with a stub executor.
//!
//! No model here on purpose. This asserts the wire behaves — that a payload
//! survives the crossing unchanged, that one stream carries a whole session,
//! and that a failing shard is distinguishable from a failing link.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ed25519_dalek::VerifyingKey;
use lattice_engine::{Activation, Payload, WireDtype};
use lattice_net::transport::PeerPolicy;
use lattice_net::{
    Connection, DeviceKey, Endpoint, Executor, RefuseControl, RemoteShard, Step, dispatch,
};

#[derive(Debug)]
struct Allow(VerifyingKey);

impl PeerPolicy for Allow {
    fn accept(&self, key: &VerifyingKey) -> bool {
        *key == self.0
    }
}

/// Adds one to every byte, so a payload that came back unchanged would fail
/// the test rather than pass it by accident.
struct Increment {
    steps: AtomicU32,
    dropped: AtomicU32,
    refuse: bool,
}

#[async_trait::async_trait]
impl Executor for Increment {
    async fn run(&self, _session: u64, step: Step) -> Result<Payload, String> {
        self.steps.fetch_add(1, Ordering::SeqCst);
        if self.refuse {
            return Err("no such layer range".into());
        }
        let (Step::Prefill { input, .. } | Step::Decode { input, .. }) = step else {
            return Err("unexpected step".into());
        };
        let Payload::Hidden(activation) = input else {
            return Err(format!("expected hidden states, got {}", input.kind()));
        };
        let bumped = activation.data.iter().map(|b| b.wrapping_add(1)).collect();
        Ok(Payload::Hidden(
            Activation::new(
                activation.n_tokens,
                activation.hidden_dim,
                activation.dtype,
                bumped,
            )
            .map_err(|e| e.to_string())?,
        ))
    }

    fn drop_session(&self, _session: u64) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct Link {
    client: Connection,
    executor: Arc<Increment>,
    _endpoints: (Endpoint, Endpoint),
    _dirs: Vec<tempfile::TempDir>,
}

async fn link(refuse: bool) -> Link {
    let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
    let a_key = DeviceKey::load_or_create(dirs[0].path()).unwrap();
    let b_key = DeviceKey::load_or_create(dirs[1].path()).unwrap();
    let addr = "127.0.0.1:0".parse().unwrap();

    let a = Endpoint::bind(addr, &a_key, Arc::new(Allow(b_key.public_key()))).unwrap();
    let b = Endpoint::bind(addr, &b_key, Arc::new(Allow(a_key.public_key()))).unwrap();
    let a_addr = a.local_addr().unwrap();

    let executor = Arc::new(Increment {
        steps: AtomicU32::new(0),
        dropped: AtomicU32::new(0),
        refuse,
    });

    let serving = executor.clone();
    let (accepted, dialed) = tokio::join!(a.accept(), b.connect(a_addr));
    let server_conn = Arc::new(accepted.unwrap().unwrap());
    tokio::spawn(async move {
        let _ = dispatch::serve(server_conn, Arc::new(RefuseControl), serving).await;
    });

    Link {
        client: dialed.unwrap(),
        executor,
        _endpoints: (a, b),
        _dirs: dirs,
    }
}

fn hidden(fill: u8, n_tokens: u32, hidden_dim: u32) -> Payload {
    let len = n_tokens as usize * hidden_dim as usize * 2;
    Payload::Hidden(
        Activation::new(n_tokens, hidden_dim, WireDtype::F16, vec![fill; len]).unwrap(),
    )
}

#[tokio::test]
async fn one_stream_carries_a_whole_session() {
    let link = link(false).await;
    let mut shard = RemoteShard::open(&link.client, 42, 0).await.unwrap();

    let out = shard
        .step(&Step::Prefill {
            pos: 0,
            input: hidden(1, 6, 2048),
        })
        .await
        .unwrap();
    assert_eq!(out, hidden(2, 6, 2048));

    // Reused, not reopened: §1's per-hop budget has no room for a stream
    // handshake per token.
    for step in 0..4u32 {
        let out = shard
            .step(&Step::Decode {
                pos: 6 + step,
                input: hidden(9, 1, 2048),
            })
            .await
            .unwrap();
        assert_eq!(out, hidden(10, 1, 2048));
    }
    assert_eq!(link.executor.steps.load(Ordering::SeqCst), 5);
}

#[tokio::test]
async fn a_prefill_sized_payload_survives_the_crossing() {
    let link = link(false).await;
    let mut shard = RemoteShard::open(&link.client, 1, 0).await.unwrap();

    // 1024 tokens at hidden 4096 in f16: 8 MB, well past anything that fits in
    // one datagram or one flow-control window.
    let out = shard
        .step(&Step::Prefill {
            pos: 0,
            input: hidden(7, 1024, 4096),
        })
        .await
        .unwrap();
    assert_eq!(out, hidden(8, 1024, 4096));
}

#[tokio::test]
async fn finishing_the_stream_drops_the_session() {
    let link = link(false).await;
    let shard = RemoteShard::open(&link.client, 5, 0).await.unwrap();
    shard.finish().await;

    for _ in 0..100 {
        if link.executor.dropped.load(Ordering::SeqCst) == 1 {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("the follower never dropped the session");
}

#[tokio::test]
async fn a_shard_that_refuses_is_not_a_broken_link() {
    let link = link(true).await;
    let mut shard = RemoteShard::open(&link.client, 3, 0).await.unwrap();

    let err = shard
        .step(&Step::Decode {
            pos: 0,
            input: hidden(1, 1, 64),
        })
        .await
        .expect_err("the executor refuses");
    assert!(err.to_string().contains("no such layer range"), "{err}");

    // The stream is still usable, which is what tells a master to re-plan
    // rather than declare the device gone.
    let again = shard
        .step(&Step::Decode {
            pos: 1,
            input: hidden(1, 1, 64),
        })
        .await;
    assert!(again.is_err());
    assert_eq!(link.executor.steps.load(Ordering::SeqCst), 2);
}
