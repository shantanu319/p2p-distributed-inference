//! Control requests over a real connection, answered by a dispatcher.

use ed25519_dalek::VerifyingKey;
use lattice_net::transport::PeerPolicy;
use lattice_net::{
    Connection, ControlHandler, DeviceId, DeviceKey, Endpoint, NoShards, Request, Response,
    control, dispatch,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug)]
struct Allow(VerifyingKey);

impl PeerPolicy for Allow {
    fn accept(&self, key: &VerifyingKey) -> bool {
        *key == self.0
    }
}

/// Records who asked and how often, so we can assert the dispatcher passes
/// the authenticated peer identity through rather than something spoofable.
struct Recorder {
    seen_from: std::sync::Mutex<Vec<DeviceId>>,
    calls: AtomicU32,
}

impl ControlHandler for Recorder {
    fn handle(&self, from: DeviceId, request: Request) -> Response {
        self.seen_from.lock().unwrap().push(from);
        self.calls.fetch_add(1, Ordering::SeqCst);
        match request {
            Request::Hello => Response::Identity {
                device_id: from,
                name: "responder".into(),
                platform: "linux-x86_64".into(),
            },
            Request::Provision(peers) => Response::Provisioned {
                added: peers.len() as u32,
                already_known: 0,
            },
            Request::LoadShard(_) => Response::Refused("this fixture holds no layers".into()),
            Request::RegisterWorker { .. } => Response::WorkerRegistered,
            Request::InferenceInfo => Response::Refused("no engine".into()),
        }
    }
}

struct Fixture {
    client: Connection,
    recorder: Arc<Recorder>,
    _endpoints: (Endpoint, Endpoint),
    _dirs: Vec<tempfile::TempDir>,
}

async fn fixture() -> Fixture {
    let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
    let a_key = DeviceKey::load_or_create(dirs[0].path()).unwrap();
    let b_key = DeviceKey::load_or_create(dirs[1].path()).unwrap();
    let addr = "127.0.0.1:0".parse().unwrap();

    let a = Endpoint::bind(addr, &a_key, Arc::new(Allow(b_key.public_key()))).unwrap();
    let b = Endpoint::bind(addr, &b_key, Arc::new(Allow(a_key.public_key()))).unwrap();
    let a_addr = a.local_addr().unwrap();

    let recorder = Arc::new(Recorder {
        seen_from: std::sync::Mutex::new(Vec::new()),
        calls: AtomicU32::new(0),
    });

    let serving = recorder.clone();
    let (accepted, dialed) = tokio::join!(a.accept(), b.connect(a_addr));
    let server_conn = Arc::new(accepted.unwrap().unwrap());
    tokio::spawn(async move {
        let _ = dispatch::serve(server_conn, serving, Arc::new(NoShards)).await;
    });

    Fixture {
        client: dialed.unwrap(),
        recorder,
        _endpoints: (a, b),
        _dirs: dirs,
    }
}

#[tokio::test]
async fn a_request_reaches_the_handler_with_the_authenticated_peer_id() {
    let f = fixture().await;
    let my_id = f.client.peer_id();

    let response = control::request(&f.client, &Request::Hello).await.unwrap();
    let Response::Identity { name, .. } = response else {
        panic!("expected an identity, got {response:?}");
    };
    assert_eq!(name, "responder");

    // The handler saw the dialer's real device id, taken from its certificate.
    let seen = f.recorder.seen_from.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_ne!(
        seen[0], my_id,
        "should be the dialer's id, not the server's"
    );
}

#[tokio::test]
async fn many_control_exchanges_share_one_connection() {
    let f = fixture().await;
    for _ in 0..12 {
        let response = control::request(&f.client, &Request::Hello).await.unwrap();
        assert!(matches!(response, Response::Identity { .. }));
    }
    assert_eq!(f.recorder.calls.load(Ordering::SeqCst), 12);
}

#[tokio::test]
async fn an_unintelligible_request_is_refused_rather_than_hanging() {
    let f = fixture().await;
    let (mut send, mut recv) = f
        .client
        .open(lattice_net::StreamHeader::control())
        .await
        .unwrap();
    lattice_net::codec::write_frame(&mut send, b"this is not postcard")
        .await
        .unwrap();
    send.finish().unwrap();

    let bytes = lattice_net::codec::read_frame(&mut recv, 64 << 10)
        .await
        .unwrap();
    let response: Response = postcard::from_bytes(&bytes).unwrap();
    assert!(matches!(response, Response::Refused(_)), "{response:?}");
}
