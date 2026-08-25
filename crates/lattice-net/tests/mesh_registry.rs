//! Connection reuse, and what happens when both devices dial at once.

use ed25519_dalek::VerifyingKey;
use lattice_net::transport::{PeerPolicy, Role};
use lattice_net::{DeviceId, DeviceKey, Endpoint, Mesh};
use std::sync::Arc;

#[derive(Debug)]
struct Allow(VerifyingKey);

impl PeerPolicy for Allow {
    fn accept(&self, key: &VerifyingKey) -> bool {
        *key == self.0
    }
}

struct Pair {
    a: Arc<Mesh>,
    b: Arc<Mesh>,
    a_id: DeviceId,
    b_id: DeviceId,
    _dirs: Vec<tempfile::TempDir>,
}

fn meshes() -> Pair {
    let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
    let a_key = DeviceKey::load_or_create(dirs[0].path()).unwrap();
    let b_key = DeviceKey::load_or_create(dirs[1].path()).unwrap();
    let addr = "127.0.0.1:0".parse().unwrap();

    let a = Endpoint::bind(addr, &a_key, Arc::new(Allow(b_key.public_key()))).unwrap();
    let b = Endpoint::bind(addr, &b_key, Arc::new(Allow(a_key.public_key()))).unwrap();
    Pair {
        a_id: a_key.id(),
        b_id: b_key.id(),
        a: Arc::new(Mesh::new(a, a_key.id())),
        b: Arc::new(Mesh::new(b, b_key.id())),
        _dirs: dirs,
    }
}

#[tokio::test]
async fn exactly_one_side_wants_to_dial() {
    let p = meshes();
    assert_ne!(
        p.a.should_dial(p.b_id),
        p.b.should_dial(p.a_id),
        "both sides dialling, or neither, defeats the rule"
    );
}

#[tokio::test]
async fn a_live_connection_is_reused_rather_than_redialled() {
    let p = meshes();
    let b_addr = p.b.local_addr().unwrap();

    let listening = {
        let b = p.b.clone();
        tokio::spawn(async move { b.accept().await })
    };
    let first = p.a.connect(p.b_id, &[b_addr]).await.unwrap();
    listening.await.unwrap().unwrap().unwrap();

    let second = p.a.connect(p.b_id, &[b_addr]).await.unwrap();
    assert!(
        Arc::ptr_eq(&first, &second),
        "a second connect must hand back the same connection"
    );
    assert_eq!(p.a.connected(), vec![p.b_id]);
}

/// Both devices dial simultaneously, so each ends up holding two connections
/// and must discard the same one without any negotiation.
#[tokio::test]
async fn a_simultaneous_open_converges_on_the_same_connection() {
    let p = meshes();
    let a_addr = p.a.local_addr().unwrap();
    let b_addr = p.b.local_addr().unwrap();

    let accept_a = {
        let a = p.a.clone();
        tokio::spawn(async move { a.accept().await })
    };
    let accept_b = {
        let b = p.b.clone();
        tokio::spawn(async move { b.accept().await })
    };
    let (a_targets, b_targets) = ([b_addr], [a_addr]);
    let (dialed_a, dialed_b) = tokio::join!(
        p.a.connect(p.b_id, &a_targets),
        p.b.connect(p.a_id, &b_targets)
    );
    dialed_a.unwrap();
    dialed_b.unwrap();
    accept_a.await.unwrap().unwrap().unwrap();
    accept_b.await.unwrap().unwrap().unwrap();

    let kept_by_a = p.a.get(p.b_id).expect("A must keep a connection to B");
    let kept_by_b = p.b.get(p.a_id).expect("B must keep a connection to A");

    // The survivor is the one dialled by the lower device id, so the lower
    // device sees itself as the dialer and the higher sees itself as listener.
    let (lower_side, higher_side) = if p.a_id < p.b_id {
        (kept_by_a, kept_by_b)
    } else {
        (kept_by_b, kept_by_a)
    };
    assert_eq!(lower_side.role(), Role::Dialer);
    assert_eq!(higher_side.role(), Role::Listener);
}

#[tokio::test]
async fn an_address_answering_as_the_wrong_device_is_refused() {
    let p = meshes();
    let b_addr = p.b.local_addr().unwrap();
    let listening = {
        let b = p.b.clone();
        tokio::spawn(async move { b.accept().await })
    };

    // Ask for a device that is not the one listening on that address.
    let wrong: DeviceId = "ffffffffffffffff".parse().unwrap();
    let err = p
        .a
        .connect(wrong, &[b_addr])
        .await
        .err()
        .expect("must not accept the wrong device")
        .to_string();
    assert!(err.contains("answered as"), "{err}");
    assert!(err.contains(&p.b_id.to_string()), "{err}");
    let _ = listening.await;
}
