//! The whole first slice: two devices that have never met pair over a code,
//! and the trust that produces is what gates every later connection.

use lattice_net::{
    AcceptAnyPeer, DeviceKey, Endpoint, PairingCode, TrustStore, TrustedPeers, pairing,
};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use tempfile::TempDir;

struct Device {
    key: DeviceKey,
    dir: TempDir,
}

fn device() -> Device {
    let dir = tempfile::tempdir().unwrap();
    Device {
        key: DeviceKey::load_or_create(dir.path()).unwrap(),
        dir,
    }
}

fn loopback() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

/// Runs both halves of a pairing concurrently, as two real devices would.
async fn attempt_pairing(
    a: &Device,
    b: &Device,
    a_code: PairingCode,
    b_code: PairingCode,
) -> (
    Result<lattice_net::PairedPeer, lattice_net::Error>,
    Result<lattice_net::PairedPeer, lattice_net::Error>,
) {
    let listener = Endpoint::bind(loopback(), &a.key, Arc::new(AcceptAnyPeer)).unwrap();
    let dialer = Endpoint::bind(loopback(), &b.key, Arc::new(AcceptAnyPeer)).unwrap();
    let addr = listener.local_addr().unwrap();

    let a_key = DeviceKey::load_or_create(a.dir.path()).unwrap();
    let listening = tokio::spawn(async move {
        let conn = listener.accept().await.unwrap().unwrap();
        pairing::exchange(
            &conn,
            &a_code,
            &a_key,
            "laptop".into(),
            "macos-aarch64".into(),
        )
        .await
    });

    let conn = dialer.connect(addr).await.unwrap();
    let dialed = pairing::exchange(
        &conn,
        &b_code,
        &b.key,
        "desktop".into(),
        "linux-x86_64".into(),
    )
    .await;

    (listening.await.unwrap(), dialed)
}

#[tokio::test]
async fn pairing_over_a_code_produces_trust_that_gates_later_connections() {
    let (a, b) = (device(), device());
    let code = PairingCode::generate();
    let (a_learned, b_learned) = attempt_pairing(&a, &b, code.clone(), code).await;
    let (a_learned, b_learned) = (a_learned.unwrap(), b_learned.unwrap());

    assert_eq!(a_learned.device_id, b.key.id());
    assert_eq!(a_learned.name, "desktop");
    assert_eq!(b_learned.device_id, a.key.id());
    assert_eq!(b_learned.name, "laptop");

    // Pin what pairing established, then reconnect on trust alone.
    let a_store = Arc::new(RwLock::new(TrustStore::load(a.dir.path()).unwrap()));
    let b_store = Arc::new(RwLock::new(TrustStore::load(b.dir.path()).unwrap()));
    a_store.write().unwrap().insert(a_learned).unwrap();
    b_store.write().unwrap().insert(b_learned).unwrap();

    assert!(
        a_store.read().unwrap().is_trusted(&b.key.public_key()),
        "A must trust B after pairing"
    );
    assert!(
        b_store.read().unwrap().is_trusted(&a.key.public_key()),
        "B must trust A after pairing"
    );

    let server = Endpoint::bind(
        loopback(),
        &a.key,
        Arc::new(TrustedPeers(a_store.clone())),
    )
    .unwrap();
    let client =
        Endpoint::bind(loopback(), &b.key, Arc::new(TrustedPeers(b_store))).unwrap();
    let addr = server.local_addr().unwrap();
    // Dropping a Connection closes it, so the accepted side is held open.
    tokio::spawn(async move {
        if let Some(Ok(accepted)) = server.accept().await {
            accepted.inner.closed().await;
        }
    });
    let conn = client.connect(addr).await.expect("paired devices reconnect");
    assert_eq!(conn.peer_id(), a.key.id());

    // A device that never paired is refused by the same store.
    let stranger = device();
    let stranger_store = Arc::new(RwLock::new(TrustStore::load(stranger.dir.path()).unwrap()));
    let server = Endpoint::bind(loopback(), &a.key, Arc::new(TrustedPeers(a_store))).unwrap();
    let intruder = Endpoint::bind(
        loopback(),
        &stranger.key,
        Arc::new(TrustedPeers(stranger_store)),
    )
    .unwrap();
    let addr = server.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = server.accept().await;
    });
    assert!(intruder.connect(addr).await.is_err());
}

#[tokio::test]
async fn mistyping_the_code_pairs_neither_device() {
    let (a, b) = (device(), device());
    let (a_result, b_result) = attempt_pairing(
        &a,
        &b,
        PairingCode::parse("314159").unwrap(),
        PairingCode::parse("271828").unwrap(),
    )
    .await;
    assert!(a_result.is_err(), "listener must not pair on a wrong code");
    assert!(b_result.is_err(), "dialer must not pair on a wrong code");
}
