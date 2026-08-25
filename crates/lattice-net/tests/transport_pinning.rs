//! Proves the transport authenticates by pinned Ed25519 key in both
//! directions, and that both ends derive the same channel binding.

use ed25519_dalek::VerifyingKey;
use lattice_net::transport::PeerPolicy;
use lattice_net::{DeviceKey, Endpoint};
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;

#[derive(Debug)]
struct Allow(Vec<VerifyingKey>);

impl PeerPolicy for Allow {
    fn accept(&self, key: &VerifyingKey) -> bool {
        self.0.contains(key)
    }
}

fn device() -> (DeviceKey, TempDir) {
    let dir = tempfile::tempdir().unwrap();
    (DeviceKey::load_or_create(dir.path()).unwrap(), dir)
}

fn loopback() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

#[tokio::test]
async fn pinned_peers_connect_and_agree_on_the_channel_binding() {
    let (a_key, _a_dir) = device();
    let (b_key, _b_dir) = device();

    let a = Endpoint::bind(loopback(), &a_key, Arc::new(Allow(vec![b_key.public_key()]))).unwrap();
    let b = Endpoint::bind(loopback(), &b_key, Arc::new(Allow(vec![a_key.public_key()]))).unwrap();
    let a_addr = a.local_addr().unwrap();

    let server = tokio::spawn(async move { a.accept().await.unwrap().unwrap() });
    let client = b.connect(a_addr).await.expect("pinned peer must connect");
    let accepted = server.await.unwrap();

    assert_eq!(client.peer_id(), a_key.id());
    assert_eq!(accepted.peer_id(), b_key.id());
    assert_eq!(
        client.channel_binding(),
        accepted.channel_binding(),
        "both ends must derive the same binding"
    );
}

#[tokio::test]
async fn a_peer_we_have_not_paired_with_is_refused() {
    let (a_key, _a_dir) = device();
    let (b_key, _b_dir) = device();
    let (stranger_key, _s_dir) = device();

    let a = Endpoint::bind(loopback(), &a_key, Arc::new(Allow(vec![b_key.public_key()]))).unwrap();
    let stranger = Endpoint::bind(
        loopback(),
        &stranger_key,
        Arc::new(Allow(vec![a_key.public_key()])),
    )
    .unwrap();
    let a_addr = a.local_addr().unwrap();

    tokio::spawn(async move {
        let _ = a.accept().await;
    });
    assert!(
        stranger.connect(a_addr).await.is_err(),
        "an unpinned device must not establish a connection"
    );
}

#[tokio::test]
async fn a_peer_that_does_not_trust_us_is_refused() {
    let (a_key, _a_dir) = device();
    let (b_key, _b_dir) = device();
    let (stranger_key, _s_dir) = device();

    // A would accept B, but B only accepts the stranger, so B rejects A's cert.
    let a = Endpoint::bind(loopback(), &a_key, Arc::new(Allow(vec![b_key.public_key()]))).unwrap();
    let b = Endpoint::bind(
        loopback(),
        &b_key,
        Arc::new(Allow(vec![stranger_key.public_key()])),
    )
    .unwrap();
    let b_addr = b.local_addr().unwrap();

    tokio::spawn(async move {
        let _ = b.accept().await;
    });
    assert!(a.connect(b_addr).await.is_err());
}
