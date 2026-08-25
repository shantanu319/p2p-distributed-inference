//! Measures a real loopback link end to end. Loopback numbers are not LAN
//! numbers, but they prove the measurement path works and the units are right.

use ed25519_dalek::VerifyingKey;
use lattice_net::probe::{self, LinkQuality};
use lattice_net::{RefuseControl, dispatch};
use lattice_net::transport::PeerPolicy;
use lattice_net::{DeviceKey, Endpoint};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
struct Allow(VerifyingKey);

impl PeerPolicy for Allow {
    fn accept(&self, key: &VerifyingKey) -> bool {
        *key == self.0
    }
}

#[tokio::test]
async fn a_link_can_be_measured_over_a_paired_connection() {
    let a_dir = tempfile::tempdir().unwrap();
    let b_dir = tempfile::tempdir().unwrap();
    let a_key = DeviceKey::load_or_create(a_dir.path()).unwrap();
    let b_key = DeviceKey::load_or_create(b_dir.path()).unwrap();
    let addr = "127.0.0.1:0".parse().unwrap();

    let a = Endpoint::bind(addr, &a_key, Arc::new(Allow(b_key.public_key()))).unwrap();
    let b = Endpoint::bind(addr, &b_key, Arc::new(Allow(a_key.public_key()))).unwrap();
    let a_addr = a.local_addr().unwrap();

    tokio::spawn(async move {
        let responder = Arc::new(a.accept().await.unwrap().unwrap());
        let _ = dispatch::serve(responder, Arc::new(RefuseControl)).await;
    });

    let conn = b.connect(a_addr).await.unwrap();
    let link = probe::measure(&conn, 1 << 20).await.unwrap();

    assert!(link.rtt > Duration::ZERO, "rtt must be positive");
    assert!(link.rtt < Duration::from_secs(5), "loopback rtt {:?}", link.rtt);
    assert!(
        link.throughput_bytes_per_sec > 1.0e6,
        "loopback throughput was {} B/s",
        link.throughput_bytes_per_sec
    );
    println!(
        "loopback: rtt {:?}, {:.1} MB/s, decode hop {:?}",
        link.rtt,
        link.throughput_bytes_per_sec / 1.0e6,
        link.decode_hop(8192)
    );
}

#[test]
fn a_decode_hop_is_the_round_trip_plus_the_activation() {
    let link = LinkQuality {
        rtt: Duration::from_millis(3),
        throughput_bytes_per_sec: 50.0e6,
    };
    // PLAN.md §1: 8192-wide fp16 hidden state is 16 KB, ~0.3 ms at 50 MB/s.
    let hop = link.decode_hop(8192);
    assert!(hop > Duration::from_micros(3_300), "{hop:?}");
    assert!(hop < Duration::from_micros(3_400), "{hop:?}");
}
