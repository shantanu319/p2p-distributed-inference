//! Exercises real multicast: one Discovery advertises, another browses.
//! Requires mDNS on the loopback/local interface, so it is marked `ignore`
//! and run explicitly by the bench harness and on developer machines.

use lattice_net::{Advertisement, Discovery, PeerEvent};
use std::time::Duration;

#[test]
#[ignore = "requires working LAN multicast"]
fn a_browsing_device_finds_an_advertising_one() {
    let id = "0123456789abcdef".parse().unwrap();
    let mut advertiser = Discovery::new().unwrap();
    advertiser
        .advertise(&Advertisement {
            device_id: id,
            name: "test-advertiser".into(),
            platform: "macos-aarch64".into(),
            total_memory: 17_179_869_184,
            port: 47_600,
        })
        .unwrap();

    let browser = Discovery::new().unwrap();
    let browser = browser.browse().unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    while std::time::Instant::now() < deadline {
        let Some(PeerEvent::Found(peer)) = browser.next_event(Duration::from_secs(5)) else {
            continue;
        };
        if peer.device_id != id {
            continue;
        }
        assert_eq!(peer.name, "test-advertiser");
        assert_eq!(peer.platform, "macos-aarch64");
        assert_eq!(peer.total_memory, 17_179_869_184);
        assert_eq!(peer.protocol, lattice_net::discovery::PROTOCOL_VERSION);
        assert!(peer.addrs.iter().all(|a| a.port() == 47_600));
        assert!(!peer.addrs.is_empty(), "peer resolved with no addresses");
        return;
    }
    panic!("advertiser was not discovered within 15s");
}
