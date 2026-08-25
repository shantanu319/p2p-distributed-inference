//! In a mesh either side may originate a channel. This proves it works in
//! both directions and that the header survives the wire.

use ed25519_dalek::VerifyingKey;
use lattice_net::stream::{StreamHeader, StreamKind};
use lattice_net::transport::PeerPolicy;
use lattice_net::{Connection, DeviceKey, Endpoint};
use std::sync::Arc;

#[derive(Debug)]
struct Allow(VerifyingKey);

impl PeerPolicy for Allow {
    fn accept(&self, key: &VerifyingKey) -> bool {
        *key == self.0
    }
}

/// A live connection seen from both ends. Holds the endpoints, because
/// dropping a quinn Endpoint tears down the connections made from it.
struct Link {
    dialer: Arc<Connection>,
    listener: Arc<Connection>,
    _endpoints: (Endpoint, Endpoint),
    _dirs: Vec<tempfile::TempDir>,
}

async fn linked_pair() -> Link {
    let dirs: Vec<_> = (0..2).map(|_| tempfile::tempdir().unwrap()).collect();
    let a_key = DeviceKey::load_or_create(dirs[0].path()).unwrap();
    let b_key = DeviceKey::load_or_create(dirs[1].path()).unwrap();
    let addr = "127.0.0.1:0".parse().unwrap();

    let a = Endpoint::bind(addr, &a_key, Arc::new(Allow(b_key.public_key()))).unwrap();
    let b = Endpoint::bind(addr, &b_key, Arc::new(Allow(a_key.public_key()))).unwrap();
    let a_addr = a.local_addr().unwrap();

    let (accepted, dialed) = tokio::join!(a.accept(), b.connect(a_addr));
    Link {
        dialer: Arc::new(dialed.unwrap()),
        listener: Arc::new(accepted.unwrap().unwrap()),
        _endpoints: (a, b),
        _dirs: dirs,
    }
}

#[tokio::test]
async fn the_listener_can_open_a_channel_to_the_dialer() {
    let link = linked_pair().await;
    let (dialer, listener) = (link.dialer.clone(), link.listener.clone());

    // The side that accepted the connection originates the stream.
    let expected = StreamHeader {
        kind: StreamKind::Activation,
        session: 42,
        generation: 9,
    };
    let listener_side = tokio::spawn(async move {
        let (mut send, _recv) = listener.open(expected).await.unwrap();
        send.write_all(b"from the listener").await.unwrap();
        send.finish().unwrap();
    });

    let (header, _send, mut recv) = dialer.accept_stream().await.unwrap();
    assert_eq!(header, expected);
    let body = recv.read_to_end(64).await.unwrap();
    assert_eq!(body, b"from the listener");
    listener_side.await.unwrap();
}

#[tokio::test]
async fn many_channels_of_different_kinds_coexist() {
    let link = linked_pair().await;
    let (dialer, listener) = (link.dialer.clone(), link.listener.clone());

    let headers: Vec<_> = (1..=8u64)
        .map(|n| StreamHeader {
            kind: match n % 3 {
                0 => StreamKind::Control,
                1 => StreamKind::Activation,
                _ => StreamKind::Bulk,
            },
            session: n,
            generation: n as u32,
        })
        .collect();

    let opening = headers.clone();
    let dialer_side = tokio::spawn(async move {
        for header in opening {
            let (mut send, _recv) = dialer.open(header).await.unwrap();
            send.write_all(&header.session.to_le_bytes()).await.unwrap();
            send.finish().unwrap();
        }
    });

    let mut seen = Vec::new();
    for _ in 0..headers.len() {
        let (header, _send, mut recv) = listener.accept_stream().await.unwrap();
        let mut body = [0u8; 8];
        recv.read_exact(&mut body).await.unwrap();
        assert_eq!(u64::from_le_bytes(body), header.session, "payload/header mismatch");
        seen.push(header);
    }
    dialer_side.await.unwrap();

    seen.sort_by_key(|h| h.session);
    assert_eq!(seen, headers, "every channel must arrive intact");
}

#[tokio::test]
async fn a_stream_that_does_not_declare_itself_is_rejected() {
    let link = linked_pair().await;
    let (dialer, listener) = (link.dialer.clone(), link.listener.clone());

    let junk = tokio::spawn(async move {
        let (mut send, _recv) = dialer.inner.open_bi().await.unwrap();
        send.write_all(&[0xffu8; 20]).await.unwrap();
        send.finish().unwrap();
    });

    assert!(listener.accept_stream().await.is_err());
    junk.await.unwrap();
}
