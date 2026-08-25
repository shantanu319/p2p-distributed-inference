//! Discovery, pairing, and authenticated transport between Lattice devices.

pub mod discovery;
pub mod identity;
pub mod pairing;
pub mod probe;
pub mod tls;
pub mod transport;
pub mod trust;

pub use discovery::{Advertisement, DiscoveredPeer, Discovery, PeerEvent};
pub use identity::{DeviceId, DeviceKey};
pub use pairing::{Pairing, PairingCode};
pub use probe::{LinkQuality, measure};
pub use transport::{Connection, Endpoint, PeerPolicy};
pub use trust::{PairedPeer, TrustStore};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("device key is not valid pkcs#8: {0}")]
    KeyEncoding(#[from] ed25519_dalek::pkcs8::Error),
    #[error("{0:?} is not a device id (expected 16 hex chars)")]
    MalformedDeviceId(String),
    #[error("mdns: {0}")]
    Mdns(#[from] mdns_sd::Error),
    #[error("trusted-devices file is corrupt: {0}")]
    TrustStore(#[from] serde_json::Error),
    #[error("pairing code must be six digits")]
    MalformedPairingCode,
    #[error("pairing failed: wrong code, or someone is between the two devices")]
    PairingFailed,
    #[error("pairing message was reflected back at us")]
    PairingReflected,
    #[error("codec: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("certificate: {0}")]
    Certificate(String),
    #[error("rcgen: {0}")]
    Rcgen(#[from] rcgen::Error),
    #[error("tls: {0}")]
    Tls(String),
    #[error("quic connect: {0}")]
    Connect(#[from] quinn::ConnectError),
    #[error("quic connection: {0}")]
    Connection(#[from] quinn::ConnectionError),
    #[error("peer refused the connection: it has not paired with this device")]
    Rejected,
    #[error("link probe failed: {0}")]
    Probe(String),
    #[error("no OS data directory available")]
    NoDataDir,
}
