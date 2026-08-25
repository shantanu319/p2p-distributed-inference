//! Discovery, pairing, and authenticated transport between Lattice devices.

pub mod identity;

pub use identity::{DeviceId, DeviceKey};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("device key is not valid pkcs#8: {0}")]
    KeyEncoding(#[from] ed25519_dalek::pkcs8::Error),
    #[error("{0:?} is not a device id (expected 16 hex chars)")]
    MalformedDeviceId(String),
    #[error("no OS data directory available")]
    NoDataDir,
}
