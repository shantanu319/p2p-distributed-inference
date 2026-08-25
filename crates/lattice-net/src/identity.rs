//! Ed25519 device identity: a keypair persisted per install, and the short
//! public ID derived from it that users compare during pairing.

use ed25519_dalek::pkcs8::spki::der::pem::LineEnding;
use ed25519_dalek::pkcs8::{DecodePrivateKey, EncodePrivateKey};
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::Error;

const KEY_FILE: &str = "device_key.pem";

/// First 8 bytes of SHA-256 over the public key, rendered as 16 hex chars.
/// Short enough to read aloud, long enough that collisions on a home LAN
/// are not a concern.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DeviceId([u8; 8]);

impl DeviceId {
    pub fn from_public_key(key: &VerifyingKey) -> Self {
        let digest = Sha256::digest(key.as_bytes());
        Self(digest[..8].try_into().expect("sha256 yields 32 bytes"))
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

impl fmt::Debug for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DeviceId({self})")
    }
}

impl FromStr for DeviceId {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        let mut bytes = [0u8; 8];
        hex::decode_to_slice(s, &mut bytes).map_err(|_| Error::MalformedDeviceId(s.to_owned()))?;
        Ok(Self(bytes))
    }
}

/// This install's long-lived signing key. Never leaves the device.
pub struct DeviceKey {
    signing: SigningKey,
    id: DeviceId,
}

impl DeviceKey {
    /// Loads the key at `dir/device_key.pem`, generating and persisting one on
    /// first run.
    pub fn load_or_create(dir: &Path) -> Result<Self, Error> {
        let path = dir.join(KEY_FILE);
        let signing = match std::fs::read_to_string(&path) {
            Ok(pem) => SigningKey::from_pkcs8_pem(&pem)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                std::fs::create_dir_all(dir)?;
                let signing = SigningKey::from_bytes(&rand::random());
                write_private(&path, &signing.to_pkcs8_pem(LineEnding::LF)?)?;
                signing
            }
            Err(e) => return Err(e.into()),
        };
        let id = DeviceId::from_public_key(&signing.verifying_key());
        Ok(Self { signing, id })
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn public_key(&self) -> VerifyingKey {
        self.signing.verifying_key()
    }

    pub fn signing_key(&self) -> &SigningKey {
        &self.signing
    }
}

/// Default data directory: `~/Library/Application Support/lattice` on macOS,
/// `~/.local/share/lattice` on Linux.
pub fn default_data_dir() -> Result<PathBuf, Error> {
    directories::ProjectDirs::from("", "", "lattice")
        .map(|d| d.data_dir().to_path_buf())
        .ok_or(Error::NoDataDir)
}

fn write_private(path: &Path, pem: &str) -> Result<(), Error> {
    std::fs::write(path, pem)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable_across_loads() {
        let dir = tempfile::tempdir().unwrap();
        let first = DeviceKey::load_or_create(dir.path()).unwrap();
        let second = DeviceKey::load_or_create(dir.path()).unwrap();
        assert_eq!(first.id(), second.id());
        assert_eq!(first.public_key(), second.public_key());
    }

    #[test]
    fn distinct_installs_get_distinct_ids() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let a = DeviceKey::load_or_create(a.path()).unwrap();
        let b = DeviceKey::load_or_create(b.path()).unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn device_id_round_trips_through_text() {
        let dir = tempfile::tempdir().unwrap();
        let key = DeviceKey::load_or_create(dir.path()).unwrap();
        let parsed: DeviceId = key.id().to_string().parse().unwrap();
        assert_eq!(parsed, key.id());
        assert_eq!(key.id().to_string().len(), 16);
    }

    #[test]
    fn malformed_device_id_is_rejected() {
        assert!("nothex".parse::<DeviceId>().is_err());
        assert!("00112233445566".parse::<DeviceId>().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn key_file_is_not_world_readable() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        DeviceKey::load_or_create(dir.path()).unwrap();
        let meta = std::fs::metadata(dir.path().join(KEY_FILE)).unwrap();
        assert_eq!(meta.permissions().mode() & 0o077, 0);
    }
}
