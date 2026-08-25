//! The set of devices this one has paired with.
//!
//! Written as readable JSON on purpose: a user should be able to open the file
//! and see exactly which machines can reach their model, and delete a line to
//! revoke one.

use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{DeviceId, Error};

const TRUST_FILE: &str = "trusted_devices.json";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PairedPeer {
    pub device_id: DeviceId,
    #[serde(with = "verifying_key_hex")]
    pub public_key: VerifyingKey,
    pub name: String,
    pub platform: String,
    /// Unix seconds, for display in the Devices screen.
    pub paired_at: u64,
}

impl PairedPeer {
    pub fn new(public_key: VerifyingKey, name: String, platform: String) -> Self {
        Self {
            device_id: DeviceId::from_public_key(&public_key),
            public_key,
            name,
            platform,
            paired_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        }
    }
}

#[derive(Debug, Default)]
pub struct TrustStore {
    path: PathBuf,
    peers: BTreeMap<DeviceId, PairedPeer>,
}

impl TrustStore {
    /// Loads `dir/trusted_devices.json`, treating absence as an empty store.
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let path = dir.join(TRUST_FILE);
        let peers = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<Vec<PairedPeer>>(&bytes)?
                .into_iter()
                .map(|p| (p.device_id, p))
                .collect(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(e) => return Err(e.into()),
        };
        Ok(Self { path, peers })
    }

    /// Re-pairing an existing device replaces its record, which is how a peer
    /// that was reinstalled (and so has a new key) gets re-trusted.
    pub fn insert(&mut self, peer: PairedPeer) -> Result<(), Error> {
        self.peers.insert(peer.device_id, peer);
        self.save()
    }

    pub fn remove(&mut self, id: DeviceId) -> Result<bool, Error> {
        let removed = self.peers.remove(&id).is_some();
        if removed {
            self.save()?;
        }
        Ok(removed)
    }

    pub fn get(&self, id: DeviceId) -> Option<&PairedPeer> {
        self.peers.get(&id)
    }

    pub fn is_trusted(&self, key: &VerifyingKey) -> bool {
        self.peers
            .get(&DeviceId::from_public_key(key))
            .is_some_and(|p| &p.public_key == key)
    }

    pub fn peers(&self) -> impl Iterator<Item = &PairedPeer> {
        self.peers.values()
    }

    fn save(&self) -> Result<(), Error> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let peers: Vec<_> = self.peers.values().collect();
        std::fs::write(&self.path, serde_json::to_vec_pretty(&peers)?)?;
        Ok(())
    }
}

/// Admits exactly the devices in the store. A poisoned lock fails closed.
#[derive(Debug, Clone)]
pub struct TrustedPeers(pub std::sync::Arc<std::sync::RwLock<TrustStore>>);

impl crate::PeerPolicy for TrustedPeers {
    fn accept(&self, key: &VerifyingKey) -> bool {
        self.0.read().is_ok_and(|store| store.is_trusted(key))
    }
}

mod verifying_key_hex {
    use ed25519_dalek::VerifyingKey;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer>(key: &VerifyingKey, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&hex::encode(key.as_bytes()))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<VerifyingKey, D::Error> {
        let text = String::deserialize(d)?;
        let mut bytes = [0u8; 32];
        hex::decode_to_slice(&text, &mut bytes).map_err(D::Error::custom)?;
        VerifyingKey::from_bytes(&bytes).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceKey;

    fn peer(dir: &Path, name: &str) -> PairedPeer {
        let key = DeviceKey::load_or_create(dir).unwrap();
        PairedPeer::new(key.public_key(), name.into(), "linux-x86_64".into())
    }

    #[test]
    fn pairings_survive_a_reload() {
        let store_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let added = peer(peer_dir.path(), "desktop");

        let mut store = TrustStore::load(store_dir.path()).unwrap();
        store.insert(added.clone()).unwrap();

        let reloaded = TrustStore::load(store_dir.path()).unwrap();
        assert_eq!(reloaded.get(added.device_id), Some(&added));
        assert!(reloaded.is_trusted(&added.public_key));
    }

    #[test]
    fn unknown_and_revoked_keys_are_untrusted() {
        let store_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let stranger_dir = tempfile::tempdir().unwrap();
        let added = peer(peer_dir.path(), "desktop");
        let stranger = peer(stranger_dir.path(), "someone-elses-laptop");

        let mut store = TrustStore::load(store_dir.path()).unwrap();
        store.insert(added.clone()).unwrap();
        assert!(!store.is_trusted(&stranger.public_key));

        assert!(store.remove(added.device_id).unwrap());
        assert!(!store.is_trusted(&added.public_key));
        assert!(!store.remove(added.device_id).unwrap());
        assert!(!TrustStore::load(store_dir.path()).unwrap().is_trusted(&added.public_key));
    }

    #[test]
    fn missing_store_loads_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(TrustStore::load(dir.path()).unwrap().peers().count(), 0);
    }

    #[test]
    fn the_file_is_human_readable() {
        let store_dir = tempfile::tempdir().unwrap();
        let peer_dir = tempfile::tempdir().unwrap();
        let added = peer(peer_dir.path(), "desktop");
        let mut store = TrustStore::load(store_dir.path()).unwrap();
        store.insert(added.clone()).unwrap();

        let text = std::fs::read_to_string(store_dir.path().join(TRUST_FILE)).unwrap();
        assert!(text.contains("desktop"), "{text}");
        assert!(text.contains(&hex::encode(added.public_key.as_bytes())), "{text}");
    }
}
