//! The control plane: small request/response exchanges between devices.
//!
//! Each exchange gets its own stream rather than sharing one long-lived
//! channel. QUIC streams cost nothing to open, and a shared channel would put
//! independent messages behind each other and leave both sides able to send a
//! request at the same time with no way to tell the replies apart.

use ed25519_dalek::VerifyingKey;
use lattice_engine::ShardSpec;
use serde::{Deserialize, Serialize};

use crate::codec::{read_frame, write_frame};
use crate::stream::StreamHeader;
use crate::{Connection, DeviceId, Error, PairedPeer};

/// Generous for a handful of introductions, far below anything worth
/// allocating blindly for.
pub const MAX_CONTROL_FRAME: usize = 64 << 10;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Request {
    /// Liveness plus identity. Costs one round trip and nothing else.
    Hello,
    /// A device the user already trusts, handing over the keys of the others
    /// so they can talk to each other directly (see docs/streams.md §7).
    Provision(Vec<Introduction>),
    /// Hold this layer range for whoever is asking. Sent before any activation
    /// stream opens, because a follower cannot execute what it has not loaded.
    LoadShard(ShardSpec),
    RegisterWorker {
        port: u16,
    },
    InferenceInfo,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Response {
    Identity {
        device_id: DeviceId,
        name: String,
        platform: String,
    },
    Provisioned {
        added: u32,
        already_known: u32,
    },
    ShardLoaded {
        /// What the shard holds resident, so the master can check its plan
        /// against what actually happened rather than what it predicted.
        weight_bytes: u64,
        kv_bytes: u64,
    },
    /// The peer understood the request and declined it.
    Refused(String),
    WorkerRegistered,
    InferenceInfo(EngineCapabilities),
}

/// One device's identity as vouched for by another.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Introduction {
    pub public_key: [u8; 32],
    pub name: String,
    pub platform: String,
}

impl Introduction {
    pub fn of(peer: &PairedPeer) -> Self {
        Self {
            public_key: peer.public_key.to_bytes(),
            name: peer.name.clone(),
            platform: peer.platform.clone(),
        }
    }

    /// Rejects a key that is not a valid Ed25519 point. This is a sanity
    /// check, not a defence: most 32-byte strings decode to *some* point, so
    /// what actually stops a forged introduction is that whoever holds the
    /// forged key cannot complete the TLS handshake with it.
    pub fn into_peer(self, introduced_by: DeviceId) -> Result<PairedPeer, Error> {
        let key = VerifyingKey::from_bytes(&self.public_key)
            .map_err(|e| Error::Certificate(format!("introduced key is unusable: {e}")))?;
        Ok(PairedPeer::introduced(
            key,
            self.name,
            self.platform,
            introduced_by,
        ))
    }
}

/// Answers control requests for one device.
pub trait ControlHandler: Send + Sync {
    fn handle(&self, from: DeviceId, request: Request) -> Response;
}

/// Sends one request and waits for its reply.
pub async fn request(conn: &Connection, request: &Request) -> Result<Response, Error> {
    let (mut send, mut recv) = conn.open(StreamHeader::control()).await?;
    write_frame(&mut send, &postcard::to_allocvec(request)?).await?;
    send.finish()
        .map_err(|e| Error::Stream(format!("finishing a control request: {e}")))?;

    let bytes = read_frame(&mut recv, MAX_CONTROL_FRAME).await?;
    Ok(postcard::from_bytes(&bytes)?)
}

/// Answers one already-accepted control stream.
pub async fn answer(
    from: DeviceId,
    handler: std::sync::Arc<dyn ControlHandler>,
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
) -> Result<(), Error> {
    let bytes = read_frame(&mut recv, MAX_CONTROL_FRAME).await?;
    let response = match postcard::from_bytes(&bytes) {
        Ok(request) => tokio::task::spawn_blocking(move || handler.handle(from, request))
            .await
            .map_err(|error| Error::Stream(format!("control handler failed: {error}")))?,
        // A peer we cannot parse gets told so rather than left hanging.
        Err(e) => Response::Refused(format!("unintelligible request: {e}")),
    };
    write_frame(&mut send, &postcard::to_allocvec(&response)?).await?;
    send.finish()
        .map_err(|e| Error::Stream(format!("finishing a control response: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceKey;

    #[test]
    fn an_introduction_round_trips_into_a_trusted_peer() {
        let peer_dir = tempfile::tempdir().unwrap();
        let voucher_dir = tempfile::tempdir().unwrap();
        let key = DeviceKey::load_or_create(peer_dir.path()).unwrap();
        let voucher = DeviceKey::load_or_create(voucher_dir.path()).unwrap();

        let original = PairedPeer::new(key.public_key(), "gpu-box".into(), "linux-x86_64".into());
        let introduction = Introduction::of(&original);
        let encoded = postcard::to_allocvec(&introduction).unwrap();
        let decoded: Introduction = postcard::from_bytes(&encoded).unwrap();
        assert_eq!(decoded, introduction);

        let peer = decoded.into_peer(voucher.id()).unwrap();
        assert_eq!(peer.device_id, original.device_id);
        assert_eq!(peer.public_key, original.public_key);
        assert_eq!(peer.introduced_by, Some(voucher.id()));
    }

    #[test]
    fn a_key_that_is_not_a_curve_point_is_rejected() {
        let voucher_dir = tempfile::tempdir().unwrap();
        let voucher = DeviceKey::load_or_create(voucher_dir.path()).unwrap();
        let mut not_a_point = [0u8; 32];
        not_a_point[0] = 2;
        let bad = Introduction {
            public_key: not_a_point,
            name: "impostor".into(),
            platform: "linux-x86_64".into(),
        };
        assert!(bad.into_peer(voucher.id()).is_err());
    }

    #[test]
    fn a_load_request_fits_a_control_frame() {
        let spec = ShardSpec {
            model_hash: "a".repeat(64),
            first_layer: 40,
            last_layer: 80,
            max_context: 8192,
            kv_dtype: lattice_engine::KvDtype::F16,
            wire_dtype: lattice_engine::WireDtype::F16,
        };
        let bytes = postcard::to_allocvec(&Request::LoadShard(spec.clone())).unwrap();
        assert!(bytes.len() < MAX_CONTROL_FRAME);
        let Request::LoadShard(decoded) = postcard::from_bytes(&bytes).unwrap() else {
            panic!("not a load request");
        };
        assert_eq!(decoded, spec);
    }

    #[test]
    fn requests_and_responses_survive_the_wire() {
        let provision = Request::Provision(vec![Introduction {
            public_key: [7; 32],
            name: "desktop".into(),
            platform: "linux-x86_64".into(),
        }]);
        let bytes = postcard::to_allocvec(&provision).unwrap();
        assert!(bytes.len() < MAX_CONTROL_FRAME);
        assert!(matches!(
            postcard::from_bytes::<Request>(&bytes).unwrap(),
            Request::Provision(peers) if peers.len() == 1
        ));

        let refused = postcard::to_allocvec(&Response::Refused("no".into())).unwrap();
        assert!(matches!(
            postcard::from_bytes::<Response>(&refused).unwrap(),
            Response::Refused(_)
        ));
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GpuDevice {
    pub name: String,
    pub description: String,
    pub kind: String,
    pub total_memory: u64,
    pub free_memory: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineCapabilities {
    pub revision: String,
    pub device: GpuDevice,
    pub busy: bool,
}
