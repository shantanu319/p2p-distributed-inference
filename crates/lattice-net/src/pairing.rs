//! SPAKE2 pairing over a short numeric code.
//!
//! Both devices are peers, so this uses symmetric SPAKE2: neither side is the
//! client. The code authenticates the exchange; the exchange then authenticates
//! each side's long-lived Ed25519 key, which is what gets pinned in the
//! `TrustStore`. A wrong code yields a different shared secret and the
//! confirmation MAC fails, so an attacker gets one guess per attempt.
//!
//! Each step consumes `self`, so a pairing code cannot be replayed.

use ed25519_dalek::VerifyingKey;
use hmac::{Hmac, KeyInit, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use spake2::{Ed25519Group, Identity, Password, Spake2};

use crate::{DeviceKey, Error, PairedPeer};

/// Domain separator. Bump if the pairing wire format ever changes.
const PAIRING_CONTEXT: &[u8] = b"lattice-pairing-v1";
const CONFIRM_LABEL: &[u8] = b"lattice-pairing-confirm-v1";

type HmacSha256 = Hmac<Sha256>;

/// The six digits the user reads from one screen and types into the other.
#[derive(Clone, PartialEq, Eq)]
pub struct PairingCode(String);

impl PairingCode {
    pub fn generate() -> Self {
        Self(format!("{:06}", rand::random_range(0..1_000_000u32)))
    }

    pub fn parse(text: &str) -> Result<Self, Error> {
        let trimmed = text.trim();
        let valid = trimmed.len() == 6 && trimmed.bytes().all(|b| b.is_ascii_digit());
        if !valid {
            return Err(Error::MalformedPairingCode);
        }
        Ok(Self(trimmed.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for PairingCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identity claim plus the MAC proving the sender knew the code.
#[derive(Serialize, Deserialize)]
pub struct Confirm {
    body: Vec<u8>,
    mac: [u8; 32],
}

#[derive(Serialize, Deserialize)]
struct ConfirmBody {
    public_key: [u8; 32],
    name: String,
    platform: String,
}

/// Awaiting the peer's SPAKE2 message.
pub struct Pairing {
    spake: Spake2<Ed25519Group>,
    ours: Vec<u8>,
    channel_binding: [u8; 32],
}

/// Awaiting the peer's identity confirmation.
pub struct AwaitingConfirm {
    mac_key: [u8; 32],
    transcript: Vec<u8>,
    our_public_key: VerifyingKey,
}

impl Pairing {
    /// `channel_binding` ties this exchange to the transport carrying it —
    /// in practice a hash of the peer's TLS certificate key. Without it, a
    /// relay could sit between two honest devices and proxy both halves.
    pub fn start(code: &PairingCode, channel_binding: [u8; 32]) -> (Self, Vec<u8>) {
        let (spake, ours) = Spake2::<Ed25519Group>::start_symmetric(
            &Password::new(code.as_str().as_bytes()),
            &Identity::new(PAIRING_CONTEXT),
        );
        let outbound = ours.clone();
        (
            Self {
                spake,
                ours,
                channel_binding,
            },
            outbound,
        )
    }

    pub fn key_exchange(
        self,
        peer_message: &[u8],
        key: &DeviceKey,
        name: String,
        platform: String,
    ) -> Result<(AwaitingConfirm, Confirm), Error> {
        // Symmetric SPAKE2 is reflection-attackable: an attacker who bounces
        // our own message back would have us agree a key with ourselves.
        if peer_message == self.ours {
            return Err(Error::PairingReflected);
        }
        let shared = self
            .spake
            .finish(peer_message)
            .map_err(|_| Error::PairingFailed)?;

        // Both sides must hash the same transcript, and symmetric SPAKE2 gives
        // no inherent ordering, so order the two messages canonically.
        let (first, second) = if self.ours <= peer_message.to_vec() {
            (self.ours.as_slice(), peer_message)
        } else {
            (peer_message, self.ours.as_slice())
        };
        let mut transcript = Vec::with_capacity(first.len() + second.len() + 32);
        transcript.extend_from_slice(first);
        transcript.extend_from_slice(second);
        transcript.extend_from_slice(&self.channel_binding);

        let mac_key = derive(&shared, CONFIRM_LABEL);
        let our_public_key = key.public_key();
        let body = postcard::to_allocvec(&ConfirmBody {
            public_key: our_public_key.to_bytes(),
            name,
            platform,
        })?;
        let mac = tag(&mac_key, &transcript, &body);

        Ok((
            AwaitingConfirm {
                mac_key,
                transcript,
                our_public_key,
            },
            Confirm { body, mac },
        ))
    }
}

impl AwaitingConfirm {
    pub fn finish(self, confirm: &Confirm) -> Result<PairedPeer, Error> {
        let expected = tag(&self.mac_key, &self.transcript, &confirm.body);
        // Constant-time: a timing oracle here would leak the code digit by digit.
        if !bool::from(constant_time_eq(&expected, &confirm.mac)) {
            return Err(Error::PairingFailed);
        }
        let body: ConfirmBody = postcard::from_bytes(&confirm.body)?;
        let public_key =
            VerifyingKey::from_bytes(&body.public_key).map_err(|_| Error::PairingFailed)?;
        if public_key == self.our_public_key {
            return Err(Error::PairingReflected);
        }
        Ok(PairedPeer::new(public_key, body.name, body.platform))
    }
}

fn derive(shared: &[u8], label: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(shared).expect("hmac accepts any key length");
    mac.update(label);
    mac.finalize().into_bytes().into()
}

fn tag(key: &[u8; 32], transcript: &[u8], body: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(transcript);
    mac.update(body);
    mac.finalize().into_bytes().into()
}

fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> subtle::Choice {
    use subtle::ConstantTimeEq;
    a.ct_eq(b)
}

/// Pairing messages are small and fixed-shape; anything larger is a peer
/// that is confused or hostile.
const MAX_FRAME: usize = 4096;

/// Runs the whole exchange over an established connection.
///
/// The connection is expected to have been made with
/// [`crate::transport::AcceptAnyPeer`]: TLS cannot authenticate a device that
/// has never been paired, which is the entire reason SPAKE2 is here. The
/// resulting peer is what the caller pins in the `TrustStore`.
pub async fn exchange(
    conn: &crate::Connection,
    code: &PairingCode,
    key: &DeviceKey,
    name: String,
    platform: String,
) -> Result<PairedPeer, Error> {
    let (mut send, mut recv) = conn.open_stream().await?;
    let (state, our_message) = Pairing::start(code, conn.channel_binding());

    write_frame(&mut send, &our_message).await?;
    let peer_message = read_frame(&mut recv).await?;

    let (awaiting, confirm) = state.key_exchange(&peer_message, key, name, platform)?;
    write_frame(&mut send, &postcard::to_allocvec(&confirm)?).await?;

    let peer_confirm: Confirm = postcard::from_bytes(&read_frame(&mut recv).await?)?;
    awaiting.finish(&peer_confirm)
}

async fn write_frame(send: &mut quinn::SendStream, payload: &[u8]) -> Result<(), Error> {
    send.write_all(&(payload.len() as u32).to_le_bytes())
        .await
        .map_err(|_| Error::PairingFailed)?;
    send.write_all(payload).await.map_err(|_| Error::PairingFailed)
}

async fn read_frame(recv: &mut quinn::RecvStream) -> Result<Vec<u8>, Error> {
    let mut header = [0u8; 4];
    recv.read_exact(&mut header)
        .await
        .map_err(|_| Error::PairingFailed)?;
    let len = u32::from_le_bytes(header) as usize;
    if len > MAX_FRAME {
        return Err(Error::PairingFailed);
    }
    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(|_| Error::PairingFailed)?;
    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Device {
        key: DeviceKey,
        _dir: tempfile::TempDir,
    }

    fn device() -> Device {
        let dir = tempfile::tempdir().unwrap();
        Device {
            key: DeviceKey::load_or_create(dir.path()).unwrap(),
            _dir: dir,
        }
    }

    /// Drives both halves to completion, returning what each side learned.
    fn pair_with(
        a_code: &PairingCode,
        b_code: &PairingCode,
        a_binding: [u8; 32],
        b_binding: [u8; 32],
    ) -> Result<(PairedPeer, PairedPeer), Error> {
        let (a, b) = (device(), device());
        let (a_state, a_msg) = Pairing::start(a_code, a_binding);
        let (b_state, b_msg) = Pairing::start(b_code, b_binding);

        let (a_await, a_confirm) =
            a_state.key_exchange(&b_msg, &a.key, "laptop".into(), "macos-aarch64".into())?;
        let (b_await, b_confirm) =
            b_state.key_exchange(&a_msg, &b.key, "desktop".into(), "linux-x86_64".into())?;

        Ok((a_await.finish(&b_confirm)?, b_await.finish(&a_confirm)?))
    }

    #[test]
    fn matching_codes_exchange_verified_identities() {
        let code = PairingCode::generate();
        let (a_learned, b_learned) = pair_with(&code, &code, [7; 32], [7; 32]).unwrap();

        assert_eq!(a_learned.name, "desktop");
        assert_eq!(a_learned.platform, "linux-x86_64");
        assert_eq!(b_learned.name, "laptop");
        assert_eq!(b_learned.platform, "macos-aarch64");
        assert_ne!(a_learned.device_id, b_learned.device_id);
        assert_eq!(
            a_learned.device_id,
            crate::DeviceId::from_public_key(&a_learned.public_key)
        );
    }

    #[test]
    fn a_wrong_code_fails_rather_than_pairing() {
        let err = pair_with(
            &PairingCode::parse("111111").unwrap(),
            &PairingCode::parse("222222").unwrap(),
            [7; 32],
            [7; 32],
        )
        .unwrap_err();
        assert!(matches!(err, Error::PairingFailed), "{err:?}");
    }

    /// A relay proxying both halves sees a different TLS key on each side, so
    /// the bindings differ and the confirmation MAC does not verify.
    #[test]
    fn a_relay_between_two_honest_devices_is_rejected() {
        let code = PairingCode::generate();
        let err = pair_with(&code, &code, [1; 32], [2; 32]).unwrap_err();
        assert!(matches!(err, Error::PairingFailed), "{err:?}");
    }

    #[test]
    fn our_own_message_reflected_back_is_rejected() {
        let dev = device();
        let code = PairingCode::generate();
        let (state, msg) = Pairing::start(&code, [7; 32]);
        let err = state
            .key_exchange(&msg, &dev.key, "laptop".into(), "macos-aarch64".into())
            .err()
            .expect("reflected message must not pair");
        assert!(matches!(err, Error::PairingReflected), "{err:?}");
    }

    #[test]
    fn our_own_confirmation_reflected_back_is_rejected() {
        let dev = device();
        let code = PairingCode::generate();
        let (a_state, a_msg) = Pairing::start(&code, [7; 32]);
        let (b_state, b_msg) = Pairing::start(&code, [7; 32]);
        let (a_await, a_confirm) = a_state
            .key_exchange(&b_msg, &dev.key, "laptop".into(), "macos-aarch64".into())
            .unwrap();
        // Same device key on both halves, so the peer's key equals our own.
        let _ = b_state.key_exchange(&a_msg, &dev.key, "x".into(), "y".into());
        let err = a_await.finish(&a_confirm).unwrap_err();
        assert!(matches!(err, Error::PairingReflected), "{err:?}");
    }

    #[test]
    fn tampering_with_a_confirmation_is_detected() {
        let (a, b) = (device(), device());
        let code = PairingCode::generate();
        let (a_state, a_msg) = Pairing::start(&code, [7; 32]);
        let (b_state, b_msg) = Pairing::start(&code, [7; 32]);
        let (a_await, _) = a_state
            .key_exchange(&b_msg, &a.key, "laptop".into(), "macos-aarch64".into())
            .unwrap();
        let (_, mut b_confirm) = b_state
            .key_exchange(&a_msg, &b.key, "desktop".into(), "linux-x86_64".into())
            .unwrap();

        // Swap in an attacker's public key, leaving the MAC untouched.
        let attacker = device();
        let forged = postcard::to_allocvec(&ConfirmBody {
            public_key: attacker.key.public_key().to_bytes(),
            name: "desktop".into(),
            platform: "linux-x86_64".into(),
        })
        .unwrap();
        b_confirm.body = forged;

        let err = a_await.finish(&b_confirm).unwrap_err();
        assert!(matches!(err, Error::PairingFailed), "{err:?}");
    }

    #[test]
    fn codes_are_six_digits_and_keep_leading_zeros() {
        assert_eq!(PairingCode::parse(" 000123 ").unwrap().as_str(), "000123");
        for _ in 0..64 {
            let code = PairingCode::generate();
            assert_eq!(code.as_str().len(), 6);
            assert!(code.as_str().bytes().all(|b| b.is_ascii_digit()));
        }
    }

    #[test]
    fn malformed_codes_are_rejected() {
        for bad in ["12345", "1234567", "12345a", "", "abcdef"] {
            assert!(PairingCode::parse(bad).is_err(), "accepted {bad:?}");
        }
    }
}
