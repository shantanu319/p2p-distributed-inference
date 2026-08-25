//! The self-signed certificate that carries a device's Ed25519 identity into
//! TLS, and the channel binding that ties a pairing exchange to it.
//!
//! There is no CA. A peer is authenticated by the Ed25519 key inside its
//! certificate matching one pinned in the `TrustStore`, so the certificate is
//! a transport-shaped envelope around the identity from `identity.rs` and
//! nothing more.

use ed25519_dalek::VerifyingKey;
use ed25519_dalek::pkcs8::EncodePrivateKey;
use rcgen::{CertificateParams, DistinguishedName, DnType, KeyPair, PKCS_ED25519};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use sha2::{Digest, Sha256};

use crate::{DeviceKey, Error};

const CHANNEL_BINDING_CONTEXT: &[u8] = b"lattice-channel-binding-v1";

pub struct DeviceCertificate {
    pub cert: CertificateDer<'static>,
    pub key: PrivateKeyDer<'static>,
}

impl DeviceCertificate {
    /// Wraps the device's long-lived key in a certificate. Regenerated on
    /// every start: nothing pins the certificate, only the key inside it.
    pub fn generate(device_key: &DeviceKey) -> Result<Self, Error> {
        let pkcs8 = device_key.signing_key().to_pkcs8_der()?;
        let key_pair =
            KeyPair::from_pkcs8_der_and_sign_algo(&pkcs8.as_bytes().into(), &PKCS_ED25519)?;

        let mut params = CertificateParams::new(vec![device_key.id().to_string()])?;
        let mut name = DistinguishedName::new();
        name.push(DnType::CommonName, device_key.id().to_string());
        params.distinguished_name = name;

        let cert = params.self_signed(&key_pair)?;
        Ok(Self {
            cert: cert.der().clone(),
            key: PrivateKeyDer::try_from(pkcs8.as_bytes().to_vec())
                .map_err(|e| Error::Certificate(e.to_string()))?,
        })
    }
}

/// Pulls the Ed25519 public key out of a peer's certificate. This is the value
/// compared against the trust store; everything else in the certificate is
/// ignored, including validity dates and the subject name.
pub fn peer_key(cert: &CertificateDer<'_>) -> Result<VerifyingKey, Error> {
    let (_, parsed) = x509_parser::parse_x509_certificate(cert.as_ref())
        .map_err(|e| Error::Certificate(e.to_string()))?;
    let spki = parsed.public_key();
    if spki.algorithm.algorithm != x509_parser::oid_registry::OID_SIG_ED25519 {
        return Err(Error::Certificate("peer key is not Ed25519".into()));
    }
    let bytes: [u8; 32] = spki
        .subject_public_key
        .data
        .as_ref()
        .try_into()
        .map_err(|_| Error::Certificate("Ed25519 key is not 32 bytes".into()))?;
    VerifyingKey::from_bytes(&bytes).map_err(|e| Error::Certificate(e.to_string()))
}

/// Binds a pairing exchange to the TLS connection carrying it.
///
/// Ordering is canonical rather than local-then-peer, so both devices derive
/// the same value. A relay terminating TLS separately on each side presents a
/// different certificate to each, so the two sides derive different bindings
/// and the pairing confirmation fails.
pub fn channel_binding(local: &CertificateDer<'_>, peer: &CertificateDer<'_>) -> [u8; 32] {
    let (a, b) = (
        Sha256::digest(local.as_ref()),
        Sha256::digest(peer.as_ref()),
    );
    let (first, second) = if a <= b { (a, b) } else { (b, a) };
    let mut hasher = Sha256::new();
    hasher.update(CHANNEL_BINDING_CONTEXT);
    hasher.update(first);
    hasher.update(second);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceId;

    fn cert_for(dir: &std::path::Path) -> (DeviceKey, DeviceCertificate) {
        let key = DeviceKey::load_or_create(dir).unwrap();
        let cert = DeviceCertificate::generate(&key).unwrap();
        (key, cert)
    }

    #[test]
    fn the_certificate_carries_the_device_key() {
        let dir = tempfile::tempdir().unwrap();
        let (key, cert) = cert_for(dir.path());
        let recovered = peer_key(&cert.cert).unwrap();
        assert_eq!(recovered, key.public_key());
        assert_eq!(DeviceId::from_public_key(&recovered), key.id());
    }

    #[test]
    fn regenerating_keeps_the_identity_stable() {
        let dir = tempfile::tempdir().unwrap();
        let (key, first) = cert_for(dir.path());
        let second = DeviceCertificate::generate(&key).unwrap();
        assert_eq!(peer_key(&first.cert).unwrap(), peer_key(&second.cert).unwrap());
    }

    #[test]
    fn both_sides_derive_the_same_binding() {
        let (a_dir, b_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let (_, a) = cert_for(a_dir.path());
        let (_, b) = cert_for(b_dir.path());
        assert_eq!(
            channel_binding(&a.cert, &b.cert),
            channel_binding(&b.cert, &a.cert)
        );
    }

    #[test]
    fn a_relay_presenting_its_own_certificate_derives_a_different_binding() {
        let dirs: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
        let (_, a) = cert_for(dirs[0].path());
        let (_, b) = cert_for(dirs[1].path());
        let (_, relay) = cert_for(dirs[2].path());
        // A sees the relay, B sees the relay; neither sees the other.
        assert_ne!(
            channel_binding(&a.cert, &relay.cert),
            channel_binding(&b.cert, &relay.cert)
        );
    }

    #[test]
    fn a_malformed_certificate_is_rejected() {
        let garbage = CertificateDer::from(vec![0u8; 64]);
        assert!(peer_key(&garbage).is_err());
    }
}
