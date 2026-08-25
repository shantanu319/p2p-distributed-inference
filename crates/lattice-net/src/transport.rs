//! QUIC transport between paired devices.
//!
//! There is no certificate authority anywhere in this path. Both directions
//! authenticate the same way: pull the Ed25519 key out of the peer's
//! certificate and ask the policy whether it is trusted. Certificate dates,
//! subject names and chains are all ignored.

use ed25519_dalek::VerifyingKey;
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::{DigitallySignedStruct, DistinguishedName, SignatureScheme};
use std::net::SocketAddr;
use std::sync::Arc;

use crate::tls::{DeviceCertificate, channel_binding, peer_key};
use crate::{DeviceKey, Error};

pub const ALPN: &[u8] = b"lattice/1";

/// Opening bytes of the control stream, exchanged in both directions.
const HELLO: [u8; 4] = *b"LTC1";

/// Decides whether a presented device key may connect. Pairing uses
/// [`AcceptAnyPeer`]; everything else consults the trust store.
pub trait PeerPolicy: Send + Sync + std::fmt::Debug {
    fn accept(&self, key: &VerifyingKey) -> bool;
}

/// Accepts any device key. Only valid for the pairing endpoint, where SPAKE2
/// — not TLS — supplies authentication.
#[derive(Debug)]
pub struct AcceptAnyPeer;

impl PeerPolicy for AcceptAnyPeer {
    fn accept(&self, _key: &VerifyingKey) -> bool {
        true
    }
}

pub struct Endpoint {
    endpoint: quinn::Endpoint,
    local_cert: CertificateDer<'static>,
}

impl Endpoint {
    /// Binds a QUIC endpoint that both listens and dials, authenticating peers
    /// in both directions with `policy`.
    pub fn bind(
        addr: SocketAddr,
        device_key: &DeviceKey,
        policy: Arc<dyn PeerPolicy>,
    ) -> Result<Self, Error> {
        let identity = DeviceCertificate::generate(device_key)?;
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let verifier = Arc::new(DeviceKeyVerifier {
            policy,
            provider: provider.clone(),
        });

        let mut server = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(tls_err)?
            .with_client_cert_verifier(verifier.clone())
            .with_single_cert(vec![identity.cert.clone()], identity.key.clone_key())
            .map_err(tls_err)?;
        server.alpn_protocols = vec![ALPN.to_vec()];

        let mut client = rustls::ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(tls_err)?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_client_auth_cert(vec![identity.cert.clone()], identity.key)
            .map_err(tls_err)?;
        client.alpn_protocols = vec![ALPN.to_vec()];

        let server_config =
            quinn::ServerConfig::with_crypto(Arc::new(QuicServerConfig::try_from(server).map_err(|e| Error::Tls(e.to_string()))?));
        let mut endpoint = quinn::Endpoint::server(server_config, addr)?;
        endpoint.set_default_client_config(quinn::ClientConfig::new(Arc::new(
            QuicClientConfig::try_from(client).map_err(|e| Error::Tls(e.to_string()))?,
        )));

        Ok(Self {
            endpoint,
            local_cert: identity.cert,
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr, Error> {
        Ok(self.endpoint.local_addr()?)
    }

    pub async fn connect(&self, addr: SocketAddr) -> Result<Connection, Error> {
        // The SNI name is unused: the peer is identified by its key, and the
        // verifier ignores names entirely.
        let connecting = self.endpoint.connect(addr, "lattice")?;
        let conn = self.wrap(connecting.await?)?;
        conn.confirm(Role::Dialer).await?;
        Ok(conn)
    }

    pub async fn accept(&self) -> Option<Result<Connection, Error>> {
        let incoming = self.endpoint.accept().await?;
        Some(async {
            let conn = self.wrap(incoming.await?)?;
            conn.confirm(Role::Listener).await?;
            Ok(conn)
        }
        .await)
    }

    pub fn wait_idle(&self) -> impl std::future::Future<Output = ()> + '_ {
        self.endpoint.wait_idle()
    }

    fn wrap(&self, inner: quinn::Connection) -> Result<Connection, Error> {
        let certs = inner
            .peer_identity()
            .and_then(|id| id.downcast::<Vec<CertificateDer<'static>>>().ok())
            .ok_or_else(|| Error::Certificate("peer presented no certificate".into()))?;
        let end_entity = certs
            .first()
            .ok_or_else(|| Error::Certificate("peer certificate chain is empty".into()))?;

        Ok(Connection {
            channel_binding: channel_binding(&self.local_cert, end_entity),
            peer_key: peer_key(end_entity)?,
            inner,
        })
    }
}

/// An authenticated connection to one peer.
pub struct Connection {
    pub inner: quinn::Connection,
    peer_key: VerifyingKey,
    channel_binding: [u8; 32],
}

enum Role {
    Dialer,
    Listener,
}

impl Connection {
    /// TLS 1.3 completes the client's handshake before the server has judged
    /// the client certificate, so a dialer whose key is not pinned still sees
    /// `connect` succeed. Exchanging bytes over a control stream is what turns
    /// that into an error at the point of connection rather than a puzzling
    /// failure on first use.
    async fn confirm(&self, role: Role) -> Result<(), Error> {
        let (mut send, mut recv) = match role {
            Role::Dialer => self.inner.open_bi().await.map_err(|_| Error::Rejected)?,
            Role::Listener => self.inner.accept_bi().await.map_err(|_| Error::Rejected)?,
        };
        send.write_all(&HELLO).await.map_err(|_| Error::Rejected)?;

        let mut greeting = [0u8; HELLO.len()];
        recv.read_exact(&mut greeting)
            .await
            .map_err(|_| Error::Rejected)?;
        if greeting != HELLO {
            return Err(Error::Rejected);
        }
        Ok(())
    }

    pub fn peer_key(&self) -> VerifyingKey {
        self.peer_key
    }

    pub fn peer_id(&self) -> crate::DeviceId {
        crate::DeviceId::from_public_key(&self.peer_key)
    }

    /// Value to feed to [`crate::Pairing::start`] on this connection.
    pub fn channel_binding(&self) -> [u8; 32] {
        self.channel_binding
    }

    pub fn remote_address(&self) -> SocketAddr {
        self.inner.remote_address()
    }
}

/// Verifies both peers the same way, in both directions.
#[derive(Debug)]
struct DeviceKeyVerifier {
    policy: Arc<dyn PeerPolicy>,
    provider: Arc<rustls::crypto::CryptoProvider>,
}

impl DeviceKeyVerifier {
    fn check(&self, cert: &CertificateDer<'_>) -> Result<(), rustls::Error> {
        let key = peer_key(cert)
            .map_err(|_| rustls::Error::InvalidCertificate(rustls::CertificateError::BadEncoding))?;
        if !self.policy.accept(&key) {
            return Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(())
    }

    fn verify_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.provider.signature_verification_algorithms,
        )
    }
}

impl ServerCertVerifier for DeviceKeyVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        self.check(end_entity)?;
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Err(rustls::Error::PeerIncompatible(
            rustls::PeerIncompatible::Tls12NotOffered,
        ))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.verify_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

impl ClientCertVerifier for DeviceKeyVerifier {
    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        &[]
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<ClientCertVerified, rustls::Error> {
        self.check(end_entity)?;
        Ok(ClientCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        Err(rustls::Error::PeerIncompatible(
            rustls::PeerIncompatible::Tls12NotOffered,
        ))
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.verify_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.provider.signature_verification_algorithms.supported_schemes()
    }
}

fn tls_err(e: rustls::Error) -> Error {
    Error::Tls(e.to_string())
}
