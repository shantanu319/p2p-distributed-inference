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
use std::time::Duration;

use crate::stream::{HEADER_LEN, StreamHeader, StreamKind};
use crate::tls::{DeviceCertificate, channel_binding, peer_key};
use crate::{DeviceKey, Error};

pub const ALPN: &[u8] = b"lattice/1";

/// Opening bytes of the control stream, exchanged in both directions.
const HELLO: [u8; 4] = *b"LTC1";

/// Stream priorities. Activation sits at quinn's default, so a stream nobody
/// classified behaves as the hot path rather than silently outranking it.
/// Control outranks it because generation bumps and drain notices decide how
/// fast a master can re-plan, and they are a few bytes each.
pub const PRIORITY_CONTROL: i32 = 1;
pub const PRIORITY_ACTIVATION: i32 = 0;
pub const PRIORITY_BULK: i32 = -1;

/// Sized to the worst link we intend to work on: gigabit ethernet (§8) at the
/// 50 ms RTT §12 warns a congested 2.4 GHz link can reach. Sustaining a rate
/// needs a window of at least bandwidth x RTT, and quinn's 1.25 MB default
/// would cap that path at ~25 MB/s — the transport bottlenecking below the
/// link, which we would misread as the network being slow.
const STREAM_RECEIVE_WINDOW: u32 = 8 << 20;

/// Caps memory per peer. Streams share it, so this is the real ceiling on how
/// much unacknowledged data one peer can make us buffer.
const CONNECTION_WINDOW: u32 = 32 << 20;

/// Idle connections are the norm between requests, and quinn ships no
/// keepalive against a 30 s idle timeout, so a warm mesh connection would die
/// half a minute after the last token and pay a full handshake on the next.
const KEEP_ALIVE: Duration = Duration::from_secs(10);
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

// These relationships are the whole point of the constants above, so they are
// checked at compile time rather than left to a test someone might delete.
const _: () = assert!(
    KEEP_ALIVE.as_secs() * 2 < IDLE_TIMEOUT.as_secs(),
    "a single lost keepalive must not close a live connection"
);
const _: () = assert!(
    CONNECTION_WINDOW > STREAM_RECEIVE_WINDOW,
    "one stream must not be able to exhaust the connection window"
);
const _: () = assert!(PRIORITY_CONTROL > PRIORITY_ACTIVATION);
const _: () = assert!(PRIORITY_ACTIVATION > PRIORITY_BULK);
const _: () = assert!(
    PRIORITY_ACTIVATION == 0,
    "activation must sit at quinn's default so unclassified streams match it"
);

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

        let transport = Arc::new(transport_config()?);
        let mut server_config =
            quinn::ServerConfig::with_crypto(Arc::new(QuicServerConfig::try_from(server).map_err(|e| Error::Tls(e.to_string()))?));
        server_config.transport_config(transport.clone());

        let mut client_config = quinn::ClientConfig::new(Arc::new(
            QuicClientConfig::try_from(client).map_err(|e| Error::Tls(e.to_string()))?,
        ));
        client_config.transport_config(transport);

        let mut endpoint = quinn::Endpoint::server(server_config, addr)?;
        endpoint.set_default_client_config(client_config);

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
        let conn = self.wrap(connecting.await?, Role::Dialer)?;
        conn.confirm().await?;
        Ok(conn)
    }

    pub async fn accept(&self) -> Option<Result<Connection, Error>> {
        let incoming = self.endpoint.accept().await?;
        Some(async {
            let conn = self.wrap(incoming.await?, Role::Listener)?;
            conn.confirm().await?;
            Ok(conn)
        }
        .await)
    }

    pub fn wait_idle(&self) -> impl std::future::Future<Output = ()> + '_ {
        self.endpoint.wait_idle()
    }

    fn wrap(&self, inner: quinn::Connection, role: Role) -> Result<Connection, Error> {
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
            role,
        })
    }
}

/// An authenticated connection to one peer.
pub struct Connection {
    pub inner: quinn::Connection,
    peer_key: VerifyingKey,
    channel_binding: [u8; 32],
    role: Role,
}

/// Which side opened the connection. Stream setup is not symmetric — one
/// side must open where the other accepts — so the role is remembered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Dialer,
    Listener,
}

impl Connection {
    /// TLS 1.3 completes the client's handshake before the server has judged
    /// the client certificate, so a dialer whose key is not pinned still sees
    /// `connect` succeed. Exchanging bytes over a control stream is what turns
    /// that into an error at the point of connection rather than a puzzling
    /// failure on first use.
    async fn confirm(&self) -> Result<(), Error> {
        let (mut send, mut recv) = self.control_stream().await?;
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

    /// Opens a channel to the peer, declaring what it carries. Either side may
    /// do this at any time — that is the point of the header.
    pub async fn open(
        &self,
        header: StreamHeader,
    ) -> Result<(quinn::SendStream, quinn::RecvStream), Error> {
        let (mut send, recv) = self
            .inner
            .open_bi()
            .await
            .map_err(|e| Error::Stream(format!("opening a channel: {e}")))?;
        let _ = send.set_priority(header.kind.priority());
        send.write_all(&header.encode())
            .await
            .map_err(|e| Error::Stream(format!("writing a stream header: {e}")))?;
        Ok((send, recv))
    }

    /// Waits for the peer to open a channel and reads what it declared.
    pub async fn accept_stream(
        &self,
    ) -> Result<(StreamHeader, quinn::SendStream, quinn::RecvStream), Error> {
        let (send, mut recv) = self
            .inner
            .accept_bi()
            .await
            .map_err(|e| Error::Stream(format!("accepting a channel: {e}")))?;
        let mut bytes = [0u8; HEADER_LEN];
        recv.read_exact(&mut bytes)
            .await
            .map_err(|e| Error::Stream(format!("reading a stream header: {e}")))?;
        let header = StreamHeader::decode(&bytes)?;
        let _ = send.set_priority(header.kind.priority());
        Ok((header, send, recv))
    }

    /// One control channel for a symmetric two-party exchange on a fresh
    /// connection, where there is no dispatcher yet: the dialer opens, the
    /// listener accepts. Used by the handshake and by pairing.
    pub async fn control_stream(&self) -> Result<(quinn::SendStream, quinn::RecvStream), Error> {
        match self.role {
            Role::Dialer => self.open(StreamHeader::control()).await,
            Role::Listener => {
                let (header, send, recv) = self.accept_stream().await?;
                if header.kind != StreamKind::Control {
                    return Err(Error::StreamHeader(format!(
                        "expected a control stream, got {:?}",
                        header.kind
                    )));
                }
                Ok((send, recv))
            }
        }
    }

    pub fn role(&self) -> Role {
        self.role
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

fn transport_config() -> Result<quinn::TransportConfig, Error> {
    let mut config = quinn::TransportConfig::default();
    config
        .stream_receive_window(STREAM_RECEIVE_WINDOW.into())
        .receive_window(CONNECTION_WINDOW.into())
        .send_window(u64::from(CONNECTION_WINDOW))
        .keep_alive_interval(Some(KEEP_ALIVE))
        .max_idle_timeout(Some(
            IDLE_TIMEOUT
                .try_into()
                .map_err(|_| Error::Tls("idle timeout out of range".into()))?,
        ));
    Ok(config)
}

fn tls_err(e: rustls::Error) -> Error {
    Error::Tls(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Locks the reasoning behind the window size. A silent regression here
    /// looks like "the network got slow" rather than a config change.
    #[test]
    fn the_stream_window_covers_the_worst_link_we_target() {
        // Gigabit ethernet (§8) at the 50 ms RTT §12 warns of: bandwidth x RTT.
        let bandwidth_delay_product = 125.0e6 * 0.050;
        assert!(
            f64::from(STREAM_RECEIVE_WINDOW) >= bandwidth_delay_product,
            "{STREAM_RECEIVE_WINDOW} bytes caps this link below its capacity"
        );
    }

    #[test]
    fn the_transport_config_builds() {
        transport_config().unwrap();
    }
}
