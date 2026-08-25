//! One connection per peer, reused.
//!
//! §5's planner makes devices adjacent in ways that change between plans, so
//! connections are established on demand and kept warm rather than torn down.
//! Followers talk to followers directly (docs/streams.md §2), which means any
//! device may both dial and be dialled.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use crate::{Connection, DeviceId, Endpoint, Error, transport::Role};

/// Sent when closing the loser of a simultaneous open.
const SUPERSEDED: u32 = 1;

pub struct Mesh {
    local: DeviceId,
    endpoint: Endpoint,
    peers: Mutex<HashMap<DeviceId, Arc<Connection>>>,
}

impl Mesh {
    pub fn new(endpoint: Endpoint, local: DeviceId) -> Self {
        Self {
            local,
            endpoint,
            peers: Mutex::new(HashMap::new()),
        }
    }

    pub fn local_addr(&self) -> Result<SocketAddr, Error> {
        self.endpoint.local_addr()
    }

    /// Whether this device should be the one to dial `peer`. Both devices
    /// evaluate this the same way, so in the common case only one dials and no
    /// resolution is needed.
    pub fn should_dial(&self, peer: DeviceId) -> bool {
        self.local < peer
    }

    pub fn get(&self, peer: DeviceId) -> Option<Arc<Connection>> {
        let peers = self.peers.lock().expect("mesh registry poisoned");
        peers.get(&peer).filter(|c| is_live(c)).cloned()
    }

    pub fn connected(&self) -> Vec<DeviceId> {
        let peers = self.peers.lock().expect("mesh registry poisoned");
        peers
            .iter()
            .filter(|(_, c)| is_live(c))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns the existing connection to `peer` if one is live, otherwise
    /// dials, trying each address in turn.
    pub async fn connect(
        &self,
        peer: DeviceId,
        addrs: &[SocketAddr],
    ) -> Result<Arc<Connection>, Error> {
        if let Some(live) = self.get(peer) {
            return Ok(live);
        }
        let mut last = None;
        for addr in addrs {
            match self.endpoint.connect(*addr).await {
                // An address can be stale or reassigned, so who answered
                // matters as much as whether anyone did.
                Ok(conn) if conn.peer_id() == peer => return Ok(self.adopt(conn)),
                Ok(conn) => {
                    last = Some(Error::Stream(format!(
                        "{addr} answered as {}, expected {peer}",
                        conn.peer_id()
                    )));
                }
                Err(e) => last = Some(e),
            }
        }
        Err(last.unwrap_or_else(|| Error::Stream(format!("no address known for {peer}"))))
    }

    /// Accepts one incoming connection and registers it.
    pub async fn accept(&self) -> Option<Result<Arc<Connection>, Error>> {
        Some(self.endpoint.accept().await?.map(|conn| self.adopt(conn)))
    }

    /// Registers a connection, resolving a simultaneous open.
    ///
    /// Both devices can dial at once and end up with two connections. They
    /// settle it without talking: keep the one whose *dialer* has the lower
    /// device id. Each side computes that identically for a given connection,
    /// so both discard the same one.
    pub fn adopt(&self, conn: Connection) -> Arc<Connection> {
        let conn = Arc::new(conn);
        let peer = conn.peer_id();
        let mut peers = self.peers.lock().expect("mesh registry poisoned");

        if let Some(existing) = peers.get(&peer).filter(|c| is_live(c)) {
            if self.dialer_of(existing) <= self.dialer_of(&conn) {
                conn.inner.close(SUPERSEDED.into(), b"duplicate connection");
                return existing.clone();
            }
            existing.inner.close(SUPERSEDED.into(), b"duplicate connection");
        }
        peers.insert(peer, conn.clone());
        conn
    }

    pub fn forget(&self, peer: DeviceId) {
        let mut peers = self.peers.lock().expect("mesh registry poisoned");
        if let Some(conn) = peers.remove(&peer) {
            conn.inner.close(SUPERSEDED.into(), b"forgotten");
        }
    }

    /// Which device opened this connection. Both ends agree on the answer.
    fn dialer_of(&self, conn: &Connection) -> DeviceId {
        match conn.role() {
            Role::Dialer => self.local,
            Role::Listener => conn.peer_id(),
        }
    }
}

fn is_live(conn: &Connection) -> bool {
    conn.inner.close_reason().is_none()
}
