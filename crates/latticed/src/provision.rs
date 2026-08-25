//! Master-provisioned trust (docs/streams.md §7).
//!
//! The user pairs each device to the master once. The master then hands every
//! device the others' keys, and each pins them exactly as if it had paired
//! directly. Afterwards nothing consults the master: the end state is the same
//! pairwise trust store we would have had, minus the typing.

use lattice_net::{
    ControlHandler, DeviceId, Introduction, PairedPeer, Request, Response, TrustStore,
};
use std::sync::{Arc, RwLock};

use crate::host::HostFacts;

/// Answers control requests, and accepts introductions from trusted peers.
pub struct Provisioner {
    store: Arc<RwLock<TrustStore>>,
    identity: (DeviceId, String, String),
}

impl Provisioner {
    pub fn new(store: Arc<RwLock<TrustStore>>, id: DeviceId, facts: &HostFacts) -> Self {
        Self {
            store,
            identity: (id, facts.name.clone(), facts.platform.clone()),
        }
    }

    fn accept_introductions(&self, from: DeviceId, peers: Vec<Introduction>) -> Response {
        // The connection was authenticated against the trust store, so `from`
        // is trusted. Checked anyway because the pairing endpoint accepts any
        // peer, and a dispatcher must never be run on that endpoint.
        if !self.knows(from) {
            return Response::Refused(format!("{from} is not a paired device"));
        }

        let (mut added, mut already_known) = (0, 0);
        for introduction in peers {
            let peer: PairedPeer = match introduction.into_peer(from) {
                Ok(peer) => peer,
                Err(e) => return Response::Refused(e.to_string()),
            };
            // An introduction to ourselves is a normal artefact of the master
            // broadcasting one roster, not an error.
            if peer.device_id == self.identity.0 || self.knows(peer.device_id) {
                already_known += 1;
                continue;
            }
            let mut store = self.store.write().expect("trust store poisoned");
            if let Err(e) = store.insert(peer) {
                return Response::Refused(format!("could not record the introduction: {e}"));
            }
            added += 1;
        }
        Response::Provisioned {
            added,
            already_known,
        }
    }

    fn knows(&self, id: DeviceId) -> bool {
        self.store
            .read()
            .expect("trust store poisoned")
            .get(id)
            .is_some()
    }
}

impl ControlHandler for Provisioner {
    fn handle(&self, from: DeviceId, request: Request) -> Response {
        match request {
            Request::Hello => Response::Identity {
                device_id: self.identity.0,
                name: self.identity.1.clone(),
                platform: self.identity.2.clone(),
            },
            Request::Provision(peers) => self.accept_introductions(from, peers),
        }
    }
}

/// What each device needs to hear: everyone the master trusts, except itself.
pub fn roster_for(target: DeviceId, all: &[PairedPeer]) -> Vec<Introduction> {
    all.iter()
        .filter(|peer| peer.device_id != target)
        .map(Introduction::of)
        .collect()
}
