//! mDNS-SD discovery of other Lattice devices on the LAN.
//!
//! Discovery is unauthenticated by design: it only tells you a device exists.
//! Nothing here grants trust — that is pairing's job (see `pairing`).

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;

use crate::{DeviceId, Error};

pub const SERVICE_TYPE: &str = "_lattice._tcp.local.";
pub const PROTOCOL_VERSION: u16 = 1;

const TXT_DEVICE_ID: &str = "id";
const TXT_NAME: &str = "name";
const TXT_PLATFORM: &str = "platform";
const TXT_MEMORY: &str = "mem";
const TXT_PROTOCOL: &str = "proto";

/// What this device publishes about itself.
#[derive(Clone, Debug)]
pub struct Advertisement {
    pub device_id: DeviceId,
    /// Human-facing name, typically the hostname.
    pub name: String,
    /// `os-arch`, e.g. `macos-aarch64`.
    pub platform: String,
    /// Total physical memory in bytes. Advisory only — the planner re-measures.
    pub total_memory: u64,
    /// Port the QUIC listener is bound to.
    pub port: u16,
}

/// A device seen on the network. Being discovered implies nothing about trust.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveredPeer {
    pub device_id: DeviceId,
    pub name: String,
    pub platform: String,
    pub total_memory: u64,
    pub protocol: u16,
    pub addrs: Vec<SocketAddr>,
}

#[derive(Clone, Debug)]
pub enum PeerEvent {
    Found(DiscoveredPeer),
    Lost(DeviceId),
}

pub struct Discovery {
    daemon: ServiceDaemon,
    advertised: Option<String>,
}

impl Discovery {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            daemon: ServiceDaemon::new()?,
            advertised: None,
        })
    }

    /// Publishes this device. The mDNS instance name is the device ID, which
    /// keeps instance names unique and lets `Lost` events identify the device
    /// without a cache lookup.
    pub fn advertise(&mut self, ad: &Advertisement) -> Result<(), Error> {
        let id = ad.device_id.to_string();
        let txt = HashMap::from([
            (TXT_DEVICE_ID.to_owned(), id.clone()),
            (TXT_NAME.to_owned(), ad.name.clone()),
            (TXT_PLATFORM.to_owned(), ad.platform.clone()),
            (TXT_MEMORY.to_owned(), ad.total_memory.to_string()),
            (TXT_PROTOCOL.to_owned(), PROTOCOL_VERSION.to_string()),
        ]);
        let info = ServiceInfo::new(
            SERVICE_TYPE,
            &id,
            &format!("{id}.local."),
            "",
            ad.port,
            txt,
        )?
        .enable_addr_auto();

        self.daemon.register(info)?;
        self.advertised = Some(format!("{id}.{SERVICE_TYPE}"));
        Ok(())
    }

    pub fn browse(&self) -> Result<Browser, Error> {
        Ok(Browser {
            events: self.daemon.browse(SERVICE_TYPE)?,
        })
    }
}

impl Drop for Discovery {
    fn drop(&mut self) {
        if let Some(fullname) = &self.advertised {
            let _ = self.daemon.unregister(fullname);
        }
        let _ = self.daemon.shutdown();
    }
}

pub struct Browser {
    events: mdns_sd::Receiver<ServiceEvent>,
}

impl Browser {
    /// Waits for the next peer appearance or disappearance. Returns `None` on
    /// timeout. mDNS events that are not resolutions are skipped silently.
    pub fn next_event(&self, timeout: Duration) -> Option<PeerEvent> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.checked_duration_since(std::time::Instant::now())?;
            match self.events.recv_timeout(remaining).ok()? {
                ServiceEvent::ServiceResolved(svc) => {
                    if let Some(peer) = to_peer(&svc) {
                        return Some(PeerEvent::Found(peer));
                    }
                }
                ServiceEvent::ServiceRemoved(_, fullname) => {
                    if let Some(id) = device_id_from_fullname(&fullname) {
                        return Some(PeerEvent::Lost(id));
                    }
                }
                _ => {}
            }
        }
    }
}

/// A record missing or mangling any field it declared is dropped rather than
/// defaulted — a half-parsed peer is worse than an invisible one.
fn to_peer(svc: &mdns_sd::ResolvedService) -> Option<DiscoveredPeer> {
    let txt = |key| svc.txt_properties.get_property_val_str(key);
    let device_id: DeviceId = txt(TXT_DEVICE_ID)?.parse().ok()?;
    Some(DiscoveredPeer {
        device_id,
        name: txt(TXT_NAME)?.to_owned(),
        platform: txt(TXT_PLATFORM)?.to_owned(),
        total_memory: txt(TXT_MEMORY)?.parse().ok()?,
        protocol: txt(TXT_PROTOCOL)?.parse().ok()?,
        addrs: svc
            .addresses
            .iter()
            .map(|a| SocketAddr::new(a.to_ip_addr(), svc.port))
            .collect(),
    })
}

fn device_id_from_fullname(fullname: &str) -> Option<DeviceId> {
    fullname.split('.').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_id() -> DeviceId {
        "0123456789abcdef".parse().unwrap()
    }

    #[test]
    fn lost_events_recover_the_device_id_from_the_instance_name() {
        let parsed = device_id_from_fullname(&format!("{}.{SERVICE_TYPE}", sample_id()));
        assert_eq!(parsed, Some(sample_id()));
    }

    #[test]
    fn non_lattice_instance_names_are_ignored() {
        assert_eq!(device_id_from_fullname("printer._ipp._tcp.local."), None);
    }
}
