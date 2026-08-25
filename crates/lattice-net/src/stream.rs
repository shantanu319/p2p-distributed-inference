//! Every stream opens by saying what it is.
//!
//! In a mesh either side may originate a channel, so streams cannot be
//! distinguished by who opened them. A fixed header on the first bytes lets the
//! accepting side route the stream without any prior agreement, which is what
//! makes "open n channels for whatever you need" work.
//!
//! This sits above §6's activation frame: the header describes the *channel*,
//! §6's frame describes each payload on it.

use crate::Error;
use crate::transport::{PRIORITY_ACTIVATION, PRIORITY_BULK, PRIORITY_CONTROL};

pub const MAGIC: u32 = u32::from_le_bytes(*b"LTCS");
pub const VERSION: u16 = 1;
pub const HEADER_LEN: usize = 20;

/// Sessions are numbered from 1; 0 means "not tied to a session".
pub const NO_SESSION: u64 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamKind {
    /// Plan pushes, provisioning, generation bumps, drain notices.
    Control = 1,
    /// §6 frames for one session across one shard boundary.
    Activation = 2,
    /// Model transfer (§8) and probe payloads. Must never delay the others.
    Bulk = 3,
}

impl StreamKind {
    fn from_wire(value: u16) -> Result<Self, Error> {
        match value {
            1 => Ok(Self::Control),
            2 => Ok(Self::Activation),
            3 => Ok(Self::Bulk),
            other => Err(Error::StreamHeader(format!("unknown stream kind {other}"))),
        }
    }

    pub fn priority(self) -> i32 {
        match self {
            Self::Control => PRIORITY_CONTROL,
            Self::Activation => PRIORITY_ACTIVATION,
            Self::Bulk => PRIORITY_BULK,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamHeader {
    pub kind: StreamKind,
    pub session: u64,
    /// §6's fence. Checked when the stream opens, so a stale channel dies
    /// before it carries anything rather than per frame.
    pub generation: u32,
}

impl StreamHeader {
    pub fn control() -> Self {
        Self {
            kind: StreamKind::Control,
            session: NO_SESSION,
            generation: 0,
        }
    }

    pub fn encode(&self) -> [u8; HEADER_LEN] {
        let mut out = [0u8; HEADER_LEN];
        out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        out[4..6].copy_from_slice(&VERSION.to_le_bytes());
        out[6..8].copy_from_slice(&(self.kind as u16).to_le_bytes());
        out[8..16].copy_from_slice(&self.session.to_le_bytes());
        out[16..20].copy_from_slice(&self.generation.to_le_bytes());
        out
    }

    pub fn decode(bytes: &[u8; HEADER_LEN]) -> Result<Self, Error> {
        let magic = u32::from_le_bytes(bytes[0..4].try_into().expect("4 bytes"));
        if magic != MAGIC {
            return Err(Error::StreamHeader("not a lattice stream".into()));
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().expect("2 bytes"));
        if version != VERSION {
            return Err(Error::StreamHeader(format!(
                "stream protocol v{version}, this device speaks v{VERSION}"
            )));
        }
        Ok(Self {
            kind: StreamKind::from_wire(u16::from_le_bytes(
                bytes[6..8].try_into().expect("2 bytes"),
            ))?,
            session: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
            generation: u32::from_le_bytes(bytes[16..20].try_into().expect("4 bytes")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StreamHeader {
        StreamHeader {
            kind: StreamKind::Activation,
            session: 0x0123_4567_89ab_cdef,
            generation: 7,
        }
    }

    #[test]
    fn headers_round_trip() {
        assert_eq!(StreamHeader::decode(&sample().encode()).unwrap(), sample());
        let control = StreamHeader::control();
        assert_eq!(StreamHeader::decode(&control.encode()).unwrap(), control);
    }

    #[test]
    fn the_header_is_a_fixed_twenty_bytes() {
        assert_eq!(sample().encode().len(), HEADER_LEN);
    }

    #[test]
    fn a_stream_that_is_not_ours_is_rejected() {
        let mut bytes = sample().encode();
        bytes[0] ^= 0xff;
        assert!(StreamHeader::decode(&bytes).is_err());
    }

    #[test]
    fn a_future_protocol_version_is_named_in_the_error() {
        let mut bytes = sample().encode();
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = StreamHeader::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("v99"), "{err}");
    }

    #[test]
    fn an_unknown_kind_is_rejected_rather_than_defaulted() {
        let mut bytes = sample().encode();
        bytes[6..8].copy_from_slice(&64u16.to_le_bytes());
        assert!(StreamHeader::decode(&bytes).is_err());
    }

    #[test]
    fn bulk_never_outranks_activation() {
        assert!(StreamKind::Control.priority() > StreamKind::Activation.priority());
        assert!(StreamKind::Activation.priority() > StreamKind::Bulk.priority());
    }
}
