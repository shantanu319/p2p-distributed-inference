//! Measures the link to a peer: round-trip time and one-way throughput.
//!
//! These are the two numbers §5's planner needs, and the two that decide
//! whether the network is a 15% tax or the bottleneck. Both are measured over
//! the same authenticated QUIC connection the shards will use, so they reflect
//! the path the activations will actually take.

use std::time::{Duration, Instant};

use crate::transport::Connection;
use crate::Error;

const PING: u8 = 0x01;
const SINK: u8 = 0x02;
const ACK: u8 = 0xAC;

/// Enough to get past slow-start on Wi-Fi without stalling the UI.
pub const DEFAULT_PROBE_BYTES: usize = 8 << 20;
const MAX_PROBE_BYTES: usize = 64 << 20;
const RTT_SAMPLES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkQuality {
    /// Best observed round trip. The minimum, not the mean: it is the closest
    /// estimate of the link's floor, and Wi-Fi's tail is noise from retries
    /// rather than a property of the path.
    pub rtt: Duration,
    pub throughput_bytes_per_sec: f64,
}

impl LinkQuality {
    /// Time to move one decode activation across this link, per §1: a
    /// `hidden_dim`-wide fp16 hidden state plus one round trip.
    pub fn decode_hop(&self, hidden_dim: usize) -> Duration {
        let bytes = (hidden_dim * 2) as f64;
        self.rtt + Duration::from_secs_f64(bytes / self.throughput_bytes_per_sec)
    }
}

/// Measures the link. The peer must be running [`serve`] on the same
/// connection.
pub async fn measure(conn: &Connection, probe_bytes: usize) -> Result<LinkQuality, Error> {
    let probe_bytes = probe_bytes.min(MAX_PROBE_BYTES);
    Ok(LinkQuality {
        rtt: measure_rtt(conn).await?,
        throughput_bytes_per_sec: measure_throughput(conn, probe_bytes).await?,
    })
}

async fn measure_rtt(conn: &Connection) -> Result<Duration, Error> {
    let (mut send, mut recv) = conn.inner.open_bi().await.map_err(|_| Error::Rejected)?;
    let mut best = Duration::MAX;
    let mut byte = [0u8; 1];
    for _ in 0..RTT_SAMPLES {
        let started = Instant::now();
        send.write_all(&[PING]).await.map_err(probe_err)?;
        recv.read_exact(&mut byte).await.map_err(probe_err)?;
        best = best.min(started.elapsed());
    }
    Ok(best)
}

async fn measure_throughput(conn: &Connection, probe_bytes: usize) -> Result<f64, Error> {
    let (mut send, mut recv) = conn.inner.open_bi().await.map_err(|_| Error::Rejected)?;
    // Bulk: a probe must not delay activations, for the same reason a model
    // transfer must not (§8).
    let _ = send.set_priority(crate::transport::PRIORITY_BULK);
    let payload = vec![0u8; probe_bytes];

    let started = Instant::now();
    send.write_all(&[SINK]).await.map_err(probe_err)?;
    send.write_all(&(probe_bytes as u32).to_le_bytes())
        .await
        .map_err(probe_err)?;
    send.write_all(&payload).await.map_err(probe_err)?;

    // Timed to the peer's acknowledgement, not to the last local write, which
    // would only measure how fast we can fill the send buffer.
    let mut ack = [0u8; 1];
    recv.read_exact(&mut ack).await.map_err(probe_err)?;
    let elapsed = started.elapsed();

    if ack[0] != ACK {
        return Err(Error::Rejected);
    }
    Ok(probe_bytes as f64 / elapsed.as_secs_f64())
}

/// Answers probes until the connection closes. Run this for every peer.
pub async fn serve(conn: &Connection) -> Result<(), Error> {
    while let Ok((send, recv)) = conn.inner.accept_bi().await {
        tokio::spawn(serve_stream(send, recv));
    }
    Ok(())
}

async fn serve_stream(mut send: quinn::SendStream, mut recv: quinn::RecvStream) {
    let mut opcode = [0u8; 1];
    while recv.read_exact(&mut opcode).await.is_ok() {
        let handled = match opcode[0] {
            PING => send.write_all(&[ACK]).await.is_ok(),
            SINK => drain(&mut send, &mut recv).await,
            _ => false,
        };
        if !handled {
            return;
        }
    }
}

async fn drain(send: &mut quinn::SendStream, recv: &mut quinn::RecvStream) -> bool {
    let mut header = [0u8; 4];
    if recv.read_exact(&mut header).await.is_err() {
        return false;
    }
    let len = u32::from_le_bytes(header) as usize;
    if len > MAX_PROBE_BYTES {
        return false;
    }
    let mut sink = vec![0u8; len];
    recv.read_exact(&mut sink).await.is_ok() && send.write_all(&[ACK]).await.is_ok()
}

fn probe_err<E: std::fmt::Display>(e: E) -> Error {
    Error::Probe(e.to_string())
}
