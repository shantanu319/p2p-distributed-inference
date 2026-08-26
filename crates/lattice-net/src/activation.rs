//! Hidden states across a shard boundary.
//!
//! One stream per session, opened once and reused for every token, because
//! opening a stream per decode would add a round trip to a hop that costs
//! 3–5 ms in total (PLAN.md §1).
//!
//! PLAN.md §6 specifies a hand-rolled frame with magic, version, positions and
//! a crc32. Almost all of it already exists elsewhere: the stream header
//! (stream.rs) carries the magic, version, session and generation, `Activation`
//! carries the shape, and QUIC checksums and orders every byte. What is left is
//! the position and the payload, so this sends a postcard-encoded `Step`
//! instead of building a second framing layer over the first.

use std::sync::Arc;

use lattice_engine::Payload;
use serde::{Deserialize, Serialize};

use crate::codec::{read_frame, write_frame};
use crate::stream::{StreamHeader, StreamKind};
use crate::{Connection, Error};

/// A 4K-token prefill at hidden 8192 in f16 is 67 MB (§1), and a prompt can be
/// longer. Generous, but still a ceiling rather than an open allocation.
pub const MAX_ACTIVATION_FRAME: usize = 512 << 20;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Step {
    Prefill { pos: u32, input: Payload },
    Decode { pos: u32, input: Payload },
    /// Free this session's KV. Not a request — nothing is sent back.
    Done,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StepResult {
    Output(Payload),
    /// The shard understood the step and could not run it. Distinct from the
    /// stream failing, which the master must treat as the device going away.
    Failed(String),
}

/// Runs a shard on behalf of a master. Implemented by the device that holds
/// the layers; `&self` because one shard serves whatever sessions it is given.
///
/// `run` is async so an implementation can hand the work to a thread that owns
/// the accelerator and await the answer. Doing that matters more than it
/// sounds: candle's Metal backend is measurably slower when its work migrates
/// between threads, and a tokio worker pool migrates constantly.
#[async_trait::async_trait]
pub trait Executor: Send + Sync {
    async fn run(&self, session: u64, step: Step) -> Result<Payload, String>;
    fn drop_session(&self, session: u64);
}

/// Declines every step. For devices that serve the control plane but hold no
/// layers — which is most of them, most of the time.
pub struct NoShards;

#[async_trait::async_trait]
impl Executor for NoShards {
    async fn run(&self, _session: u64, _step: Step) -> Result<Payload, String> {
        Err("this device holds no shards".into())
    }
    fn drop_session(&self, _session: u64) {}
}

/// The master's end of a boundary: a shard that happens to be elsewhere.
pub struct RemoteShard {
    send: quinn::SendStream,
    recv: quinn::RecvStream,
}

impl RemoteShard {
    pub async fn open(conn: &Connection, session: u64, generation: u32) -> Result<Self, Error> {
        let header = StreamHeader {
            kind: StreamKind::Activation,
            session,
            generation,
        };
        let (send, recv) = conn.open(header).await?;
        Ok(Self { send, recv })
    }

    /// One boundary crossing: §1's `transfer_b`, twice, plus the far side's
    /// compute.
    pub async fn step(&mut self, step: &Step) -> Result<Payload, Error> {
        write_frame(&mut self.send, &postcard::to_allocvec(step)?).await?;
        let bytes = read_frame(&mut self.recv, MAX_ACTIVATION_FRAME).await?;
        match postcard::from_bytes(&bytes)? {
            StepResult::Output(payload) => Ok(payload),
            StepResult::Failed(why) => Err(Error::Stream(format!("remote shard: {why}"))),
        }
    }

    /// Tells the far side to drop the session's KV. Best effort: if the peer
    /// has already gone, the KV went with it.
    pub async fn finish(mut self) {
        if let Ok(bytes) = postcard::to_allocvec(&Step::Done) {
            let _ = write_frame(&mut self.send, &bytes).await;
        }
        let _ = self.send.finish();
    }
}

/// Serves one session's activation stream until the master closes it.
pub async fn serve_stream(
    session: u64,
    executor: Arc<dyn Executor>,
    mut send: quinn::SendStream,
    mut recv: quinn::RecvStream,
) {
    loop {
        let Ok(bytes) = read_frame(&mut recv, MAX_ACTIVATION_FRAME).await else {
            break;
        };
        let step: Step = match postcard::from_bytes(&bytes) {
            Ok(step) => step,
            Err(e) => {
                let _ = reply(&mut send, StepResult::Failed(format!("unintelligible: {e}"))).await;
                break;
            }
        };
        if matches!(step, Step::Done) {
            break;
        }
        // Off the connection's task: a decode is tens of milliseconds of
        // compute, and running it here would hold up every other stream on
        // this connection for exactly that long.
        let result = match executor.run(session, step).await {
            Ok(payload) => StepResult::Output(payload),
            Err(why) => StepResult::Failed(why),
        };
        if reply(&mut send, result).await.is_err() {
            break;
        }
    }
    executor.drop_session(session);
}

async fn reply(send: &mut quinn::SendStream, result: StepResult) -> Result<(), Error> {
    write_frame(send, &postcard::to_allocvec(&result)?).await
}
