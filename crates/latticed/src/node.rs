//! This device answering for itself: the trust it will accept, and the layers
//! it will hold.
//!
//! Master-provisioned trust (docs/streams.md §7).
//!
//! The user pairs each device to the master once. The master then hands every
//! device the others' keys, and each pins them exactly as if it had paired
//! directly. Afterwards nothing consults the master: the end state is the same
//! pairwise trust store we would have had, minus the typing.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{Arc, RwLock};

use lattice_engine::{CandleShard, Payload, Shard, ShardSpec, hash_file};
use lattice_net::{
    ControlHandler, DeviceId, Executor, Introduction, PairedPeer, Request, Response, Step,
    TrustStore,
};

use crate::host::HostFacts;

/// Answers control requests, accepts introductions from trusted peers, and
/// executes whatever layer range it has been given.
pub struct Node {
    store: Arc<RwLock<TrustStore>>,
    identity: (DeviceId, String, String),
    /// The one GGUF this device is willing to serve. §8's peer transfer does
    /// not exist yet, so the file gets here by whatever means the user chose.
    model: Option<PathBuf>,
    shard: ShardThread,
}

/// Everything touching the shard happens on one thread.
///
/// Not for exclusion — a `Mutex` would give that. For affinity: candle's Metal
/// backend runs roughly three times slower when its work moves between threads,
/// and a tokio worker pool moves it on almost every step. Measured at 21.6 vs
/// 71 tok/s on TinyLlama; see bench/RESULTS.md.
struct ShardThread {
    jobs: mpsc::Sender<Job>,
}

enum Job {
    Load {
        path: PathBuf,
        spec: ShardSpec,
        reply: mpsc::Sender<Response>,
    },
    Run {
        session: u64,
        step: Step,
        reply: tokio::sync::oneshot::Sender<Result<Payload, String>>,
    },
    Drop {
        session: u64,
    },
}

impl ShardThread {
    fn spawn() -> Self {
        let (jobs, inbox) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("lattice-shard".into())
            .spawn(move || {
                let mut held: Option<CandleShard> = None;
                while let Ok(job) = inbox.recv() {
                    match job {
                        Job::Load { path, spec, reply } => {
                            let _ = reply.send(match CandleShard::open(&path, spec) {
                                Ok(shard) => {
                                    let stats = shard.stats();
                                    held = Some(shard);
                                    Response::ShardLoaded {
                                        weight_bytes: stats.weight_bytes,
                                        kv_bytes: stats.kv_bytes,
                                    }
                                }
                                Err(e) => Response::Refused(format!("could not load the shard: {e}")),
                            });
                        }
                        Job::Run { session, step, reply } => {
                            let _ = reply.send(run_step(held.as_mut(), session, step));
                        }
                        Job::Drop { session } => {
                            if let Some(shard) = held.as_mut() {
                                shard.drop_session(session);
                            }
                        }
                    }
                }
            })
            .expect("spawning the shard thread");
        Self { jobs }
    }
}

fn run_step(shard: Option<&mut CandleShard>, session: u64, step: Step) -> Result<Payload, String> {
    let shard = shard.ok_or("this device has been given no layers")?;
    match step {
        Step::Prefill { pos, input } => shard.prefill(session, pos, input),
        Step::Decode { pos, input } => shard.decode(session, pos, input),
        Step::Done => return Err("Done is handled by the stream, not the shard".into()),
    }
    .map_err(|e| e.to_string())
}

impl Node {
    pub fn new(
        store: Arc<RwLock<TrustStore>>,
        id: DeviceId,
        facts: &HostFacts,
        model: Option<PathBuf>,
    ) -> Self {
        Self {
            store,
            identity: (id, facts.name.clone(), facts.platform.clone()),
            model,
            shard: ShardThread::spawn(),
        }
    }

    fn load_shard(&self, spec: ShardSpec) -> Response {
        let Some(path) = &self.model else {
            return Response::Refused("this device has no model to serve".into());
        };
        // Content addressing is the whole point of the hash: a master and a
        // follower silently running different weights would show up as
        // nonsense output, not as an error.
        match hash_file(path) {
            Ok(hash) if hash == spec.model_hash => {}
            Ok(hash) => {
                return Response::Refused(format!(
                    "this device holds {}, not {}",
                    &hash[..16],
                    &spec.model_hash[..16.min(spec.model_hash.len())]
                ));
            }
            Err(e) => return Response::Refused(format!("could not hash the model: {e}")),
        }

        let (reply, answer) = mpsc::channel();
        let job = Job::Load {
            path: path.clone(),
            spec,
            reply,
        };
        if self.shard.jobs.send(job).is_err() {
            return Response::Refused("the shard thread is gone".into());
        }
        answer
            .recv()
            .unwrap_or_else(|_| Response::Refused("the shard thread died loading".into()))
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

impl ControlHandler for Node {
    fn handle(&self, from: DeviceId, request: Request) -> Response {
        match request {
            Request::Hello => Response::Identity {
                device_id: self.identity.0,
                name: self.identity.1.clone(),
                platform: self.identity.2.clone(),
            },
            Request::Provision(peers) => self.accept_introductions(from, peers),
            Request::LoadShard(spec) => self.load_shard(spec),
        }
    }
}

#[async_trait::async_trait]
impl Executor for Node {
    async fn run(&self, session: u64, step: Step) -> Result<Payload, String> {
        let (reply, answer) = tokio::sync::oneshot::channel();
        let job = Job::Run { session, step, reply };
        self.shard
            .jobs
            .send(job)
            .map_err(|_| "the shard thread is gone".to_string())?;
        answer.await.map_err(|_| "the shard thread died mid-step".to_string())?
    }

    fn drop_session(&self, session: u64) {
        let _ = self.shard.jobs.send(Job::Drop { session });
    }
}

/// What each device needs to hear: everyone the master trusts, except itself.
pub fn roster_for(target: DeviceId, all: &[PairedPeer]) -> Vec<Introduction> {
    all.iter()
        .filter(|peer| peer.device_id != target)
        .map(Introduction::of)
        .collect()
}
