use super::*;
use lattice_net::{AcceptAnyPeer, ControlHandler, PairingCode, Request, Response, control, probe};
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::mpsc;

struct Joined {
    conn: Arc<Connection>,
    port: u16,
}

struct MasterHandler {
    node: Arc<Node>,
    conn: Arc<Connection>,
    joined: mpsc::Sender<Joined>,
}

impl ControlHandler for MasterHandler {
    fn handle(&self, from: DeviceId, request: Request) -> Response {
        match request {
            Request::RegisterWorker { port } if port != 0 => {
                match self.joined.try_send(Joined {
                    conn: self.conn.clone(),
                    port,
                }) {
                    Ok(()) => Response::WorkerRegistered,
                    Err(_) => Response::Refused(MASTER_BUSY.into()),
                }
            }
            Request::RegisterWorker { .. } => {
                Response::Refused("worker port must be nonzero".into())
            }
            request => self.node.handle(from, request),
        }
    }
}

pub async fn run(
    dir: &Path,
    key: &DeviceKey,
    facts: &HostFacts,
    options: MasterOptions,
) -> Result<()> {
    if options.common.port == options.pairing_port {
        bail!("--port and --pairing-port must differ");
    }
    let pairing = Endpoint::bind(
        bind_addr(options.pairing_port),
        key,
        Arc::new(AcceptAnyPeer),
    )
    .with_context(|| format!("binding pairing UDP port {}", options.pairing_port))?;
    let service = Arc::new(Service::start(
        dir,
        key,
        facts,
        &options.common,
        Some(options.pairing_port),
    )?);
    let (joined, events) = mpsc::channel(32);
    let probe_endpoint = Arc::new(Endpoint::bind(
        bind_addr(0),
        key,
        Arc::new(TrustedPeers(service.store.clone())),
    )?);
    tokio::spawn(serve(service.clone(), joined));
    tokio::spawn(workers(service.clone(), probe_endpoint, events));
    println!("On each other machine, run: latticed worker");
    println!("Keep this terminal open. Ctrl-C stops the master.");
    loop {
        let code = PairingCode::generate();
        println!("pairing code: {code}");
        let conn = loop {
            match tokio::time::timeout(REQUEST_TIMEOUT, pairing.accept()).await {
                Ok(Some(Ok(conn))) => break conn,
                Ok(Some(Err(error))) => eprintln!("pairing connection refused: {error}"),
                Ok(None) => return Ok(()),
                Err(_) => {}
            }
        };
        match crate::pairing_session::exchange(&conn, key, facts, &service.store, &code).await {
            Ok(id) => println!("paired with {id}; ready to connect"),
            Err(error) => eprintln!("pairing failed: {error:#}; use the new code below"),
        }
    }
}

async fn serve(service: Arc<Service>, joined: mpsc::Sender<Joined>) {
    loop {
        match tokio::time::timeout(REQUEST_TIMEOUT, service.endpoint.accept()).await {
            Ok(Some(Ok(conn))) => {
                let conn = Arc::new(conn);
                let handler = Arc::new(MasterHandler {
                    node: service.node.clone(),
                    conn: conn.clone(),
                    joined: joined.clone(),
                });
                let node = service.node.clone();
                tokio::spawn(async move {
                    let _ = dispatch::serve(conn, handler, node).await;
                });
            }
            Ok(Some(Err(error))) => eprintln!("refused a connection: {error}"),
            Ok(None) => return,
            Err(_) => {}
        }
    }
}

async fn workers(
    service: Arc<Service>,
    probe_endpoint: Arc<Endpoint>,
    mut events: mpsc::Receiver<Joined>,
) {
    let live = Arc::new(Mutex::new(HashMap::<DeviceId, Arc<Connection>>::new()));
    while let Some(joined) = events.recv().await {
        let id = joined.conn.peer_id();
        live.lock()
            .expect("worker registry poisoned")
            .insert(id, joined.conn.clone());
        println!(
            "worker {id} connected from {}",
            joined.conn.remote_address()
        );
        let peers: Vec<_> = service
            .store
            .read()
            .expect("trust store poisoned")
            .peers()
            .cloned()
            .collect();
        let connections: Vec<_> = live
            .lock()
            .expect("worker registry poisoned")
            .values()
            .cloned()
            .collect();
        for conn in connections {
            let roster = crate::node::roster_for(conn.peer_id(), &peers);
            tokio::spawn(async move {
                match tokio::time::timeout(
                    REQUEST_TIMEOUT,
                    control::request(&conn, &Request::Provision(roster)),
                )
                .await
                {
                    Ok(Ok(Response::Provisioned { .. })) => {}
                    other => eprintln!("worker {} introductions failed: {other:?}", conn.peer_id()),
                }
            });
        }
        let endpoint = probe_endpoint.clone();
        let conn = joined.conn;
        let live = live.clone();
        tokio::spawn(async move {
            let mut addr = conn.remote_address();
            addr.set_port(joined.port);
            let tested = tokio::time::timeout(REQUEST_TIMEOUT, async {
                let incoming = endpoint.connect(addr).await?;
                if incoming.peer_id() != id {
                    bail!("worker identity changed");
                }
                Ok::<_, anyhow::Error>(probe::measure(&incoming, 1 << 20).await?)
            })
            .await;
            match tested {
                Ok(Ok(link)) => println!(
                    "worker {id} verified both ways: RTT {:.2} ms, {:.1} MB/s",
                    link.rtt.as_secs_f64() * 1e3,
                    link.throughput_bytes_per_sec / 1e6
                ),
                other => eprintln!(
                    "worker {id} connected, but its listener at {addr} could not be verified: {other:?}; check inbound UDP {}",
                    joined.port
                ),
            }
            conn.inner.closed().await;
            let mut registry = live.lock().expect("worker registry poisoned");
            if registry
                .get(&id)
                .is_some_and(|current| Arc::ptr_eq(current, &conn))
            {
                registry.remove(&id);
                println!("worker {id} disconnected; waiting for reconnection");
            }
        });
    }
}
