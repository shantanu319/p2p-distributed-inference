use super::*;
use lattice_net::{
    AcceptAnyPeer, DiscoveredPeer, PairingCode, PeerEvent, Request, Response, control,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Target {
    id: Option<DeviceId>,
    addrs: Vec<SocketAddr>,
    pairing_port: u16,
}

pub async fn run(
    dir: &Path,
    key: &DeviceKey,
    facts: &HostFacts,
    options: WorkerOptions,
) -> Result<()> {
    tokio::select! {
        result = run_inner(dir, key, facts, options) => result,
        result = shutdown_signal() => { result?; Ok(()) },
    }
}

async fn shutdown_signal() -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result,
            _ = terminate.recv() => Ok(()),
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await
}

struct RuntimeGuard(Arc<Node>);

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        self.0.authorize_master(None);
    }
}

async fn run_inner(
    dir: &Path,
    key: &DeviceKey,
    facts: &HostFacts,
    options: WorkerOptions,
) -> Result<()> {
    let mut supplied_code = options
        .code
        .as_deref()
        .map(PairingCode::parse)
        .transpose()?;
    let mut saved = read_saved(dir)?;
    let service = Arc::new(Service::start(dir, key, facts, &options.common, None)?);
    let runtime =
        crate::runtime::Engine::locate(options.engine_dir.as_deref()).and_then(|engine| {
            println!("Detecting GPUs; first initialization can take a minute.");
            let device =
                crate::runtime::select_device(&engine.devices()?, options.device.as_deref())?;
            println!(
                "GPU inference enabled on {} ({})",
                device.name, device.description
            );
            crate::runtime::WorkerRuntime::new(engine, device, dir)
        });
    match runtime {
        Ok(runtime) => service.node.set_runtime(Arc::new(runtime)),
        Err(error)
            if options.require_gpu || options.engine_dir.is_some() || options.device.is_some() =>
        {
            return Err(error);
        }
        Err(error) => eprintln!(
            "Network-only worker: {error:#}. Run ./scripts/setup-engine.sh, then restart with --require-gpu."
        ),
    }
    let _runtime_guard = RuntimeGuard(service.node.clone());
    tokio::spawn(serve_worker(service.clone()));
    println!("Looking for a master. Keep this terminal open; Ctrl-C stops the worker.");
    let mut last_error = String::new();
    loop {
        let mut target = match find_target(&options, saved.as_ref(), key.id()).await? {
            Some(target) => target,
            None => {
                report_once(
                    &mut last_error,
                    "No master found; retrying. Start latticed master, or use --master <IPv4> if multicast is blocked.",
                );
                tokio::time::sleep(Duration::from_secs(3)).await;
                continue;
            }
        };
        let trusted = target.id.is_some_and(|id| {
            service
                .store
                .read()
                .expect("trust store poisoned")
                .get(id)
                .is_some()
        });
        let connection = if supplied_code.is_none() {
            service.connect(&target.addrs, target.id).await
        } else {
            Err(anyhow::anyhow!("pairing requested"))
        };
        let conn = match connection {
            Ok(conn) => conn,
            Err(error) if trusted && supplied_code.is_none() => {
                report_once(
                    &mut last_error,
                    &format!(
                        "Master unavailable: {error:#}. Retrying; existing pairing is preserved."
                    ),
                );
                tokio::time::sleep(Duration::from_secs(3)).await;
                continue;
            }
            Err(_) => {
                match pair(&service, &mut target, key, facts, supplied_code.take()).await {
                    Ok(()) => {}
                    Err(error) => bail!(
                        "could not pair with the master: {error:#}; check its current code and rerun latticed worker"
                    ),
                }
                saved = Some(target.clone());
                match service.connect(&target.addrs, target.id).await {
                    Ok(conn) => conn,
                    Err(error) => {
                        report_once(
                            &mut last_error,
                            &format!("Pairing saved; waiting for master: {error:#}"),
                        );
                        tokio::time::sleep(Duration::from_secs(3)).await;
                        continue;
                    }
                }
            }
        };
        let node = service.node.clone();
        let serving_conn = conn.clone();
        let serving = tokio::spawn(async move {
            dispatch::serve_with_rpc(serving_conn, node.clone(), node.clone(), node).await
        });
        let registration = tokio::time::timeout(
            REQUEST_TIMEOUT,
            control::request(
                &conn,
                &Request::RegisterWorker {
                    port: options.common.port,
                },
            ),
        )
        .await;
        let retry = match registration {
            Ok(Ok(Response::WorkerRegistered)) => None,
            Ok(Ok(Response::Refused(why))) if why == MASTER_BUSY => Some(why),
            Ok(Err(error)) => Some(format!("Master disconnected during registration: {error}")),
            Err(_) => Some("Master registration timed out".into()),
            other => {
                serving.abort();
                conn.inner.close(0u32.into(), b"registration failed");
                bail!(
                    "master did not accept worker registration: {other:?}; run the updated latticed master on that device"
                );
            }
        };
        if let Some(reason) = retry {
            serving.abort();
            conn.inner.close(0u32.into(), b"retrying registration");
            saved = Some(Target {
                id: Some(conn.peer_id()),
                ..target
            });
            report_once(
                &mut last_error,
                &format!("{reason}; retrying automatically"),
            );
            tokio::time::sleep(Duration::from_secs(3)).await;
            continue;
        }
        service.node.authorize_master(Some(conn.peer_id()));
        target.id = Some(conn.peer_id());
        write_saved(dir, &target)?;
        saved = Some(target);
        println!(
            "Connected to master {} at {}; pairing saved",
            conn.peer_id(),
            conn.remote_address()
        );
        last_error.clear();
        conn.inner.closed().await;
        serving.abort();
        service.node.authorize_master(None);
        println!("Master disconnected; reconnecting automatically");
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

async fn pair(
    service: &Service,
    target: &mut Target,
    key: &DeviceKey,
    facts: &HostFacts,
    code: Option<PairingCode>,
) -> Result<()> {
    println!("Requesting pairing with master at {}", target.addrs[0]);
    let code = match code {
        Some(code) => code,
        None => {
            print!("Enter the six-digit code shown by the master: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            PairingCode::parse(&input)
                .context("enter the master's code, or pass --code for a noninteractive start")?
        }
    };
    let addrs: Vec<_> = target
        .addrs
        .iter()
        .map(|addr| SocketAddr::new(addr.ip(), target.pairing_port))
        .collect();
    let (_endpoint, conn) = crate::connect_any(key, Arc::new(AcceptAnyPeer), &addrs).await?;
    if target.id.is_some_and(|id| conn.peer_id() != id) {
        bail!("pairing endpoint identity differs from the selected master");
    }
    let id = crate::pairing_session::exchange(&conn, key, facts, &service.store, &code).await?;
    target.id = Some(id);
    Ok(())
}

fn report_once(previous: &mut String, message: &str) {
    if previous != message {
        eprintln!("{message}");
        *previous = message.to_owned();
    }
}

async fn find_target(
    options: &WorkerOptions,
    saved: Option<&Target>,
    self_id: DeviceId,
) -> Result<Option<Target>> {
    let requested_id = match options.master.as_deref() {
        Some(text) => match explicit_address(text)? {
            Some(addr) => {
                let id = saved
                    .filter(|target| target.addrs.contains(&addr))
                    .and_then(|target| target.id);
                return Ok(Some(Target {
                    id,
                    addrs: vec![addr],
                    pairing_port: options.pairing_port,
                }));
            }
            None => Some(
                text.parse::<DeviceId>()
                    .context("--master must be a device ID or IPv4[:port]")?,
            ),
        },
        None => saved.and_then(|target| target.id),
    };
    if requested_id == Some(self_id) {
        bail!("the selected master is this device");
    }
    let peers = tokio::task::spawn_blocking(move || browse(self_id)).await??;
    let found = select_master(peers, requested_id)?;
    if let Some(peer) = found {
        return Ok(Some(Target {
            id: Some(peer.device_id),
            addrs: crate::usable(peer.addrs),
            pairing_port: peer.pairing_port.unwrap_or(options.pairing_port),
        }));
    }
    Ok(saved.filter(|target| target.id == requested_id).cloned())
}

fn explicit_address(text: &str) -> Result<Option<SocketAddr>> {
    let addr = text.parse::<SocketAddr>().ok().or_else(|| {
        text.parse::<Ipv4Addr>()
            .ok()
            .map(|ip| SocketAddr::from((ip, 47900)))
    });
    if addr.is_some_and(|addr| !addr.is_ipv4() || addr.port() == 0 || addr.ip().is_unspecified()) {
        bail!("--master requires a usable IPv4 address and nonzero port");
    }
    Ok(addr)
}

fn browse(self_id: DeviceId) -> Result<Vec<DiscoveredPeer>> {
    let discovery = Discovery::new()?;
    let browser = discovery.browse()?;
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    let mut peers = BTreeMap::new();
    while let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) {
        match browser.next_event(remaining) {
            Some(PeerEvent::Found(peer)) if peer.device_id != self_id => {
                peers.insert(peer.device_id, peer);
            }
            Some(PeerEvent::Lost(id)) => {
                peers.remove(&id);
            }
            Some(_) => {}
            None => break,
        }
    }
    Ok(peers.into_values().collect())
}

fn select_master(
    peers: Vec<DiscoveredPeer>,
    wanted: Option<DeviceId>,
) -> Result<Option<DiscoveredPeer>> {
    let mut candidates = peers.into_iter().filter(|peer| {
        peer.role.as_deref() == Some("master")
            && peer.protocol == lattice_net::discovery::PROTOCOL_VERSION
            && wanted.is_none_or(|id| peer.device_id == id)
            && peer.addrs.iter().any(SocketAddr::is_ipv4)
    });
    let first = candidates.next();
    if let Some(second) = candidates.next() {
        bail!(
            "multiple masters found ({} and {}); choose one with --master <device-id>",
            first.as_ref().unwrap().device_id,
            second.device_id
        );
    }
    Ok(first)
}

fn read_saved(dir: &Path) -> Result<Option<Target>> {
    match std::fs::read(dir.join("worker-master.json")) {
        Ok(bytes) => {
            let target: Target = serde_json::from_slice(&bytes).context("reading saved master")?;
            if target.id.is_none()
                || target.addrs.is_empty()
                || target.pairing_port == 0
                || target
                    .addrs
                    .iter()
                    .any(|addr| !addr.is_ipv4() || addr.port() == 0)
            {
                bail!(
                    "saved master is invalid; remove {} and reconnect",
                    dir.join("worker-master.json").display()
                );
            }
            Ok(Some(target))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn write_saved(dir: &Path, target: &Target) -> Result<()> {
    let pending = dir.join("worker-master.json.tmp");
    std::fs::write(&pending, serde_json::to_vec_pretty(target)?)?;
    std::fs::rename(pending, dir.join("worker-master.json"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(id: &str, role: Option<&str>) -> DiscoveredPeer {
        DiscoveredPeer {
            device_id: id.parse().unwrap(),
            name: "computer".into(),
            platform: "linux-x86_64".into(),
            total_memory: 0,
            protocol: lattice_net::discovery::PROTOCOL_VERSION,
            addrs: vec!["192.168.7.40:47900".parse().unwrap()],
            role: role.map(str::to_owned),
            pairing_port: Some(47901),
        }
    }

    #[test]
    fn discovery_requires_a_master_and_never_silently_switches_identity() {
        let a = peer("1111111111111111", Some("master"));
        let b = peer("2222222222222222", Some("master"));
        assert!(select_master(vec![a.clone(), b.clone()], None).is_err());
        assert_eq!(
            select_master(vec![a.clone(), b.clone()], Some(a.device_id))
                .unwrap()
                .unwrap()
                .device_id,
            a.device_id
        );
        assert!(select_master(vec![b], Some(a.device_id)).unwrap().is_none());
        assert!(
            select_master(
                vec![
                    peer("3333333333333333", Some("worker")),
                    peer("4444444444444444", None)
                ],
                None
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn explicit_master_uses_fixed_ports_and_ipv4() {
        assert_eq!(
            explicit_address("192.168.7.40").unwrap().unwrap().port(),
            47900
        );
        assert_eq!(
            explicit_address("192.168.7.40:4242")
                .unwrap()
                .unwrap()
                .port(),
            4242
        );
        assert!(explicit_address("192.168.7.40:0").is_err());
        assert!(explicit_address("[::1]:47900").is_err());
        assert!(explicit_address("1111111111111111").unwrap().is_none());
    }

    #[test]
    fn remembered_master_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_saved(dir.path()).unwrap().is_none());
        let target = Target {
            id: Some("1111111111111111".parse().unwrap()),
            addrs: vec!["192.168.7.40:4242".parse().unwrap()],
            pairing_port: 4243,
        };
        write_saved(dir.path(), &target).unwrap();
        let saved = read_saved(dir.path()).unwrap().unwrap();
        assert_eq!(saved.id, target.id);
        assert_eq!(saved.addrs, target.addrs);
        assert_eq!(saved.pairing_port, target.pairing_port);
    }
}
