mod plan;

use anyhow::{Context, Result, bail};
use clap::Args;
use lattice_engine::ModelInventory;
use lattice_net::{
    Connection, DeviceId, DeviceKey, Discovery, Endpoint, EngineCapabilities, PeerEvent, Request,
    Response, RpcProxy, TrustStore, TrustedPeers, control,
};
use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::runtime::{Engine, select_device};

#[derive(Args)]
pub struct Options {
    #[arg(
        long,
        help = "GGUF on the master; assigned weights transfer automatically"
    )]
    model: PathBuf,
    #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(128..=65536))]
    context: u32,
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(1..=65536))]
    tokens: u32,
    #[arg(
        long,
        help = "Generate one response and exit; omit for interactive chat"
    )]
    prompt: Option<String>,
    #[arg(long, help = "Worker device ID or IPv4:port; repeat to select several")]
    worker: Vec<String>,
    #[arg(
        long,
        conflicts_with = "worker",
        help = "Explicitly run on the local GPU alone"
    )]
    local: bool,
    #[arg(long, help = "Local GPU name from latticed devices")]
    device: Option<String>,
    #[arg(long)]
    engine_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(256..), help = "Memory to leave free on each selected GPU, in MiB")]
    reserve_mib: u32,
}

struct Worker {
    id: DeviceId,
    name: String,
    conn: Arc<Connection>,
    caps: EngineCapabilities,
    _endpoint: Endpoint,
}

pub async fn run(dir: &Path, key: &DeviceKey, options: Options) -> Result<()> {
    let engine = Engine::locate(options.engine_dir.as_deref())?;
    let model = options
        .model
        .canonicalize()
        .context("opening the master's GGUF")?;
    let inventory = ModelInventory::read_file(&model)?;
    let facts = &inventory.facts;
    if options.context > facts.train_context {
        bail!(
            "--context {} exceeds the model's trained context {}",
            options.context,
            facts.train_context
        );
    }
    eprintln!("Detecting GPUs; first initialization can take a minute.");
    let local = select_device(&engine.devices()?, options.device.as_deref())?;
    let workers = if options.local {
        Vec::new()
    } else {
        connect_workers(dir, key, &options.worker, engine.revision()).await?
    };
    let devices: Vec<_> = workers
        .iter()
        .map(|worker| worker.caps.device.clone())
        .chain(std::iter::once(local))
        .collect();
    let plan = plan::build(
        &inventory,
        options.context,
        &devices,
        u64::from(options.reserve_mib) << 20,
    )
    .with_context(|| {
        let memory = devices
            .iter()
            .map(|device| {
                format!(
                    "{}: {:.0} MiB available",
                    device.description,
                    device.free_memory as f64 / 1048576.0
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        format!(
            "GPU placement failed with a reserve of at least {} MiB per device ({memory})",
            options.reserve_mib
        )
    })?;
    eprintln!(
        "Using {} GPU(s); {} layers including output, context {}.",
        devices.len(),
        facts.layers + 1,
        options.context
    );
    for (index, (device, layers)) in devices.iter().zip(&plan.layers).enumerate() {
        let name = workers
            .get(index)
            .map_or("this machine", |worker| worker.name.as_str());
        eprintln!(
            "  {name}: {} — {layers} layer slots, {:.1} GiB free",
            device.description,
            device.free_memory as f64 / 1073741824.0
        );
    }
    let mut proxies = Vec::new();
    for worker in &workers {
        proxies.push(RpcProxy::bind(worker.conn.clone()).await?);
    }
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let log_dir = dir.join("logs");
    std::fs::create_dir_all(&log_dir)?;
    let log_path = log_dir.join(format!("chat-{stamp}-{}.log", std::process::id()));
    let log = std::fs::File::create(&log_path)?;
    eprintln!("Engine log: {}", log_path.display());
    if !workers.is_empty() {
        eprintln!(
            "Loading and transferring assigned weights. First load can take several minutes."
        );
    }
    let rpc = proxies
        .iter()
        .map(|proxy| proxy.address().to_string())
        .collect::<Vec<_>>()
        .join(",");
    let names = (0..workers.len())
        .map(|index| format!("RPC{index}"))
        .chain(std::iter::once(devices.last().unwrap().name.clone()))
        .collect::<Vec<_>>()
        .join(",");
    let split = plan
        .layers
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut command = tokio::process::Command::new(engine.cli());
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("LLAMA_ARG_") {
            command.env_remove(name);
        }
    }
    if !rpc.is_empty() {
        command.args(["--rpc", &rpc]);
    }
    command
        .arg("--model")
        .arg(&model)
        .args([
            "--device",
            &names,
            "--split-mode",
            "layer",
            "--gpu-layers",
            "all",
            "--tensor-split",
            &split,
            "--fit",
            "off",
            "--simple-io",
            "--log-verbosity",
            "4",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
        ])
        .arg("--ctx-size")
        .arg(options.context.to_string())
        .arg("--predict")
        .arg(options.tokens.to_string())
        .env("GGML_RPC_NO_RDMA", "1")
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::from(log))
        .kill_on_drop(true);
    if let Some(prompt) = &options.prompt {
        command.args(["--prompt", prompt, "--single-turn", "--temp", "0"]);
    }
    let mut child = command.spawn().context("starting GPU inference")?;
    let mut disconnects = tokio::task::JoinSet::new();
    for worker in &workers {
        let conn = worker.conn.clone();
        let id = worker.id;
        disconnects.spawn(async move { (id, conn.inner.closed().await) });
    }
    let outcome = tokio::select! {
        status = child.wait() => status.context("waiting for the inference engine"),
        failure = disconnects.join_next(), if !workers.is_empty() => {
            let _ = child.kill().await;
            Err(anyhow::anyhow!("worker connection closed during inference: {failure:?}"))
        }
        signal = shutdown_signal() => {
            signal?;
            let _ = child.kill().await;
            eprintln!("Chat stopped.");
            disconnects.abort_all();
            return Ok(());
        }
    };
    disconnects.abort_all();
    for worker in &workers {
        worker.conn.inner.close(0u32.into(), b"chat finished");
    }
    let status = outcome.with_context(|| format!("see {}", log_path.display()))?;
    if !status.success() {
        let tail = log_tail(&log_path)?;
        bail!(
            "inference engine exited with {status}\n{tail}\nFull log: {}",
            log_path.display()
        );
    }
    Ok(())
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => { result?; }
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

async fn connect_workers(
    dir: &Path,
    key: &DeviceKey,
    requested: &[String],
    revision: &str,
) -> Result<Vec<Worker>> {
    let store = Arc::new(RwLock::new(TrustStore::load(dir)?));
    let trusted: BTreeSet<_> = store
        .read()
        .expect("trust store poisoned")
        .peers()
        .map(|peer| peer.device_id)
        .collect();
    let targets = requested.to_vec();
    let self_id = key.id();
    let located =
        tokio::task::spawn_blocking(move || find_workers(&targets, &trusted, self_id)).await??;
    if located.is_empty() {
        bail!(
            "no paired workers found; start latticed worker --require-gpu, use --worker <IPv4:port>, or choose --local"
        );
    }
    let mut workers = Vec::new();
    let mut seen = BTreeSet::new();
    for (expected, addrs) in located {
        let (endpoint, conn) =
            crate::connect_any(key, Arc::new(TrustedPeers(store.clone())), &addrs).await?;
        let id = conn.peer_id();
        if expected.is_some_and(|wanted| wanted != id) || id == key.id() {
            bail!("worker identity differs from the selected device");
        }
        if !seen.insert(id) {
            bail!("worker {id} was selected more than once");
        }
        let reply = tokio::time::timeout(
            Duration::from_secs(120),
            control::request(&conn, &Request::InferenceInfo),
        )
        .await
        .context("worker capability request timed out")??;
        let caps = match reply {
            Response::InferenceInfo(caps) => caps,
            Response::Refused(why) if why.starts_with("unintelligible request:") => {
                bail!(
                    "worker {id} is running an older latticed; run ./scripts/setup.sh --inference there and restart its worker service"
                );
            }
            Response::Refused(why) => bail!("worker {id}: {why}"),
            other => {
                bail!("worker {id} does not support inference: {other:?}; update its installation")
            }
        };
        check_worker(&caps, revision).with_context(|| format!("worker {id}"))?;
        let name = store
            .read()
            .expect("trust store poisoned")
            .get(id)
            .map_or_else(|| id.to_string(), |peer| peer.name.clone());
        workers.push(Worker {
            id,
            name,
            conn: Arc::new(conn),
            caps,
            _endpoint: endpoint,
        });
    }
    Ok(workers)
}

fn check_worker(caps: &EngineCapabilities, revision: &str) -> Result<()> {
    if caps.revision != revision {
        bail!("engine versions differ; run scripts/setup-engine.sh on both machines");
    }
    if caps.busy {
        bail!("GPU is busy with another chat; wait for it to finish");
    }
    if !matches!(caps.device.kind.as_str(), "gpu" | "igpu") {
        bail!("worker selected a non-GPU device");
    }
    Ok(())
}

type Located = Vec<(Option<DeviceId>, Vec<SocketAddr>)>;

fn find_workers(
    requested: &[String],
    trusted: &BTreeSet<DeviceId>,
    self_id: DeviceId,
) -> Result<Located> {
    let mut explicit = Vec::new();
    let mut wanted = BTreeSet::new();
    for target in requested {
        if let Ok(addr) = target.parse::<SocketAddr>() {
            if !addr.is_ipv4() || addr.port() == 0 || addr.ip().is_unspecified() {
                bail!("--worker requires a usable IPv4:port");
            }
            explicit.push((None, vec![addr]));
        } else {
            let id: DeviceId = target
                .parse()
                .context("--worker expects a device ID or IPv4:port")?;
            if id == self_id || !trusted.contains(&id) {
                bail!("{id} is not a paired worker");
            }
            wanted.insert(id);
        }
    }
    if wanted.is_empty() && !explicit.is_empty() {
        return Ok(explicit);
    }
    let discovery = Discovery::new()?;
    let browser = discovery.browse()?;
    let deadline = std::time::Instant::now() + Duration::from_secs(4);
    let mut found = BTreeMap::new();
    while let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) {
        match browser.next_event(remaining) {
            Some(PeerEvent::Found(peer))
                if peer.device_id != self_id
                    && trusted.contains(&peer.device_id)
                    && peer.role.as_deref() == Some("worker")
                    && peer.protocol == lattice_net::discovery::PROTOCOL_VERSION
                    && (requested.is_empty() || wanted.contains(&peer.device_id)) =>
            {
                let addrs = crate::usable(peer.addrs);
                if !addrs.is_empty() {
                    found.insert(peer.device_id, addrs);
                }
            }
            Some(PeerEvent::Lost(id)) => {
                found.remove(&id);
            }
            Some(_) => {}
            None => break,
        }
    }
    for id in wanted {
        if !found.contains_key(&id) {
            bail!("worker {id} is unavailable; start it or supply --worker <IPv4:port>");
        }
    }
    explicit.extend(found.into_iter().map(|(id, addrs)| (Some(id), addrs)));
    Ok(explicit)
}

fn log_tail(path: &Path) -> Result<String> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let size = file.metadata()?.len();
    file.seek(SeekFrom::Start(size.saturating_sub(8192)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_net::GpuDevice;

    #[test]
    fn incompatible_busy_and_cpu_workers_are_refused() {
        let mut caps = EngineCapabilities {
            revision: "same".into(),
            busy: false,
            device: GpuDevice {
                name: "Vulkan0".into(),
                description: "RX 7600".into(),
                kind: "gpu".into(),
                total_memory: 8 << 30,
                free_memory: 7 << 30,
            },
        };
        assert!(check_worker(&caps, "same").is_ok());
        assert!(check_worker(&caps, "different").is_err());
        caps.busy = true;
        assert!(check_worker(&caps, "same").is_err());
        caps.busy = false;
        caps.device.kind = "cpu".into();
        assert!(check_worker(&caps, "same").is_err());
    }
}
