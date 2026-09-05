mod master;
mod worker;

pub use master::run as master;
pub use worker::run as worker;

use anyhow::{Context, Result, bail};
use clap::{Args, ValueEnum};
use lattice_net::{
    Advertisement, Connection, DeviceId, DeviceKey, Discovery, Endpoint, TrustStore, TrustedPeers,
    dispatch,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::{bind_addr, host::HostFacts, node::Node};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const MASTER_BUSY: &str = "master is busy; retry shortly";

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Firewall {
    Auto,
    Print,
    Off,
}

#[derive(Args)]
pub struct CommonOptions {
    #[arg(long, default_value_t = 47900, value_parser = clap::value_parser!(u16).range(1..), help = "Fixed UDP serving port")]
    port: u16,
    #[arg(long, help = "GGUF model to offer to paired devices")]
    model: Option<PathBuf>,
    #[arg(
        long,
        value_enum,
        default_value = "auto",
        help = "Configure an active Linux firewall, print rules, or skip"
    )]
    firewall: Firewall,
}

#[derive(Args)]
pub struct MasterOptions {
    #[command(flatten)]
    common: CommonOptions,
    #[arg(long, default_value_t = 47901, value_parser = clap::value_parser!(u16).range(1..), help = "Fixed UDP pairing port")]
    pairing_port: u16,
}

#[derive(Args)]
pub struct WorkerOptions {
    #[arg(long, help = "Inference engine installation directory")]
    engine_dir: Option<PathBuf>,
    #[arg(long, help = "Require a working GPU inference engine")]
    require_gpu: bool,
    #[arg(long, help = "Exact GPU name reported by lattice-engine-info")]
    device: Option<String>,
    #[command(flatten)]
    common: CommonOptions,
    #[arg(
        long,
        help = "Master device ID or IPv4[:port]; otherwise discover or reconnect to the saved master"
    )]
    master: Option<String>,
    #[arg(
        long,
        help = "First-time six-digit pairing code; otherwise ask interactively"
    )]
    code: Option<String>,
    #[arg(long, default_value_t = 47901, value_parser = clap::value_parser!(u16).range(1..), help = "Pairing port when connecting by explicit IPv4")]
    pairing_port: u16,
}

struct Service {
    endpoint: Arc<Endpoint>,
    store: Arc<RwLock<TrustStore>>,
    node: Arc<Node>,
    _discovery: Discovery,
}

impl Service {
    fn start(
        dir: &Path,
        key: &DeviceKey,
        facts: &HostFacts,
        options: &CommonOptions,
        pairing_port: Option<u16>,
    ) -> Result<Self> {
        if let Some(model) = &options.model {
            lattice_engine::ModelFacts::read_file(model)
                .with_context(|| format!("reading model {}", model.display()))?;
        }
        let store = Arc::new(RwLock::new(TrustStore::load(dir)?));
        let endpoint = Arc::new(Endpoint::bind(bind_addr(options.port), key, Arc::new(TrustedPeers(store.clone())))
            .with_context(|| format!("UDP port {} is unavailable; stop the existing latticed process or choose --port", options.port))?);
        configure_firewall(options, pairing_port)?;
        let role = if pairing_port.is_some() {
            "master"
        } else {
            "worker"
        };
        let mut discovery = Discovery::new()?;
        discovery.advertise_role(
            &Advertisement {
                device_id: key.id(),
                name: facts.name.clone(),
                platform: facts.platform.clone(),
                total_memory: facts.total_memory,
                port: options.port,
            },
            role,
            pairing_port,
        )?;
        println!(
            "{} {role} serving as {} on UDP port {}",
            facts.name,
            key.id(),
            options.port
        );
        let node = Arc::new(Node::new(
            store.clone(),
            key.id(),
            facts,
            options.model.clone(),
        ));
        Ok(Self {
            endpoint,
            store,
            node,
            _discovery: discovery,
        })
    }

    async fn connect(
        &self,
        addrs: &[std::net::SocketAddr],
        expected: Option<DeviceId>,
    ) -> Result<Arc<Connection>> {
        let mut failures = Vec::new();
        for addr in addrs {
            match self.endpoint.connect(*addr).await {
                Ok(conn) if expected.is_none_or(|id| conn.peer_id() == id) => {
                    return Ok(Arc::new(conn));
                }
                Ok(_) => {
                    failures.push(format!("{addr}: identity differs from the selected master"))
                }
                Err(error) => failures.push(format!("{addr}: {error}")),
            }
        }
        bail!("could not connect:\n{}", failures.join("\n"))
    }
}

fn configure_firewall(options: &CommonOptions, pairing_port: Option<u16>) -> Result<()> {
    let mode = match options.firewall {
        Firewall::Auto => "auto",
        Firewall::Print => "print",
        Firewall::Off => "off",
    };
    let mut command = std::process::Command::new("bash");
    command.args([
        "-c",
        include_str!("../../../scripts/firewall.sh"),
        "lattice-firewall",
        mode,
    ]);
    command.arg(options.port.to_string());
    if let Some(port) = pairing_port {
        command.arg(port.to_string());
    }
    if !command
        .status()
        .context("running firewall setup")?
        .success()
    {
        bail!(
            "firewall setup failed; fix the reported problem or use --firewall print for manual setup"
        );
    }
    Ok(())
}

async fn serve_worker(service: Arc<Service>) {
    loop {
        match tokio::time::timeout(REQUEST_TIMEOUT, service.endpoint.accept()).await {
            Ok(Some(Ok(conn))) => {
                let node = service.node.clone();
                tokio::spawn(async move {
                    let _ =
                        dispatch::serve_with_rpc(Arc::new(conn), node.clone(), node.clone(), node)
                            .await;
                });
            }
            Ok(Some(Err(error))) => eprintln!("refused a connection: {error}"),
            Ok(None) => return,
            Err(_) => {}
        }
    }
}
