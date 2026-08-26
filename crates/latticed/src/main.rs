//! `latticed` — the Lattice daemon and its CLI.
//!
//! Devices find each other, pair, and measure the link (§7); `generate` runs a
//! model on this device alone. Nothing yet runs one across two devices.

mod generate;
mod host;
mod provision;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use lattice_net::discovery::{Advertisement, Discovery, PeerEvent};
use lattice_net::identity::default_data_dir;
use lattice_net::{
    AcceptAnyPeer, DeviceId, DeviceKey, Endpoint, Mesh, PairedPeer, PairingCode, Request, Response,
    TrustStore, TrustedPeers, control, dispatch, pairing, probe,
};
use std::collections::{BTreeMap, HashMap};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

#[derive(Parser)]
#[command(name = "latticed", about = "Pool the machines you already own")]
struct Cli {
    /// Where the device key and trusted-devices file live.
    #[arg(long, global = true)]
    data_dir: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Print this device's identity.
    Id,
    /// List Lattice devices on the network. Discovery implies no trust.
    Discover {
        #[arg(long, default_value_t = 5)]
        seconds: u64,
    },
    /// Wait for another device to pair with this one, showing a code.
    Host {
        #[arg(long, default_value_t = 0)]
        port: u16,
    },
    /// Pair with a device that is showing a code.
    Pair {
        /// A `host:port`, or the device id of something `discover` found.
        target: String,
        #[arg(long)]
        code: String,
    },
    /// List paired devices.
    Peers,
    /// Introduce every paired device to the others, so they can talk directly.
    Provision,
    /// Forget a paired device.
    Unpair { device: DeviceId },
    /// Advertise on the LAN and serve paired peers.
    Serve {
        #[arg(long, default_value_t = 0)]
        port: u16,
    },
    /// Generate tokens on this device alone, greedily. Token ids in, ids out.
    Generate {
        #[arg(long)]
        model: PathBuf,
        /// Prompt token ids, comma separated. A tokenizer arrives with the API.
        #[arg(long, value_delimiter = ',')]
        prompt: Vec<u32>,
        #[arg(long, default_value_t = 32)]
        tokens: u32,
        #[arg(long, default_value_t = 2048)]
        context: u32,
        /// Layer indices to cut at, comma separated. Each piece becomes a
        /// shard. Omit for one shard holding the whole model.
        #[arg(long, value_delimiter = ',')]
        split: Vec<u32>,
        /// What crosses a shard boundary. §6 ships f16; f32 separates a wrong
        /// split from a lossy one.
        #[arg(long, default_value = "f16")]
        wire: String,
    },
    /// Measure the link to a paired device.
    Probe {
        target: String,
        #[arg(long, default_value_t = 8)]
        mib: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let dir = match cli.data_dir {
        Some(dir) => dir,
        None => default_data_dir()?,
    };
    let key = DeviceKey::load_or_create(&dir).context("loading device key")?;
    let facts = host::detect();

    match cli.command {
        Command::Generate { model, prompt, tokens, context, split, wire } => {
            let wire_dtype = match wire.as_str() {
                "f16" => lattice_engine::WireDtype::F16,
                "f32" => lattice_engine::WireDtype::F32,
                other => bail!("unknown wire dtype {other:?}; expected f16 or f32"),
            };
            let options = generate::Options {
                max_new: tokens,
                max_context: context,
                cuts: split,
                wire_dtype,
            };
            generate::run(&model, &prompt, &options)?
        }
        Command::Id => {
            println!("{}  {}  {}", key.id(), facts.name, facts.platform);
            println!("{:.1} GB memory", facts.total_memory as f64 / 1e9);
        }
        Command::Discover { seconds } => discover(Duration::from_secs(seconds), key.id())?,
        Command::Host { port } => host_pairing(&dir, &key, &facts, port).await?,
        Command::Pair { target, code } => {
            pair(&dir, &key, &facts, &target, &code).await?;
        }
        Command::Peers => {
            let store = TrustStore::load(&dir)?;
            let mut any = false;
            for peer in store.peers() {
                println!(
                    "{}  {}  {}  ({})",
                    peer.device_id,
                    peer.name,
                    peer.platform,
                    peer.origin()
                );
                any = true;
            }
            if !any {
                println!("no paired devices — run `latticed host` on one machine");
            }
        }
        Command::Provision => provision_mesh(&dir, &key).await?,
        Command::Unpair { device } => {
            let mut store = TrustStore::load(&dir)?;
            if store.remove(device)? {
                println!("forgot {device}");
            } else {
                bail!("{device} was not paired");
            }
        }
        Command::Serve { port } => serve(&dir, &key, &facts, port).await?,
        Command::Probe { target, mib } => probe_peer(&dir, &key, &target, mib << 20).await?,
    }
    Ok(())
}

/// Browses for the given window, printing each device once.
fn discover(window: Duration, self_id: DeviceId) -> Result<()> {
    let discovery = Discovery::new()?;
    let browser = discovery.browse()?;
    let deadline = std::time::Instant::now() + window;
    let mut seen = BTreeMap::new();

    while let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) {
        match browser.next_event(remaining) {
            Some(PeerEvent::Found(peer)) if peer.device_id != self_id => {
                seen.insert(peer.device_id, peer);
            }
            Some(_) => {}
            None => break,
        }
    }

    if seen.is_empty() {
        println!("no devices found — is `latticed serve` running elsewhere?");
    }
    for peer in seen.values() {
        // Show the address a dial would use, not whatever the peer listed
        // first — a device advertises link-local addresses we cannot reach,
        // and printing one invites the user to paste it into `pair`.
        let addr = usable(peer.addrs.clone())
            .first()
            .map_or_else(|| "no reachable address".to_owned(), |a| a.to_string());
        println!(
            "{}  {}  {}  {:.0} GB  {}",
            peer.device_id,
            peer.name,
            peer.platform,
            peer.total_memory as f64 / 1e9,
            addr
        );
    }
    Ok(())
}

async fn host_pairing(dir: &std::path::Path, key: &DeviceKey, facts: &host::HostFacts, port: u16) -> Result<()> {
    let code = PairingCode::generate();
    let endpoint = Endpoint::bind(bind_addr(port), key, Arc::new(AcceptAnyPeer))?;
    let local = endpoint.local_addr()?;

    // Advertise so the other side can pair by device id rather than address.
    let mut discovery = Discovery::new()?;
    discovery.advertise(&Advertisement {
        device_id: key.id(),
        name: facts.name.clone(),
        platform: facts.platform.clone(),
        total_memory: facts.total_memory,
        port: local.port(),
    })?;

    println!("pairing code: {code}");
    println!("on the other device, run:");
    println!("    latticed pair {} --code {code}", key.id());
    println!("(listening on port {})", local.port());

    let conn = endpoint
        .accept()
        .await
        .context("endpoint closed before anyone connected")??;
    let peer = pairing::exchange(
        &conn,
        &code,
        key,
        facts.name.clone(),
        facts.platform.clone(),
    )
    .await
    .context("pairing failed — wrong code, or someone is between the two devices")?;

    TrustStore::load(dir)?.insert(peer.clone())?;
    println!("paired with {} ({})", peer.name, peer.device_id);
    Ok(())
}

async fn pair(
    dir: &std::path::Path,
    key: &DeviceKey,
    facts: &host::HostFacts,
    target: &str,
    code: &str,
) -> Result<()> {
    let code = PairingCode::parse(code)?;
    let addrs = resolve(target, key.id())?;
    let (_endpoint, conn) = connect_any(key, Arc::new(AcceptAnyPeer), &addrs).await?;

    let peer = pairing::exchange(
        &conn,
        &code,
        key,
        facts.name.clone(),
        facts.platform.clone(),
    )
    .await
    .context("pairing failed — wrong code, or someone is between the two devices")?;

    TrustStore::load(dir)?.insert(peer.clone())?;
    println!("paired with {} ({})", peer.name, peer.device_id);
    Ok(())
}

async fn serve(dir: &std::path::Path, key: &DeviceKey, facts: &host::HostFacts, port: u16) -> Result<()> {
    let store = Arc::new(RwLock::new(TrustStore::load(dir)?));
    let endpoint = Endpoint::bind(bind_addr(port), key, Arc::new(TrustedPeers(store.clone())))?;
    let local = endpoint.local_addr()?;
    let handler = Arc::new(provision::Provisioner::new(store, key.id(), facts));

    let mut discovery = Discovery::new()?;
    discovery.advertise(&Advertisement {
        device_id: key.id(),
        name: facts.name.clone(),
        platform: facts.platform.clone(),
        total_memory: facts.total_memory,
        port: local.port(),
    })?;

    println!("{} serving as {} on port {}", facts.name, key.id(), local.port());
    while let Some(incoming) = endpoint.accept().await {
        match incoming {
            Ok(conn) => {
                println!("peer {} connected from {}", conn.peer_id(), conn.remote_address());
                let conn = Arc::new(conn);
                let handler = handler.clone();
                tokio::spawn(async move {
                    let _ = dispatch::serve(conn, handler, Arc::new(lattice_net::NoShards)).await;
                });
            }
            // An unpaired device reaching us is expected on a shared network.
            Err(e) => println!("refused a connection: {e}"),
        }
    }
    Ok(())
}

async fn probe_peer(dir: &std::path::Path, key: &DeviceKey, target: &str, bytes: usize) -> Result<()> {
    let store = Arc::new(RwLock::new(TrustStore::load(dir)?));
    let addrs = resolve(target, key.id())?;
    let (_endpoint, conn) = connect_any(key, Arc::new(TrustedPeers(store)), &addrs).await?;

    let link = probe::measure(&conn, bytes).await?;
    println!("peer      {}", conn.peer_id());
    println!("rtt       {:.2} ms", link.rtt.as_secs_f64() * 1e3);
    println!("bandwidth {:.1} MB/s", link.throughput_bytes_per_sec / 1e6);
    println!(
        "decode hop {:.2} ms per boundary at hidden_dim 8192",
        link.decode_hop(8192).as_secs_f64() * 1e3
    );
    Ok(())
}

/// Hands every paired device the keys of the others. Run on the device the
/// user paired everything to.
async fn provision_mesh(dir: &std::path::Path, key: &DeviceKey) -> Result<()> {
    let store = TrustStore::load(dir)?;
    let peers: Vec<PairedPeer> = store.peers().cloned().collect();
    if peers.len() < 2 {
        bail!(
            "provisioning needs at least two paired devices, this one has {} — \
             pair the others to this device first",
            peers.len()
        );
    }

    let wanted: Vec<DeviceId> = peers.iter().map(|p| p.device_id).collect();
    let located = locate(&wanted)?;

    let shared = Arc::new(RwLock::new(store));
    let endpoint = Endpoint::bind(bind_addr(0), key, Arc::new(TrustedPeers(shared)))?;
    let mesh = Mesh::new(endpoint, key.id());

    for peer in &peers {
        let Some(addrs) = located.get(&peer.device_id) else {
            println!("{}: not on the network, skipped", peer.name);
            continue;
        };
        let conn = mesh.connect(peer.device_id, addrs).await?;
        let roster = provision::roster_for(peer.device_id, &peers);
        let introduced = roster.len();
        match control::request(&conn, &Request::Provision(roster)).await? {
            Response::Provisioned {
                added,
                already_known,
            } => println!(
                "{}: {added} of {introduced} introductions new, {already_known} already known",
                peer.name
            ),
            Response::Refused(why) => println!("{}: refused — {why}", peer.name),
            other => println!("{}: unexpected reply {other:?}", peer.name),
        }
    }
    Ok(())
}

/// One mDNS browse for several devices at once, rather than one per device.
fn locate(wanted: &[DeviceId]) -> Result<HashMap<DeviceId, Vec<SocketAddr>>> {
    let discovery = Discovery::new()?;
    let browser = discovery.browse()?;
    let deadline = std::time::Instant::now() + Duration::from_secs(6);
    let mut found = HashMap::new();

    while found.len() < wanted.len() {
        let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
            break;
        };
        match browser.next_event(remaining) {
            // As in resolve(): ignore a record with nothing dialable in it and
            // wait for a fuller one.
            Some(PeerEvent::Found(peer)) if wanted.contains(&peer.device_id) => {
                let addrs = usable(peer.addrs);
                if !addrs.is_empty() {
                    found.insert(peer.device_id, addrs);
                }
            }
            Some(_) => {}
            None => break,
        }
    }
    Ok(found)
}

/// Accepts either a `host:port` or a device id to look up over mDNS.
///
/// A device advertises every address it is reachable on, so this returns all
/// of them; picking one up front is how a peer that is perfectly reachable on
/// its LAN address ends up unreachable because it also published a loopback.
fn resolve(target: &str, self_id: DeviceId) -> Result<Vec<SocketAddr>> {
    if let Ok(addr) = target.parse::<SocketAddr>() {
        return Ok(vec![addr]);
    }
    let wanted: DeviceId = target
        .parse()
        .context("target must be a host:port or a device id")?;

    let discovery = Discovery::new()?;
    let browser = discovery.browse()?;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut seen_but_unreachable = false;
    while let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) {
        match browser.next_event(remaining) {
            // mDNS resolves a device's addresses incrementally, and the first
            // record for a peer often carries only its loopback and
            // link-local ones. Returning on that would report a device we can
            // see as unreachable, so keep waiting for a record we can use.
            Some(PeerEvent::Found(peer)) if peer.device_id == wanted => {
                let addrs = usable(peer.addrs);
                if !addrs.is_empty() {
                    return Ok(addrs);
                }
                seen_but_unreachable = true;
            }
            Some(_) => {}
            None => break,
        }
    }
    if wanted == self_id {
        bail!("{wanted} is this device");
    }
    if seen_but_unreachable {
        bail!(
            "{wanted} is on the network but advertised no address this device \
             can reach — pass a host:port instead"
        );
    }
    bail!("could not find {wanted} on the network — pass a host:port instead")
}

/// The advertised addresses this device can actually dial, best first.
///
/// Link-local IPv6 is dropped because reaching it needs the interface scope,
/// which is lost when an mDNS record is turned into a plain SocketAddr. The
/// rest of IPv6 is dropped because listeners bind IPv4 (see bench/README.md);
/// remove that filter when they no longer do.
fn usable(addrs: Vec<SocketAddr>) -> Vec<SocketAddr> {
    let mut usable: Vec<_> = addrs.into_iter().filter(|a| a.is_ipv4()).collect();
    usable.sort_by_key(|a| a.ip().is_loopback());
    usable
}

/// Tries each address in turn. A peer publishes every interface it has, and
/// most of them cannot reach any given caller.
async fn connect_any(
    key: &DeviceKey,
    policy: Arc<dyn lattice_net::PeerPolicy>,
    addrs: &[SocketAddr],
) -> Result<(Endpoint, lattice_net::Connection)> {
    if addrs.is_empty() {
        bail!("no reachable address for that device");
    }
    let mut failures = Vec::new();
    for addr in addrs {
        let endpoint = Endpoint::bind(bind_addr(0), key, policy.clone())?;
        match endpoint.connect(*addr).await {
            Ok(conn) => return Ok((endpoint, conn)),
            // Report every attempt: "it did not connect" is useless when a
            // peer advertised four addresses.
            Err(e) => failures.push(format!("  {addr}: {e}")),
        }
    }
    bail!("could not reach that device:\n{}", failures.join("\n"))
}

fn bind_addr(port: u16) -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], port))
}
