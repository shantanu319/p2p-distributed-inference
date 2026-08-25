//! `latticed` — the Lattice daemon and its CLI.
//!
//! This slice covers §7 only: find devices, pair with them, and measure the
//! link. Nothing here loads a model.

mod host;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use lattice_net::discovery::{Discovery, PeerEvent};
use lattice_net::identity::default_data_dir;
use lattice_net::{DeviceId, DeviceKey, TrustStore};
use std::collections::BTreeMap;
use std::path::PathBuf;
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
    /// List paired devices.
    Peers,
    /// Forget a paired device.
    Unpair { device: DeviceId },
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
        Command::Id => {
            println!("{}  {}  {}", key.id(), facts.name, facts.platform);
            println!("{:.1} GB memory", facts.total_memory as f64 / 1e9);
        }
        Command::Discover { seconds } => discover(Duration::from_secs(seconds), key.id())?,
        Command::Peers => {
            let store = TrustStore::load(&dir)?;
            let mut any = false;
            for peer in store.peers() {
                println!("{}  {}  {}", peer.device_id, peer.name, peer.platform);
                any = true;
            }
            if !any {
                println!("no paired devices — run `latticed host` on one machine");
            }
        }
        Command::Unpair { device } => {
            let mut store = TrustStore::load(&dir)?;
            if store.remove(device)? {
                println!("forgot {device}");
            } else {
                bail!("{device} was not paired");
            }
        }
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
        let addr = peer.addrs.first().map_or_else(String::new, |a| a.to_string());
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
