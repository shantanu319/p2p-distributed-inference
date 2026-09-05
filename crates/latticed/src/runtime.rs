use anyhow::{Context, Result, bail};
use lattice_net::{DeviceId, EngineCapabilities, GpuDevice, RpcHandler, RpcSession};
use std::io::{Read, Seek, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::Semaphore;

pub const ENGINE_REVISION: &str = "85d5703a3b1b47243213a39059a6e3076c92733a";

#[derive(Clone)]
pub struct Engine {
    dir: PathBuf,
}

impl Engine {
    pub fn locate(dir: Option<&Path>) -> Result<Self> {
        let dir = match dir {
            Some(dir) => dir.to_owned(),
            None => match std::env::var_os("LATTICE_ENGINE_DIR") {
                Some(dir) => PathBuf::from(dir),
                None => PathBuf::from(
                    std::env::var_os("HOME").context("HOME is unset; use --engine-dir")?,
                )
                .join(".local/share/lattice/engine"),
            },
        };
        let revision = std::fs::read_to_string(dir.join("engine-revision"))
            .context("inference engine missing; run ./scripts/setup-engine.sh")?;
        if revision.trim() != ENGINE_REVISION {
            bail!("inference engine revision mismatch; rerun ./scripts/setup-engine.sh");
        }
        let engine = Self { dir };
        for path in [engine.cli(), engine.rpc_server(), engine.info()] {
            if !path.is_file() {
                bail!(
                    "missing {}; rerun ./scripts/setup-engine.sh",
                    path.display()
                );
            }
        }
        Ok(engine)
    }

    pub fn cli(&self) -> PathBuf {
        self.dir.join("bin/llama-cli")
    }
    pub fn rpc_server(&self) -> PathBuf {
        self.dir.join("bin/ggml-rpc-server")
    }
    pub fn info(&self) -> PathBuf {
        self.dir.join("bin/lattice-engine-info")
    }
    pub fn revision(&self) -> &'static str {
        ENGINE_REVISION
    }

    pub fn devices(&self) -> Result<Vec<GpuDevice>> {
        let mut child = Command::new(self.info())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("detecting inference GPUs")?;
        let stdout = drain(
            child
                .stdout
                .take()
                .context("GPU helper stdout unavailable")?,
        );
        let stderr = drain(
            child
                .stderr
                .take()
                .context("GPU helper stderr unavailable")?,
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(90);
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break Some(status);
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(25));
        };
        let stdout = stdout
            .join()
            .map_err(|_| anyhow::anyhow!("GPU helper output reader failed"))??;
        let stderr = stderr
            .join()
            .map_err(|_| anyhow::anyhow!("GPU helper error reader failed"))??;
        let status =
            status.context("GPU discovery timed out; check the GPU driver installation")?;
        if !status.success() {
            bail!("GPU detection failed: {}", String::from_utf8_lossy(&stderr));
        }
        serde_json::from_slice(&stdout).context("invalid GPU discovery output")
    }
}

fn drain(
    mut stream: impl Read + Send + 'static,
) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>> {
    std::thread::spawn(move || {
        let mut output = Vec::new();
        let mut buffer = [0; 8192];
        loop {
            let count = stream.read(&mut buffer)?;
            if count == 0 {
                return Ok(output);
            }
            let keep = count.min((1024 * 1024usize).saturating_sub(output.len()));
            output.extend_from_slice(&buffer[..keep]);
        }
    })
}

pub fn select_device(devices: &[GpuDevice], explicit: Option<&str>) -> Result<GpuDevice> {
    let gpu = |device: &&GpuDevice| matches!(device.kind.as_str(), "gpu" | "igpu");
    if let Some(name) = explicit {
        return devices
            .iter()
            .filter(gpu)
            .find(|device| device.name == name)
            .cloned()
            .with_context(|| {
                format!(
                    "GPU {name:?} unavailable; available GPUs: {}",
                    devices
                        .iter()
                        .filter(gpu)
                        .map(|d| d.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            });
    }
    devices
        .iter()
        .filter(gpu)
        .max_by_key(|device| (device.kind == "gpu", device.free_memory))
        .cloned()
        .context(
            "no supported GPU found; install the GPU engine and driver; CPU inference is disabled",
        )
}

struct Server {
    child: Child,
    address: SocketAddr,
    logger: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(logger) = self.logger.take() {
            let _ = logger.join();
        }
    }
}

struct State {
    master: Option<DeviceId>,
    server: Option<Server>,
}

pub struct WorkerRuntime {
    engine: Engine,
    device: GpuDevice,
    dir: PathBuf,
    state: Mutex<State>,
    slots: Arc<Semaphore>,
}

impl WorkerRuntime {
    pub fn new(engine: Engine, device: GpuDevice, dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir.join("rpc-cache"))?;
        Ok(Self {
            engine,
            device,
            dir: dir.to_owned(),
            state: Mutex::new(State {
                master: None,
                server: None,
            }),
            slots: Arc::new(Semaphore::new(1)),
        })
    }

    pub fn authorize(&self, master: Option<DeviceId>) {
        let mut state = self.state.lock().expect("runtime state poisoned");
        if state.master != master {
            state.server = None;
            state.master = master;
        }
    }

    pub fn capabilities(&self, from: DeviceId) -> Result<EngineCapabilities> {
        self.check_health(from)?;
        let permit = self.slots.try_acquire();
        let device = if permit.is_ok() {
            select_device(&self.engine.devices()?, Some(&self.device.name))?
        } else {
            self.device.clone()
        };
        self.check_health(from)?;
        Ok(EngineCapabilities {
            revision: ENGINE_REVISION.into(),
            device,
            busy: permit.is_err(),
        })
    }

    fn check_health(&self, from: DeviceId) -> Result<()> {
        let mut state = self.state.lock().expect("runtime state poisoned");
        Self::check_master(&state, from)?;
        let exited = state
            .server
            .as_mut()
            .map(|server| server.child.try_wait())
            .transpose()?
            .flatten();
        if let Some(status) = exited {
            state.server = None;
            bail!(
                "GPU RPC server exited ({status}); inspect {}",
                self.dir.join("rpc-server.log").display()
            );
        }
        Ok(())
    }

    fn check_master(state: &State, from: DeviceId) -> Result<()> {
        if state.master != Some(from) {
            bail!("inference is restricted to the registered master");
        }
        Ok(())
    }

    async fn acquire(&self, from: DeviceId) -> Result<tokio::sync::OwnedSemaphorePermit, String> {
        self.check_health(from).map_err(|e| e.to_string())?;
        let permit =
            tokio::time::timeout(Duration::from_secs(2), self.slots.clone().acquire_owned())
                .await
                .map_err(|_| "GPU worker is busy".to_owned())?
                .map_err(|_| "GPU worker is shutting down".to_owned())?;
        self.check_health(from).map_err(|e| e.to_string())?;
        Ok(permit)
    }

    fn address(&self, from: DeviceId) -> Result<SocketAddr> {
        let mut state = self.state.lock().expect("runtime state poisoned");
        Self::check_master(&state, from)?;
        if let Some(server) = state.server.as_mut() {
            if server.child.try_wait()?.is_none() {
                return Ok(server.address);
            }
            state.server = None;
        }
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let address = listener.local_addr()?;
        let mut log = std::fs::File::create(self.dir.join("rpc-server.log"))?;
        drop(listener);
        let mut child = Command::new(self.engine.rpc_server())
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &address.port().to_string(),
                "--device",
                &self.device.name,
                "--cache",
            ])
            .env("LLAMA_CACHE", self.dir.join("rpc-cache"))
            .env("GGML_RPC_NO_RDMA", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .context("starting GPU RPC server")?;
        let mut stderr = child.stderr.take().context("RPC stderr unavailable")?;
        let logger = std::thread::spawn(move || {
            let mut buffer = [0; 8192];
            let mut written = 0;
            while let Ok(count) = stderr.read(&mut buffer) {
                if count == 0 {
                    break;
                }
                if written + count > 1024 * 1024 {
                    if log.set_len(0).is_err() || log.rewind().is_err() {
                        break;
                    }
                    written = 0;
                }
                if log.write_all(&buffer[..count]).is_err() {
                    break;
                }
                written += count;
            }
        });
        state.server = Some(Server {
            child,
            address,
            logger: Some(logger),
        });
        Ok(address)
    }
}

#[async_trait::async_trait]
impl RpcHandler for WorkerRuntime {
    async fn connect(&self, from: DeviceId) -> Result<RpcSession, String> {
        let permit = self.acquire(from).await?;
        let address = self.address(from).map_err(|e| e.to_string())?;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(90);
        loop {
            self.check_health(from).map_err(|e| e.to_string())?;
            match TcpStream::connect(address).await {
                Ok(stream) => {
                    self.check_health(from).map_err(|e| e.to_string())?;
                    return Ok(RpcSession { stream, permit });
                }
                Err(error) if tokio::time::Instant::now() >= deadline => {
                    return Err(format!(
                        "GPU RPC server did not become ready: {error}; inspect {}",
                        self.dir.join("rpc-server.log").display()
                    ));
                }
                Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn device(name: &str, kind: &str, free_memory: u64) -> GpuDevice {
        GpuDevice {
            name: name.into(),
            description: name.into(),
            kind: kind.into(),
            total_memory: free_memory,
            free_memory,
        }
    }
    #[test]
    fn discrete_gpu_wins_over_large_integrated_memory() {
        let devices = vec![
            device("CPU", "cpu", 1000),
            device("integrated", "igpu", 900),
            device("discrete", "gpu", 100),
        ];
        assert_eq!(select_device(&devices, None).unwrap().name, "discrete");
        assert_eq!(
            select_device(&devices, Some("integrated")).unwrap().name,
            "integrated"
        );
        assert!(select_device(&devices, Some("CPU")).is_err());
        assert!(select_device(&devices, Some("missing")).is_err());
    }
    #[test]
    fn only_gpu_and_registered_master_are_accepted() {
        assert!(select_device(&[device("CPU", "cpu", 1000)], None).is_err());
        assert_eq!(
            select_device(&[device("Metal", "igpu", 1000)], None)
                .unwrap()
                .name,
            "Metal"
        );
        let master = "1111111111111111".parse().unwrap();
        assert!(
            WorkerRuntime::check_master(
                &State {
                    master: None,
                    server: None
                },
                master
            )
            .is_err()
        );
        assert!(
            WorkerRuntime::check_master(
                &State {
                    master: Some(master),
                    server: None
                },
                master
            )
            .is_ok()
        );
    }
    #[test]
    fn mismatched_engine_revision_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("engine-revision"), "wrong").unwrap();
        assert!(Engine::locate(Some(dir.path())).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn replacing_master_stops_and_reaps_old_gpu_process() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = WorkerRuntime::new(
            Engine {
                dir: dir.path().into(),
            },
            device("gpu", "gpu", 100),
            dir.path(),
        )
        .unwrap();
        let first = "1111111111111111".parse().unwrap();
        let second = "2222222222222222".parse().unwrap();
        runtime.authorize(Some(first));
        let child = Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id();
        runtime.state.lock().unwrap().server = Some(Server {
            child,
            address: "127.0.0.1:1".parse().unwrap(),
            logger: None,
        });
        runtime.authorize(Some(second));
        assert!(runtime.capabilities(first).is_err());
        assert!(runtime.check_health(second).is_ok());
        assert_eq!(unsafe { libc::kill(pid as i32, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
        runtime.authorize(None);
        assert!(runtime.capabilities(second).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn idle_capabilities_refresh_memory_and_busy_skips_helper() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("bin")).unwrap();
        let helper = dir.path().join("bin/lattice-engine-info");
        let fresh = device("gpu", "gpu", 37);
        std::fs::write(
            &helper,
            format!(
                "#!/bin/sh\nprintf '%s' '{}'\n",
                serde_json::to_string(&vec![fresh]).unwrap()
            ),
        )
        .unwrap();
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755)).unwrap();
        let runtime = WorkerRuntime::new(
            Engine {
                dir: dir.path().into(),
            },
            device("gpu", "gpu", 100),
            dir.path(),
        )
        .unwrap();
        let master = "1111111111111111".parse().unwrap();
        runtime.authorize(Some(master));
        let info = runtime.capabilities(master).unwrap();
        assert_eq!(info.device.free_memory, 37);
        assert!(!info.busy);
        let _permit = runtime.slots.try_acquire().unwrap();
        std::fs::remove_file(helper).unwrap();
        assert!(runtime.capabilities(master).unwrap().busy);
    }

    #[test]
    fn helper_output_is_drained_even_when_retention_is_full() {
        let output = drain(std::io::Cursor::new(vec![42; 2 * 1024 * 1024]))
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(output.len(), 1024 * 1024);
    }
    #[tokio::test]
    async fn rpc_waits_for_previous_session_and_rechecks_owner() {
        let dir = tempfile::tempdir().unwrap();
        let runtime = WorkerRuntime::new(
            Engine {
                dir: dir.path().into(),
            },
            device("gpu", "gpu", 100),
            dir.path(),
        )
        .unwrap();
        let master = "1111111111111111".parse().unwrap();
        let other = "2222222222222222".parse().unwrap();
        runtime.authorize(Some(master));
        let permit = runtime.slots.clone().acquire_owned().await.unwrap();
        assert!(runtime.acquire(other).await.is_err());
        let release = async {
            tokio::time::sleep(Duration::from_millis(25)).await;
            drop(permit);
        };
        let (next, _) = tokio::join!(runtime.acquire(master), release);
        assert!(next.is_ok());
        drop(next);
        let permit = runtime.slots.clone().acquire_owned().await.unwrap();
        let revoke = async {
            tokio::time::sleep(Duration::from_millis(25)).await;
            runtime.authorize(Some(other));
            drop(permit);
        };
        let (next, _) = tokio::join!(runtime.acquire(master), revoke);
        assert!(next.is_err());
    }
}
