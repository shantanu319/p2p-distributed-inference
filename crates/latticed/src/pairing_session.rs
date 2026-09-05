use anyhow::{Context, Result, bail};
use lattice_net::codec::{read_frame, write_frame};
use lattice_net::{Connection, DeviceId, DeviceKey, PairingCode, TrustStore, pairing};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use crate::host::HostFacts;

const TIMEOUT: Duration = Duration::from_secs(30);
const SAVED: &[u8] = b"saved";

pub async fn exchange(
    conn: &Connection,
    key: &DeviceKey,
    facts: &HostFacts,
    store: &Arc<RwLock<TrustStore>>,
    code: &PairingCode,
) -> Result<DeviceId> {
    tokio::time::timeout(TIMEOUT, async {
        let peer = pairing::exchange(conn, code, key, facts.name.clone(), facts.platform.clone())
            .await
            .context("pairing failed: check the six-digit code on the master")?;
        if peer.device_id != conn.peer_id() {
            bail!("paired identity did not match the connection's device key");
        }
        let id = peer.device_id;
        store
            .write()
            .expect("trust store poisoned")
            .insert(peer)
            .context("saving the paired device")?;

        let (mut send, mut recv) = conn.control_stream().await?;
        write_frame(&mut send, SAVED).await?;
        send.finish().context("finishing pairing confirmation")?;
        if read_frame(&mut recv, SAVED.len()).await? != SAVED {
            bail!("peer did not confirm saving the pairing");
        }
        recv.read_to_end(0)
            .await
            .context("reading the end of pairing confirmation")?;
        if send
            .stopped()
            .await
            .context("delivering pairing confirmation")?
            .is_some()
        {
            bail!("peer stopped the pairing confirmation");
        }
        Ok(id)
    })
    .await
    .context("pairing did not finish within 30 seconds")?
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_net::{AcceptAnyPeer, Endpoint};
    use tempfile::TempDir;

    struct Device {
        key: DeviceKey,
        dir: TempDir,
        store: Arc<RwLock<TrustStore>>,
        facts: HostFacts,
    }

    fn device(name: &str) -> Device {
        let dir = tempfile::tempdir().unwrap();
        Device {
            key: DeviceKey::load_or_create(dir.path()).unwrap(),
            store: Arc::new(RwLock::new(TrustStore::load(dir.path()).unwrap())),
            facts: HostFacts {
                name: name.into(),
                platform: "test-loopback".into(),
                total_memory: 1 << 30,
            },
            dir,
        }
    }

    fn endpoint(key: &DeviceKey) -> Endpoint {
        Endpoint::bind("127.0.0.1:0".parse().unwrap(), key, Arc::new(AcceptAnyPeer)).unwrap()
    }

    async fn attempt(
        master: &Device,
        worker: &Device,
        master_code: &PairingCode,
        worker_code: &PairingCode,
    ) -> (Result<DeviceId>, Result<DeviceId>) {
        let listener = endpoint(&master.key);
        let dialer = endpoint(&worker.key);
        let addr = listener.local_addr().unwrap();
        let master = async move {
            let conn = listener.accept().await.unwrap().unwrap();
            exchange(
                &conn,
                &master.key,
                &master.facts,
                &master.store,
                master_code,
            )
            .await
        };
        let worker = async move {
            let conn = dialer.connect(addr).await.unwrap();
            exchange(
                &conn,
                &worker.key,
                &worker.facts,
                &worker.store,
                worker_code,
            )
            .await
        };
        tokio::join!(master, worker)
    }

    #[tokio::test]
    async fn matching_codes_persist_and_confirm_both_devices() {
        let master = device("master");
        let worker = device("worker");
        let code = PairingCode::generate();
        let (master_result, worker_result) = attempt(&master, &worker, &code, &code).await;

        assert_eq!(master_result.unwrap(), worker.key.id());
        assert_eq!(worker_result.unwrap(), master.key.id());
        for (owner, peer) in [(&master, &worker), (&worker, &master)] {
            assert!(
                owner
                    .store
                    .read()
                    .unwrap()
                    .is_trusted(&peer.key.public_key())
            );
            let persisted = TrustStore::load(owner.dir.path()).unwrap();
            let saved = persisted.get(peer.key.id()).unwrap();
            assert_eq!(saved.name, peer.facts.name);
            assert_eq!(saved.public_key, peer.key.public_key());
        }
    }

    #[tokio::test]
    async fn wrong_codes_leave_neither_device_trusted() {
        let master = device("master");
        let worker = device("worker");
        let (master_result, worker_result) = attempt(
            &master,
            &worker,
            &PairingCode::parse("123456").unwrap(),
            &PairingCode::parse("654321").unwrap(),
        )
        .await;

        assert!(master_result.is_err());
        assert!(worker_result.is_err());
        for owner in [&master, &worker] {
            assert_eq!(owner.store.read().unwrap().peers().count(), 0);
            assert_eq!(
                TrustStore::load(owner.dir.path()).unwrap().peers().count(),
                0
            );
        }
    }

    #[tokio::test]
    async fn a_peer_that_does_not_confirm_saving_cannot_report_success() {
        let master = device("master");
        let worker = device("worker");
        let code = PairingCode::generate();
        let listener = endpoint(&master.key);
        let dialer = endpoint(&worker.key);
        let master_run = async {
            let conn = listener.accept().await.unwrap().unwrap();
            pairing::exchange(
                &conn,
                &code,
                &master.key,
                master.facts.name.clone(),
                master.facts.platform.clone(),
            )
            .await
            .unwrap();
            let (mut send, mut recv) = conn.control_stream().await.unwrap();
            assert_eq!(read_frame(&mut recv, SAVED.len()).await.unwrap(), SAVED);
            write_frame(&mut send, b"no").await.unwrap();
            send.finish().unwrap();
            let _ = send.stopped().await;
        };
        let worker_run = async {
            let conn = dialer
                .connect(listener.local_addr().unwrap())
                .await
                .unwrap();
            exchange(&conn, &worker.key, &worker.facts, &worker.store, &code).await
        };
        let (_, result) = tokio::join!(master_run, worker_run);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("did not confirm saving")
        );
    }
}
