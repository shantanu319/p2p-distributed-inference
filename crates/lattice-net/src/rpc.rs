use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncWriteExt, copy};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::OwnedSemaphorePermit;
use tokio::task::{JoinHandle, JoinSet};

use crate::codec::{read_frame, write_frame};
use crate::{Connection, DeviceId, Error, StreamHeader, StreamKind};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_ACK: usize = 4096;

pub struct RpcSession {
    pub stream: TcpStream,
    pub permit: OwnedSemaphorePermit,
}

#[async_trait::async_trait]
pub trait RpcHandler: Send + Sync {
    async fn connect(&self, from: DeviceId) -> Result<RpcSession, String>;
}

pub struct DenyRpc;

#[async_trait::async_trait]
impl RpcHandler for DenyRpc {
    async fn connect(&self, _from: DeviceId) -> Result<RpcSession, String> {
        Err("inference RPC is unavailable on this device".into())
    }
}

pub struct RpcProxy {
    address: SocketAddr,
    task: JoinHandle<()>,
}

impl RpcProxy {
    pub async fn bind(conn: Arc<Connection>) -> Result<Self, Error> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let task = tokio::spawn(async move {
            let mut sessions = JoinSet::new();
            loop {
                tokio::select! {
                    accepted = listener.accept() => {
                        let Ok((tcp, _)) = accepted else { break };
                        let conn = conn.clone();
                        sessions.spawn(async move {
                            if let Err(error) = proxy(tcp, &conn).await {
                                eprintln!("inference RPC connection failed: {error}");
                            }
                        });
                    }
                    _ = sessions.join_next(), if !sessions.is_empty() => {}
                    _ = conn.inner.closed() => break,
                }
            }
        });
        Ok(Self { address, task })
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }
}

impl Drop for RpcProxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn proxy(tcp: TcpStream, conn: &Connection) -> Result<(), Error> {
    let (send, recv) = tokio::time::timeout(HANDSHAKE_TIMEOUT, async {
        let (send, mut recv) = conn
            .open(StreamHeader {
                kind: StreamKind::InferenceRpc,
                session: 0,
                generation: 0,
            })
            .await?;
        let bytes = read_frame(&mut recv, MAX_ACK).await?;
        let ack: Result<(), String> = postcard::from_bytes(&bytes)?;
        ack.map_err(Error::Stream)?;
        Ok::<_, Error>((send, recv))
    })
    .await
    .map_err(|_| Error::Stream("inference RPC handshake timed out".into()))??;
    tunnel(tcp, send, recv).await
}

pub(crate) async fn serve(
    from: DeviceId,
    handler: Arc<dyn RpcHandler>,
    mut send: quinn::SendStream,
    recv: quinn::RecvStream,
) -> Result<(), Error> {
    let session = tokio::time::timeout(HANDSHAKE_TIMEOUT, handler.connect(from))
        .await
        .unwrap_or_else(|_| Err("inference RPC connection timed out".into()));
    let ack = session
        .as_ref()
        .map(|_| ())
        .map_err(|why| why.chars().take(512).collect::<String>());
    write_frame(&mut send, &postcard::to_allocvec(&ack)?).await?;
    match session {
        Ok(RpcSession { stream, permit }) => {
            let result = tunnel(stream, send, recv).await;
            drop(permit);
            result
        }
        Err(_) => {
            let _ = send.finish();
            Ok(())
        }
    }
}

struct TunnelStreams {
    send: quinn::SendStream,
    recv: quinn::RecvStream,
    completed: bool,
}

impl Drop for TunnelStreams {
    fn drop(&mut self) {
        if !self.completed {
            let _ = self.send.reset(0u32.into());
            let _ = self.recv.stop(0u32.into());
        }
    }
}

async fn tunnel(
    tcp: TcpStream,
    send: quinn::SendStream,
    recv: quinn::RecvStream,
) -> Result<(), Error> {
    let mut streams = TunnelStreams {
        send,
        recv,
        completed: false,
    };
    let (mut reader, mut writer) = tcp.into_split();
    let outbound = async {
        copy(&mut reader, &mut streams.send).await?;
        streams
            .send
            .finish()
            .map_err(|e| Error::Stream(e.to_string()))?;
        Ok::<_, Error>(())
    };
    let inbound = async {
        copy(&mut streams.recv, &mut writer).await?;
        writer.shutdown().await?;
        Ok::<_, Error>(())
    };
    tokio::try_join!(outbound, inbound)?;
    streams.completed = true;
    Ok(())
}
