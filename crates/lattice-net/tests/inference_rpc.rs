use lattice_net::{
    AcceptAnyPeer, DeviceId, DeviceKey, Endpoint, NoShards, RefuseControl, RpcHandler, RpcProxy,
    RpcSession, dispatch,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

struct Handler {
    allowed: DeviceId,
    address: std::net::SocketAddr,
    slots: Arc<Semaphore>,
}

#[async_trait::async_trait]
impl RpcHandler for Handler {
    async fn connect(&self, from: DeviceId) -> Result<RpcSession, String> {
        if from != self.allowed {
            return Err("unauthorized".into());
        }
        let permit = self.slots.clone().try_acquire_owned().map_err(|_| "busy")?;
        let stream = TcpStream::connect(self.address)
            .await
            .map_err(|e| e.to_string())?;
        Ok(RpcSession { stream, permit })
    }
}

async fn exercise(authorized: bool) {
    let a_dir = tempfile::tempdir().unwrap();
    let b_dir = tempfile::tempdir().unwrap();
    let a_key = DeviceKey::load_or_create(a_dir.path()).unwrap();
    let b_key = DeviceKey::load_or_create(b_dir.path()).unwrap();
    let a = Endpoint::bind(
        "127.0.0.1:0".parse().unwrap(),
        &a_key,
        Arc::new(AcceptAnyPeer),
    )
    .unwrap();
    let b = Endpoint::bind(
        "127.0.0.1:0".parse().unwrap(),
        &b_key,
        Arc::new(AcceptAnyPeer),
    )
    .unwrap();
    let backend = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let slots = Arc::new(Semaphore::new(1));
    let handler = Arc::new(Handler {
        allowed: if authorized { b_key.id() } else { a_key.id() },
        address: backend.local_addr().unwrap(),
        slots: slots.clone(),
    });
    let (accepted, connected) = tokio::join!(a.accept(), b.connect(a.local_addr().unwrap()));
    let serve = tokio::spawn(dispatch::serve_with_rpc(
        Arc::new(accepted.unwrap().unwrap()),
        Arc::new(RefuseControl),
        Arc::new(NoShards),
        handler,
    ));
    let proxy = RpcProxy::bind(Arc::new(connected.unwrap())).await.unwrap();
    assert!(proxy.address().ip().is_loopback());
    let mut client = TcpStream::connect(proxy.address()).await.unwrap();
    if !authorized {
        let mut byte = [0];
        assert_eq!(client.read(&mut byte).await.unwrap(), 0);
        assert_eq!(slots.available_permits(), 1);
        serve.abort();
        return;
    }
    let (mut server, _) = backend.accept().await.unwrap();
    assert_eq!(slots.available_permits(), 0);
    let mut busy = TcpStream::connect(proxy.address()).await.unwrap();
    let mut byte = [0];
    assert_eq!(busy.read(&mut byte).await.unwrap(), 0);
    let payload: Vec<u8> = (0..131072).map(|i| (i % 256) as u8).collect();
    let expected = payload.clone();
    let responder = tokio::spawn(async move {
        let mut input = Vec::new();
        server.read_to_end(&mut input).await.unwrap();
        assert_eq!(input, expected);
        server.write_all(&input).await.unwrap();
        server.shutdown().await.unwrap();
    });
    client.write_all(&payload).await.unwrap();
    client.shutdown().await.unwrap();
    let mut output = Vec::new();
    client.read_to_end(&mut output).await.unwrap();
    assert_eq!(output, payload);
    responder.await.unwrap();
    while slots.available_permits() != 1 {
        tokio::task::yield_now().await;
    }
    let _client = TcpStream::connect(proxy.address()).await.unwrap();
    let (mut backend_client, _) = backend.accept().await.unwrap();
    drop(proxy);
    assert_eq!(backend_client.read(&mut byte).await.unwrap(), 0);
    while slots.available_permits() != 1 {
        tokio::task::yield_now().await;
    }
    serve.abort();
}

#[tokio::test]
async fn binary_half_close_busy_and_proxy_cleanup() {
    tokio::time::timeout(Duration::from_secs(10), exercise(true))
        .await
        .unwrap();
}

#[tokio::test]
async fn rpc_handler_refuses_an_unauthorized_peer() {
    tokio::time::timeout(Duration::from_secs(10), exercise(false))
        .await
        .unwrap();
}
