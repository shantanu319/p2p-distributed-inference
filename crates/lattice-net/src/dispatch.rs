//! The single accept loop for a connection: read each stream's declared kind
//! and hand it to whatever serves that kind.
//!
//! One loop per connection, because two would race for the same streams.

use std::sync::Arc;

use crate::control::{self, ControlHandler, Request, Response};
use crate::stream::StreamKind;
use crate::{Connection, DeviceId, Error, probe};

/// Serves a peer until the connection closes. Returns `Ok` on a clean close;
/// a peer going away is ordinary, not an error.
pub async fn serve(conn: Arc<Connection>, control: Arc<dyn ControlHandler>) -> Result<(), Error> {
    let from = conn.peer_id();
    loop {
        let Ok((header, send, recv)) = conn.accept_stream().await else {
            return Ok(());
        };
        let control = control.clone();
        tokio::spawn(async move {
            match header.kind {
                StreamKind::Control => {
                    let _ = control::answer(from, control.as_ref(), send, recv).await;
                }
                // Until the engine lands there is nothing to execute, so both
                // data kinds go to the probe responder.
                StreamKind::Activation | StreamKind::Bulk => {
                    probe::serve_stream(send, recv).await;
                }
            }
        });
    }
}

/// Declines every control request. For peers we serve data to but take no
/// instructions from.
#[derive(Debug)]
pub struct RefuseControl;

impl ControlHandler for RefuseControl {
    fn handle(&self, _from: DeviceId, _request: Request) -> Response {
        Response::Refused("this device takes no control requests".into())
    }
}
