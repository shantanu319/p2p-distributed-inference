//! Length-prefixed frames on a stream.
//!
//! Control and pairing messages are small and fixed-shape. The caller supplies
//! the ceiling, and anything above it is a peer that is confused or hostile —
//! never something we allocate for.

use crate::Error;

pub async fn write_frame(send: &mut quinn::SendStream, payload: &[u8]) -> Result<(), Error> {
    let len = u32::try_from(payload.len())
        .map_err(|_| Error::Stream("frame exceeds 4 GiB".into()))?;
    send.write_all(&len.to_le_bytes())
        .await
        .map_err(|e| Error::Stream(format!("writing frame length: {e}")))?;
    send.write_all(payload)
        .await
        .map_err(|e| Error::Stream(format!("writing frame: {e}")))
}

pub async fn read_frame(recv: &mut quinn::RecvStream, max: usize) -> Result<Vec<u8>, Error> {
    let mut header = [0u8; 4];
    recv.read_exact(&mut header)
        .await
        .map_err(|e| Error::Stream(format!("reading frame length: {e}")))?;
    let len = u32::from_le_bytes(header) as usize;
    if len > max {
        return Err(Error::Stream(format!(
            "peer announced a {len} byte frame, ceiling is {max}"
        )));
    }
    let mut payload = vec![0u8; len];
    recv.read_exact(&mut payload)
        .await
        .map_err(|e| Error::Stream(format!("reading frame: {e}")))?;
    Ok(payload)
}
