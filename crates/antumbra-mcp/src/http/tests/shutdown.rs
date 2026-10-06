//! Stopping: a request in flight when the stop comes is answered, and a
//! stream that never ends (a client's GET /mcp) does not hold the stop open
//! past the drain. Over a real socket, since the drain is about connections.

use super::*;
use std::time::{Duration, Instant};
use tokio::net::TcpStream;
use tokio::sync::{oneshot, Notify};

/// Serves `app` on a free port until the returned sender fires, with `drain`.
async fn start(
    app: Router,
    drain: Duration,
) -> (
    std::net::SocketAddr,
    oneshot::Sender<()>,
    tokio::task::JoinHandle<Result<()>>,
) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel::<()>();
    let stop_future = async move {
        let _ = stopped.await;
    };
    let server = tokio::spawn(serve_until(listener, app, stop_future, drain));
    (addr, stop, server)
}

/// Opens a connection and sends a bare GET for `path` on it.
async fn send_get(addr: std::net::SocketAddr, path: &str) -> TcpStream {
    let conn = TcpStream::connect(addr).await.unwrap();
    let request = format!("GET {path} HTTP/1.1\r\nhost: localhost\r\n\r\n");
    let mut sent = 0;
    while sent < request.len() {
        conn.writable().await.unwrap();
        match conn.try_write(&request.as_bytes()[sent..]) {
            Ok(n) => sent += n,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => panic!("sending the request: {e}"),
        }
    }
    conn
}

/// Reads from `conn` until what arrived satisfies `done` or the server closes.
async fn read_until(conn: &TcpStream, done: impl Fn(&str) -> bool) -> String {
    let mut got = Vec::new();
    let mut buf = [0u8; 1024];
    while !done(&String::from_utf8_lossy(&got)) {
        conn.readable().await.unwrap();
        match conn.try_read(&mut buf) {
            Ok(0) => break,
            Ok(n) => got.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => panic!("reading the response: {e}"),
        }
    }
    String::from_utf8_lossy(&got).into_owned()
}

#[tokio::test]
async fn a_stop_with_nothing_open_returns_at_once() {
    let (_, stop, server) = start(Router::new(), Duration::from_secs(60)).await;
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("returned without waiting out the drain")
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn a_request_in_flight_when_the_stop_comes_is_answered() {
    let started = Arc::new(Notify::new());
    let reached = started.clone();
    let app = Router::new().route(
        "/slow",
        axum::routing::get(move || {
            let reached = reached.clone();
            async move {
                reached.notify_one();
                tokio::time::sleep(Duration::from_millis(300)).await;
                "done"
            }
        }),
    );
    let (addr, stop, server) = start(app, Duration::from_secs(60)).await;
    let conn = send_get(addr, "/slow").await;
    started.notified().await;

    stop.send(()).unwrap();
    let answer = tokio::time::timeout(
        Duration::from_secs(5),
        read_until(&conn, |got| got.ends_with("done")),
    )
    .await
    .expect("answered");
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    assert!(answer.ends_with("done"), "{answer}");
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("returned once the request was answered, not after the drain")
        .unwrap()
        .unwrap();
}

/// What a client's GET /mcp is to the server: a response that has started and
/// whose body never ends. The stop gives it the drain and then returns anyway.
#[tokio::test]
async fn a_stream_that_never_ends_does_not_hold_the_stop_open() {
    let app = Router::new().route(
        "/stream",
        axum::routing::get(|| async {
            Body::from_stream(futures::stream::pending::<
                Result<axum::body::Bytes, std::convert::Infallible>,
            >())
        }),
    );
    let drain = Duration::from_millis(300);
    let (addr, stop, server) = start(app, drain).await;
    let conn = send_get(addr, "/stream").await;
    let head = read_until(&conn, |got| got.contains("\r\n\r\n")).await;
    assert!(
        head.starts_with("HTTP/1.1 200"),
        "the stream started: {head}"
    );

    let asked = Instant::now();
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(5), server)
        .await
        .expect("the stop returned with the stream still open")
        .unwrap()
        .unwrap();
    assert!(
        asked.elapsed() >= drain,
        "the open stream had its drain first ({:?})",
        asked.elapsed()
    );
}
