//! Tests for the connection close / deregistration path.
//!
//! These tests guard against handle leaks when many connections are closed in succession and
//! make sure that in-flight readiness operations are torn down (their wakers released) instead
//! of being left waiting on a closed handle.

#![cfg(unix)]

use std::io;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Wake};
use std::time::{Duration, Instant};

use async_io::Async;
use futures_lite::{future, prelude::*};

/// Number of connections opened and closed in a row.
const CONNECTIONS: usize = 1000;

/// How long to wait for the descriptor count to fall back before declaring a leak.
const SETTLE: Duration = Duration::from_secs(3);

/// Upper bound for closing a thousand connections. Deregistration is a quick syscall with only a
/// few bounded retries, so a full teardown must stay well below this even on a slow CI machine.
const CLOSE_TIME_BUDGET: Duration = Duration::from_secs(5);

/// Counts the file descriptors currently open by this process.
fn open_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("read /proc/self/fd")
        .count()
}

/// Repeatedly measures the open descriptor count and waits until it drops back to the baseline
/// (or the timeout elapses). Returns the last measured count.
fn wait_for_fd_count(baseline: usize) -> usize {
    let deadline = Instant::now() + SETTLE;
    let mut last = open_fd_count();
    while last > baseline && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
        last = open_fd_count();
    }
    last
}

/// Opening and immediately dropping a thousand TCP connections must not leak descriptors.
#[test]
fn close_thousand_tcp_connections_releases_handles() -> io::Result<()> {
    let listener = Async::<std::net::TcpListener>::bind(([127, 0, 0, 1], 0))?;
    let addr = listener.get_ref().local_addr()?;

    // Give the reactor thread a moment to settle, then record the steady-state baseline.
    std::thread::sleep(Duration::from_millis(50));
    let baseline = open_fd_count();

    for _ in 0..CONNECTIONS {
        let client = future::block_on(Async::<std::net::TcpStream>::connect(addr))?;
        let (incoming, _peer) = future::block_on(listener.accept())?;
        drop(client);
        drop(incoming);
    }

    let after = wait_for_fd_count(baseline);
    assert!(
        after <= baseline,
        "descriptor count did not fall back after closing {CONNECTIONS} connections: \
         baseline={baseline}, after={after}"
    );
    Ok(())
}

/// Same check using Unix-domain socket pairs, which do not depend on the network stack.
#[test]
fn close_thousand_uds_pairs_releases_handles() -> io::Result<()> {
    // Touch the reactor once so its own descriptors are part of the baseline.
    let _ = Async::<UnixStream>::pair()?;
    std::thread::sleep(Duration::from_millis(50));
    let baseline = open_fd_count();

    for _ in 0..CONNECTIONS {
        let (a, b) = Async::<UnixStream>::pair()?;
        drop(a);
        drop(b);
    }

    let after = wait_for_fd_count(baseline);
    assert!(
        after <= baseline,
        "descriptor count did not fall back after closing {CONNECTIONS} socket pairs: \
         baseline={baseline}, after={after}"
    );
    Ok(())
}

/// Closing a thousand connections in a row must complete in bounded time; teardown may retry
/// transient failures but is not allowed to block indefinitely on an in-flight operation.
#[test]
fn closing_many_connections_is_bounded_in_time() -> io::Result<()> {
    let conns: Vec<_> = (0..CONNECTIONS)
        .map(|_| Async::<UnixStream>::pair())
        .collect::<io::Result<Vec<_>>>()?;

    let start = Instant::now();
    drop(conns);
    let elapsed = start.elapsed();

    assert!(
        elapsed < CLOSE_TIME_BUDGET,
        "closing {CONNECTIONS} connections took {elapsed:?}, exceeding the {CLOSE_TIME_BUDGET:?} budget"
    );
    Ok(())
}

/// A waker that counts how many times it has been woken.
#[derive(Default)]
struct CountingWaker {
    wakes: AtomicUsize,
}

impl Wake for CountingWaker {
    fn wake(self: Arc<Self>) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.wakes.fetch_add(1, Ordering::SeqCst);
    }
}

/// Closing a connection while a readiness future is still registered must end that in-flight
/// operation by waking the waiting task, rather than leaving the waker parked on a dead handle.
#[test]
fn closing_connection_wakes_in_flight_readiness() -> io::Result<()> {
    let (a, mut b) = Async::<UnixStream>::pair()?;

    let counting = Arc::new(CountingWaker::default());
    let waker = counting.clone().into();
    let mut cx = Context::from_waker(&waker);

    // Register an in-flight readability wait through the `poll_readable` path, which stores the
    // task's waker directly on the source. It must be pending because nothing was written.
    assert!(a.poll_readable(&mut cx).is_pending());
    assert_eq!(counting.wakes.load(Ordering::SeqCst), 0);

    // Closing the source must end the in-flight operation and wake the waiting task instead of
    // leaving it parked on a closed descriptor.
    drop(a);
    assert_eq!(
        counting.wakes.load(Ordering::SeqCst),
        1,
        "in-flight readiness wait was not woken when the connection was closed"
    );

    // The peer stays usable and observes end-of-file now that its peer is gone.
    let mut buf = [0u8; 16];
    let n = future::block_on(b.read(&mut buf))?;
    assert_eq!(n, 0);
    Ok(())
}

/// `into_inner()` deregisters the source and hands the still-open handle back to the caller.
#[test]
fn into_inner_keeps_handle_and_deregisters() -> io::Result<()> {
    let (a, _b) = Async::<UnixStream>::pair()?;
    let raw: RawFd = a.as_raw_fd();
    let inner = a.into_inner()?;
    assert_eq!(inner.as_raw_fd(), raw);
    drop(inner);
    Ok(())
}
