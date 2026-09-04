#![recursion_limit = "256"]
//! mnemos-mcp-http: HTTP host serving ALL rmcp streamable-HTTP services on
//! ONE port.
//!
//! | Path | Service |
//! |------|---------|
//! | `/mcp` | protocol tools ([`ProtocolTools`]: `recall`/`store`/`contradiction_check`/`consolidate`) |
//! | `/mcp/tools` | multi-tool ([`MnemosMcpTools`]: `mnemos_ingest`/`mnemos_recall`/`mnemos_reward`/`mnemos_consolidate`/`mnemos_stats`/`help`) |
//! | `/mcp/cli` | single-tool CLI ([`MnemosServer`]: `mnemos_cli`) |
//!
//! Wiring (grep-verified against `rmcp-3.2.0`
//! `src/transport/streamable_http_server/tower.rs`): [`StreamableHttpService`]
//! is built with `StreamableHttpService::new(service_factory, session_manager,
//! config)` where the factory is `Fn() -> Result<S, std::io::Error>`, and the
//! resulting service implements `tower-service`'s `Service<http::Request<B>>`.
//! Routing is a plain `hyper` (`server` + `http1` features only, no axum)
//! `service_fn` that matches the request path manually and forwards with
//! `tower::ServiceExt::oneshot`; anything else is a 404.
//!
//! Env: `MNEMOS_MCP_PORT` (default `4545`), `MNEMOS_MCP_HOST` (default
//! `127.0.0.1`), `MNEMOS_MCP_TOKEN` (optional bearer token for all `/mcp/*`).
//! Use [`serve`] to bind `HOST:PORT` and serve forever.
//!
//! [`MnemosServer`]: mnemos_mcp_server::MnemosServer
//! [`ProtocolTools`]: mnemos_mcp_protocol::ProtocolTools
//! [`StreamableHttpService`]: rmcp::transport::streamable_http_server::StreamableHttpService

use std::{
    convert::Infallible,
    net::SocketAddr,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use bytes::Bytes;
use http_body_util::{BodyExt as _, Full, combinators::BoxBody};
use mnemos_cli::Cli;
use mnemos_mcp_protocol::ProtocolTools;
use mnemos_mcp_server::MnemosServer;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use tower::ServiceExt as _;

/// Default port when `MNEMOS_MCP_PORT` is unset or unparsable.
pub const DEFAULT_MCP_PORT: u16 = 4545;

/// Default bind host when `MNEMOS_MCP_HOST` is unset or blank.
pub const DEFAULT_MCP_HOST: &str = "127.0.0.1";

/// Path serving the protocol tools ([`ProtocolTools`]).
///
/// [`ProtocolTools`]: mnemos_mcp_protocol::ProtocolTools
pub const PROTOCOL_PATH: &str = "/mcp";

/// Path serving the multi-tool MCP server ([`MnemosMcpTools`]).
///
/// [`MnemosMcpTools`]: mnemos_mcp_tools::MnemosMcpTools
pub const TOOLS_PATH: &str = "/mcp/tools";

/// Path serving the single-tool CLI ([`MnemosServer`]).
///
/// [`MnemosServer`]: mnemos_mcp_server::MnemosServer
pub const CLI_PATH: &str = "/mcp/cli";

/// Path for direct CLI RPC (`POST {"command": ...}`) against the running
/// daemon. Lets shell/`curl`/agents hit the persistent `Cli` without
/// spawning a new process per command.
pub const CLI_RPC_PATH: &str = "/cli";

/// Liveness probe for thin clients (`GET` → `{"status":"ok"}`).
pub const HEALTH_PATH: &str = "/health";

/// Which rmcp service a request path routes to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceKey {
    /// [`PROTOCOL_PATH`] → [`ProtocolTools`].
    ///
    /// [`ProtocolTools`]: mnemos_mcp_protocol::ProtocolTools
    Protocol,
    /// [`TOOLS_PATH`] → [`MnemosMcpTools`].
    ///
    /// [`MnemosMcpTools`]: mnemos_mcp_tools::MnemosMcpTools
    Tools,
    /// [`CLI_PATH`] → [`MnemosServer`].
    ///
    /// [`MnemosServer`]: mnemos_mcp_server::MnemosServer
    Cli,
}

/// Extract the service key from a request path (query strings and fragments
/// stripped). Returns `None` for unknown paths (the caller answers 404).
#[must_use]
pub fn route_for_path(path: &str) -> Option<ServiceKey> {
    let clean = path.split(['?', '#']).next().unwrap_or(path);
    match clean {
        PROTOCOL_PATH => Some(ServiceKey::Protocol),
        TOOLS_PATH => Some(ServiceKey::Tools),
        CLI_PATH => Some(ServiceKey::Cli),
        _ => None,
    }
}

/// Parse a port value with fallback to [`DEFAULT_MCP_PORT`].
#[must_use]
pub fn parse_port(raw: Option<&str>) -> u16 {
    raw.and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(DEFAULT_MCP_PORT)
}

/// Parse a host value with fallback to [`DEFAULT_MCP_HOST`].
#[must_use]
pub fn parse_host(raw: Option<&str>) -> String {
    raw.map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_MCP_HOST)
        .to_string()
}

/// Port from `MNEMOS_MCP_PORT` (default `4545`).
#[must_use]
pub fn mcp_port_from_env() -> u16 {
    parse_port(std::env::var("MNEMOS_MCP_PORT").ok().as_deref())
}

/// Bind host from `MNEMOS_MCP_HOST` (default `127.0.0.1`).
#[must_use]
pub fn mcp_host_from_env() -> String {
    parse_host(std::env::var("MNEMOS_MCP_HOST").ok().as_deref())
}

/// Bearer token from `MNEMOS_MCP_TOKEN` (default: open access).
///
/// Empty / unset → `None` (no auth). Set → `Some(token)` required as
/// `Authorization: Bearer <token>` on every `/mcp/*` and `/telemetry*` request.
#[must_use]
pub fn mcp_token_from_env() -> Option<String> {
    std::env::var("MNEMOS_MCP_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Whether `req` satisfies the bearer token requirement (`None` → always true).
#[must_use]
pub fn is_authorized(req: &hyper::Request<hyper::body::Incoming>, token: Option<&str>) -> bool {
    let Some(expected) = token.filter(|s| !s.is_empty()) else {
        return true;
    };
    let Some(header) = req
        .headers()
        .get(hyper::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    else {
        return false;
    };
    // Accept `Bearer <token>` (case-sensitive scheme per RFC 6750, but trim).
    let Some(suffix) = header.strip_prefix("Bearer ") else {
        return false;
    };
    suffix.trim() == expected
}

/// Response body shared by the rmcp services and the local 404.
type HttpBody = BoxBody<Bytes, Infallible>;

/// Plain 404 for paths other than [`PROTOCOL_PATH`] / [`CLI_PATH`].
fn not_found() -> hyper::Response<HttpBody> {
    hyper::Response::builder()
        .status(hyper::StatusCode::NOT_FOUND)
        .body(
            Full::new(Bytes::from_static(b"not found"))
                .boxed(),
        )
        .expect("static 404 response builds")
}

/// 401 for missing/invalid bearer token.
fn unauthorized() -> hyper::Response<HttpBody> {
    hyper::Response::builder()
        .status(hyper::StatusCode::UNAUTHORIZED)
        .header(hyper::header::WWW_AUTHENTICATE, "Bearer")
        .body(Full::new(Bytes::from_static(b"unauthorized")).boxed())
        .expect("static 401 response builds")
}

/// JSON response helper for telemetry.
fn json_response(value: serde_json::Value) -> hyper::Response<HttpBody> {
    hyper::Response::builder()
        .status(hyper::StatusCode::OK)
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(serde_json::to_vec(&value).unwrap_or_default())).boxed())
        .expect("telemetry json response builds")
}

/// Handle `GET /telemetry*` paths (dashboard poll). Returns `Some(response)` if handled.
fn handle_telemetry(path: &str, method: &hyper::Method) -> Option<hyper::Response<HttpBody>> {
    if *method != hyper::Method::GET {
        return Some(
            hyper::Response::builder()
                .status(hyper::StatusCode::METHOD_NOT_ALLOWED)
                .body(Full::new(Bytes::from_static(b"method not allowed")).boxed())
                .expect("405 builds"),
        );
    }
    let clean = path.split(['?', '#']).next().unwrap_or(path);
    match clean {
        "/telemetry" => Some(json_response(mnemos_telemetry::global().full_snapshot())),
        "/telemetry/diagnose" => Some(json_response(serde_json::to_value(mnemos_telemetry::global().diagnose(20)).unwrap_or_default())),
        "/telemetry/counters" => Some(json_response(serde_json::to_value(mnemos_telemetry::global().counters_snapshot()).unwrap_or_default())),
        "/telemetry/events" => Some(json_response(serde_json::to_value(mnemos_telemetry::global().snapshot()).unwrap_or_default())),
        "/telemetry/weights" => Some(json_response(serde_json::to_value(mnemos_telemetry::global().weights_history_snapshot()).unwrap_or_default())),
        "/telemetry/system" => Some(json_response(serde_json::to_value(mnemos_telemetry::global().system_history_snapshot()).unwrap_or_default())),
        _ if clean.starts_with("/telemetry") => Some(not_found()),
        _ => None,
    }
}

/// Handle `GET /health` and `POST /cli` against the persistent daemon
/// `Cli`. Returns `Some(response)` if the path is a local route.
async fn handle_local(
    cli: &Arc<Cli>,
    path: &str,
    method: &hyper::Method,
    req: hyper::Request<hyper::body::Incoming>,
) -> Option<hyper::Response<HttpBody>> {
    let clean = path.split(['?', '#']).next().unwrap_or(path);
    if clean == HEALTH_PATH {
        return Some(if *method == hyper::Method::GET {
            json_response(serde_json::json!({"status": "ok", "service": "engram-daemon"}))
        } else {
            hyper::Response::builder()
                .status(hyper::StatusCode::METHOD_NOT_ALLOWED)
                .body(Full::new(Bytes::from_static(b"method not allowed")).boxed())
                .expect("405 builds")
        });
    }
    if clean != CLI_RPC_PATH {
        return None;
    }
    if *method != hyper::Method::POST {
        return Some(
            hyper::Response::builder()
                .status(hyper::StatusCode::METHOD_NOT_ALLOWED)
                .body(Full::new(Bytes::from_static(b"method not allowed")).boxed())
                .expect("405 builds"),
        );
    }
    let body = match http_body_util::BodyExt::collect(req.into_body()).await {
        Ok(b) => b.to_bytes(),
        Err(e) => {
            return Some(json_response(serde_json::json!({"ok": false, "error": format!("read body: {e}")})));
        }
    };
    Some(dispatch_cli_rpc(cli, &body).await)
}

/// Execute one CLI command against the running daemon's `Cli`.
///
/// Request: `{"command": "ingest"|"recall"|"reward"|"consolidate"|"stats",
/// "text"?, "query"?, "limit"?, "attributions"?, "score"?, "recall_id"?, "aggressive"?}`.
/// Always HTTP 200 with `{"ok": true, "data": ...}` or `{"ok": false, "error": ...}`;
/// failures are recorded via telemetry (`mnemos-mcp-http` / `cli_rpc`).
async fn dispatch_cli_rpc(cli: &Arc<Cli>, body: &[u8]) -> hyper::Response<HttpBody> {
    let req: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            mnemos_telemetry::global().record("mnemos-mcp-http", "cli_rpc", false, &format!("bad json: {e}"));
            return json_response(serde_json::json!({"ok": false, "error": format!("bad json: {e}")}));
        }
    };
    let command = req.get("command").and_then(|v| v.as_str()).unwrap_or("");
    let out: Result<serde_json::Value, String> = match command {
        "ingest" => match req.get("text").and_then(|v| v.as_str()) {
            Some(text) => cli.ingest(text).await.map(|id| serde_json::json!({"engram_id": id})).map_err(|e| e.to_string()),
            None => Err("ingest needs {text}".to_string()),
        },
        "recall" => {
            let query = req.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let limit = req.get("limit").and_then(serde_json::Value::as_u64).unwrap_or(5) as usize;
            match cli.recall(query, limit).await {
                Ok(results) => {
                    let recall_id = cli.last_recall_id().await;
                    serde_json::to_value(&serde_json::json!({"results": results, "recall_id": recall_id})).map_err(|e| e.to_string())
                }
                Err(e) => Err(e.to_string()),
            }
        }
        "reward" => {
            let score = req.get("score").and_then(serde_json::Value::as_f64).unwrap_or(0.0);
            let res = match req.get("recall_id").and_then(serde_json::Value::as_u64) {
                Some(id) => cli.reward_with_id(id, score).await,
                None => {
                    let attr: Vec<f64> = req.get("attributions").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
                    cli.reward(&attr, score).await
                }
            };
            res.map(|()| serde_json::json!({"ok": true})).map_err(|e| e.to_string())
        }
        "consolidate" => {
            let aggressive = req.get("aggressive").and_then(serde_json::Value::as_bool).unwrap_or(false);
            cli.consolidate_aggressive(aggressive).await.map(|r| serde_json::to_value(&r).unwrap_or_default()).map_err(|e| e.to_string())
        }
        "stats" => cli.stats().await.map(|s| serde_json::to_value(&s).unwrap_or_default()).map_err(|e| e.to_string()),
        "setup" => {
            // Dimension comes from env (EMBEDDING_DIM), never from the request.
            let dimension = mnemos_core::embedding_dim_from_env();
            cli.setup_vector_index(dimension).await.map(|s| serde_json::json!({"dimension": dimension, "message": s})).map_err(|e| e.to_string())
        }
        other => Err(format!("unknown command {other:?} (ingest|recall|reward|consolidate|stats|setup)")),
    };
    match out {
        Ok(data) => json_response(serde_json::json!({"ok": true, "data": data})),
        Err(error) => {
            mnemos_telemetry::global().record("mnemos-mcp-http", "cli_rpc", false, &error);
            json_response(serde_json::json!({"ok": false, "error": error}))
        }
    }
}

/// One request against the persistent daemon: auth → local routes
/// (`/health`, `/cli`, `/telemetry*`) → rmcp services → 404.
async fn handle_request(
    protocol_service: StreamableHttpService<Arc<ProtocolTools>, LocalSessionManager>,
    tools_service: StreamableHttpService<mnemos_mcp_tools::MnemosMcpTools, LocalSessionManager>,
    cli_service: StreamableHttpService<mnemos_mcp_server::MnemosServer, LocalSessionManager>,
    rpc_cli: Arc<Cli>,
    token: Option<String>,
    req: hyper::Request<hyper::body::Incoming>,
) -> Result<hyper::Response<HttpBody>, Infallible> {
    if !is_authorized(&req, token.as_deref()) {
        return Ok(unauthorized());
    }
    // Local daemon routes (/health, /cli) take precedence over rmcp
    // services; they hit the persistent Cli.
    let path = req.uri().path().to_string();
    let clean = path.split(['?', '#']).next().unwrap_or(&path);
    if clean == HEALTH_PATH || clean == CLI_RPC_PATH {
        let method = req.method().clone();
        if let Some(resp) = handle_local(&rpc_cli, &path, &method, req).await {
            return Ok(resp);
        }
        return Ok(not_found());
    }
    if let Some(resp) = handle_telemetry(req.uri().path(), req.method()) {
        return Ok(resp);
    }
    match route_for_path(req.uri().path()) {
        Some(ServiceKey::Protocol) => match protocol_service.oneshot(req).await {
            Ok(response) => Ok(response),
            Err(never) => match never {},
        },
        Some(ServiceKey::Tools) => match tools_service.oneshot(req).await {
            Ok(response) => Ok(response),
            Err(never) => match never {},
        },
        Some(ServiceKey::Cli) => match cli_service.oneshot(req).await {
            Ok(response) => Ok(response),
            Err(never) => match never {},
        },
        None => Ok(not_found()),
    }
}

/// Record a serve-side failure via telemetry (stderr logging stays inline).
fn record_serve_error(detail: &str) {
    mnemos_telemetry::global().record("mnemos-mcp-http", "serve", false, detail);
}

/// Minimal Tokio ↔ hyper IO adapter (avoids a `hyper-util` dependency).
struct TokioIo<T>(T);

impl<T> TokioIo<T> {
    fn new(inner: T) -> Self {
        Self(inner)
    }
}

impl<T: tokio::io::AsyncRead + Unpin> hyper::rt::Read for TokioIo<T> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        mut buf: hyper::rt::ReadBufCursor<'_>,
    ) -> Poll<std::io::Result<()>> {
        // Mirrors `hyper-util`'s `TokioIo`: fill the cursor, then advance it
        // by the number of bytes read.
        let filled = unsafe {
            let mut read_buf = tokio::io::ReadBuf::uninit(buf.as_mut());
            match tokio::io::AsyncRead::poll_read(
                Pin::new(&mut self.get_mut().0),
                cx,
                &mut read_buf,
            ) {
                Poll::Ready(Ok(())) => read_buf.filled().len(),
                Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                Poll::Pending => return Poll::Pending,
            }
        };
        unsafe {
            buf.advance(filled);
        }
        Poll::Ready(Ok(()))
    }
}

impl<T: tokio::io::AsyncWrite + Unpin> hyper::rt::Write for TokioIo<T> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        tokio::io::AsyncWrite::poll_write(Pin::new(&mut self.get_mut().0), cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        tokio::io::AsyncWrite::poll_flush(Pin::new(&mut self.get_mut().0), cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        tokio::io::AsyncWrite::poll_shutdown(Pin::new(&mut self.get_mut().0), cx)
    }
}

/// Serve ALL surfaces on one port: rmcp MCP (`/mcp`, `/mcp/tools`,
/// `/mcp/cli`), telemetry (`/telemetry*`), daemon CLI RPC (`POST /cli`)
/// and liveness (`GET /health`).
///
/// `HOST`/`PORT` come from `MNEMOS_MCP_HOST` / `MNEMOS_MCP_PORT`. The
/// `protocol` tools answer on `/mcp`; the single-tool CLI (built fresh per
/// connection from `cli`) answers on `/mcp/cli`. Runs until the listener
/// fails; per-connection failures are logged to stderr.
///
/// Each factory closure returns a fresh instance per connection: a cloned
/// `Arc<ProtocolTools>` on the protocol side (`Arc<T: ServerHandler>`
/// implements `ServerHandler` in rmcp, so no `Clone` bound on
/// [`ProtocolTools`] itself is needed) and `MnemosServer::new` on shared
/// `Arc<Cli>` clones on the CLI side.
///
/// [`ProtocolTools`]: mnemos_mcp_protocol::ProtocolTools
///
/// # Errors
///
/// Returns [`mnemos_core::MnemosError::Http`] if the bind address is invalid,
/// the listener fails to bind, or accepting connections fails. Failures are
/// also recorded via telemetry (`mnemos-mcp-http` / `serve`).
pub async fn serve(protocol: ProtocolTools, cli: Arc<Cli>) -> mnemos_core::Result<()> {
    let host = mcp_host_from_env();
    let port = mcp_port_from_env();
    let addr: SocketAddr = format!("{host}:{port}").parse().map_err(|err| {
        let failure =
            mnemos_core::MnemosError::Http(format!("invalid bind addr {host}:{port}: {err}"));
        record_serve_error(&failure.to_string());
        failure
    })?;
    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|err| {
        let failure = mnemos_core::MnemosError::Http(format!("bind {addr}: {err}"));
        record_serve_error(&failure.to_string());
        failure
    })?;
    eprintln!("mnemos-mcp-http listening on http://{addr}{PROTOCOL_PATH} (protocol tools)");
    eprintln!("mnemos-mcp-http listening on http://{addr}{TOOLS_PATH} (multi-tool)");
    eprintln!("mnemos-mcp-http listening on http://{addr}{CLI_PATH} (cli single-tool)");
    eprintln!("mnemos-mcp-http listening on http://{addr}{CLI_RPC_PATH} (daemon CLI RPC) + {HEALTH_PATH}");
    let token = mcp_token_from_env();
    if token.is_some() {
        eprintln!("mnemos-mcp-http auth: bearer token required for /mcp/* (MNEMOS_MCP_TOKEN set)");
    }

    let protocol = Arc::new(protocol);
    let protocol_service = StreamableHttpService::new(
        {
            let protocol = Arc::clone(&protocol);
            move || Ok(Arc::clone(&protocol))
        },
        Arc::new(LocalSessionManager::default()),
        rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default(),
    );
    let tools_service = StreamableHttpService::new(
        {
            let cli = Arc::clone(&cli);
            move || Ok(mnemos_mcp_tools::MnemosMcpTools::new(Arc::clone(&cli)))
        },
        Arc::new(LocalSessionManager::default()),
        rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default(),
    );
    let cli_service = StreamableHttpService::new(
        {
            let cli = Arc::clone(&cli);
            move || Ok(MnemosServer::new(Arc::clone(&cli)))
        },
        Arc::new(LocalSessionManager::default()),
        rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default(),
    );

    loop {
        let (stream, peer) = listener.accept().await.map_err(|err| {
            let failure = mnemos_core::MnemosError::Http(format!("accept: {err}"));
            record_serve_error(&failure.to_string());
            failure
        })?;
        let protocol_service = protocol_service.clone();
        let tools_service = tools_service.clone();
        let cli_service = cli_service.clone();
        let rpc_cli = Arc::clone(&cli);
        let token = token.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let router = hyper::service::service_fn(
                move |req: hyper::Request<hyper::body::Incoming>| {
                    let protocol_service = protocol_service.clone();
                    let tools_service = tools_service.clone();
                    let cli_service = cli_service.clone();
                    let rpc_cli = Arc::clone(&rpc_cli);
                    let token = token.clone();
                    handle_request(
                        protocol_service,
                        tools_service,
                        cli_service,
                        rpc_cli,
                        token,
                        req,
                    )
                },
            );
            if let Err(err) = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, router)
                .await
            {
                eprintln!("mnemos-mcp-http connection from {peer} failed: {err}");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_all_service_paths() {
        assert_eq!(route_for_path("/mcp"), Some(ServiceKey::Protocol));
        assert_eq!(route_for_path("/mcp/tools"), Some(ServiceKey::Tools));
        assert_eq!(route_for_path("/mcp/cli"), Some(ServiceKey::Cli));
    }

    #[test]
    fn route_strips_query_and_fragment() {
        assert_eq!(route_for_path("/mcp?x=1"), Some(ServiceKey::Protocol));
        assert_eq!(route_for_path("/mcp/tools?v=2"), Some(ServiceKey::Tools));
        assert_eq!(route_for_path("/mcp/cli#frag"), Some(ServiceKey::Cli));
    }

    #[test]
    fn route_rejects_unknown_paths() {
        for path in [
            "",
            "/",
            "/mcp/",
            "/mcp/tools/",
            "/mcp/cli/",
            "/mcp/other",
            "/other",
            "/MCP",
        ] {
            assert_eq!(route_for_path(path), None, "path `{path}` must 404");
        }
    }

    #[test]
    fn port_parsing_defaults_and_trims() {
        assert_eq!(parse_port(None), DEFAULT_MCP_PORT);
        assert_eq!(parse_port(Some("")), DEFAULT_MCP_PORT);
        assert_eq!(parse_port(Some("not-a-port")), DEFAULT_MCP_PORT);
        assert_eq!(parse_port(Some("99999")), DEFAULT_MCP_PORT);
        assert_eq!(parse_port(Some("4545")), 4545);
        assert_eq!(parse_port(Some(" 8080 ")), 8080);
    }

    #[test]
    fn host_parsing_defaults_on_blank() {
        assert_eq!(parse_host(None), DEFAULT_MCP_HOST);
        assert_eq!(parse_host(Some("")), DEFAULT_MCP_HOST);
        assert_eq!(parse_host(Some("   ")), DEFAULT_MCP_HOST);
        assert_eq!(parse_host(Some("0.0.0.0")), "0.0.0.0");
    }

    #[test]
    fn not_found_is_404() {
        let response = not_found();
        assert_eq!(response.status(), hyper::StatusCode::NOT_FOUND);
    }

    /// Manual smoke test: `serve` binds a real port and runs forever, and
    /// needs a live `ProtocolTools` + `Cli` (pipelines + storage), so there
    /// is nothing to construct here. Run explicitly via
    /// `cargo test -p mnemos-mcp-http -- --ignored` with a binary harness.
    #[ignore = "binds a real port and serves forever; requires live backends"]
    #[tokio::test]
    async fn serve_http_smoke() {
        // Intentionally empty: `serve` never returns while healthy, so this
        // documents the manual harness rather than executing it in CI.
    }
}
