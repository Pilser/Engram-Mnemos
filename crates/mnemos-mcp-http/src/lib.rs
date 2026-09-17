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

/// Tool catalog for humans and agents (`GET` → every MCP tool on every
/// surface with its endpoint). No DB access, no auth bypass (same bearer
/// rule as everything else).
pub const TOOLS_LIST_PATH: &str = "/tools";

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

/// Static catalog of every tool on every surface (for `GET /tools`).
#[must_use]
pub fn tools_catalog() -> serde_json::Value {
    serde_json::json!([
        {"endpoint": "/mcp", "transport": "mcp-streamable-http", "tools": [
            {"name": "recall", "params": "query*, limit?=5, type?=auto, since?=all"},
            {"name": "store", "params": "content*, importance?=auto, type?=auto"},
            {"name": "contradiction_check", "params": "claim*"},
            {"name": "consolidate", "params": "aggressive?=false"}
        ]},
        {"endpoint": "/mcp/tools", "transport": "mcp-streamable-http", "tools": [
            {"name": "engram_ingest", "params": "text*, prev_id?, seq_pos? (sequential TemporalSequence)"},
            {"name": "engram_recall", "params": "query*, limit?=10, follow_seq?, seq_depth?, seq_dir? (sequential walk)"},
            {"name": "engram_reward", "params": "attributions?, score*, recall_id?"},
            {"name": "engram_consolidate", "params": "none"},
            {"name": "engram_stats", "params": "none"},
            {"name": "engram_status", "params": "none"},
            {"name": "help", "params": "tool?"}
        ]},
        {"endpoint": "/mcp/cli", "transport": "mcp-streamable-http", "tools": [
            {"name": "engram_cli", "params": "command*, text?, query?, limit?, prev_id?, seq_pos?, follow_seq?, seq_depth?, seq_dir?, attributions?, score?, recall_id?, args? (commands: help, ingest, recall, reward, consolidate, stats)"}
        ]}
    ])
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

/// Parse a `k=v&k2=v2` query string into a map (no decoding — dashboard
/// params are `[A-Za-z0-9_-]`).
fn parse_query(query: Option<&str>) -> std::collections::HashMap<String, String> {
    query
        .unwrap_or("")
        .split('&')
        .filter_map(|pair| {
            let mut parts = pair.splitn(2, '=');
            Some((parts.next()?.to_string(), parts.next().unwrap_or("").to_string()))
        })
        .collect()
}

/// Handle `GET /telemetry*` paths (dashboard poll) plus file management
/// (`GET /telemetry/files`, `GET /telemetry/file?date=&limit=&offset=&ok=`,
/// `DELETE /telemetry/file?date=`). Returns `Some(response)` if handled.
fn handle_telemetry(
    uri: &hyper::Uri,
    method: &hyper::Method,
) -> Option<hyper::Response<HttpBody>> {
    let path = uri.path();
    let clean = path.split(['?', '#']).next().unwrap_or(path);
    if !clean.starts_with("/telemetry") {
        return None;
    }
    let query = parse_query(uri.query());
    let tele = mnemos_telemetry::global();
    // File deletion (dashboard file manager).
    if clean == "/telemetry/file" && *method == hyper::Method::DELETE {
        let date = query.get("date").map(String::as_str).unwrap_or("");
        let deleted = tele.delete_file(date);
        let status = if deleted {
            hyper::StatusCode::OK
        } else {
            hyper::StatusCode::NOT_FOUND
        };
        return Some(
            hyper::Response::builder()
                .status(status)
                .header(hyper::header::CONTENT_TYPE, "application/json")
                .body(
                    Full::new(
                        Bytes::from(
                            serde_json::to_vec(&serde_json::json!({
                                "deleted": deleted,
                                "date": date,
                            }))
                            .unwrap_or_default(),
                        ),
                    )
                    .boxed(),
                )
                .expect("delete response builds"),
        );
    }
    if *method != hyper::Method::GET {
        return Some(
            hyper::Response::builder()
                .status(hyper::StatusCode::METHOD_NOT_ALLOWED)
                .body(Full::new(Bytes::from_static(b"method not allowed")).boxed())
                .expect("405 builds"),
        );
    }
    match clean {
        "/telemetry" => Some(json_response(tele.full_snapshot())),
        "/telemetry/diagnose" => Some(json_response(serde_json::to_value(tele.diagnose(20)).unwrap_or_default())),
        "/telemetry/counters" => Some(json_response(serde_json::to_value(tele.counters_snapshot()).unwrap_or_default())),
        "/telemetry/events" => Some(json_response(serde_json::to_value(tele.snapshot()).unwrap_or_default())),
        "/telemetry/weights" => Some(json_response(serde_json::to_value(tele.weights_history_snapshot()).unwrap_or_default())),
        "/telemetry/system" => Some(json_response(serde_json::to_value(tele.system_history_snapshot()).unwrap_or_default())),
        "/telemetry/files" => Some(json_response(serde_json::to_value(tele.telemetry_files()).unwrap_or_default())),
        "/telemetry/file" => {
            let date = query.get("date").map(String::as_str).unwrap_or("");
            let limit = query
                .get("limit")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(200);
            let offset = query
                .get("offset")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);
            let ok_only = query.get("ok").map(|s| s == "true" || s == "1");
            match tele.read_file(date, limit, offset, ok_only) {
                Some(events) => Some(json_response(
                    serde_json::to_value(&events).unwrap_or_default(),
                )),
                None => Some(not_found()),
            }
        }
        _ => Some(not_found()),
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
            // Wake check: must not touch the DB.
            json_response(serde_json::json!({"ok": true, "data": {"status": "up"}}))
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

/// Read a string field from a request object.
fn get_str<'a>(v: &'a serde_json::Value, k: &str) -> Option<&'a str> {
    v.get(k).and_then(serde_json::Value::as_str)
}

/// Read an unsigned integer field (number or numeric string).
fn get_u64(v: &serde_json::Value, k: &str) -> Option<u64> {
    v.get(k).and_then(|x| {
        x.as_u64()
            .or_else(|| x.as_i64().and_then(|i| u64::try_from(i).ok()))
            .or_else(|| x.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
    })
}

/// Read a float field (number or numeric string).
fn get_f64(v: &serde_json::Value, k: &str) -> Option<f64> {
    v.get(k).and_then(|x| {
        x.as_f64()
            .or_else(|| x.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
    })
}

/// Read a boolean field (bool or `1/true/yes/on`).
fn get_bool(v: &serde_json::Value, k: &str) -> Option<bool> {
    v.get(k).and_then(|x| {
        x.as_bool().or_else(|| {
            x.as_str().map(|s| {
                matches!(
                    s.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
        })
    })
}

/// Read a float array field (JSON array or comma-separated string).
fn get_f64_vec(v: &serde_json::Value, k: &str) -> Vec<f64> {
    match v.get(k) {
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(|x| {
                x.as_f64()
                    .or_else(|| x.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
            })
            .collect(),
        Some(serde_json::Value::String(s)) => s
            .split(',')
            .filter_map(|p| p.trim().parse::<f64>().ok())
            .collect(),
        _ => Vec::new(),
    }
}

/// Coerce a GET query string into a request object (all values as strings;
/// the typed getters above coerce them).
fn query_to_json(query: Option<&str>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in parse_query(query) {
        map.insert(k, serde_json::json!(v));
    }
    serde_json::Value::Object(map)
}

/// Execute one CLI command against the running daemon's `Cli`, returning the
/// command's structured output or `(http_status, error_message)`.
///
/// Mirrors the CLI exactly: same defaults, same result keys, same messages.
async fn execute_cli_value(
    cli: &Arc<Cli>,
    req: &serde_json::Value,
) -> Result<serde_json::Value, (u16, String)> {
    let command = get_str(req, "command").unwrap_or("");
    match command {
        "ingest" => {
            let text = get_str(req, "text")
                .ok_or_else(|| (400, "ingest needs {text}".to_string()))?;
            if text.trim().is_empty() {
                return Err((
                    400,
                    "ingest needs text: engram ingest <text...>".to_string(),
                ));
            }
            let prev_id = get_u64(req, "prev_id").or_else(|| get_u64(req, "seq"));
            let seq_pos = get_u64(req, "seq_pos")
                .or_else(|| get_u64(req, "pos"))
                .map(|v| v as i64);
            let res = if let Some(pid) = prev_id {
                cli.ingest_sequential(text, Some(pid), seq_pos).await
            } else {
                cli.ingest(text).await
            };
            res.map(|id| serde_json::json!({"engram_id": id}))
                .map_err(|e| (500, e.to_string()))
        }
        "recall" => {
            if let Some(sid) = get_u64(req, "follow_seq") {
                let depth = get_u64(req, "depth")
                    .or_else(|| get_u64(req, "seq_depth"))
                    .unwrap_or(10) as usize;
                let dir = get_str(req, "dir")
                    .or_else(|| get_str(req, "seq_dir"))
                    .unwrap_or("down");
                if !matches!(dir, "up" | "down" | "both") {
                    return Err((400, "--dir must be up|down|both".to_string()));
                }
                return cli
                    .follow_sequence(sid, depth, dir)
                    .await
                    .map(|chain| {
                        serde_json::json!({
                            "sequential_chain": chain,
                            "start_id": sid,
                            "depth": depth,
                            "dir": dir,
                        })
                    })
                    .map_err(|e| (500, e.to_string()));
            }
            let query = get_str(req, "query")
                .ok_or_else(|| (400, "recall needs a query: engram recall <query...> [--limit N]".to_string()))?;
            if query.trim().is_empty() {
                return Err((
                    400,
                    "recall needs a query: engram recall <query...> [--limit N]".to_string(),
                ));
            }
            let limit = get_u64(req, "limit").unwrap_or(5) as usize;
            let results = cli.recall(query, limit).await.map_err(|e| (500, e.to_string()))?;
            if results.is_empty() {
                return Ok(serde_json::json!({
                    "results": [],
                    "recall_id": serde_json::Value::Null,
                    "message": "I don't know",
                }));
            }
            let recall_id = cli.last_recall_id().await;
            Ok(serde_json::json!({"results": results, "recall_id": recall_id}))
        }
        "reward" => {
            let score = get_f64(req, "score")
                .ok_or_else(|| (400, "reward needs a score: engram reward <score> [attributions csv]".to_string()))?;
            let res = match get_u64(req, "recall_id") {
                Some(id) => cli.reward_with_id(id, score).await,
                None => {
                    let attr = get_f64_vec(req, "attributions");
                    cli.reward(&attr, score).await
                }
            };
            res.map(|()| serde_json::json!({"applied": true}))
                .map_err(|e| (500, e.to_string()))
        }
        "consolidate" => {
            let aggressive = get_bool(req, "aggressive").unwrap_or(false);
            cli.consolidate_aggressive(aggressive)
                .await
                .map(|r| serde_json::to_value(&r).unwrap_or_default())
                .map_err(|e| (500, e.to_string()))
        }
        "stats" => cli
            .stats()
            .await
            .map(|s| serde_json::to_value(&s).unwrap_or_default())
            .map_err(|e| (500, e.to_string())),
        "status" => match cli.stats().await {
            Ok(stats) => Ok(serde_json::json!({
                "storage": {"ok": true, "stats": stats},
                "learning": learning_summary(),
                "embedding": {"ok": true, "note": "daemon: live ping via provider"},
                "llm": {"ok": true, "note": "daemon: live ping via provider"},
            })),
            Err(e) => Err((500, e.to_string())),
        },
        "setup" => {
            // Dimension comes from env (EMBEDDING_DIM), never from the request.
            let dimension = mnemos_core::embedding_dim_from_env();
            cli.setup_vector_index(dimension)
                .await
                .map(|s| serde_json::json!({"dimension": dimension, "message": s}))
                .map_err(|e| (500, e.to_string()))
        }
        "train-reranker" => {
            let log_path = std::env::var("MNEMOS_FEEDBACK_LOG")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "./data/helix/feedback.jsonl".to_string());
            let model_path = std::env::var("MNEMOS_RERANKER_MODEL")
                .ok()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "./data/helix/reranker.json".to_string());
            mnemos_reranker::train_from_log(&log_path, &model_path).map_err(|e| (400, e))
        }
        other => Err((
            400,
            format!(
                "unknown command {other:?} (ingest|recall|reward|consolidate|setup|stats|status|train-reranker)"
            ),
        )),
    }
}

/// `POST /cli` — command name in the body's `command` field. Always HTTP 200
/// with `{"ok": true|false, ...}` (legacy behaviour, kept for compatibility).
async fn dispatch_cli_rpc(cli: &Arc<Cli>, body: &[u8]) -> hyper::Response<HttpBody> {
    let req: serde_json::Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            mnemos_telemetry::global().record(
                "mnemos-mcp-http",
                "cli_rpc",
                false,
                &format!("bad json: {e}"),
            );
            return json_response(
                serde_json::json!({"ok": false, "error": format!("bad json: {e}")}),
            );
        }
    };
    match execute_cli_value(cli, &req).await {
        Ok(data) => json_response(serde_json::json!({"ok": true, "data": data})),
        Err((_status, error)) => {
            mnemos_telemetry::global().record("mnemos-mcp-http", "cli_rpc", false, &error);
            json_response(serde_json::json!({"ok": false, "error": error}))
        }
    }
}

/// Map a mirror path to its CLI command (callable commands only).
///
/// Process modes (`serve`, `mcp-server`, `mcp-tools`) are not RPCs and are
/// intentionally absent. `/help` is the mirror of `engram --help`.
#[must_use]
pub fn mirror_command(path: &str) -> Option<&'static str> {
    match path {
        "/ingest" => Some("ingest"),
        "/recall" => Some("recall"),
        "/reward" => Some("reward"),
        "/consolidate" => Some("consolidate"),
        "/setup" => Some("setup"),
        "/stats" => Some("stats"),
        "/status" => Some("status"),
        "/train-reranker" => Some("train-reranker"),
        "/help" => Some("help"),
        _ => None,
    }
}

/// Machine-readable mirror of `engram --help` (endpoints + fields).
fn mirror_help() -> serde_json::Value {
    serde_json::json!({
        "usage": "engram <command> [args]  |  HTTP: <METHOD> http://<host>:4545/<command>",
        "capability": "episodic memory with optional sequential story chains (TemporalSequence); every recall must be rewarded via reward (use recall_id from recall); recall answers \"I don't know\" when nothing is relevant; learning is automatic (just recall+reward)",
        "response": "{\"ok\": true, \"data\": <cli-json>} | {\"ok\": false, \"error\": \"<cli message>\"}",
        "commands": {
            "ingest": {"methods": ["GET","POST"], "fields": {"text*": "string", "prev_id?": "u64 (== --seq)", "seq_pos?": "i64 (== --seq-pos)"}, "returns": {"engram_id": "u64"}},
            "recall": {"methods": ["GET","POST"], "fields": {"query*": "string", "limit?": "usize=5", "follow_seq?": "u64 (== --follow-seq)", "depth?": "usize=10 (== --depth)", "dir?": "up|down|both=down (== --dir)"}, "returns": {"results": "[...]", "recall_id": "u64|null"} },
            "reward": {"methods": ["GET","POST"], "fields": {"score*": "f64 -1.0..1.0", "recall_id?": "u64", "attributions?": "[f64] or csv"}, "returns": {"applied": true}},
            "consolidate": {"methods": ["GET","POST"], "fields": {"aggressive?": "bool=false"}, "returns": {"pruned": "u64", "compressed": "u64", "promoted": "u64", "contradictions_linked": "u64"}},
            "setup": {"methods": ["GET","POST"], "fields": {}, "returns": {"dimension": "usize", "message": "string"}},
            "stats": {"methods": ["GET","POST"], "fields": {}, "returns": {"total_engrams": "u64", "contradictions": "u64", "concepts": "u64", "identities": "u64"}},
            "status": {"methods": ["GET","POST"], "fields": {}, "returns": {"storage": "object", "learning": "object", "embedding": "object", "llm": "object"}},
            "train-reranker": {"methods": ["GET","POST"], "fields": {}, "returns": {"model": "string", "pairs": "usize", "trained_samples": "u64"}},
            "help": {"methods": ["GET","POST"], "fields": {}, "returns": "this document"},
            "health": {"methods": ["GET"], "fields": {}, "returns": {"status": "up"}},
        },
        "not_exposed": ["serve", "mcp-server", "mcp-tools"],
        "note": "learning is load-bearing: recall returns recall_id; reward it (score -1.0..1.0) exactly like the local CLI. Unrewarded remote recall == unrewarded local recall.",
    })
}

/// Handle a mirrored CLI endpoint (GET query or POST JSON body).
async fn handle_mirror(
    cli: &Arc<Cli>,
    command: &str,
    method: &hyper::Method,
    req: hyper::Request<hyper::body::Incoming>,
    peer: &str,
) -> hyper::Response<HttpBody> {
    if !matches!(*method, hyper::Method::GET | hyper::Method::POST) {
        return mirror_response(
            405,
            serde_json::json!({"ok": false, "error": "method not allowed (use GET or POST)"}),
        );
    }
    let path = req.uri().path().to_string();
    let query = req.uri().query().map(str::to_string);
    let mut payload = query_to_json(query.as_deref());
    if *method == hyper::Method::POST {
        match http_body_util::BodyExt::collect(req.into_body()).await {
            Ok(body) => {
                let bytes = body.to_bytes();
                if !bytes.is_empty() {
                    match serde_json::from_slice::<serde_json::Value>(&bytes) {
                        Ok(serde_json::Value::Object(map)) => {
                            if let Some(obj) = payload.as_object_mut() {
                                for (k, v) in map {
                                    obj.insert(k, v);
                                }
                            }
                        }
                        Ok(_) => {
                            return mirror_response(
                                400,
                                serde_json::json!({"ok": false, "error": "body must be a JSON object"}),
                            );
                        }
                        Err(e) => {
                            return mirror_response(
                                400,
                                serde_json::json!({"ok": false, "error": format!("bad json: {e}")}),
                            );
                        }
                    }
                }
            }
            Err(e) => {
                return mirror_response(
                    400,
                    serde_json::json!({"ok": false, "error": format!("read body: {e}")}),
                );
            }
        }
    }
    // Log every remote call to stdout (host telemetry reads it).
    eprintln!(
        "engram-api: {} {} from {} at {}",
        method,
        path,
        peer,
        chrono::Utc::now().to_rfc3339()
    );
    if command == "help" {
        return mirror_response(200, serde_json::json!({"ok": true, "data": mirror_help()}));
    }
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("command".to_string(), serde_json::json!(command));
    }
    match execute_cli_value(cli, &payload).await {
        Ok(data) => mirror_response(200, serde_json::json!({"ok": true, "data": data})),
        Err((status, error)) => {
            mnemos_telemetry::global().record("mnemos-mcp-http", "mirror", false, &error);
            mirror_response(status, serde_json::json!({"ok": false, "error": error}))
        }
    }
}

/// JSON response with an explicit HTTP status.
fn mirror_response(status: u16, value: serde_json::Value) -> hyper::Response<HttpBody> {
    hyper::Response::builder()
        .status(hyper::StatusCode::from_u16(status).unwrap_or(hyper::StatusCode::INTERNAL_SERVER_ERROR))
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(Full::new(Bytes::from(serde_json::to_vec(&value).unwrap_or_default())).boxed())
        .expect("mirror json response builds")
}

/// One request against the persistent daemon: auth → local routes
/// (`/health`, `/cli`, `/telemetry*`) → rmcp services → 404.
async fn handle_request(
    protocol_service: StreamableHttpService<Arc<ProtocolTools>, LocalSessionManager>,
    tools_service: StreamableHttpService<mnemos_mcp_tools::MnemosMcpTools, LocalSessionManager>,
    cli_service: StreamableHttpService<mnemos_mcp_server::MnemosServer, LocalSessionManager>,
    rpc_cli: Arc<Cli>,
    token: Option<String>,
    peer: String,
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
    // Mirrored CLI HTTP API (one endpoint per command), in-process.
    if let Some(command) = mirror_command(clean) {
        let method = req.method().clone();
        return Ok(handle_mirror(&rpc_cli, command, &method, req, &peer).await);
    }
    if clean == TOOLS_LIST_PATH {
        if *req.method() != hyper::Method::GET {
            return Ok(hyper::Response::builder()
                .status(hyper::StatusCode::METHOD_NOT_ALLOWED)
                .body(Full::new(Bytes::from_static(b"method not allowed")).boxed())
                .expect("405 builds"));
        }
        return Ok(json_response(tools_catalog()));
    }
    if let Some(resp) = handle_telemetry(req.uri(), req.method()) {
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

/// Compact learning-state summary read from the local reranker model file, so
/// the shell `status` (which forwards here) still shows learning progress.
fn learning_summary() -> serde_json::Value {
    let path = std::env::var("MNEMOS_RERANKER_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "./data/helix/reranker.json".to_string());
    let model = std::fs::read_to_string(&path)
        .ok()
        .and_then(|d| serde_json::from_str::<serde_json::Value>(&d).ok());
    let (state, reward_events) = match &model {
        Some(m) => (
            "local",
            m.get("reward_events")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0),
        ),
        None => ("seed", 0),
    };
    let min_pairs = std::env::var("MNEMOS_RERANKER_MIN_PAIRS")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(500.0);
    let max_alpha = std::env::var("MNEMOS_RERANKER_MAX_ALPHA")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| (0.0..=1.0).contains(v))
        .unwrap_or(0.5);
    let raw = std::env::var("MNEMOS_RERANKER_ALPHA").ok();
    let mode = raw
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("auto");
    let alpha = if mode.eq_ignore_ascii_case("auto") {
        ((reward_events as f64) / min_pairs).clamp(0.0, 1.0) * max_alpha
    } else {
        mode.parse::<f64>()
            .ok()
            .filter(|v| (0.0..=1.0).contains(v))
            .unwrap_or(0.0)
    };
    serde_json::json!({
        "model": state,
        "reward_events": reward_events,
        "alpha": (alpha * 1000.0).round() / 1000.0,
        "alpha_mode": mode,
        "state": if alpha <= 0.0 { "shadow" } else { "active" },
    })
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
/// Returns [`mnemos_core::MnemosError::Http`] if a bind address is invalid,
/// a listener fails to bind, or accepting connections fails. Failures are
/// also recorded via telemetry (`mnemos-mcp-http` / `serve`).
/// Bind the HTTP listener from `MNEMOS_MCP_HOST`/`MNEMOS_MCP_PORT`.
///
/// Call this **before** opening storage: the daemon's embedded DB invalidates
/// any older handle when a second process opens the same path, so a failed
/// start (e.g. port already in use) must fail *here*, not after it has already
/// broken a running daemon.
///
/// # Errors
///
/// Returns [`mnemos_core::MnemosError::Http`] on an invalid address or bind
/// failure (also recorded via telemetry).
pub async fn bind_listener() -> mnemos_core::Result<tokio::net::TcpListener> {
    let host = mcp_host_from_env();
    let port = mcp_port_from_env();
    let addr: SocketAddr = format!("{host}:{port}").parse().map_err(|err| {
        let failure =
            mnemos_core::MnemosError::Http(format!("invalid bind addr {host}:{port}: {err}"));
        record_serve_error(&failure.to_string());
        failure
    })?;
    tokio::net::TcpListener::bind(addr).await.map_err(|err| {
        let failure = mnemos_core::MnemosError::Http(format!("bind {addr}: {err}"));
        record_serve_error(&failure.to_string());
        failure
    })
}

/// Serve all surfaces on an already-bound listener (see [`bind_listener`]).
///
/// # Errors
///
/// Returns [`mnemos_core::MnemosError::Http`] when accepting connections fails.
pub async fn serve_on(
    listener: tokio::net::TcpListener,
    protocol: ProtocolTools,
    cli: Arc<Cli>,
) -> mnemos_core::Result<()> {
    let addr = listener
        .local_addr()
        .map_or_else(|_| "?".to_string(), |a| a.to_string());
    eprintln!("mnemos-mcp-http listening on http://{addr}{PROTOCOL_PATH} (protocol tools)");
    eprintln!("mnemos-mcp-http listening on http://{addr}{TOOLS_PATH} (multi-tool)");
    eprintln!("mnemos-mcp-http listening on http://{addr}{CLI_PATH} (cli single-tool)");
    eprintln!("mnemos-mcp-http listening on http://{addr}{CLI_RPC_PATH} (daemon CLI RPC) + {HEALTH_PATH}");
    eprintln!("mnemos-mcp-http catalog at http://{addr}{TOOLS_LIST_PATH}");
    let token = mcp_token_from_env();
    if token.is_some() {
        eprintln!("mnemos-mcp-http auth: bearer token required for /mcp/* (MNEMOS_MCP_TOKEN set)");
    }
    let services = build_services(protocol, Arc::clone(&cli));

    loop {
        let (stream, peer) = listener.accept().await.map_err(|err| {
            let failure = mnemos_core::MnemosError::Http(format!("accept: {err}"));
            record_serve_error(&failure.to_string());
            failure
        })?;
        let services = services.clone();
        let token = token.clone();
        tokio::spawn(async move {
            serve_stream(TokioIo::new(stream), &services, token, &peer.to_string()).await;
        });
    }
}

/// Bind and serve in one call (convenience; prefer [`bind_listener`] +
/// [`serve_on`] so the bind happens before storage is opened).
///
/// # Errors
///
/// See [`bind_listener`] and [`serve_on`].
pub async fn serve(protocol: ProtocolTools, cli: Arc<Cli>) -> mnemos_core::Result<()> {
    let listener = bind_listener().await?;
    serve_on(listener, protocol, cli).await
}

/// Cloneable bundle of the three rmcp services plus the daemon `Cli`.
#[derive(Clone)]
struct Services {
    protocol_service: StreamableHttpService<Arc<ProtocolTools>, LocalSessionManager>,
    tools_service:
        StreamableHttpService<mnemos_mcp_tools::MnemosMcpTools, LocalSessionManager>,
    cli_service: StreamableHttpService<MnemosServer, LocalSessionManager>,
    rpc_cli: Arc<Cli>,
}

/// Build one instance of each service (factories clone per connection).
fn build_services(protocol: ProtocolTools, cli: Arc<Cli>) -> Services {
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
    Services {
        protocol_service,
        tools_service,
        cli_service,
        rpc_cli: cli,
    }
}

/// Serve one accepted connection through the router.
async fn serve_stream<S>(
    io: TokioIo<S>,
    services: &Services,
    token: Option<String>,
    peer: &str,
) where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let protocol_service = services.protocol_service.clone();
    let tools_service = services.tools_service.clone();
    let cli_service = services.cli_service.clone();
    let rpc_cli = Arc::clone(&services.rpc_cli);
    let peer_label = peer.to_string();
    let peer_for_closure = peer_label.clone();
    let router = hyper::service::service_fn(
        move |req: hyper::Request<hyper::body::Incoming>| {
            let protocol_service = protocol_service.clone();
            let tools_service = tools_service.clone();
            let cli_service = cli_service.clone();
            let rpc_cli = Arc::clone(&rpc_cli);
            let token = token.clone();
            let peer = peer_for_closure.clone();
            handle_request(
                protocol_service,
                tools_service,
                cli_service,
                rpc_cli,
                token,
                peer,
                req,
            )
        },
    );
    if let Err(err) = hyper::server::conn::http1::Builder::new()
        .serve_connection(io, router)
        .await
    {
        eprintln!("mnemos-mcp-http connection from {peer_label} failed: {err}");
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
