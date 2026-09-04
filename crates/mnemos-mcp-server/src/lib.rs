#![recursion_limit = "256"]
//! mnemos-mcp-server: MCP server exposing the MNEMOS memory CLI as ONE tool.
//!
//! The server owns a `std::sync::Arc<mnemos_cli::Cli>` (built by `mnemos-app`)
//! and exposes a single tool, `engram_cli`, over stdio:
//!
//! - `ingest` → `Cli::ingest(text)` → `{ "engram_id": … }`
//! - `recall` → `Cli::recall(query, limit)` → `{ "results": […] }`
//! - `reward` → `Cli::reward(attributions, score)` → `{ "ok": true }`
//! - `consolidate` → `Cli::consolidate()` → `{ "report": … }`
//! - `stats` → `Cli::stats()` → `{ "stats": … }`
//! - `help` → usage text → `{ "help": … }` (no backend touch)
//!
//! Help has two layers: `{"command":"help"}` lists every command (one line
//! each); `{"command":"help","args":["<command>"]}` gives per-command usage
//! (params + example JSON).
//!
//! Shell parity: the same commands exist as `mnemos <command>` on the shell,
//! implemented by the `mnemos-app` binary (`ingest`, `recall`, `reward`,
//! `consolidate`, `stats`).
//!
//! Argument parsing/validation is pure (free functions below) so it can be
//! unit-tested without a server, a database, or pipelines. Only the final
//! dispatch touches the pipelines/storage behind `Cli`.

use std::sync::Arc;

use mnemos_cli::Cli;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// `limit` used for `recall` when the caller omits it.
const DEFAULT_RECALL_LIMIT: usize = 5;

/// Wire parameters for the single `engram_cli` tool.
///
/// `command` is one of `ingest`, `recall`, `reward`, `consolidate`, `stats`,
/// `help`. The remaining fields are per-command; missing required fields are
/// reported as MCP invalid-params errors. `args` carries the optional `help`
/// topic: `{"command":"help","args":["recall"]}`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct MnemosCliParams {
    /// Command to run: `ingest` | `recall` | `reward` | `consolidate` |
    /// `stats` | `help`.
    command: String,
    /// Text to ingest (`ingest`).
    #[serde(default)]
    text: Option<String>,
    /// Query string (`recall`).
    #[serde(default)]
    query: Option<String>,
    /// Max results to return (`recall`). Defaults to 5.
    #[serde(default)]
    limit: Option<usize>,
    /// Per-engram reward attributions (`reward`).
    #[serde(default)]
    attributions: Option<Vec<f64>>,
    /// Reward score (`reward`).
    #[serde(default)]
    score: Option<f64>,
    /// Optional `help` topic, e.g. `["recall"]` (`help`).
    #[serde(default)]
    args: Option<Vec<String>>,
}

/// Parsed `command` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Ingest,
    Recall,
    Reward,
    Consolidate,
    Stats,
    Help,
}

/// Valid `help` topics (the five commands plus `help` itself).
const HELP_TOPICS: [&str; 6] = ["help", "ingest", "recall", "reward", "consolidate", "stats"];

/// Parse the raw `command` string.
///
/// Unknown commands become MCP invalid-params errors.
fn parse_command(raw: &str) -> Result<Command, McpError> {
    match raw {
        "ingest" => Ok(Command::Ingest),
        "recall" => Ok(Command::Recall),
        "reward" => Ok(Command::Reward),
        "consolidate" => Ok(Command::Consolidate),
        "stats" => Ok(Command::Stats),
        "help" => Ok(Command::Help),
        other => Err(McpError::invalid_params(
            format!("unknown command: {other}"),
            None,
        )),
    }
}

/// Missing-required-argument error for `command`.
fn missing(param: &str, command: &str) -> McpError {
    McpError::invalid_params(format!("command '{command}' requires '{param}'"), None)
}

/// Backend failure mapped to an MCP internal error.
fn internal(err: impl std::fmt::Display) -> McpError {
    McpError::internal_error(err.to_string(), None)
}

/// Validated `ingest` arguments: the text to store.
fn ingest_text(params: &MnemosCliParams) -> Result<&str, McpError> {
    params
        .text
        .as_deref()
        .ok_or_else(|| missing("text", "ingest"))
}

/// Validated `recall` arguments: `(query, limit)`.
fn recall_args(params: &MnemosCliParams) -> Result<(&str, usize), McpError> {
    let query = params
        .query
        .as_deref()
        .ok_or_else(|| missing("query", "recall"))?;
    Ok((query, params.limit.unwrap_or(DEFAULT_RECALL_LIMIT)))
}

/// Validated `reward` arguments: `(attributions, score)`.
fn reward_args(params: &MnemosCliParams) -> Result<(&[f64], f64), McpError> {
    let attributions = params
        .attributions
        .as_deref()
        .ok_or_else(|| missing("attributions", "reward"))?;
    let score = params.score.ok_or_else(|| missing("score", "reward"))?;
    Ok((attributions, score))
}

/// Global usage: every command, one line each.
///
/// Same commands exist as `mnemos <command>` on the shell (implemented by
/// the `mnemos-app` binary).
fn global_help() -> String {
    [
        "engram_cli commands (same as `mnemos <command>` on the shell via mnemos-app):",
        "- ingest: store one text episode",
        "- recall: search memory by resonance",
        "- reward: apply a scalar reward signal",
        "- consolidate: run one consolidation (sleep) cycle",
        "- stats: show aggregate memory counts",
        "Call with {\"command\":\"help\",\"args\":[\"<command>\"]} for per-command usage.",
    ]
    .join("\n")
}

/// Detailed usage for one `help` topic (params + example JSON).
fn topic_help(topic: &str) -> Option<String> {
    match topic {
        "help" => Some(
            [
                "help: show usage text.",
                "params: args (optional string array; args[0] is the topic).",
                "example: {\"command\":\"help\",\"args\":[\"recall\"]}",
            ]
            .join("\n"),
        ),
        "ingest" => Some(
            [
                "ingest: store one text episode into memory.",
                "params: text (string, required).",
                "example: {\"command\":\"ingest\",\"text\":\"the sky is blue\"}",
            ]
            .join("\n"),
        ),
        "recall" => Some(
            [
                "recall: search memory by resonance with a natural-language query.",
                "params: query (string, required), limit (integer, optional, default 5).",
                "example: {\"command\":\"recall\",\"query\":\"blue sky\",\"limit\":5}",
            ]
            .join("\n"),
        ),
        "reward" => Some(
            [
                "reward: apply a scalar reward signal with per-engram attributions.",
                "params: attributions (array of numbers, required), score (number, required).",
                "example: {\"command\":\"reward\",\"attributions\":[0.5,0.5],\"score\":1.0}",
            ]
            .join("\n"),
        ),
        "consolidate" => Some(
            [
                "consolidate: run one memory consolidation (sleep) cycle.",
                "params: none.",
                "example: {\"command\":\"consolidate\"}",
            ]
            .join("\n"),
        ),
        "stats" => Some(
            [
                "stats: show aggregate memory counts (engrams, concepts, identities).",
                "params: none.",
                "example: {\"command\":\"stats\"}",
            ]
            .join("\n"),
        ),
        _ => None,
    }
}

/// Resolve a `help` invocation: `None` topic → global usage; known topic →
/// detailed usage; unknown topic → invalid-params listing valid topics.
fn help_for_topic(topic: Option<&str>) -> Result<String, McpError> {
    match topic {
        None => Ok(global_help()),
        Some(name) => topic_help(name).ok_or_else(|| {
            McpError::invalid_params(
                format!(
                    "unknown help topic: {name}. valid topics: {}",
                    HELP_TOPICS.join(", ")
                ),
                None,
            )
        }),
    }
}

/// MCP server exposing the MNEMOS CLI as the single `engram_cli` tool.
pub struct MnemosServer {
    cli: Arc<Cli>,
    tool_router: ToolRouter<Self>,
}

impl MnemosServer {
    /// Build a server around an already-constructed CLI handle.
    #[must_use]
    pub fn new(cli: Arc<Cli>) -> Self {
        Self {
            cli,
            tool_router: Self::tool_router(),
        }
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MnemosServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("MNEMOS memory CLI: ingest/recall/reward/consolidate/stats")
    }
}

#[tool_router(router = tool_router)]
impl MnemosServer {
    /// Single entry point for the MNEMOS memory CLI.
    ///
    /// Two-layer help:
    /// - `{"command":"help"}` → command list (one line each).
    /// - `{"command":"help","args":["<command>"]}` → per-command usage
    ///   (params, types, example JSON) — like `--help` per command.
    ///
    /// Commands: ingest, recall, reward, consolidate, stats.
    /// Shell parity: same commands exist as `mnemos <command>` on the shell.
    #[tool(
        name = "engram_cli",
        description = "Single entry point for the MNEMOS memory CLI. Call {\"command\":\"help\"} for the command list, or {\"command\":\"help\",\"args\":[\"<command>\"]} for per-command usage (params, types, example JSON — like --help per command). Commands: ingest, recall, reward, consolidate, stats. Shell parity: same commands exist as `mnemos <command>` on the shell."
    )]
    async fn engram_cli(
        &self,
        Parameters(params): Parameters<MnemosCliParams>,
    ) -> Result<String, McpError> {
        let payload = match parse_command(&params.command)? {
            Command::Help => {
                let topic = params
                    .args
                    .as_ref()
                    .and_then(|args| args.first())
                    .map(String::as_str);
                let text = help_for_topic(topic)?;
                serde_json::json!({ "help": text })
            }
            Command::Ingest => {
                let id = self
                    .cli
                    .ingest(ingest_text(&params)?)
                    .await
                    .map_err(internal)?;
                serde_json::json!({ "engram_id": id })
            }
            Command::Recall => {
                let (query, limit) = recall_args(&params)?;
                let results = self.cli.recall(query, limit).await.map_err(internal)?;
                serde_json::json!({ "results": results })
            }
            Command::Reward => {
                let (attributions, score) = reward_args(&params)?;
                self.cli
                    .reward(attributions, score)
                    .await
                    .map_err(internal)?;
                serde_json::json!({ "ok": true })
            }
            Command::Consolidate => {
                let report = self.cli.consolidate().await.map_err(internal)?;
                serde_json::json!({ "report": report })
            }
            Command::Stats => {
                let stats = self.cli.stats().await.map_err(internal)?;
                serde_json::json!({ "stats": stats })
            }
        };
        serde_json::to_string(&payload).map_err(internal)
    }
}

/// Serve the CLI as an MCP server over stdio.
///
/// Blocks until the client disconnects. Called by the `mnemos-app` binary.
///
/// # Errors
///
/// Returns [`mnemos_core::MnemosError::Internal`] if the stdio transport fails
/// to start or the service task fails.
pub async fn run(cli: Arc<Cli>) -> mnemos_core::Result<()> {
    let service = MnemosServer::new(cli)
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|err| mnemos_core::MnemosError::Internal(err.to_string()))?;
    service
        .waiting()
        .await
        .map_err(|err| mnemos_core::MnemosError::Internal(err.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn params(value: serde_json::Value) -> MnemosCliParams {
        serde_json::from_value(value).expect("test params must deserialize")
    }

    #[test]
    fn tool_attr_matches_spec() {
        let attr = MnemosServer::engram_cli_tool_attr();
        assert_eq!(attr.name.as_ref(), "engram_cli");
        let desc = attr.description.expect("description present");
        assert!(desc.contains("command\":\"help\""), "desc should mention help");
        assert!(desc.contains("ingest, recall, reward, consolidate, stats"), "desc should list commands");
        assert!(desc.contains("--help"), "desc should mention --help");
        let schema = serde_json::to_value(&attr.input_schema).expect("schema serializes");
        for field in [
            "command",
            "text",
            "query",
            "limit",
            "attributions",
            "score",
            "args",
        ] {
            assert!(
                schema["properties"].get(field).is_some(),
                "input schema missing property `{field}`"
            );
        }
        let required = schema["required"].as_array().expect("schema has required");
        assert!(
            required.iter().any(|v| v == "command"),
            "`command` must be required"
        );
    }

    #[test]
    fn params_deserialize_with_optional_fields_missing() {
        let p = params(json!({ "command": "stats" }));
        assert_eq!(p.command, "stats");
        assert!(p.text.is_none());
        assert!(p.query.is_none());
        assert!(p.limit.is_none());
        assert!(p.attributions.is_none());
        assert!(p.score.is_none());
        assert!(p.args.is_none());
    }

    #[test]
    fn params_deserialize_help_with_args() {
        let p = params(json!({ "command": "help", "args": ["recall"] }));
        assert_eq!(p.command, "help");
        assert_eq!(
            p.args,
            Some(vec!["recall".to_string()])
        );
    }

    #[test]
    fn params_deserialize_full_object() {
        let p = params(json!({
            "command": "reward",
            "text": "x",
            "query": "y",
            "limit": 3,
            "attributions": [0.1, 0.9],
            "score": 0.5,
        }));
        assert_eq!(p.command, "reward");
        assert_eq!(p.text.as_deref(), Some("x"));
        assert_eq!(p.query.as_deref(), Some("y"));
        assert_eq!(p.limit, Some(3));
        assert_eq!(p.attributions, Some(vec![0.1, 0.9]));
        assert_eq!(p.score, Some(0.5));
    }

    #[test]
    fn parse_command_accepts_all_spec_commands() {
        assert_eq!(parse_command("ingest"), Ok(Command::Ingest));
        assert_eq!(parse_command("recall"), Ok(Command::Recall));
        assert_eq!(parse_command("reward"), Ok(Command::Reward));
        assert_eq!(parse_command("consolidate"), Ok(Command::Consolidate));
        assert_eq!(parse_command("stats"), Ok(Command::Stats));
        assert_eq!(parse_command("help"), Ok(Command::Help));
    }

    #[test]
    fn parse_command_rejects_unknown_with_invalid_params() {
        for raw in ["", "forget", "INGEST", "ingest recall"] {
            let err = parse_command(raw).expect_err("unknown command must fail");
            assert_eq!(
                err.code,
                rmcp::model::ErrorCode::INVALID_PARAMS,
                "wrong code for `{raw}`"
            );
        }
    }

    #[test]
    fn ingest_requires_text() {
        let ok = params(json!({ "command": "ingest", "text": "hello" }));
        assert_eq!(ingest_text(&ok), Ok("hello"));
        let missing_arg = params(json!({ "command": "ingest" }));
        let err = ingest_text(&missing_arg).expect_err("missing text must fail");
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn recall_defaults_limit_and_requires_query() {
        let defaulted = params(json!({ "command": "recall", "query": "q" }));
        assert_eq!(recall_args(&defaulted), Ok(("q", DEFAULT_RECALL_LIMIT)));
        let custom = params(json!({ "command": "recall", "query": "q", "limit": 2 }));
        assert_eq!(recall_args(&custom), Ok(("q", 2)));
        let missing_arg = params(json!({ "command": "recall" }));
        let err = recall_args(&missing_arg).expect_err("missing query must fail");
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
    }

    #[test]
    fn reward_requires_attributions_and_score() {
        let ok = params(json!({
            "command": "reward",
            "attributions": [1.0],
            "score": 0.7,
        }));
        assert_eq!(reward_args(&ok), Ok((&[1.0][..], 0.7)));
        let no_attr = params(json!({ "command": "reward", "score": 0.7 }));
        assert_eq!(
            reward_args(&no_attr)
                .expect_err("missing attributions must fail")
                .code,
            rmcp::model::ErrorCode::INVALID_PARAMS
        );
        let no_score = params(json!({ "command": "reward", "attributions": [1.0] }));
        assert_eq!(
            reward_args(&no_score)
                .expect_err("missing score must fail")
                .code,
            rmcp::model::ErrorCode::INVALID_PARAMS
        );
    }

    #[test]
    fn help_no_topic_lists_five_commands() {
        let text = help_for_topic(None).expect("global help must succeed");
        for command in ["ingest", "recall", "reward", "consolidate", "stats"] {
            assert!(
                text.contains(command),
                "global help missing `{command}`: {text}"
            );
        }
    }

    #[test]
    fn help_recall_mentions_query_and_limit() {
        let text = help_for_topic(Some("recall")).expect("recall help must succeed");
        assert!(text.contains("query"), "recall help missing `query`: {text}");
        assert!(text.contains("limit"), "recall help missing `limit`: {text}");
    }

    #[test]
    fn help_unknown_topic_errors_with_valid_topics() {
        let err = help_for_topic(Some("forget")).expect_err("unknown topic must fail");
        assert_eq!(err.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        let message = err.message.to_string();
        for topic in ["ingest", "recall", "reward", "consolidate", "stats"] {
            assert!(
                message.contains(topic),
                "error must list valid topics, missing `{topic}`: {message}"
            );
        }
    }

    #[test]
    fn help_empty_args_means_global_help() {
        let p = params(json!({ "command": "help", "args": [] }));
        let topic = p.args.as_ref().and_then(|args| args.first());
        assert!(topic.is_none());
        help_for_topic(topic.map(String::as_str)).expect("empty args → global help");
    }

    /// Manual smoke test: serve a real CLI over stdio.
    ///
    /// Requires a live `Cli` (pipelines + storage), which is built by
    /// `mnemos-app`, so there is nothing to construct here. Run explicitly via
    /// `cargo test -p mnemos-mcp-server -- --ignored` with a binary harness.
    #[ignore = "needs a live Cli backed by storage; run manually"]
    #[tokio::test]
    async fn serve_stdio_smoke() {
        // Intentionally empty: `run` blocks on stdio, so this documents the
        // manual harness rather than executing it in CI.
    }
}
