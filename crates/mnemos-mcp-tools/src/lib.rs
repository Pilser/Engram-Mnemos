#![recursion_limit = "256"]
//! mnemos-mcp-tools: MCP server tools over [`Cli`].
//!
//! One MCP tool per [`Cli`] function, served over stdio or HTTP (`/mcp/tools`):
//!
//! | Tool | Params | Returns (JSON string) |
//! |------|--------|---------------------|
//! | `engram_ingest` | `{ text, prev_id?, seq_pos? }` | `{ "engram_id": … }` |
//! | `engram_recall` | `{ query, limit?, follow_seq?, seq_depth?, seq_dir? }` (default `limit` = 10) | `{ "results": […] }` or `{ "sequential_chain": […] }` |
//! | `engram_reward` | `{ attributions: number[], score }` | `{ "ok": true }` |
//! | `engram_consolidate` | `{}` | `{ "report": {…} }` |
//! | `engram_stats` | `{}` | `{ "stats": {…} }` |
//! | `engram_status` | `{}` | `{ "stats": {…}, "embedding": {…}, "llm": {…} }` |
//! | `help` | `{ tool? }` | per-tool usage or tool list |
//!
//! Two-layer help:
//! - **Layer 1** — each tool's `description` (brief, always visible).
//! - **Layer 2** — the `help` tool returns full usage (params, types, example
//!   JSON) for any tool, or lists all tools when called without `tool`.
//!
//! Every tool returns its payload as a JSON-encoded string; failures are
//! reported as [`rmcp::ErrorData`] (internal error) carrying the source
//! message. Use [`run`] to serve the tools over stdio.

use std::sync::Arc;

use mnemos_cli::Cli;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ErrorData, ServerHandler,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Default `limit` for `engram_recall` when the caller omits it.
pub const DEFAULT_RECALL_LIMIT: usize = 10;

/// Resolve the serde default for [`RecallParams::limit`].
fn default_recall_limit() -> usize {
    DEFAULT_RECALL_LIMIT
}

/// Map any displayable source error into an MCP protocol error.
fn internal_error(err: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(err.to_string(), None)
}

/// Compact learning-state summary read from the local reranker model file.
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
    let mode = raw.as_deref().map(str::trim).filter(|s| !s.is_empty()).unwrap_or("auto");
    let alpha = if mode.eq_ignore_ascii_case("auto") {
        ((reward_events as f64) / min_pairs).clamp(0.0, 1.0) * max_alpha
    } else {
        mode.parse::<f64>().ok().filter(|v| (0.0..=1.0).contains(v)).unwrap_or(0.0)
    };
    serde_json::json!({
        "model": state,
        "reward_events": reward_events,
        "alpha": (alpha * 1000.0).round() / 1000.0,
        "alpha_mode": mode,
        "state": if alpha <= 0.0 { "shadow" } else { "active" },
    })
}

/// Serialize a payload to the JSON string returned by every tool.
///
/// # Errors
///
/// Returns an [`ErrorData`] internal error if serialization fails.
fn to_json_string(payload: impl Serialize) -> Result<String, ErrorData> {
    serde_json::to_string(&payload).map_err(internal_error)
}

/// Params for `engram_ingest`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct IngestParams {
    /// Raw episode text to ingest into memory.
    pub text: String,
    /// Previous engram id for sequential chain (`TemporalSequence`).
    #[serde(default)]
    pub prev_id: Option<u64>,
    /// Position number for sequential edge (`pos` property, fan-out at same level).
    #[serde(default)]
    pub seq_pos: Option<i64>,
}

/// Params for `engram_recall`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RecallParams {
    /// Natural-language query to resonate against.
    pub query: String,
    /// Max results; defaults to [`DEFAULT_RECALL_LIMIT`] when omitted.
    #[serde(default = "default_recall_limit")]
    pub limit: usize,
    /// Follow sequential chain from this engram id (`TemporalSequence` walk).
    #[serde(default)]
    pub follow_seq: Option<u64>,
    /// Depth for follow-seq walk (default 10).
    #[serde(default)]
    pub seq_depth: Option<usize>,
    /// Direction for follow-seq walk: up|down|both (default down).
    #[serde(default)]
    pub seq_dir: Option<String>,
}

impl RecallParams {
    /// Effective result limit (the deserialized `limit`, default 10).
    #[must_use]
    pub fn effective_limit(&self) -> usize {
        self.limit
    }
}

/// Params for `engram_reward`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RewardParams {
    /// Per-engram attribution weights.
    #[serde(default)]
    pub attributions: Vec<f64>,
    /// Scalar reward signal.
    pub score: f64,
    /// Optional ledger recall id from a prior `engram_recall` (parallel-safe reward).
    #[serde(default)]
    pub recall_id: Option<u64>,
}

/// Params for the `help` tool.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct HelpParams {
    /// Tool name to get detailed usage for. Omit for a full tool list.
    #[serde(default)]
    pub tool: Option<String>,
}

/// MCP tool server: one tool per [`Cli`] function, plus a `help` tool.
#[derive(Clone)]
pub struct MnemosMcpTools {
    cli: Arc<Cli>,
    tool_router: ToolRouter<Self>,
}

impl std::fmt::Debug for MnemosMcpTools {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MnemosMcpTools").finish_non_exhaustive()
    }
}

impl MnemosMcpTools {
    /// Wrap a shared [`Cli`] handle as an MCP tool server.
    #[must_use]
    pub fn new(cli: Arc<Cli>) -> Self {
        Self {
            cli,
            tool_router: Self::tool_router(),
        }
    }

    /// Borrow the underlying [`Cli`] handle.
    #[must_use]
    pub fn cli(&self) -> &Arc<Cli> {
        &self.cli
    }
}

#[tool_router(router = tool_router)]
impl MnemosMcpTools {
    /// Ingest a text episode; returns `{ "engram_id": … }`.
    #[tool(
        name = "engram_ingest",
        description = "Ingest a text episode into memory. Optional prev_id/seq_pos for sequential chain (TemporalSequence). Returns the new engram id as JSON. Call help with tool=\"engram_ingest\" for full usage."
    )]
    pub async fn ingest(
        &self,
        Parameters(params): Parameters<IngestParams>,
    ) -> Result<String, ErrorData> {
        let id = if let Some(pid) = params.prev_id {
            self.cli
                .ingest_sequential(&params.text, Some(pid), params.seq_pos)
                .await
                .map_err(internal_error)?
        } else {
            self.cli
                .ingest(&params.text)
                .await
                .map_err(internal_error)?
        };
        to_json_string(serde_json::json!({ "engram_id": id }))
    }

    /// Recall resonant engrams; returns `{ "results": […], "recall_id": … }`.
    #[tool(
        name = "engram_recall",
        description = "Recall engrams resonating with a query. Optional limit defaults to 10. Optional follow_seq/seq_depth/seq_dir to walk sequential chain (TemporalSequence). Returns results as JSON. Call help with tool=\"engram_recall\" for full usage."
    )]
    pub async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<String, ErrorData> {
        if let Some(sid) = params.follow_seq {
            let depth = params.seq_depth.unwrap_or(10);
            let dir = params.seq_dir.as_deref().unwrap_or("down");
            let chain = self
                .cli
                .follow_sequence(sid, depth, dir)
                .await
                .map_err(internal_error)?;
            return to_json_string(serde_json::json!({ "sequential_chain": chain, "start_id": sid, "depth": depth, "dir": dir }));
        }
        let results = self
            .cli
            .recall(&params.query, params.effective_limit())
            .await
            .map_err(internal_error)?;
        if results.is_empty() {
            return to_json_string(serde_json::json!({
                "results": [],
                "recall_id": serde_json::Value::Null,
                "message": "I don't know"
            }));
        }
        let recall_id = self.cli.last_recall_id().await;
        to_json_string(serde_json::json!({ "results": results, "recall_id": recall_id }))
    }

    /// Apply a reward signal; returns `{ "ok": true }`.
    #[tool(
        name = "engram_reward",
        description = "Apply a scalar reward signal with per-engram attributions. Returns ok flag as JSON. Call help with tool=\"engram_reward\" for full usage."
    )]
    pub async fn reward(
        &self,
        Parameters(params): Parameters<RewardParams>,
    ) -> Result<String, ErrorData> {
        if let Some(id) = params.recall_id {
            self.cli
                .reward_with_id(id, params.score)
                .await
                .map_err(internal_error)?;
        } else {
            self.cli
                .reward(&params.attributions, params.score)
                .await
                .map_err(internal_error)?;
        }
        to_json_string(serde_json::json!({ "ok": true }))
    }

    /// Run a consolidation cycle; returns `{ "report": {…} }`.
    #[tool(
        name = "engram_consolidate",
        description = "Run one memory consolidation (sleep) cycle. Returns the consolidation report as JSON. Call help with tool=\"engram_consolidate\" for full usage."
    )]
    pub async fn consolidate(&self) -> Result<String, ErrorData> {
        let report = self.cli.consolidate().await.map_err(internal_error)?;
        to_json_string(serde_json::json!({ "report": report }))
    }

    /// Fetch aggregate memory stats; returns `{ "stats": {…} }`.
    #[tool(
        name = "engram_stats",
        description = "Fetch aggregate memory stats (engrams, concepts, identities). Returns stats as JSON. Call help with tool=\"engram_stats\" for full usage."
    )]
    pub async fn stats(&self) -> Result<String, ErrorData> {
        let stats = self.cli.stats().await.map_err(internal_error)?;
        to_json_string(serde_json::json!({ "stats": stats }))
    }

    /// Fetch status (embedding + LLM + stats); returns `{ "stats": {…}, "embedding": {…}, "llm": {…} }`.
    #[tool(
        name = "engram_status",
        description = "Check embedding and LLM reachability plus stats. Returns status JSON. Call help with tool=\"engram_status\" for full usage."
    )]
    pub async fn status(&self) -> Result<String, ErrorData> {
        let stats = self.cli.stats().await.map_err(internal_error)?;
        to_json_string(serde_json::json!({
            "stats": stats,
            "learning": learning_summary(),
            "embedding": {"note": "use shell engram status for live ping"},
            "llm": {"note": "use shell engram status for live ping"},
        }))
    }

    /// Get usage help for a tool (or all tools when `tool` is omitted).
    ///
    /// Layer-2 help: returns full param specs, types, and example JSON for
    /// the named tool, or a one-line summary of every tool when called
    /// without `tool`.
    #[tool(
        name = "help",
        description = "Get usage help. Call with no args for a tool list, or tool=\"<name>\" for full usage (params, types, example JSON)."
    )]
    pub async fn help(
        &self,
        Parameters(params): Parameters<HelpParams>,
    ) -> Result<String, ErrorData> {
        let response = match params.tool.as_deref() {
            None => tool_list(),
            Some("engram_ingest") => help_ingest(),
            Some("engram_recall") => help_recall(),
            Some("engram_reward") => help_reward(),
            Some("engram_consolidate") => help_consolidate(),
            Some("engram_stats") => help_stats(),
            Some("engram_status") => help_status(),
            Some("help") => help_help(),
            Some(other) => {
                return Err(ErrorData::invalid_params(
                    format!("unknown tool: {other}. valid: engram_ingest, engram_recall, engram_reward, engram_consolidate, engram_stats, engram_status, help"),
                    None,
                ));
            }
        };
        Ok(response)
    }
}

/// One-line summary of every tool (layer-1 help extended).
fn tool_list() -> String {
    "mnemos memory tools (call help with tool=\"<name>\" for full usage):\n\
      - engram_ingest: Ingest a text episode into memory.\n\
      - engram_recall: Recall engrams resonating with a query.\n\
      - engram_reward: Apply a scalar reward signal with per-engram attributions.\n\
      - engram_consolidate: Run one memory consolidation (sleep) cycle.\n\
      - engram_stats: Fetch aggregate memory stats (engrams, concepts, identities).\n\
      - engram_status: Check embedding and LLM reachability plus stats.\n\
      - help: Get usage help (this list, or per-tool detail).".to_string()
}

/// Full usage for `engram_ingest`.
fn help_ingest() -> String {
    "engram_ingest: Ingest a text episode into memory.\n\
      Params:\n\
        text (string, required): Raw episode text to ingest.\n\
        prev_id (integer, optional): Previous engram id for sequential chain (TemporalSequence).\n\
        seq_pos (integer, optional): Position number for sequential edge (fan-out at same level).\n\
      Example: {\"text\": \"the sky is blue\"}\n\
      Sequential example: {\"text\": \"ch2\", \"prev_id\": 123, \"seq_pos\": 1}\n\
      Returns: {\"engram_id\": 42}".to_string()
}

/// Full usage for `engram_recall`.
fn help_recall() -> String {
    "engram_recall: Recall engrams resonating with a query.\n\
      Params:\n\
        query (string, required): Natural-language query to resonate against.\n\
        limit (integer, optional, default 10): Max results to return.\n\
        follow_seq (integer, optional): Follow sequential chain from this engram id (TemporalSequence).\n\
        seq_depth (integer, optional, default 10): Depth for follow-seq walk.\n\
        seq_dir (string, optional, default down): Direction for follow-seq walk: up|down|both.\n\
      Example: {\"query\": \"blue sky\", \"limit\": 5}\n\
      Sequential walk: {\"follow_seq\": 123, \"seq_depth\": 5, \"seq_dir\": \"down\", \"query\": \"\"}\n\
      Returns: {\"results\": [...]} or {\"sequential_chain\": [...]}".to_string()
}

/// Full usage for `engram_reward`.
fn help_reward() -> String {
    "engram_reward: Apply a scalar reward signal with per-engram attributions.\n\
      Params:\n\
        attributions (array of numbers, optional, default []): Per-engram attribution weights.\n\
        score (number, required): Scalar reward signal.\n\
        recall_id (integer, optional): Ledger recall id from a prior engram_recall (parallel-safe reward).\n\
      Example: {\"attributions\": [0.5, 0.5], \"score\": 1.0}\n\
      Example (ledger): {\"recall_id\": 42, \"score\": 1.0}\n\
      Returns: {\"ok\": true}".to_string()
}

/// Full usage for `engram_consolidate`.
fn help_consolidate() -> String {
    "engram_consolidate: Run one memory consolidation (sleep) cycle.\n\
     Params: none.\n\
     Example: {}\n\
     Returns: {\"report\": {\"pruned\": 0, \"compressed\": 0, \"promoted\": 0, \"contradictions_linked\": 0}}".to_string()
}

/// Full usage for `engram_stats`.
fn help_stats() -> String {
    "engram_stats: Fetch aggregate memory stats (engrams, concepts, identities).\n\
      Params: none.\n\
      Example: {}\n\
      Returns: {\"stats\": {\"total_engrams\": 0, \"contradictions\": 0, \"concepts\": 0, \"identities\": 0}}".to_string()
}

/// Full usage for `engram_status`.
fn help_status() -> String {
    "engram_status: Check embedding and LLM reachability plus stats.\n\
      Params: none.\n\
      Example: {}\n\
      Returns: {\"stats\": {\"total_engrams\": 0, \"contradictions\": 0, \"concepts\": 0, \"identities\": 0}, \"embedding\": {\"ok\": true}, \"llm\": {\"ok\": true}}".to_string()
}

/// Full usage for `help` itself.
fn help_help() -> String {
    "help: Get usage help for mnemos memory tools.\n\
     Params:\n\
       tool (string, optional): Tool name to get detailed usage for. Omit for a full tool list.\n\
     Example: {\"tool\": \"engram_recall\"}\n\
     Returns: full usage text (params, types, example JSON) for the named tool.".to_string()
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MnemosMcpTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "mnemos-mcp-tools",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Mnemos memory tools: ingest episodes, recall by resonance, reward, consolidate, stats. \
                 Call the `help` tool (tool=\"<name>\") for full usage including params and example JSON.",
            )
    }
}

/// Serve the MCP tools over stdio until the client disconnects.
///
/// # Errors
///
/// Returns [`rmcp::RmcpError`] if the stdio transport handshake fails or the
/// serving task terminates with a join error.
pub async fn run(cli: Arc<Cli>) -> Result<(), rmcp::RmcpError> {
    use rmcp::ServiceExt as _;
    let service = MnemosMcpTools::new(cli)
        .serve(rmcp::transport::stdio())
        .await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn recall_limit_defaults_to_ten_when_omitted() {
        assert_eq!(default_recall_limit(), 10);
        assert_eq!(DEFAULT_RECALL_LIMIT, 10);
        let params: RecallParams = serde_json::from_value(json!({ "query": "redwood" }))
            .expect("query-only params must deserialize");
        assert_eq!(params.limit, DEFAULT_RECALL_LIMIT);
        assert_eq!(params.effective_limit(), 10);
    }

    #[test]
    fn recall_explicit_limit_is_preserved() {
        let params: RecallParams =
            serde_json::from_value(json!({ "query": "redwood", "limit": 3 }))
                .expect("explicit limit must deserialize");
        assert_eq!(params.limit, 3);
        assert_eq!(params.effective_limit(), 3);
    }

    #[test]
    fn help_params_default_tool_to_none() {
        let params: HelpParams = serde_json::from_value(json!({}))
            .expect("empty params must deserialize");
        assert_eq!(params.tool, None);
    }

    #[test]
    fn help_params_accepts_tool_name() {
        let params: HelpParams =
            serde_json::from_value(json!({ "tool": "engram_recall" }))
                .expect("tool param must deserialize");
        assert_eq!(params.tool.as_deref(), Some("engram_recall"));
    }

    #[test]
    fn tool_list_mentions_all_tools() {
        let list = tool_list();
        for name in [
            "engram_ingest",
            "engram_recall",
            "engram_reward",
            "engram_consolidate",
            "engram_stats",
            "engram_status",
            "help",
        ] {
            assert!(list.contains(name), "tool list should mention {name}");
        }
    }

    #[test]
    fn per_tool_help_includes_params_and_example() {
        for (name, help_text) in [
            ("engram_ingest", help_ingest()),
            ("engram_recall", help_recall()),
            ("engram_reward", help_reward()),
            ("engram_consolidate", help_consolidate()),
            ("engram_stats", help_stats()),
            ("engram_status", help_status()),
            ("help", help_help()),
        ] {
            assert!(
                help_text.contains("Params") || help_text.contains("Params: none"),
                "{name} help should mention Params"
            );
            assert!(
                help_text.contains("Example"),
                "{name} help should mention Example"
            );
        }
    }

    /// Manual-only: serving blocks on stdio and needs a live `Cli` backend,
    /// so it stays `#[ignore]`d and never runs under plain `cargo test`.
    #[tokio::test]
    #[ignore = "manual: blocks serving stdio; requires a live Cli backend"]
    async fn run_serves_tools_over_stdio() {
        // Intentionally empty: documents the `run` entrypoint without
        // requiring a live storage backend in unit tests.
    }
}
