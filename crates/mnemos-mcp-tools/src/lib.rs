#![recursion_limit = "256"]
//! mnemos-mcp-tools: MCP server tools over [`Cli`].
//!
//! One MCP tool per [`Cli`] function, served over stdio or HTTP (`/mcp/tools`):
//!
//! | Tool | Params | Returns (JSON string) |
//! |------|--------|---------------------|
//! | `mnemos_ingest` | `{ text }` | `{ "engram_id": … }` |
//! | `mnemos_recall` | `{ query, limit? }` (default `limit` = 10) | `{ "results": […] }` |
//! | `mnemos_reward` | `{ attributions: number[], score }` | `{ "ok": true }` |
//! | `mnemos_consolidate` | `{}` | `{ "report": {…} }` |
//! | `mnemos_stats` | `{}` | `{ "stats": {…} }` |
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

/// Default `limit` for `mnemos_recall` when the caller omits it.
pub const DEFAULT_RECALL_LIMIT: usize = 10;

/// Resolve the serde default for [`RecallParams::limit`].
fn default_recall_limit() -> usize {
    DEFAULT_RECALL_LIMIT
}

/// Map any displayable source error into an MCP protocol error.
fn internal_error(err: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(err.to_string(), None)
}

/// Serialize a payload to the JSON string returned by every tool.
///
/// # Errors
///
/// Returns an [`ErrorData`] internal error if serialization fails.
fn to_json_string(payload: impl Serialize) -> Result<String, ErrorData> {
    serde_json::to_string(&payload).map_err(internal_error)
}

/// Params for `mnemos_ingest`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct IngestParams {
    /// Raw episode text to ingest into memory.
    pub text: String,
}

/// Params for `mnemos_recall`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RecallParams {
    /// Natural-language query to resonate against.
    pub query: String,
    /// Max results; defaults to [`DEFAULT_RECALL_LIMIT`] when omitted.
    #[serde(default = "default_recall_limit")]
    pub limit: usize,
}

impl RecallParams {
    /// Effective result limit (the deserialized `limit`, default 10).
    #[must_use]
    pub fn effective_limit(&self) -> usize {
        self.limit
    }
}

/// Params for `mnemos_reward`.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub struct RewardParams {
    /// Per-engram attribution weights.
    #[serde(default)]
    pub attributions: Vec<f64>,
    /// Scalar reward signal.
    pub score: f64,
    /// Optional ledger recall id from a prior `mnemos_recall` (parallel-safe reward).
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
        name = "mnemos_ingest",
        description = "Ingest a text episode into memory. Returns the new engram id as JSON. Call help with tool=\"mnemos_ingest\" for full usage."
    )]
    pub async fn ingest(
        &self,
        Parameters(params): Parameters<IngestParams>,
    ) -> Result<String, ErrorData> {
        let id = self
            .cli
            .ingest(&params.text)
            .await
            .map_err(internal_error)?;
        to_json_string(serde_json::json!({ "engram_id": id }))
    }

    /// Recall resonant engrams; returns `{ "results": […], "recall_id": … }`.
    #[tool(
        name = "mnemos_recall",
        description = "Recall engrams resonating with a query. Optional limit defaults to 10. Returns results as JSON. Call help with tool=\"mnemos_recall\" for full usage."
    )]
    pub async fn recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<String, ErrorData> {
        let results = self
            .cli
            .recall(&params.query, params.effective_limit())
            .await
            .map_err(internal_error)?;
        let recall_id = self.cli.last_recall_id().await;
        to_json_string(serde_json::json!({ "results": results, "recall_id": recall_id }))
    }

    /// Apply a reward signal; returns `{ "ok": true }`.
    #[tool(
        name = "mnemos_reward",
        description = "Apply a scalar reward signal with per-engram attributions. Returns ok flag as JSON. Call help with tool=\"mnemos_reward\" for full usage."
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
        name = "mnemos_consolidate",
        description = "Run one memory consolidation (sleep) cycle. Returns the consolidation report as JSON. Call help with tool=\"mnemos_consolidate\" for full usage."
    )]
    pub async fn consolidate(&self) -> Result<String, ErrorData> {
        let report = self.cli.consolidate().await.map_err(internal_error)?;
        to_json_string(serde_json::json!({ "report": report }))
    }

    /// Fetch aggregate memory stats; returns `{ "stats": {…} }`.
    #[tool(
        name = "mnemos_stats",
        description = "Fetch aggregate memory stats (engrams, concepts, identities). Returns stats as JSON. Call help with tool=\"mnemos_stats\" for full usage."
    )]
    pub async fn stats(&self) -> Result<String, ErrorData> {
        let stats = self.cli.stats().await.map_err(internal_error)?;
        to_json_string(serde_json::json!({ "stats": stats }))
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
            Some("mnemos_ingest") => help_ingest(),
            Some("mnemos_recall") => help_recall(),
            Some("mnemos_reward") => help_reward(),
            Some("mnemos_consolidate") => help_consolidate(),
            Some("mnemos_stats") => help_stats(),
            Some("help") => help_help(),
            Some(other) => {
                return Err(ErrorData::invalid_params(
                    format!("unknown tool: {other}. valid: mnemos_ingest, mnemos_recall, mnemos_reward, mnemos_consolidate, mnemos_stats, help"),
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
     - mnemos_ingest: Ingest a text episode into memory.\n\
     - mnemos_recall: Recall engrams resonating with a query.\n\
     - mnemos_reward: Apply a scalar reward signal with per-engram attributions.\n\
     - mnemos_consolidate: Run one memory consolidation (sleep) cycle.\n\
     - mnemos_stats: Fetch aggregate memory stats (engrams, concepts, identities).\n\
     - help: Get usage help (this list, or per-tool detail).".to_string()
}

/// Full usage for `mnemos_ingest`.
fn help_ingest() -> String {
    "mnemos_ingest: Ingest a text episode into memory.\n\
     Params:\n\
       text (string, required): Raw episode text to ingest.\n\
     Example: {\"text\": \"the sky is blue\"}\n\
     Returns: {\"engram_id\": 42}".to_string()
}

/// Full usage for `mnemos_recall`.
fn help_recall() -> String {
    "mnemos_recall: Recall engrams resonating with a query.\n\
     Params:\n\
       query (string, required): Natural-language query to resonate against.\n\
       limit (integer, optional, default 10): Max results to return.\n\
     Example: {\"query\": \"blue sky\", \"limit\": 5}\n\
     Returns: {\"results\": [...]}".to_string()
}

/// Full usage for `mnemos_reward`.
fn help_reward() -> String {
    "mnemos_reward: Apply a scalar reward signal with per-engram attributions.\n\
      Params:\n\
        attributions (array of numbers, optional, default []): Per-engram attribution weights.\n\
        score (number, required): Scalar reward signal.\n\
        recall_id (integer, optional): Ledger recall id from a prior mnemos_recall (parallel-safe reward).\n\
      Example: {\"attributions\": [0.5, 0.5], \"score\": 1.0}\n\
      Example (ledger): {\"recall_id\": 42, \"score\": 1.0}\n\
      Returns: {\"ok\": true}".to_string()
}

/// Full usage for `mnemos_consolidate`.
fn help_consolidate() -> String {
    "mnemos_consolidate: Run one memory consolidation (sleep) cycle.\n\
     Params: none.\n\
     Example: {}\n\
     Returns: {\"report\": {\"pruned\": 0, \"compressed\": 0, \"promoted\": 0, \"contradictions_linked\": 0}}".to_string()
}

/// Full usage for `mnemos_stats`.
fn help_stats() -> String {
    "mnemos_stats: Fetch aggregate memory stats (engrams, concepts, identities).\n\
     Params: none.\n\
     Example: {}\n\
     Returns: {\"stats\": {\"total_engrams\": 0, \"contradictions\": 0, \"concepts\": 0, \"identities\": 0}}".to_string()
}

/// Full usage for `help` itself.
fn help_help() -> String {
    "help: Get usage help for mnemos memory tools.\n\
     Params:\n\
       tool (string, optional): Tool name to get detailed usage for. Omit for a full tool list.\n\
     Example: {\"tool\": \"mnemos_recall\"}\n\
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
            serde_json::from_value(json!({ "tool": "mnemos_recall" }))
                .expect("tool param must deserialize");
        assert_eq!(params.tool.as_deref(), Some("mnemos_recall"));
    }

    #[test]
    fn tool_list_mentions_all_tools() {
        let list = tool_list();
        for name in [
            "mnemos_ingest",
            "mnemos_recall",
            "mnemos_reward",
            "mnemos_consolidate",
            "mnemos_stats",
            "help",
        ] {
            assert!(list.contains(name), "tool list should mention {name}");
        }
    }

    #[test]
    fn per_tool_help_includes_params_and_example() {
        for (name, help_text) in [
            ("mnemos_ingest", help_ingest()),
            ("mnemos_recall", help_recall()),
            ("mnemos_reward", help_reward()),
            ("mnemos_consolidate", help_consolidate()),
            ("mnemos_stats", help_stats()),
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
